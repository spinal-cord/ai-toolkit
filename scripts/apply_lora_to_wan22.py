"""
Apply a LoRA to both HIGH and LOW noise Wan 2.2 I2V 14B models and save the merged models.

Usage:
    # Using a single folder and single LoRA for both models:
    python scripts/apply_lora_to_wan22.py \
        --model-path /path/to/folder/with/safetensors \
        --lora-path /path/to/lora.safetensors \
        --output-path /path/to/output/folder \
        --scale 1.0

    # Using separate model paths and separate LoRAs for high/low:
    python scripts/apply_lora_to_wan22.py \
        --model-path-high /path/to/high_model.safetensors \
        --model-path-low /path/to/low_model.safetensors \
        --lora-path-high /path/to/lora_high.safetensors \
        --lora-path-low /path/to/lora_low.safetensors \
        --output-path /path/to/output/folder \
        --scale 1.0
"""

import argparse
import gc
import os
import re
import sys
import json
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from diffusers import WanTransformer3DModel

# Add repo root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from extensions_built_in.diffusion_models.wan22.wan22_14b_model import (
    find_safetensors_files_local,
    load_transformer_from_safetensors,
    download_config_for_model,
)


BLOCK_INDEX_RE = re.compile(r"blocks\.(\d+)")


def count_blocks_from_safetensors(safetensors_path: str) -> int:
    """
    Return the number of transformer blocks in a model safetensors file (max block index + 1).

    Reads only the header (no tensor data) so it is cheap. Returns 0 if no block keys are
    found (caller should then fall back to the config's num_layers).
    """
    block_indices = set()
    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            match = BLOCK_INDEX_RE.search(key)
            if match:
                block_indices.add(int(match.group(1)))
    return (max(block_indices) + 1) if block_indices else 0


def compute_fp32_prefixes(num_blocks: int, fp32_front_layers: int, fp32_tail_layers: int) -> set:
    """
    Build the set of diffusers module-name prefixes that should be kept in fp32.

    This encodes the desired mixed-precision layout:
      - front: condition_embedder (text/time embeddings + time projection) + patch_embedding
      - tail:  proj_out + norm_out + head scale_shift_table
      - the first ``fp32_front_layers`` blocks (entire)
      - the last ``fp32_tail_layers`` blocks (entire)
      - every block's attention norms, norm2 and per-block scale_shift_table (modulation)
    """
    prefixes = set()

    # Front / condition (text embeddings, time embeddings, time projection, patch embedding)
    prefixes.add("condition_embedder")
    prefixes.add("patch_embedding")

    # Tail (output projection, output norm, head modulation)
    prefixes.add("proj_out")
    prefixes.add("norm_out")
    prefixes.add("scale_shift_table")

    num_blocks = max(num_blocks, 0)

    # First N blocks (entire)
    front = min(max(fp32_front_layers, 0), num_blocks)
    for i in range(front):
        prefixes.add(f"blocks.{i}")

    # Last M blocks (entire)
    tail = min(max(fp32_tail_layers, 0), num_blocks)
    for i in range(max(0, num_blocks - tail), num_blocks):
        prefixes.add(f"blocks.{i}")

    # Norms + per-block modulation in EVERY block (so middle blocks' norms also stay fp32)
    for i in range(num_blocks):
        prefixes.add(f"blocks.{i}.attn1.norm_q")
        prefixes.add(f"blocks.{i}.attn1.norm_k")
        prefixes.add(f"blocks.{i}.attn2.norm_q")
        prefixes.add(f"blocks.{i}.attn2.norm_k")
        prefixes.add(f"blocks.{i}.norm2")
        prefixes.add(f"blocks.{i}.scale_shift_table")

    return prefixes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply LoRA to both HIGH and LOW noise Wan 2.2 14B models"
    )
    
    # Model paths
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local folder containing high and low noise .safetensors files",
    )
    parser.add_argument(
        "--model-path-high",
        type=str,
        default=None,
        help="Path to HIGH noise .safetensors file (or folder containing it)",
    )
    parser.add_argument(
        "--model-path-low",
        type=str,
        default=None,
        help="Path to LOW noise .safetensors file (or folder containing it)",
    )
    
    # LoRA paths
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA .safetensors file (BF16) to apply to both models",
    )
    parser.add_argument(
        "--lora-path-high",
        type=str,
        default=None,
        help="Path to LoRA .safetensors file for HIGH noise model",
    )
    parser.add_argument(
        "--lora-path-low",
        type=str,
        default=None,
        help="Path to LoRA .safetensors file for LOW noise model",
    )
    
    # Other arguments
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output directory for merged models. Defaults to same folder as input.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="LoRA scale/multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help=(
            "Base dtype for the merged model. By default, text/time embeddings, projections, "
            "norms and the front/tail blocks are additionally kept in float32 (see the "
            "--fp32-* flags). Use --no-fp32 to force the entire model into this dtype. "
            "(default: bfloat16)"
        ),
    )
    parser.add_argument(
        "--fp32-front-layers",
        type=int,
        default=1,
        help="Number of leading transformer blocks to keep in float32 (default: 1)",
    )
    parser.add_argument(
        "--fp32-tail-layers",
        type=int,
        default=4,
        help="Number of trailing transformer blocks to keep in float32 (default: 4)",
    )
    parser.add_argument(
        "--no-fp32",
        action="store_true",
        help=(
            "Disable the automatic float32 layout. The entire model is saved in --dtype "
            "(no fp32 front/tail/embedding/norm layers)."
        ),
    )
    
    args = parser.parse_args()
    
    # --- Validation Logic ---
    
    # Validate Model Paths
    if args.model_path_high is not None or args.model_path_low is not None:
        if args.model_path_high is None or args.model_path_low is None:
            print("Error: Both --model-path-high and --model-path-low must be provided together.")
            sys.exit(1)
    else:
        if args.model_path is None:
            print("Error: Either --model-path or both --model-path-high and --model-path-low must be provided.")
            sys.exit(1)

    # Validate LoRA Paths
    if args.lora_path_high is not None or args.lora_path_low is not None:
        if args.lora_path_high is None or args.lora_path_low is None:
            print("Error: Both --lora-path-high and --lora-path-low must be provided together.")
            sys.exit(1)
    else:
        if args.lora_path is None:
            print("Error: Either --lora-path or both --lora-path-high and --lora-path-low must be provided.")
            sys.exit(1)
            
    return args


def load_config_from_path(safetensors_path: str, model_path: str = None) -> dict:
    """
    Try to load config.json from the same folder as the safetensors file.
    Falls back to default Wan 2.2 14B config if not found.
    """
    config_path = os.path.join(os.path.dirname(safetensors_path), "config.json")

    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            return json.load(f)

    print("config.json not found, using default Wan 2.2 14B config")
    return download_config_for_model("", "config.json")


def lora_key_to_model_key(lora_key: str) -> str:
    """
    Map a LoRA tensor key to the corresponding Wan 2.2 Diffusers model tensor key.
    
    Handles both legacy kohya format and the new per-expert format (with transformer_N. prefix).
    
    The model's state_dict (after _process_state_dict_for_fp8 remapping) has:
    - No "diffusion_model." prefix (stripped during loading)
    - cross_attn -> attn2 (replacement, NOT nesting)
    - self_attn -> attn1 (replacement, NOT nesting)
    - ffn.0/ffn.2 -> ffn.net.0.proj / ffn.net.2
    - head.head -> proj_out
    - text_embedding.{0,2} -> condition_embedder.text_embedder.linear_{1,2}
    - time_embedding.{0,2} -> condition_embedder.time_embedder.linear_{1,2}
    - time_projection.1 -> condition_embedder.time_proj
    """
    # Remove transformer_N. prefix if present (per-expert format from lora_special.py)
    lora_key = re.sub(r'^diffusion_model\.transformer_\d+\.', 'diffusion_model.', lora_key)
    
    # Handle kohya-style keys: lora_unet_blocks_N_sub_key
    if "_blocks_" in lora_key:
        match = re.match(r"lora_unet_blocks_(\d+)_(.+)", lora_key)
        if match:
            block_idx = match.group(1)
            remainder = match.group(2)
            
            if remainder.startswith("cross_attn_"):
                attn_type = "attn2"
                sub = remainder[len("cross_attn_"):]
                if sub == "k": return f"blocks.{block_idx}.{attn_type}.to_k"
                elif sub == "q": return f"blocks.{block_idx}.{attn_type}.to_q"
                elif sub == "v": return f"blocks.{block_idx}.{attn_type}.to_v"
                elif sub == "o": return f"blocks.{block_idx}.{attn_type}.to_out.0"
            elif remainder.startswith("self_attn_"):
                attn_type = "attn1"
                sub = remainder[len("self_attn_"):]
                if sub == "k": return f"blocks.{block_idx}.{attn_type}.to_k"
                elif sub == "q": return f"blocks.{block_idx}.{attn_type}.to_q"
                elif sub == "v": return f"blocks.{block_idx}.{attn_type}.to_v"
                elif sub == "o": return f"blocks.{block_idx}.{attn_type}.to_out.0"
            elif remainder == "ffn_0":
                return f"blocks.{block_idx}.ffn.net.0.proj"
            elif remainder == "ffn_2":
                return f"blocks.{block_idx}.ffn.net.2"
    
    # Handle new per-expert format: diffusion_model.blocks.N.sub_key.sub_sub_key
    if "diffusion_model.blocks." in lora_key:
        match = re.match(r"diffusion_model\.blocks\.(\d+)\.(.+)", lora_key)
        if match:
            block_idx = match.group(1)
            remainder = match.group(2)
            
            # Handle attention keys -> actual linear projection layers
            if remainder.startswith("cross_attn."):
                sub = remainder[len("cross_attn."):]
                if sub == "k": return f"blocks.{block_idx}.attn2.to_k"
                elif sub == "q": return f"blocks.{block_idx}.attn2.to_q"
                elif sub == "v": return f"blocks.{block_idx}.attn2.to_v"
                elif sub == "o": return f"blocks.{block_idx}.attn2.to_out.0"
                # Handle .diff tensors for norm_k/norm_q (additive adjustments)
                elif sub == "norm_k": return f"blocks.{block_idx}.attn2.norm_k"
                elif sub == "norm_q": return f"blocks.{block_idx}.attn2.norm_q"
            elif remainder.startswith("self_attn."):
                sub = remainder[len("self_attn."):]
                if sub == "k": return f"blocks.{block_idx}.attn1.to_k"
                elif sub == "q": return f"blocks.{block_idx}.attn1.to_q"
                elif sub == "v": return f"blocks.{block_idx}.attn1.to_v"
                elif sub == "o": return f"blocks.{block_idx}.attn1.to_out.0"
                # Handle .diff tensors for norm_k/norm_q (additive adjustments)
                elif sub == "norm_k": return f"blocks.{block_idx}.attn1.norm_k"
                elif sub == "norm_q": return f"blocks.{block_idx}.attn1.norm_q"
            elif remainder.startswith("ffn."):
                sub = remainder[len("ffn."):]
                if sub == "0": return f"blocks.{block_idx}.ffn.net.0.proj"
                elif sub == "2": return f"blocks.{block_idx}.ffn.net.2"
            # Handle .diff/.diff_b for norm2
            elif remainder.startswith("norm2"):
                return f"blocks.{block_idx}.norm2"
    
    # New format: diffusion_model.condition_embedder.text_embedder.linear_{idx}
    match = re.match(r"diffusion_model\.condition_embedder\.text_embedder\.linear_(\d+)", lora_key)
    if match:
        idx = match.group(1)
        return f"condition_embedder.text_embedder.linear_{idx}"
    
    # New format: diffusion_model.condition_embedder.time_embedder.linear_{idx}
    match = re.match(r"diffusion_model\.condition_embedder\.time_embedder\.linear_(\d+)", lora_key)
    if match:
        idx = match.group(1)
        return f"condition_embedder.time_embedder.linear_{idx}"
    
    # New format: diffusion_model.condition_embedder.time_proj
    if lora_key.startswith("diffusion_model.condition_embedder.time_proj"):
        return "condition_embedder.time_proj"
    
    # New format: diffusion_model.proj_out
    if lora_key.startswith("diffusion_model.proj_out"):
        return "proj_out"
    
    # New format: diffusion_model.patch_embedding (for .diff and .diff_b)
    if lora_key.startswith("diffusion_model.patch_embedding"):
        return "patch_embedding"
    
    # Legacy kohya format for head
    if lora_key == "lora_unet_head_head":
        return "proj_out"
    
    # Legacy kohya format for text embedding
    match = re.match(r"lora_unet_text_embedding_(\d+)", lora_key)
    if match:
        idx = match.group(1)
        return f"condition_embedder.text_embedder.linear_{idx}"
    
    # Legacy kohya format for time embedding
    match = re.match(r"lora_unet_time_embedding_(\d+)", lora_key)
    if match:
        idx = match.group(1)
        return f"condition_embedder.time_embedder.linear_{idx}"
    
    if lora_key == "lora_unet_time_projection_1":
        return "condition_embedder.time_proj"
    
    # Fallback: try to handle other legacy kohya formats
    if lora_key.startswith("lora_unet_") or ("unet_" in lora_key and not lora_key.startswith("diffusion_model")):
        key = lora_key.replace("lora_", "", 1)
        key = key.replace("unet_", "diffusion_model.", 1)
        key = re.sub(r"^diffusion_model\.blocks_(\d+)_", r"blocks.\1.", key)
        last_underscore = key.rfind("_")
        if last_underscore != -1:
            key = key[:last_underscore] + "." + key[last_underscore + 1:]
        return key
    
    return lora_key


def _find_base_param_key(model_key: str, base_state_dict: dict, suffix: str = ".weight") -> str:
    """
    Find the base parameter key in the model state dict for a given model key.
    
    Args:
        model_key: The mapped model key (e.g., "blocks.0.attn2.norm_k")
        base_state_dict: The base model's state dict
        suffix: The expected parameter suffix (".weight" or ".bias")
    
    Returns:
        The matching key in base_state_dict, or None if not found.
    """
    # Try exact match first
    candidate = model_key + suffix
    if candidate in base_state_dict and "lora" not in candidate:
        return candidate
    
    # Try to find a key that starts with model_key and ends with suffix
    for key in base_state_dict:
        if key.startswith(model_key) and key.endswith(suffix) and "lora" not in key:
            return key
    
    return None


def merge_lora_into_state_dict(
    base_state_dict: dict,
    lora_state_dict: dict,
    scale: float = 1.0,
) -> dict:
    """
    Merge LoRA weights into the base model state dict.
    
    Handles:
    - Standard LoRA pairs (lora_A/lora_B or lora_down/lora_up)
    - Additive adjustment tensors (.diff for weights, .diff_b for biases)
    """
    merged = {}

    # --- Collect LoRA pairs (standard lora_A/lora_B format) ---
    lora_pairs = {}
    for key in lora_state_dict:
        if ".lora_A.weight" in key:
            base_key = key.replace(".lora_A.weight", "")
            lora_pairs.setdefault(base_key, {})["A"] = key
        elif ".lora_B.weight" in key:
            base_key = key.replace(".lora_B.weight", "")
            lora_pairs.setdefault(base_key, {})["B"] = key
        elif ".lora_down.weight" in key:
            base_key = key.replace(".lora_down.weight", "")
            lora_pairs.setdefault(base_key, {})["A"] = key
        elif ".lora_up.weight" in key:
            base_key = key.replace(".lora_up.weight", "")
            lora_pairs.setdefault(base_key, {})["B"] = key

    # --- Collect alpha values ---
    alphas = {}
    for key in lora_state_dict:
        if ".alpha" in key:
            base_key = key.replace(".alpha", "")
            alphas[base_key] = lora_state_dict[key]

    merged_keys = set()

    # --- Process standard LoRA pairs ---
    for base_key, pairs in lora_pairs.items():
        if "A" not in pairs or "B" not in pairs:
            continue

        model_key = lora_key_to_model_key(base_key)
        
        base_weight_key = _find_base_param_key(model_key, base_state_dict, ".weight")

        if base_weight_key is None:
            print(f"  Warning: Could not find base weight for key: {base_key}")
            print(f"    (Mapped to model key: {model_key})")
            continue

        if base_weight_key in merged_keys:
            continue

        lora_A = lora_state_dict[pairs["A"]].to(torch.float32)
        lora_B = lora_state_dict[pairs["B"]].to(torch.float32)

        if base_key in alphas:
            alpha = alphas[base_key].item()
        else:
            alpha = lora_A.shape[0]

        scale_factor = scale * (alpha / lora_A.shape[0])
        delta = scale_factor * (lora_B @ lora_A)

        base_weight = base_state_dict[base_weight_key].to(torch.float32)
        merged_weight = base_weight + delta

        merged[base_weight_key] = merged_weight.to(base_state_dict[base_weight_key].dtype)
        merged_keys.add(base_weight_key)

        # Also copy the bias if it exists (will be updated later if .diff_b is present)
        bias_key = base_weight_key.replace(".weight", ".bias")
        if bias_key in base_state_dict:
            merged[bias_key] = base_state_dict[bias_key]
            merged_keys.add(bias_key)

    # --- Process .diff tensors (additive adjustments to weights) ---
    for key in lora_state_dict:
        if not key.endswith(".diff"):
            continue
        
        # Extract base key by removing .diff suffix
        base_key = key[:-len(".diff")]
        model_key = lora_key_to_model_key(base_key)
        
        # Find the target weight parameter in base_state_dict
        target_key = _find_base_param_key(model_key, base_state_dict, ".weight")
        
        if target_key is None:
            print(f"  Warning: Could not find base weight for .diff key: {key}")
            print(f"    (Mapped to model key: {model_key})")
            continue
        
        diff_tensor = lora_state_dict[key]
        
        # If the base weight hasn't been processed yet, copy it first
        if target_key not in merged:
            merged[target_key] = base_state_dict[target_key]
            merged_keys.add(target_key)
        
        # Apply the additive adjustment
        base_weight = merged[target_key].to(torch.float32)
        adjusted_weight = base_weight + diff_tensor.to(torch.float32)
        merged[target_key] = adjusted_weight.to(base_state_dict[target_key].dtype)

    # --- Process .diff_b tensors (additive adjustments to biases) ---
    for key in lora_state_dict:
        if not key.endswith(".diff_b"):
            continue
        
        # Extract base key by removing .diff_b suffix
        base_key = key[:-len(".diff_b")]
        model_key = lora_key_to_model_key(base_key)
        
        # Find the target bias parameter in base_state_dict
        target_key = _find_base_param_key(model_key, base_state_dict, ".bias")
        
        if target_key is None:
            print(f"  Warning: Could not find base bias for .diff_b key: {key}")
            print(f"    (Mapped to model key: {model_key})")
            continue
        
        diff_tensor = lora_state_dict[key]
        
        # If the base bias hasn't been processed yet, copy it first
        if target_key not in merged:
            merged[target_key] = base_state_dict[target_key]
            merged_keys.add(target_key)
        
        # Apply the additive adjustment
        base_bias = merged[target_key].to(torch.float32)
        adjusted_bias = base_bias + diff_tensor.to(torch.float32)
        merged[target_key] = adjusted_bias.to(base_state_dict[target_key].dtype)

    # --- Copy all remaining base parameters ---
    for key, value in base_state_dict.items():
        if key not in merged_keys and "lora" not in key:
            merged[key] = value

    return merged


def main():
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    target_dtype = dtype_map[args.dtype]

    # Step 1: Resolve the high and low noise safetensors files
    if args.model_path_high is not None and args.model_path_low is not None:
        high_path = args.model_path_high
        low_path = args.model_path_low
        
        # If directories were provided, search inside them
        if os.path.isdir(high_path):
            high_files = find_safetensors_files_local(high_path)
            high_path = high_files.get("high") or (list(high_files.values())[0] if high_files else None)
                
        if os.path.isdir(low_path):
            low_files = find_safetensors_files_local(low_path)
            low_path = low_files.get("low") or (list(low_files.values())[0] if low_files else None)
            
        if not high_path or not os.path.isfile(high_path):
            print(f"Error: Could not resolve HIGH noise model path from {args.model_path_high}")
            sys.exit(1)
        if not low_path or not os.path.isfile(low_path):
            print(f"Error: Could not resolve LOW noise model path from {args.model_path_low}")
            sys.exit(1)
            
        safetensor_files = {"high": high_path, "low": low_path}
        print(f"Using specified HIGH noise model: {safetensor_files['high']}")
        print(f"Using specified LOW noise model: {safetensor_files['low']}")
    else:
        print(f"Searching for safetensors files in {args.model_path}")
        safetensor_files = find_safetensors_files_local(args.model_path)

        if "high" not in safetensor_files:
            print(f"Error: Could not find a .safetensors file with 'high' in the name.")
            print(f"Found files: {list(safetensor_files.keys())}")
            sys.exit(1)

        if "low" not in safetensor_files:
            print(f"Error: Could not find a .safetensors file with 'low' in the name.")
            print(f"Found files: {list(safetensor_files.keys())}")
            sys.exit(1)

        print(f"HIGH noise model: {safetensor_files['high']}")
        print(f"LOW noise model: {safetensor_files['low']}")

    # Step 2: Load config and compute the mixed-precision fp32 layout.
    # (No model tensors are loaded yet — this is pure config/path work.)
    config = load_config_from_path(safetensor_files["high"], args.model_path)

    # Compute the mixed-precision fp32 layout (text/time embeddings, projections, norms and
    # front/tail blocks stay in float32; the rest uses --dtype).
    fp32_prefixes = None
    if not args.no_fp32 and target_dtype != torch.float32:
        num_blocks = count_blocks_from_safetensors(safetensor_files["high"]) or config.get("num_layers", 40)
        fp32_prefixes = compute_fp32_prefixes(num_blocks, args.fp32_front_layers, args.fp32_tail_layers)
        fp32_block_indices = sorted({int(p.split(".")[1]) for p in fp32_prefixes if p.startswith("blocks.") and p.count(".") == 1})
        print(
            f"Keeping text/time embeddings, projections, norms and front/tail blocks in float32 "
            f"(front={args.fp32_front_layers}, tail={args.fp32_tail_layers}, total blocks={num_blocks})"
        )
        print(f"Entire blocks kept in float32: {fp32_block_indices}")
    elif args.no_fp32:
        print("--no-fp32 set: entire model will be saved in the base dtype.")

    # Step 3: Resolve output paths (pure path math, no tensors).
    if args.output_path:
        output_path = args.output_path
    else:
        # Default to the directory of the high noise model
        output_path = os.path.dirname(os.path.abspath(safetensor_files["high"]))
    os.makedirs(output_path, exist_ok=True)

    def _output_path_for(model_file: str) -> str:
        name, ext = os.path.splitext(os.path.basename(model_file))
        return os.path.join(output_path, f"{name}_lora{ext}")

    high_output = _output_path_for(safetensor_files["high"])
    low_output = _output_path_for(safetensor_files["low"])

    # Step 4: Resolve per-expert LoRA paths. If both experts share one LoRA file, load it
    # once and reuse it (it is small, a few hundred MB).
    if args.lora_path_high is not None and args.lora_path_low is not None:
        lora_paths = {"high": args.lora_path_high, "low": args.lora_path_low}
    else:
        lora_paths = {"high": args.lora_path, "low": args.lora_path}

    shared_lora_sd = None
    if lora_paths["high"] == lora_paths["low"]:
        print(f"\nLoading LoRA from {lora_paths['high']}")
        shared_lora_sd = load_file(lora_paths["high"])

    # Step 5: Process each expert SEQUENTIALLY: load -> merge -> save -> free.
    # Only one expert's model + merged weights are resident at a time, so peak RAM is a
    # single expert (~60GB for a 14B model) instead of both experts at once (~125GB+).
    experts = [
        ("HIGH", safetensor_files["high"], lora_paths["high"], high_output),
        ("LOW",  safetensor_files["low"],  lora_paths["low"],  low_output),
    ]
    for label, model_file, lora_file, out_file in experts:
        print(f"\nLoading {label} noise transformer...")
        model = load_transformer_from_safetensors(
            model_file, config, target_dtype, torch.device("cpu"),
            fp32_prefixes=fp32_prefixes,
        )

        if shared_lora_sd is not None:
            lora_sd = shared_lora_sd
        else:
            print(f"Loading {label} noise LoRA from {lora_file}")
            lora_sd = load_file(lora_file)

        print(f"Merging LoRA into {label} noise transformer...")
        base_sd = model.state_dict()
        merged_sd = merge_lora_into_state_dict(base_sd, lora_sd, scale=args.scale)
        del base_sd

        print(f"Saving merged {label} noise model to {out_file}")
        save_file(merged_sd, out_file, metadata={"format": "pt"})

        # Free this expert before loading the next one to keep peak RAM bounded.
        del model, merged_sd
        if shared_lora_sd is None:
            del lora_sd
        gc.collect()

    if shared_lora_sd is not None:
        del shared_lora_sd
        gc.collect()

    # Copy config.json if it exists
    config_path = os.path.join(os.path.dirname(safetensor_files["high"]), "config.json")
    if os.path.exists(config_path):
        import shutil
        config_output = os.path.join(output_path, "config.json")
        if not os.path.exists(config_output):
            shutil.copy2(config_path, config_output)
            print(f"Copied config.json to {config_output}")

    print("\nDone! Merged models saved successfully.")


if __name__ == "__main__":
    main()