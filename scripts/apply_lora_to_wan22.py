"""
Apply a LoRA to both HIGH and LOW noise Wan 2.2 I2V 14B models and save the merged models.

Usage:
    python scripts/apply_lora_to_wan22.py \
        --model-path /path/to/folder/with/safetensors \
        --lora-path /path/to/lora.safetensors \
        --output-path /path/to/output/folder \
        --scale 1.0
"""

import argparse
import os
import re
import sys
import json
from collections import OrderedDict

import torch
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply LoRA to both HIGH and LOW noise Wan 2.2 14B models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to local folder containing high and low noise .safetensors files",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to LoRA .safetensors file (BF16)",
    )
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
        help="LoRA scale/multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Target dtype for merged model (default: bfloat16)",
    )
    return parser.parse_args()


def load_config_from_path(safetensors_path: str, model_path: str) -> dict:
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
    Map a LoRA tensor key to the corresponding Wan 2.2 model tensor key.
    
    Handles TWO naming conventions:
    1. Old format (Wan2.1-style Unet): lora_unet_blocks_{N}_ffn_0
    2. New format (Standard diffusers): diffusion_model.blocks.{N}.ffn.0
    
    For the old format, translates from unet_* naming to diffusion_model.* naming.
    For the new format, returns the key as-is (it's already in the correct format).
    """
    # Check if this is the old format (starts with "lora_unet_" or contains "unet_")
    if lora_key.startswith("lora_unet_") or ("unet_" in lora_key and not lora_key.startswith("diffusion_model")):
        # Old format: lora_unet_blocks_N_ffn_0 -> model.diffusion_model.blocks.N.ffn.0
        key = lora_key.replace("lora_", "", 1)
        key = key.replace("unet_", "diffusion_model.", 1)
        key = re.sub(r"^diffusion_model\.blocks_(\d+)_", r"diffusion_model.blocks.\1.", key)
        last_underscore = key.rfind("_")
        if last_underscore != -1:
            key = key[:last_underscore] + "." + key[last_underscore + 1:]
        key = "model." + key
        return key
    else:
        # New format: already in diffusers naming, just add "model." prefix
        # e.g., "diffusion_model.blocks.0.cross_attn.k" -> "model.diffusion_model.blocks.0.cross_attn.k"
        if not lora_key.startswith("model."):
            return "model." + lora_key
        return lora_key


def merge_lora_into_state_dict(
    base_state_dict: dict,
    lora_state_dict: dict,
    scale: float = 1.0,
) -> dict:
    """
    Merge LoRA weights into the base model state dict.

    Supports two LoRA naming conventions:
    1. Old format (Wan2.1-style): lora_unet_blocks_{N}_ffn_0.lora_down.weight
    2. New format (Standard diffusers): diffusion_model.blocks.{N}.ffn.0.lora_A.weight
    
    Merged weight = base_weight + scale * (lora_up @ lora_down) * (alpha / rank)
    """
    merged = {}

    # Group LoRA weights by base key prefix
    lora_pairs = {}
    for key in lora_state_dict:
        # Handle different LoRA naming conventions
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

    # Find alpha values
    alphas = {}
    for key in lora_state_dict:
        if ".alpha" in key:
            base_key = key.replace(".alpha", "")
            alphas[base_key] = lora_state_dict[key]

    # Merge each pair
    merged_keys = set()
    for base_key, pairs in lora_pairs.items():
        if "A" not in pairs or "B" not in pairs:
            continue

        # Translate LoRA key to model key
        model_key = lora_key_to_model_key(base_key)
        
        # Try direct match first
        candidate_keys = [
            model_key + ".weight",
            model_key + ".bias",
        ]

        base_weight_key = None
        for ck in candidate_keys:
            if ck in base_state_dict and "lora" not in ck:
                base_weight_key = ck
                break

        if base_weight_key is None:
            # Fallback: search for partial matches
            for key in base_state_dict:
                if key.startswith(model_key) and "lora" not in key and key.endswith(".weight"):
                    base_weight_key = key
                    break

        if base_weight_key is None:
            print(f"  Warning: Could not find base weight for key: {base_key}")
            print(f"    (Mapped to model key: {model_key})")
            continue

        if base_weight_key in merged_keys:
            continue

        lora_A = lora_state_dict[pairs["A"]].to(torch.float32)
        lora_B = lora_state_dict[pairs["B"]].to(torch.float32)

        # Get alpha (default to rank if not present)
        if base_key in alphas:
            alpha = alphas[base_key].item()
        else:
            alpha = lora_A.shape[0]  # rank

        scale_factor = scale * (alpha / lora_A.shape[0])

        # Compute the delta weight
        delta = scale_factor * (lora_B @ lora_A)

        # Add to base weight
        base_weight = base_state_dict[base_weight_key].to(torch.float32)
        merged_weight = base_weight + delta

        merged[base_weight_key] = merged_weight.to(base_state_dict[base_weight_key].dtype)
        merged_keys.add(base_weight_key)

        # Also merge any biases if present
        bias_key = base_weight_key.replace(".weight", ".bias")
        if bias_key in base_state_dict:
            merged[bias_key] = base_state_dict[bias_key]
            merged_keys.add(bias_key)

    # Copy all non-merged base weights
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

    # Step 1: Find the high and low noise safetensors files
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

    # Step 2: Load the base models
    print("\nLoading HIGH noise transformer...")
    config = load_config_from_path(safetensor_files["high"], args.model_path)
    high_model = load_transformer_from_safetensors(
        safetensor_files["high"], config, target_dtype, torch.device("cpu")
    )

    print("Loading LOW noise transformer...")
    low_model = load_transformer_from_safetensors(
        safetensor_files["low"], config, target_dtype, torch.device("cpu")
    )

    # Step 3: Load the LoRA
    print(f"\nLoading LoRA from {args.lora_path}")
    lora_state_dict = load_file(args.lora_path)

    # Step 4: Merge LoRA into both models
    print("\nMerging LoRA into HIGH noise transformer...")
    high_state_dict = high_model.state_dict()
    merged_high_sd = merge_lora_into_state_dict(high_state_dict, lora_state_dict, scale=args.scale)

    print("Merging LoRA into LOW noise transformer...")
    low_state_dict = low_model.state_dict()
    merged_low_sd = merge_lora_into_state_dict(low_state_dict, lora_state_dict, scale=args.scale)

    # Step 5: Determine output path
    output_path = args.output_path if args.output_path else args.model_path

    # Step 6: Save the merged models
    os.makedirs(output_path, exist_ok=True)

    # Save HIGH noise model
    base_high = os.path.basename(safetensor_files["high"])
    name_high, ext_high = os.path.splitext(base_high)
    high_output = os.path.join(output_path, f"{name_high}_lora{ext_high}")

    print(f"\nSaving merged HIGH noise model to {high_output}")
    save_file(merged_high_sd, high_output, metadata={"format": "pt"})

    # Save LOW noise model
    base_low = os.path.basename(safetensor_files["low"])
    name_low, ext_low = os.path.splitext(base_low)
    low_output = os.path.join(output_path, f"{name_low}_lora{ext_low}")

    print(f"Saving merged LOW noise model to {low_output}")
    save_file(merged_low_sd, low_output, metadata={"format": "pt"})

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
