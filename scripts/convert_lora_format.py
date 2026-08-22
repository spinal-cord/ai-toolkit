"""
Convert LoRA format from per-expert (with transformer_N. prefix) to simplified format.

The per-expert format has keys like:
  diffusion_model.transformer_1.blocks.0.cross_attn.k.lora_A.weight
  diffusion_model.transformer_2.blocks.0.cross_attn.k.lora_A.weight

This script removes the transformer_N. prefix to produce:
  diffusion_model.blocks.0.cross_attn.k.lora_A.weight

Precision handling:
  By default, large block attention/FFN LoRA tensors are converted to bfloat16,
  while small tensors such as norms, biases, modulation, text/time embeddings,
  time projection, patch embedding, and head/proj output tensors are kept in
  float32.

  In addition, the first 1 and last 4 transformer blocks (front/tail layers) are
  always kept in float32 for numerical stability, regardless of their size class.
  This can be tuned with --fp32_front_layers and --fp32_tail_layers.

Usage:
    python scripts/convert_lora_format.py /path/to/lora.safetensors
    python scripts/convert_lora_format.py /path/to/lora.safetensors --dtype float16
    python scripts/convert_lora_format.py /path/to/lora.safetensors --small_tensors_precision fp16
    python scripts/convert_lora_format.py /path/to/lora.safetensors --dtype fp16 --small_tensors_precision fp16
"""

import argparse
import os
import re
import sys
from collections import Counter

import torch
from safetensors.torch import load_file, save_file


PRECISION_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

PRECISION_ABBR = {
    "bfloat16": "BF16",
    "float16": "FP16",
    "float32": "FP32",
}


def normalize_precision(value: str):
    """
    Normalize precision aliases:
      bf16/bfloat16 -> bfloat16
      fp16/float16  -> float16
      fp32/float32  -> float32
    """
    if value is None:
        return None

    value = str(value).strip().lower()
    if value in PRECISION_ALIASES:
        return PRECISION_ALIASES[value]

    raise argparse.ArgumentTypeError(
        f"Invalid precision '{value}'. "
        "Use bf16/bfloat16, fp16/float16, or fp32/float32."
    )


def precision_abbr(precision: str) -> str:
    return PRECISION_ABBR.get(precision, precision.upper())


TRANSFORMER_PREFIX_RE = re.compile(
    r"^(?P<prefix>model\.)?diffusion_model\.transformer_\d+\."
)

STRIP_MODEL_PREFIX_RE = re.compile(r"^model\.")

NORM_RE = re.compile(
    r"(?:^|\.)norm(?:[_\.]?(?:q|k|[123]))?(?:$|\.)"
)

SMALL_MODULE_RES = (
    re.compile(r"(?:^|\.)patch_embedding(?:$|\.)"),
    re.compile(r"(?:^|\.)text_embedding(?:$|\.)"),
    re.compile(r"(?:^|\.)text_embedder(?:$|\.)"),
    re.compile(r"(?:^|\.)time_embedding(?:$|\.)"),
    re.compile(r"(?:^|\.)time_embedder(?:$|\.)"),
    re.compile(r"(?:^|\.)time_projection(?:$|\.)"),
    re.compile(r"(?:^|\.)time_proj(?:$|\.)"),
    re.compile(r"(?:^|\.)condition_embedder(?:$|\.)"),
    re.compile(r"(?:^|\.)head(?:$|\.)"),
    re.compile(r"(?:^|\.)proj_out(?:$|\.)"),
    re.compile(r"(?:^|\.)final_layer(?:$|\.)"),
    re.compile(r"(?:^|\.)modulation(?:$|\.)"),
    NORM_RE,
)

LARGE_BLOCK_LINEAR_RE = re.compile(
    r"^diffusion_model\.blocks\.\d+\.(?:self_attn|cross_attn)\.(?:q|k|v|o)\.weight$"
    r"|"
    r"^diffusion_model\.blocks\.\d+\.ffn\.(?:0|2)\.weight$"
)

BLOCK_INDEX_RE = re.compile(r"blocks\.(\d+)\.")


def _remove_transformer_prefix(key: str) -> str:
    """
    Remove transformer_N. prefix while preserving a leading model. prefix if present.

    Examples:
      diffusion_model.transformer_2.blocks... -> diffusion_model.blocks...
      model.diffusion_model.transformer_2.blocks... -> model.diffusion_model.blocks...
    """

    def repl(match):
        return (match.group("prefix") or "") + "diffusion_model."

    return TRANSFORMER_PREFIX_RE.sub(repl, key)


def convert_lora_keys(state_dict: dict) -> dict:
    """
    Remove transformer_N. prefix from LoRA keys.

    Converts:
      diffusion_model.transformer_1.blocks.0.cross_attn.k.lora_A.weight
      -> diffusion_model.blocks.0.cross_attn.k.lora_A.weight
    """
    converted = {}
    duplicate_keys = set()

    for key, value in state_dict.items():
        new_key = _remove_transformer_prefix(key)

        # Fallback for keys that start directly with transformer_N.
        if new_key == key:
            new_key = re.sub(r"^transformer_\d+\.", "", new_key)

        if new_key in converted:
            duplicate_keys.add(new_key)

        converted[new_key] = value

    if duplicate_keys:
        print(
            f"Warning: {len(duplicate_keys)} duplicate keys were found after "
            "removing transformer_N. prefixes. Kept the last occurrence for each."
        )
        for key in sorted(duplicate_keys)[:5]:
            print(f"  {key}")
        if len(duplicate_keys) > 5:
            print("  ...")

    return converted


def _strip_for_classification(key: str) -> str:
    """
    Normalize key only for precision classification.
    This does not affect the saved output key names.
    """
    key = STRIP_MODEL_PREFIX_RE.sub("", key)
    key = _remove_transformer_prefix(key)
    return key


def canonical_base_key(key: str) -> str:
    """
    Convert a LoRA/diff tensor key into an approximate base-model tensor key
    for precision classification.

    Examples:
      diffusion_model.blocks.0.cross_attn.k.lora_A.weight
      -> diffusion_model.blocks.0.cross_attn.k.weight

      diffusion_model.blocks.0.cross_attn.norm_k.diff
      -> diffusion_model.blocks.0.cross_attn.norm_k.weight

      diffusion_model.patch_embedding.diff_b
      -> diffusion_model.patch_embedding.bias

      diffusion_model.condition_embedder.time_proj.lora_A.weight
      -> diffusion_model.time_projection.1.weight
    """
    key = _strip_for_classification(key)

    # LoRA weight suffixes.
    key = re.sub(r"\.lora_(?:A|B|down|up)\.bias$", ".bias", key)
    key = re.sub(
        r"\.lora_(?:A|B|down|up)(?:\.(?:weight|default))?$",
        ".weight",
        key,
    )

    # Diff-style tensors.
    key = re.sub(r"\.diff_b$", ".bias", key)
    key = re.sub(r"\.diff$", ".weight", key)

    # Map condition embedder names to base-model-ish names.
    key = re.sub(
        r"^diffusion_model\.condition_embedder\.text_embedder\.linear_1\.",
        "diffusion_model.text_embedding.0.",
        key,
    )
    key = re.sub(
        r"^diffusion_model\.condition_embedder\.text_embedder\.linear_2\.",
        "diffusion_model.text_embedding.2.",
        key,
    )
    key = re.sub(
        r"^diffusion_model\.condition_embedder\.time_embedder\.linear_1\.",
        "diffusion_model.time_embedding.0.",
        key,
    )
    key = re.sub(
        r"^diffusion_model\.condition_embedder\.time_embedder\.linear_2\.",
        "diffusion_model.time_embedding.2.",
        key,
    )
    key = re.sub(
        r"^diffusion_model\.condition_embedder\.time_proj\.",
        "diffusion_model.time_projection.1.",
        key,
    )

    # Common output projection naming.
    key = re.sub(
        r"^diffusion_model\.proj_out\.",
        "diffusion_model.head.head.",
        key,
    )

    return key


def extract_block_index(key: str):
    """
    Return the transformer block index for a key, or None if the key does not
    belong to a transformer block (e.g. condition embedder, proj_out, etc.).
    """
    canonical_key = canonical_base_key(key)
    match = BLOCK_INDEX_RE.search(canonical_key)
    if match:
        return int(match.group(1))
    return None


def is_small_tensor(key: str, shape) -> bool:
    """
    Heuristic for tensors that should usually stay in float32:
      - 1D tensors: biases, norms, modulation vectors, diff vectors
      - patch embedding
      - text/time embeddings
      - time projection
      - condition embedder
      - head/proj_out/final_layer
      - norms and modulation
    """
    if len(shape) <= 1:
        return True

    canonical_key = canonical_base_key(key)
    raw_key = _strip_for_classification(key)

    for pattern in SMALL_MODULE_RES:
        if pattern.search(canonical_key) or pattern.search(raw_key):
            return True

    if canonical_key.endswith(".bias") or raw_key.endswith(".bias"):
        return True

    return False


def is_large_block_linear(key: str) -> bool:
    """
    Large block linears that are bfloat16 in the base model:
      blocks.*.self_attn.{q,k,v,o}.weight
      blocks.*.cross_attn.{q,k,v,o}.weight
      blocks.*.ffn.{0,2}.weight
    """
    canonical_key = canonical_base_key(key)
    return bool(LARGE_BLOCK_LINEAR_RE.search(canonical_key))


def choose_precision(
    key: str,
    tensor: torch.Tensor,
    small_precision: str,
    large_precision: str,
    fp32_layer_indices: set = None,
) -> str:
    """
    Choose target precision for a tensor.

    Default intended behavior:
      - front/tail transformer blocks (fp32_layer_indices) -> float32
      - small tensors -> float32 unless overridden
      - large block attention/FFN weights -> bfloat16 unless --dtype changes it
    """
    shape = tuple(tensor.shape)

    # Front/tail layers are always kept in float32 for numerical stability,
    # regardless of whether the tensor would otherwise be classified as small
    # or large.
    if fp32_layer_indices:
        block_index = extract_block_index(key)
        if block_index is not None and block_index in fp32_layer_indices:
            return "float32"

    if is_small_tensor(key, shape):
        return small_precision

    if is_large_block_linear(key):
        return large_precision

    canonical_key = canonical_base_key(key)

    if len(shape) <= 1:
        return small_precision

    # Inside transformer blocks, unknown multi-dimensional weights are treated
    # as large tensors by default.
    if "blocks." in canonical_key:
        return large_precision

    # Outside blocks, unknown tensors are treated as small/condition tensors
    # by default, matching the provided base precision table.
    return small_precision


def main():
    parser = argparse.ArgumentParser(
        description="Convert LoRA from per-expert format to simplified format "
        "with mixed precision support."
    )

    parser.add_argument(
        "lora_input",
        type=str,
        help="Path to input LoRA .safetensors file",
    )

    parser.add_argument(
        "--dtype",
        type=normalize_precision,
        default="bfloat16",
        help=(
            "Precision for large/regular tensors. "
            "Accepted: bf16/bfloat16, fp16/float16, fp32/float32. "
            "Default: bfloat16"
        ),
    )

    parser.add_argument(
        "--small_tensors_precision",
        type=normalize_precision,
        default=None,
        help=(
            "Precision for small tensors such as norms, biases, modulation, "
            "text/time embeddings, time projection, patch embedding, and head/proj output. "
            "Accepted: bf16/bfloat16, fp16/float16, fp32/float32. "
            "Default: float32"
        ),
    )

    parser.add_argument(
        "--fp32_front_layers",
        type=int,
        default=1,
        help=(
            "Number of leading transformer blocks to keep in float32. "
            "Default: 1"
        ),
    )

    parser.add_argument(
        "--fp32_tail_layers",
        type=int,
        default=4,
        help=(
            "Number of trailing transformer blocks to keep in float32. "
            "Default: 4"
        ),
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output directory (defaults to same folder as input)",
    )

    args = parser.parse_args()

    # Validate input.
    if not os.path.isfile(args.lora_input):
        print(f"Error: Input file not found: {args.lora_input}")
        sys.exit(1)

    if not args.lora_input.endswith(".safetensors"):
        print(f"Error: Input file must be a .safetensors file: {args.lora_input}")
        sys.exit(1)

    large_precision = args.dtype

    small_was_specified = args.small_tensors_precision is not None
    if small_was_specified:
        small_precision = args.small_tensors_precision
    else:
        small_precision = "float32"

    # Load LoRA.
    print(f"Loading LoRA from: {args.lora_input}")
    lora_state_dict = load_file(args.lora_input)

    # Convert keys.
    print("Converting keys (removing transformer_N. prefix)...")
    converted_state_dict = convert_lora_keys(lora_state_dict)

    # Determine which transformer blocks to keep in float32 (front + tail layers).
    block_indices = set()
    for key in converted_state_dict:
        idx = extract_block_index(key)
        if idx is not None:
            block_indices.add(idx)
    num_layers = (max(block_indices) + 1) if block_indices else 0

    front_count = min(max(args.fp32_front_layers, 0), num_layers)
    tail_count = min(max(args.fp32_tail_layers, 0), num_layers)
    fp32_layer_indices = (
        set(range(front_count)) | set(range(num_layers - tail_count, num_layers))
    ) if num_layers > 0 else set()

    if fp32_layer_indices:
        print(
            f"Keeping {len(fp32_layer_indices)} front/tail layers in float32 "
            f"(front={front_count}, tail={tail_count}, total blocks={num_layers}): "
            f"{sorted(fp32_layer_indices)}"
        )

    # Convert tensors to selected precisions.
    print(
        "Converting tensors "
        f"(large/default: {large_precision}, small: {small_precision})..."
    )

    final_state_dict = {}
    precision_counter = Counter()

    for key, value in converted_state_dict.items():
        target_precision = choose_precision(
            key=key,
            tensor=value,
            small_precision=small_precision,
            large_precision=large_precision,
            fp32_layer_indices=fp32_layer_indices,
        )
        target_dtype = DTYPE_MAP[target_precision]

        final_state_dict[key] = value.to(target_dtype)
        precision_counter[target_precision] += 1

    # Determine output path.
    if args.output_path:
        output_dir = args.output_path
    else:
        output_dir = os.path.dirname(os.path.abspath(args.lora_input))

    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename.
    base_name = os.path.splitext(os.path.basename(args.lora_input))[0]

    used_precisions = set(precision_counter.keys())

    if len(used_precisions) == 1:
        suffix = precision_abbr(next(iter(used_precisions)))
    elif (
        not small_was_specified
        and small_precision == "float32"
        and used_precisions <= {"bfloat16", "float32"}
    ):
        # Preserve the traditional output name for the default mixed behavior:
        # large tensors BF16, small/front-tail tensors FP32.
        suffix = precision_abbr(large_precision)
    else:
        suffix = (
            f"{precision_abbr(large_precision)}"
            f"_small_{precision_abbr(small_precision)}"
        )

    output_name = f"{base_name}_{suffix}_renamed.safetensors"
    output_path = os.path.join(output_dir, output_name)

    # Save converted LoRA.
    print(f"Saving converted LoRA to: {output_path}")
    save_file(final_state_dict, output_path, metadata={"format": "pt"})

    # Print summary.
    print("\nConversion complete!")
    print(f"  Input:  {args.lora_input}")
    print(f"  Output: {output_path}")
    print(f"  Large/default precision: {large_precision}")
    print(f"  Small precision: {small_precision}")
    if fp32_layer_indices:
        print(
            f"  FP32 front/tail layers ({len(fp32_layer_indices)} blocks): "
            f"{sorted(fp32_layer_indices)}"
        )
    print(f"  Keys: {len(final_state_dict)} tensors")

    print("\nTensor counts by precision:")
    for precision, count in sorted(precision_counter.items()):
        print(f"  {precision}: {count}")

    # Show a few example keys.
    print("\nExample converted keys:")
    for key in sorted(final_state_dict.keys())[:5]:
        print(f"  {key} ({final_state_dict[key].dtype})")


if __name__ == "__main__":
    main()
