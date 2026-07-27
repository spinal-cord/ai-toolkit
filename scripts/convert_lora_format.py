"""
Convert LoRA format from per-expert (with transformer_N. prefix) to simplified format.

The per-expert format has keys like:
  diffusion_model.transformer_1.blocks.0.cross_attn.k.lora_A.weight
  diffusion_model.transformer_2.blocks.0.cross_attn.k.lora_A.weight

This script removes the transformer_N. prefix to produce:
  diffusion_model.blocks.0.cross_attn.k.lora_A.weight

Usage:
    python scripts/convert_lora_format.py /path/to/lora.safetensors
    python scripts/convert_lora_format.py /path/to/lora.safetensors --dtype float16
"""

import argparse
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file


def convert_lora_keys(state_dict: dict) -> dict:
    """
    Remove transformer_N. prefix from LoRA keys.
    
    Converts:
      diffusion_model.transformer_1.blocks.0.cross_attn.k.lora_A.weight
      -> diffusion_model.blocks.0.cross_attn.k.lora_A.weight
    """
    converted = {}
    for key, value in state_dict.items():
        # Remove transformer_N. prefix if present
        new_key = re.sub(r'^diffusion_model\.transformer_\d+\.', 'diffusion_model.', key)
        converted[new_key] = value
    
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert LoRA from per-expert format to simplified format"
    )
    
    parser.add_argument(
        'lora_input',
        type=str,
        help="Path to input LoRA .safetensors file"
    )
    
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float16', 'float32'],
        help="Output dtype (default: bfloat16)"
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help="Output directory (defaults to same folder as input)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.isfile(args.lora_input):
        print(f"Error: Input file not found: {args.lora_input}")
        sys.exit(1)
    
    if not args.lora_input.endswith('.safetensors'):
        print(f"Error: Input file must be a .safetensors file: {args.lora_input}")
        sys.exit(1)
    
    # Load LoRA
    print(f"Loading LoRA from: {args.lora_input}")
    lora_state_dict = load_file(args.lora_input)
    
    # Convert keys
    print("Converting keys (removing transformer_N. prefix)...")
    converted_state_dict = convert_lora_keys(lora_state_dict)
    
    # Determine output dtype
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    output_dtype = dtype_map[args.dtype]
    
    # Convert tensors to output dtype
    print(f"Converting tensors to {args.dtype}...")
    for key in list(converted_state_dict.keys()):
        converted_state_dict[key] = converted_state_dict[key].to(output_dtype)
    
    # Determine output path
    if args.output_path:
        output_dir = args.output_path
    else:
        output_dir = os.path.dirname(os.path.abspath(args.lora_input))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(args.lora_input))[0]
    
    if args.dtype == 'bfloat16':
        output_name = f"{base_name}_BF16_renamed.safetensors"
    elif args.dtype == 'float16':
        output_name = f"{base_name}_FP16_renamed.safetensors"
    elif args.dtype == 'float32':
        output_name = f"{base_name}_FP32_renamed.safetensors"
    
    output_path = os.path.join(output_dir, output_name)
    
    # Save converted LoRA
    print(f"Saving converted LoRA to: {output_path}")
    save_file(converted_state_dict, output_path, metadata={'format': 'pt'})
    
    # Print summary
    print(f"\nConversion complete!")
    print(f"  Input:  {args.lora_input}")
    print(f"  Output: {output_path}")
    print(f"  Dtype:  {args.dtype}")
    print(f"  Keys:   {len(converted_state_dict)} tensors")
    
    # Show a few example keys
    print(f"\nExample converted keys:")
    for i, key in enumerate(sorted(converted_state_dict.keys())[:5]):
        print(f"  {key}")


if __name__ == '__main__':
    main()
