#!/usr/bin/env python3
"""
truncate_lora_gates.py
======================

Take a LoRA checkpoint that was saved WITH rank gates (mid-training
checkpoint from rank-gate annealing, i.e. a state dict containing
`...rank_gate.gates` / `...rank_gate_b.gates` keys) and produce a new LoRA
whose rank is PHYSICALLY reduced according to those gates.

Unlike a gate-folded save (which only zeroes dead components and keeps the
original tensor shapes), this script RESHAPES the tensors to the surviving
rank:

  - LoRA pairs: dead rows of `lora_down.weight` and the matching columns of
    `lora_up.weight` are removed, surviving rows are multiplied by their
    gate values, and `alpha` is rescaled so the per-rank scaling
    (alpha / rank) is preserved. Works for linear (r, in) / (out, r) and
    conv (r, in, k, k) / (out, r, 1, 1).
  - `.diff` tensors (full finetune): no rank axis exists, so the gates are
    simply folded element-wise (dead elements -> 0 diff = base weight).

The output is a standard, smaller LoRA that any loader can consume.

Usage:
    python scripts/truncate_lora_gates.py INPUT.safetensors [OUTPUT.safetensors]
        [--threshold 0.5] [--min-rank 1] [--dtype keep|fp16|bf16|fp32]

    INPUT   LoRA safetensors containing rank gate keys (checked first; if no
            gates are present the script reports this and exits without
            writing anything).
    OUTPUT  Output path (default: <INPUT stem>_truncated.safetensors).

Examples:
    python scripts/truncate_lora_gates.py my_lora_00010000.safetensors
    python scripts/truncate_lora_gates.py my_lora.safetensors my_lora_rank32.safetensors --threshold 0.5
"""

import argparse
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

# LoRA weight key suffixes, in the order to try. (down, up) pairs share the
# same module prefix; the rank axis is dim 0 of "down" and dim 1 of "up" for
# both linear and conv layouts (and for peft-style lora_A / lora_B).
LORA_PAIR_SUFFIXES = [
    ('.lora_down.weight', '.lora_up.weight'),
    ('.lora_A.weight', '.lora_B.weight'),
]
DIFF_SUFFIXES = ('.diff', '.diff_b')

DTYPE_MAP = {
    'keep': None,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
}


def find_gate_keys(state_dict):
    """Return (lora_gate_keys, diff_gate_keys) found in the state dict."""
    lora_gate_keys, diff_gate_keys = [], []
    for key in state_dict:
        if key.endswith('.rank_gate.gates'):
            lora_gate_keys.append(key)
        elif key.endswith('.rank_gate_b.gates'):
            diff_gate_keys.append(key)
    return lora_gate_keys, diff_gate_keys


def truncate_lora_pair(state_dict, prefix, gates, threshold, min_rank, summary):
    """Physically reduce the rank of one LoRA pair in place in state_dict."""
    for down_suf, up_suf in LORA_PAIR_SUFFIXES:
        key_A, key_B = prefix + down_suf, prefix + up_suf
        if key_A in state_dict and key_B in state_dict:
            break
    else:
        print(f"  [WARN] gates for '{prefix}' but no lora_down/lora_up pair found; dropping gates")
        return False

    A = state_dict[key_A]
    B = state_dict[key_B]
    r = A.shape[0]
    if B.shape[1] != r:
        print(f"  [WARN] '{prefix}': rank mismatch down={A.shape[0]} vs up={B.shape[1]}; skipping")
        return False

    g = gates.detach().float().view(-1)
    if g.numel() != r:
        print(f"  [WARN] '{prefix}': gate size {g.numel()} != rank {r}; skipping")
        return False

    keep = (g > threshold).nonzero(as_tuple=True)[0].tolist()
    if len(keep) < min_rank:
        # keep at least the strongest ranks
        n = min(min_rank, r)
        keep = g.topk(n).indices.sort().values.tolist()
    k = len(keep)

    # Fold gates into the surviving rows of A (dropped rows are removed, so
    # only the kept rows need scaling). Rank axis is dim 0.
    g_keep = g[keep]
    for _ in range(A.dim() - 1):
        g_keep = g_keep.unsqueeze(-1)
    A_new = (A.float() * g_keep).to(A.dtype)
    A_new = A_new[keep].contiguous()

    # B: keep the matching columns (dim 1) for linear (out, r) and
    # conv (out, r, 1, 1).
    B_new = B[:, keep].contiguous().to(B.dtype)

    state_dict[key_A] = A_new
    state_dict[key_B] = B_new

    # Rescale alpha so that alpha / rank is preserved.
    key_alpha = prefix + '.alpha'
    if key_alpha in state_dict and r > 0:
        state_dict[key_alpha] = (state_dict[key_alpha].float() * (k / r)).to(
            state_dict[key_alpha].dtype)

    summary.append((prefix, r, k))
    print(f"  {prefix}: rank {r} -> {k}")
    return True


def fold_diff(state_dict, prefix, gates, gate_key):
    """Fold per-element gates into a .diff / .diff_b tensor (no dim removal)."""
    suffix = '.diff_b' if gate_key.endswith('.rank_gate_b.gates') else '.diff'
    key = prefix + suffix
    if key not in state_dict:
        print(f"  [WARN] gate key '{gate_key}' but no '{key}' tensor; dropping gates")
        return False
    v = state_dict[key]
    m = gates.detach().float().view(v.shape)
    state_dict[key] = (v.float() * m).to(v.dtype)
    kept = int((m > 0.5).sum().item())
    print(f"  {prefix}{suffix}: folded {kept}/{v.numel()} active elements")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Truncate a gate-annotated LoRA to its physical surviving rank.')
    parser.add_argument('input_path', type=str, help='Input LoRA safetensors (with rank gates)')
    parser.add_argument('output_path', type=str, nargs='?', default=None,
                        help='Output path (default: <input>_truncated.safetensors)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Gate value above which a rank is kept (default 0.5)')
    parser.add_argument('--min-rank', type=int, default=1,
                        help='Minimum ranks to keep per LoRA (default 1)')
    parser.add_argument('--dtype', choices=list(DTYPE_MAP.keys()), default='keep',
                        help='Output dtype (default: keep original per-tensor dtypes)')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    if not os.path.exists(input_path):
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    if args.output_path is None:
        stem, ext = os.path.splitext(input_path)
        output_path = f"{stem}_truncated{ext or '.safetensors'}"
    else:
        output_path = os.path.abspath(args.output_path)

    print(f"Loading {input_path}...")
    meta = {}
    state_dict = load_file(input_path, metadata=meta)
    print(f"  {len(state_dict)} tensors")

    lora_gate_keys, diff_gate_keys = find_gate_keys(state_dict)
    if not lora_gate_keys and not diff_gate_keys:
        print("No rank gate keys found in this LoRA (no '.rank_gate.gates' / "
              "'.rank_gate_b.gates'). Nothing to truncate — file left untouched.")
        sys.exit(1)

    print(f"Found {len(lora_gate_keys)} LoRA gate sets and "
          f"{len(diff_gate_keys)} diff gate sets. Truncating (threshold={args.threshold})...")

    # Work on a copy of the non-gate tensors; gate keys are never written out.
    out = {k: v for k, v in state_dict.items()
           if not k.endswith('.rank_gate.gates') and not k.endswith('.rank_gate_b.gates')}

    summary = []
    for gate_key in lora_gate_keys + diff_gate_keys:
        prefix = re.sub(r'\.rank_gate(_b)?\.gates$', '', gate_key)
        gates = state_dict[gate_key]
        if gate_key.endswith('.rank_gate_b.gates'):
            fold_diff(out, prefix, gates, gate_key)
        else:
            # A '.rank_gate.gates' gates either a LoRA pair or (for FullModule
            # layers) a .diff tensor — try the pair first, then the diff.
            if not truncate_lora_pair(out, prefix, gates, args.threshold, args.min_rank, summary):
                fold_diff(out, prefix, gates, gate_key)

    out_dtype = DTYPE_MAP[args.dtype]
    if out_dtype is not None:
        out = {k: v.to(out_dtype) if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    # Preserve the original safetensors metadata (sssd_metadata, format, ...).
    save_file(out, output_path, metadata=meta or None)

    old_total = sum(o for _, o, _ in summary)
    new_total = sum(n for _, _, n in summary)
    print(f"\nTruncated {len(summary)} LoRA modules: {old_total} -> {new_total} total ranks "
          f"({100.0 * new_total / max(1, old_total):.1f}%)")
    print(f"Saved truncated LoRA to {output_path}")


if __name__ == '__main__':
    main()
