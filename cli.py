"""
Command-line interface for tensorshrink.

    python -m tensorshrink.cli quantize --model-dir <hf checkpoint dir> --out <out.tsk>
    python -m tensorshrink.cli inspect <container.tsk>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .container import ContainerReader, stream_quantize_safetensors_dir


def _cmd_quantize(args: argparse.Namespace) -> int:
    stats = stream_quantize_safetensors_dir(
        args.model_dir,
        args.out,
        group_size=args.group_size,
        outlier_frac=args.outlier_frac,
        error_threshold=args.error_threshold,
        min_numel=args.min_numel,
        progress=not args.quiet,
        vram_budget_mb=args.vram_budget_mb,
    )
    orig_gb = stats["orig_bytes"] / 1e9
    container_gb = stats["container_file_bytes"] / 1e9
    print()
    print(f"tensors      : {stats['n_tensors']}")
    print(f"original     : {orig_gb:.3f} GB")
    print(f"container    : {container_gb:.3f} GB")
    print(f"ratio        : {orig_gb/container_gb:.2f}x")
    print(f"written to   : {args.out}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    r = ContainerReader(args.container)
    stats = r.stats()
    print(json.dumps(stats, indent=2))
    if args.list:
        for name in r.names():
            entry = r.manifest["tensors"][name]
            tag = f"{entry['bits']}b" if entry["kind"] == "quant" else "raw"
            print(f"  [{tag:>4s}] {name}  shape={entry['orig_shape']}")
    r.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tensorshrink", description="Weight compression for local transformer/diffusion models.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_q = sub.add_parser("quantize", help="quantize a HF safetensors checkpoint directory into a .tsk container")
    p_q.add_argument("--model-dir", required=True, type=Path)
    p_q.add_argument("--out", required=True, type=Path)
    p_q.add_argument("--group-size", type=int, default=128)
    p_q.add_argument("--outlier-frac", type=float, default=0.003)
    p_q.add_argument("--error-threshold", type=float, default=0.06)
    p_q.add_argument("--min-numel", type=int, default=16384)
    p_q.add_argument("--quiet", action="store_true")
    p_q.set_defaults(func=_cmd_quantize)

    p_i = sub.add_parser("inspect", help="print summary stats for a .tsk container")
    p_i.add_argument("container", type=Path)
    p_i.add_argument("--list", action="store_true", help="also list every tensor")
    p_i.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
