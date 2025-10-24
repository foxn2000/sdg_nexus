from __future__ import annotations
import argparse
import sys
from .runner import run


def build_run_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--yaml", required=True, help="YAML blueprint path")
    p.add_argument("--input", required=True, help="Input dataset (.jsonl or .csv)")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--max-batch", type=int, default=8, help="Max concurrent requests")
    p.add_argument("--min-batch", type=int, default=1, help="Min concurrent requests")
    p.add_argument("--target-latency-ms", type=int, default=3000, help="Target average latency per request")
    p.add_argument("--save-intermediate", action="store_true", help="Save intermediate outputs")
    return p


def main():
    argv = sys.argv[1:]

    # Backward compatibility: support legacy usage `sdg --yaml ...`
    legacy_mode = len(argv) > 0 and not argv[0] in {"run"} and argv[0].startswith("--")

    if legacy_mode:
        p = argparse.ArgumentParser(description="SDG (Scalable Data Generator) CLI [legacy mode: sdg --yaml ...]")
        build_run_parser(p)
        args = p.parse_args(argv)
        run(
            args.yaml,
            args.input,
            args.output,
            max_batch=args.max_batch,
            min_batch=args.min_batch,
            target_latency_ms=args.target_latency_ms,
            save_intermediate=args.save_intermediate,
        )
        return

    # Subcommand style: `sdg run --yaml ...`
    p = argparse.ArgumentParser(description="SDG (Scalable Data Generator) CLI")
    sub = p.add_subparsers(dest="command")
    # Python 3.10 supports required for subparsers
    try:
        sub.required = True  # type: ignore[attr-defined]
    except Exception:
        pass

    run_p = sub.add_parser("run", help="Run a YAML blueprint over an input dataset")
    build_run_parser(run_p)

    args = p.parse_args(argv)

    if args.command == "run":
        run(
            args.yaml,
            args.input,
            args.output,
            max_batch=args.max_batch,
            min_batch=args.min_batch,
            target_latency_ms=args.target_latency_ms,
            save_intermediate=args.save_intermediate,
        )
