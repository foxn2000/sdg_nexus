from __future__ import annotations
import argparse
import sys
from .runner import run, run_streaming


def build_run_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--yaml", required=True, help="YAML blueprint path")
    p.add_argument("--input", required=True, help="Input dataset (.jsonl or .csv)")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--save-intermediate", action="store_true", help="Save intermediate outputs")
    
    # Streaming mode options (default mode)
    p.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent rows to process (streaming mode, default)")
    p.add_argument("--no-progress", action="store_true", help="Disable progress display (streaming mode)")
    
    # Batch mode options (block-by-block processing)
    p.add_argument("--batch-mode", action="store_true", help="Use batch mode (block-by-block processing instead of streaming)")
    p.add_argument("--max-batch", type=int, default=8, help="Max concurrent requests per block (batch mode only)")
    p.add_argument("--min-batch", type=int, default=1, help="Min concurrent requests per block (batch mode only)")
    p.add_argument("--target-latency-ms", type=int, default=3000, help="Target average latency per request (batch mode only)")
    
    # Legacy option (hidden, for backward compatibility)
    p.add_argument("--streaming", action="store_true", help=argparse.SUPPRESS)  # Now default, kept for compatibility
    p.add_argument("--max-concurrent-rows", type=int, default=None, help=argparse.SUPPRESS)  # Alias for --max-concurrent
    return p


def _execute_run(args):
    """Execute the run command based on args"""
    # Batch mode is now opt-in; streaming is default
    if args.batch_mode:
        # Batch mode: block-by-block processing
        run(
            args.yaml,
            args.input,
            args.output,
            max_batch=args.max_batch,
            min_batch=args.min_batch,
            target_latency_ms=args.target_latency_ms,
            save_intermediate=args.save_intermediate,
        )
    else:
        # Streaming mode: row-by-row processing (default)
        # Support legacy --max-concurrent-rows option
        max_concurrent = args.max_concurrent_rows if args.max_concurrent_rows is not None else args.max_concurrent
        run_streaming(
            args.yaml,
            args.input,
            args.output,
            max_concurrent=max_concurrent,
            save_intermediate=args.save_intermediate,
            show_progress=not args.no_progress,
        )


def main():
    argv = sys.argv[1:]

    # Backward compatibility: support legacy usage `sdg --yaml ...`
    legacy_mode = len(argv) > 0 and not argv[0] in {"run"} and argv[0].startswith("--")

    if legacy_mode:
        p = argparse.ArgumentParser(description="SDG (Scalable Data Generator) CLI [legacy mode: sdg --yaml ...]")
        build_run_parser(p)
        args = p.parse_args(argv)
        _execute_run(args)
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
        _execute_run(args)
