from __future__ import annotations
import argparse
from .runner import run

def main():
    p = argparse.ArgumentParser(description="SDG (Scalable Data Generator) CLI")
    p.add_argument("--yaml", required=True, help="YAML blueprint path")
    p.add_argument("--input", required=True, help="Input dataset (.jsonl or .csv)")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--max-batch", type=int, default=8, help="Max concurrent requests")
    p.add_argument("--min-batch", type=int, default=1, help="Min concurrent requests")
    p.add_argument("--target-latency-ms", type=int, default=3000, help="Target average latency per request")
    p.add_argument("--save-intermediate", action="store_true", help="Save intermediate outputs")
    args = p.parse_args()
    run(args.yaml, args.input, args.output, max_batch=args.max_batch, min_batch=args.min_batch, target_latency_ms=args.target_latency_ms, save_intermediate=args.save_intermediate)
