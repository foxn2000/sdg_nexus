from __future__ import annotations
import asyncio, csv, json, os
from typing import Any, Dict, Iterable, List
from .config import load_config
from .executors import run_pipeline

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [ json.loads(line) for line in f if line.strip() ]

def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def run(yaml_path: str, input_path: str, output_path: str, max_batch: int, min_batch: int, target_latency_ms: int, save_intermediate: bool):
    cfg = load_config(yaml_path)
    # load data
    if input_path.endswith(".jsonl"):
        ds = read_jsonl(input_path)
    elif input_path.endswith(".csv"):
        ds = read_csv(input_path)
    else:
        raise ValueError("Unsupported input format. Use .jsonl or .csv")
    # run
    res = asyncio.run(run_pipeline(cfg, ds, max_batch=max_batch, min_batch=min_batch, target_latency_ms=target_latency_ms, save_intermediate=save_intermediate))
    write_jsonl(output_path, res)
