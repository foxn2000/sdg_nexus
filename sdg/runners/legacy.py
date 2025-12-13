from __future__ import annotations
import asyncio
from typing import Dict, Optional

from ..config import load_config
from ..executors import run_pipeline
from ..logger import get_logger
from ..io import (
    read_jsonl,
    read_csv,
    read_hf_dataset,
    apply_mapping,
    write_jsonl,
)


def run(
    yaml_path: str,
    input_path: Optional[str],
    output_path: str,
    max_batch: int,
    min_batch: int,
    target_latency_ms: int,
    save_intermediate: bool,
    # Data limit options
    max_inputs: Optional[int] = None,
    # HF Dataset options
    dataset_name: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "train",
    mapping: Optional[Dict[str, str]] = None,
):
    """
    従来のブロック単位一括処理パイプライン実行（後方互換性のため維持）
    """
    cfg = load_config(yaml_path)
    # load data
    if input_path:
        if input_path.endswith(".jsonl"):
            ds = read_jsonl(input_path, max_inputs=max_inputs)
        elif input_path.endswith(".csv"):
            ds = read_csv(input_path, max_inputs=max_inputs)
        else:
            raise ValueError("Unsupported input format. Use .jsonl or .csv")
    elif dataset_name:
        ds = read_hf_dataset(dataset_name, subset, split, max_inputs=max_inputs)
    else:
        raise ValueError("Either input_path or dataset_name must be provided")

    # Apply mapping
    if mapping:
        ds = apply_mapping(ds, mapping)

    # Print dataset info
    logger = get_logger()
    subtitle = "Legacy batch processing mode" if logger.locale == "en" else "レガシーバッチ処理モード"
    logger.header("SDG Pipeline - Legacy Batch Mode", subtitle)
    
    if logger.locale == "ja":
        config_info = {
            "入力データ数": f"{len(ds)}" + (f" (--max-inputs {max_inputs}で制限)" if max_inputs else ""),
            "最大バッチ": max_batch,
            "最小バッチ": min_batch,
            "目標レイテンシ": f"{target_latency_ms}ms",
        }
        logger.table("実行設定", config_info)
    else:
        config_info = {
            "Input Data Count": f"{len(ds)}" + (f" (limited by --max-inputs {max_inputs})" if max_inputs else ""),
            "Max Batch": max_batch,
            "Min Batch": min_batch,
            "Target Latency": f"{target_latency_ms}ms",
        }
        logger.table("Execution Configuration", config_info)

    # run
    res = asyncio.run(
        run_pipeline(
            cfg,
            ds,
            max_batch=max_batch,
            min_batch=min_batch,
            target_latency_ms=target_latency_ms,
            save_intermediate=save_intermediate,
        )
    )
    write_jsonl(output_path, res)
