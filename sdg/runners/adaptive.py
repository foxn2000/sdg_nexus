from __future__ import annotations
import asyncio
from typing import Any, Dict, Iterable, Optional

from ..config import load_config
from ..executors import run_pipeline_streaming_adaptive
from ..logger import get_logger
from ..io import (
    AsyncBufferedWriter,
    count_lines_fast,
    read_jsonl,
    read_csv,
    read_hf_dataset,
    apply_mapping,
)


async def _run_streaming_adaptive_async(
    cfg,
    dataset: Iterable[Dict[str, Any]],
    output_path: str,
    max_concurrent: int,
    min_concurrent: int,
    target_latency_ms: int,
    target_queue_depth: int,
    metrics_type: str,
    save_intermediate: bool,
    show_progress: bool = True,
    buffer_size: int = AsyncBufferedWriter.DEFAULT_BUFFER_SIZE,
    flush_interval: float = AsyncBufferedWriter.DEFAULT_FLUSH_INTERVAL,
    clean_output: bool = True,
    # Phase 2: Scheduling options
    enable_scheduling: bool = False,
    max_pending_tasks: int = 1000,
    chunk_size: int = 100,
    # Phase 2: Memory optimization options
    enable_memory_optimization: bool = False,
    max_cache_size: int = 500,
    enable_memory_monitoring: bool = False,
    # Total count (optional, for progress display)
    total: Optional[int] = None,
):
    """
    適応的並行性制御付きストリーミング版パイプライン実行（非同期）。

    AsyncBufferedWriterを使用して非同期でファイルに書き込み、
    バッファリングによるI/O最適化を行う。

    Args:
        cfg: パイプライン設定
        dataset: 入力データセット（イテラブル）
        output_path: 出力ファイルパス
        max_concurrent: 最大並行処理数
        min_concurrent: 最小並行処理数
        target_latency_ms: 目標レイテンシ（ミリ秒）
        target_queue_depth: 目標キュー深度
        metrics_type: メトリクスタイプ
        save_intermediate: 中間結果を保存するか
        show_progress: 進捗表示を行うか
        buffer_size: バッファサイズ（件数）
        flush_interval: 定期フラッシュ間隔（秒）
        clean_output: 出力をクリーニングするか
        total: 総データ数（不明な場合はNone）

    Returns:
        (完了数, エラー数) のタプル
    """
    completed = 0
    errors = 0

    logger = get_logger()
    progress = logger.create_progress() if show_progress else None

    # AsyncBufferedWriterを使用して非同期で書き込み
    async with AsyncBufferedWriter(
        output_path,
        buffer_size=buffer_size,
        flush_interval=flush_interval,
        clean_output=clean_output,
    ) as writer:
        if progress:
            with progress:
                # 総数が不明な場合は「処理済み件数」のみを表示
                if total is not None:
                    task = progress.add_task(
                        f"[cyan]Processing {total} rows (adaptive)...", total=total
                    )
                else:
                    task = progress.add_task(
                        "[cyan]Processing rows (adaptive)...", total=None
                    )

                async for result in run_pipeline_streaming_adaptive(
                    cfg,
                    dataset,
                    max_concurrent=max_concurrent,
                    min_concurrent=min_concurrent,
                    target_latency_ms=target_latency_ms,
                    target_queue_depth=target_queue_depth,
                    metrics_type=metrics_type,
                    save_intermediate=save_intermediate,
                    enable_scheduling=enable_scheduling,
                    max_pending_tasks=max_pending_tasks,
                    chunk_size=chunk_size,
                    enable_memory_optimization=enable_memory_optimization,
                    max_cache_size=max_cache_size,
                    enable_memory_monitoring=enable_memory_monitoring,
                ):
                    completed += 1

                    if result.error:
                        errors += 1
                        logger.debug(f"Error in row {result.row_index}: {result.error}")
                        # エラー時も空の結果を書き込む
                        result_with_error = {
                            "_row_index": result.row_index,
                            "_error": str(result.error),
                            **result.data,
                        }
                        await writer.write(result_with_error)
                    else:
                        result_data = {
                            "_row_index": result.row_index,
                            **result.data,
                        }
                        await writer.write(result_data)

                    progress.update(task, advance=1)
        else:
            # プログレス表示なしの場合
            async for result in run_pipeline_streaming_adaptive(
                cfg,
                dataset,
                max_concurrent=max_concurrent,
                min_concurrent=min_concurrent,
                target_latency_ms=target_latency_ms,
                target_queue_depth=target_queue_depth,
                metrics_type=metrics_type,
                save_intermediate=save_intermediate,
                enable_scheduling=enable_scheduling,
                max_pending_tasks=max_pending_tasks,
                chunk_size=chunk_size,
                enable_memory_optimization=enable_memory_optimization,
                max_cache_size=max_cache_size,
                enable_memory_monitoring=enable_memory_monitoring,
            ):
                completed += 1

                if result.error:
                    errors += 1
                    result_with_error = {
                        "_row_index": result.row_index,
                        "_error": str(result.error),
                        **result.data,
                    }
                    await writer.write(result_with_error)
                else:
                    result_data = {
                        "_row_index": result.row_index,
                        **result.data,
                    }
                    await writer.write(result_data)

    logger = get_logger()
    if show_progress:
        if writer.total_cleaned > 0:
            logger.info(f"Cleaned {writer.total_cleaned} invalid JSON lines")

        stats = {
            "total": total if total is not None else completed,
            "completed": completed,
            "errors": errors,
        }
        logger.print_stats(stats)

    return completed, errors


def run_streaming_adaptive(
    yaml_path: str,
    input_path: Optional[str],
    output_path: str,
    max_concurrent: int = 64,
    min_concurrent: int = 1,
    target_latency_ms: int = 3000,
    target_queue_depth: int = 32,
    metrics_type: str = "none",
    save_intermediate: bool = False,
    show_progress: bool = True,
    use_shared_transport: bool = False,
    http2: bool = True,
    # LLM retry options
    retry_on_empty: bool = True,
    # JSONL cleaning options
    clean_output: bool = True,
    # Phase 2: Scheduling options
    enable_scheduling: bool = False,
    max_pending_tasks: int = 1000,
    chunk_size: int = 100,
    # Phase 2: Memory optimization options
    enable_memory_optimization: bool = False,
    max_cache_size: int = 500,
    enable_memory_monitoring: bool = False,
    # Data limit options
    max_inputs: Optional[int] = None,
    # HF Dataset options
    dataset_name: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "train",
    mapping: Optional[Dict[str, str]] = None,
):
    """
    適応的並行性制御付きストリーミング版パイプライン実行

    レイテンシとオプションのバックエンドメトリクスに基づいて、
    並行処理数を動的に調整しながら実行する。

    Args:
        yaml_path: YAMLブループリントのパス
        input_path: 入力データセット (.jsonl or .csv)
        output_path: 出力JSONLファイルのパス
        max_concurrent: 同時処理行数の上限 (デフォルト: 64)
        min_concurrent: 同時処理行数の下限 (デフォルト: 1)
        target_latency_ms: 目標レイテンシ (ミリ秒、デフォルト: 3000)
        target_queue_depth: 目標バックエンドキュー深度 (デフォルト: 32)
        metrics_type: メトリクスタイプ ("none", "vllm", or "sglang")
        save_intermediate: 中間結果を保存するか
        show_progress: 進捗表示を行うか
        use_shared_transport: 共有HTTPトランスポートを使用するか
        http2: HTTP/2を有効にするか
        retry_on_empty: 空返答時にリトライするか（デフォルト: True）
        enable_scheduling: 階層的タスクスケジューリングを有効化
        max_pending_tasks: 最大保留タスク数（スケジューリング有効時）
        chunk_size: データセット分割サイズ（スケジューリング有効時）
        enable_memory_optimization: メモリ最適化を有効化
        max_cache_size: コンテキストキャッシュの最大サイズ
        enable_memory_monitoring: メモリ使用状況監視を有効化
        clean_output: 出力JSONLをクリーニングするか（デフォルト: True）
        dataset_name: Hugging Faceデータセット名
        subset: データセットサブセット
        split: データセットスプリット
        mapping: キーマッピング辞書

    Note:
        出力順序は処理完了順となるため、入力順序と異なる場合がある。
        元の順序が必要な場合は _row_index フィールドでソートすること。
    """
    cfg = load_config(yaml_path)

    # 最適化オプションを設定
    cfg.optimization = {
        "use_shared_transport": use_shared_transport,
        "http2": http2,
        "retry_on_empty": retry_on_empty,
    }

    # load data and count lines for progress display
    total: Optional[int] = None
    if input_path:
        if input_path.endswith(".jsonl"):
            # 高速行数カウント（wcコマンド使用）
            total = count_lines_fast(input_path)
            if max_inputs is not None and total is not None:
                total = min(total, max_inputs)
            ds = read_jsonl(input_path, max_inputs=max_inputs)
        elif input_path.endswith(".csv"):
            # CSVの場合は行数カウント（ヘッダー行を除く）
            line_count = count_lines_fast(input_path)
            if line_count is not None:
                total = line_count - 1  # ヘッダー行を除く
                if max_inputs is not None:
                    total = min(total, max_inputs)
            ds = read_csv(input_path, max_inputs=max_inputs)
        else:
            raise ValueError("Unsupported input format. Use .jsonl or .csv")
    elif dataset_name:
        ds = read_hf_dataset(dataset_name, subset, split, max_inputs=max_inputs)
        # HF Datasetの場合は総数が不明（streaming=True）
        total = max_inputs  # max_inputsが指定されていればそれを使用
    else:
        raise ValueError("Either input_path or dataset_name must be provided")

    # Apply mapping
    if mapping:
        ds = apply_mapping(ds, mapping)

    # Print dataset info
    logger = get_logger()
    if show_progress:
        subtitle = (
            "Adaptive concurrency control"
            if logger.locale == "en"
            else "適応的並行性制御"
        )
        logger.header("SDG Pipeline - Adaptive Mode", subtitle)

        # 総数の表示（不明な場合は「不明」と表示）
        total_str = str(total) if total is not None else "unknown"
        if logger.locale == "ja":
            config_info = {
                "入力データ数": total_str
                + (f" (--max-inputs {max_inputs}で制限)" if max_inputs else ""),
                "最大並行数": max_concurrent,
                "最小並行数": min_concurrent,
                "目標レイテンシ": f"{target_latency_ms}ms",
                "メトリクスタイプ": metrics_type,
            }
            logger.table("実行設定", config_info)
        else:
            config_info = {
                "Input Data Count": total_str
                + (f" (limited by --max-inputs {max_inputs})" if max_inputs else ""),
                "Max Concurrency": max_concurrent,
                "Min Concurrency": min_concurrent,
                "Target Latency": f"{target_latency_ms}ms",
                "Metrics Type": metrics_type,
            }
            logger.table("Execution Configuration", config_info)

    # run
    asyncio.run(
        _run_streaming_adaptive_async(
            cfg,
            ds,
            output_path,
            max_concurrent=max_concurrent,
            min_concurrent=min_concurrent,
            target_latency_ms=target_latency_ms,
            target_queue_depth=target_queue_depth,
            metrics_type=metrics_type,
            save_intermediate=save_intermediate,
            show_progress=show_progress,
            clean_output=clean_output,
            enable_scheduling=enable_scheduling,
            max_pending_tasks=max_pending_tasks,
            chunk_size=chunk_size,
            enable_memory_optimization=enable_memory_optimization,
            max_cache_size=max_cache_size,
            enable_memory_monitoring=enable_memory_monitoring,
            total=total,
        )
    )
