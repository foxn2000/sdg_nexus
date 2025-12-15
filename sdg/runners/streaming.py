from __future__ import annotations
import asyncio
import os
from typing import Any, Dict, Iterable, Optional, Set

from ..config import load_config
from ..executors import run_pipeline_streaming
from ..logger import get_logger
from ..io import (
    AsyncBufferedWriter,
    count_lines_fast,
    read_jsonl,
    read_csv,
    read_hf_dataset,
    apply_mapping,
    load_processed_indices,
)


async def _run_streaming_async(
    cfg,
    dataset: Iterable[Dict[str, Any]],
    output_path: str,
    max_concurrent: int,
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
    # Resume options
    processed_indices: Optional[Set[int]] = None,
    append: bool = False,
):
    """
    ストリーミング版パイプライン実行（非同期）。

    AsyncBufferedWriterを使用して非同期でファイルに書き込み、
    バッファリングによるI/O最適化を行う。

    Args:
        cfg: パイプライン設定
        dataset: 入力データセット（イテラブル）
        output_path: 出力ファイルパス
        max_concurrent: 最大並行処理数
        save_intermediate: 中間結果を保存するか
        show_progress: 進捗表示を行うか
        buffer_size: バッファサイズ（件数）
        flush_interval: 定期フラッシュ間隔（秒）
        clean_output: 出力をクリーニングするか
        total: 総データ数（不明な場合はNone）
        processed_indices: 処理済み行インデックスのセット（再開時に使用）
        append: 追記モードで開くか（再開時はTrue）

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
        append=append,
    ) as writer:
        if progress:
            with progress:
                # 総数が不明な場合は「処理済み件数」のみを表示
                if total is not None:
                    task = progress.add_task(
                        f"[cyan]Processing {total} rows...", total=total
                    )
                else:
                    task = progress.add_task("[cyan]Processing rows...", total=None)

                async for result in run_pipeline_streaming(
                    cfg,
                    dataset,
                    max_concurrent=max_concurrent,
                    save_intermediate=save_intermediate,
                    enable_scheduling=enable_scheduling,
                    max_pending_tasks=max_pending_tasks,
                    chunk_size=chunk_size,
                    enable_memory_optimization=enable_memory_optimization,
                    max_cache_size=max_cache_size,
                    enable_memory_monitoring=enable_memory_monitoring,
                    processed_indices=processed_indices,
                ):
                    completed += 1

                    if result.error:
                        errors += 1
                        logger.debug(f"Error in row {result.row_index}: {result.error}")
                        # エラー時も空の結果を書き込む（行の順序を保持するため後でソートする場合に備え）
                        result_with_error = {
                            "_row_index": result.row_index,
                            "_error": str(result.error),
                            **result.data,
                        }
                        await writer.write(result_with_error)
                    else:
                        # 行インデックスを結果に含める（オプション）
                        result_data = {
                            "_row_index": result.row_index,
                            **result.data,
                        }
                        await writer.write(result_data)

                    progress.update(task, advance=1)
        else:
            # プログレス表示なしの場合
            async for result in run_pipeline_streaming(
                cfg,
                dataset,
                max_concurrent=max_concurrent,
                save_intermediate=save_intermediate,
                enable_scheduling=enable_scheduling,
                max_pending_tasks=max_pending_tasks,
                chunk_size=chunk_size,
                enable_memory_optimization=enable_memory_optimization,
                max_cache_size=max_cache_size,
                enable_memory_monitoring=enable_memory_monitoring,
                processed_indices=processed_indices,
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


def run_streaming(
    yaml_path: str,
    input_path: Optional[str],
    output_path: str,
    max_concurrent: int = 8,
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
    gc_interval: int = 100,
    memory_threshold_mb: int = 1024,
    # Data limit options
    max_inputs: Optional[int] = None,
    # HF Dataset options
    dataset_name: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "train",
    mapping: Optional[Dict[str, str]] = None,
):
    """
    ストリーミング版パイプライン実行

    各データ行を並列処理し、完了した行から順次JSONL出力ファイルへ書き込む。
    途中結果が失われにくく、大量データ処理時のメモリ効率が良い。

    Args:
        yaml_path: YAMLブループリントのパス
        input_path: 入力データセット (.jsonl or .csv)
        output_path: 出力JSONLファイルのパス
        max_concurrent: 同時処理行数の上限
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
        gc_interval: ガベージコレクション実行間隔（処理行数）
        memory_threshold_mb: メモリ使用量警告閾値（MB）
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

    # 初期化: 再開機能用変数
    logger = get_logger()
    processed_indices: Set[int] = set()
    processed_count = 0
    append_mode = False

    # 既存の出力ファイルをチェック（再開機能）
    if os.path.exists(output_path):
        processed_indices, processed_count = load_processed_indices(output_path)
        if processed_count > 0:
            append_mode = True
            if show_progress:
                if logger.locale == "ja":
                    logger.info(
                        f"既存の出力ファイルを検出: {processed_count}件の処理済みデータがあります。続きから再開します。"
                    )
                else:
                    logger.info(
                        f"Existing output file detected: {processed_count} records already processed. Resuming from where it left off."
                    )

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

    # 残り処理件数を計算（プログレス表示用）
    remaining_count: Optional[int] = None
    if total is not None:
        remaining_count = total - processed_count
        if remaining_count < 0:
            remaining_count = 0

    # Print dataset info
    if show_progress:
        subtitle = (
            "Fixed concurrency streaming processing"
            if logger.locale == "en"
            else "固定並行数ストリーミング処理"
        )
        logger.header("SDG Pipeline - Streaming Mode", subtitle)

        # 総数の表示（不明な場合は「不明」と表示）
        total_str = str(total) if total is not None else "unknown"
        remaining_str = (
            str(remaining_count) if remaining_count is not None else "unknown"
        )
        if logger.locale == "ja":
            config_info = {
                "入力データ数": total_str
                + (f" (--max-inputs {max_inputs}で制限)" if max_inputs else ""),
                "処理済み": processed_count,
                "残り": remaining_str,
                "並行処理数": max_concurrent,
            }
            logger.table("実行設定", config_info)
        else:
            config_info = {
                "Input Data Count": total_str
                + (f" (limited by --max-inputs {max_inputs})" if max_inputs else ""),
                "Already Processed": processed_count,
                "Remaining": remaining_str,
                "Concurrency": max_concurrent,
            }
            logger.table("Execution Configuration", config_info)

    # run
    asyncio.run(
        _run_streaming_async(
            cfg,
            ds,
            output_path,
            max_concurrent=max_concurrent,
            save_intermediate=save_intermediate,
            show_progress=show_progress,
            clean_output=clean_output,
            enable_scheduling=enable_scheduling,
            max_pending_tasks=max_pending_tasks,
            chunk_size=chunk_size,
            enable_memory_optimization=enable_memory_optimization,
            max_cache_size=max_cache_size,
            enable_memory_monitoring=enable_memory_monitoring,
            total=remaining_count,
            processed_indices=processed_indices,
            append=append_mode,
        )
    )
