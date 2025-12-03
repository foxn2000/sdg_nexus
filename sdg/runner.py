from __future__ import annotations
import asyncio, csv, json, os, sys
from typing import Any, Dict, Iterable, List, Optional
from .config import load_config
from .executors import (
    run_pipeline,
    run_pipeline_streaming,
    run_pipeline_streaming_adaptive,
    StreamingResult,
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def _run_streaming_async(
    cfg,
    dataset: List[Dict[str, Any]],
    output_path: str,
    max_concurrent: int,
    save_intermediate: bool,
    show_progress: bool = True,
):
    """ストリーミング版パイプライン実行（非同期）"""
    # 出力ディレクトリ作成
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # ファイル書き込み用ロック
    write_lock = asyncio.Lock()

    total = len(dataset)
    completed = 0
    errors = 0

    # 出力ファイルを開く（追記モードではなく新規作成）
    with open(output_path, "w", encoding="utf-8") as f:
        async for result in run_pipeline_streaming(
            cfg,
            dataset,
            max_concurrent=max_concurrent,
            save_intermediate=save_intermediate,
        ):
            completed += 1

            if result.error:
                errors += 1
                if show_progress:
                    print(
                        f"\r[{completed}/{total}] Error in row {result.row_index}: {result.error}",
                        file=sys.stderr,
                    )
                # エラー時も空の結果を書き込む（行の順序を保持するため後でソートする場合に備え）
                result_with_error = {
                    "_row_index": result.row_index,
                    "_error": str(result.error),
                    **result.data,
                }
                async with write_lock:
                    f.write(json.dumps(result_with_error, ensure_ascii=False) + "\n")
                    f.flush()  # 即座にディスクに書き込み
            else:
                # 行インデックスを結果に含める（オプション）
                result_data = {
                    "_row_index": result.row_index,
                    **result.data,
                }
                async with write_lock:
                    f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                    f.flush()  # 即座にディスクに書き込み

                if show_progress:
                    print(
                        f"\r[{completed}/{total}] Completed row {result.row_index}",
                        end="",
                        file=sys.stderr,
                    )

    if show_progress:
        print(file=sys.stderr)  # 最後に改行
        if errors > 0:
            print(
                f"Completed with {errors} errors out of {total} rows.", file=sys.stderr
            )
        else:
            print(f"Successfully processed all {total} rows.", file=sys.stderr)

    return completed, errors


async def _run_streaming_adaptive_async(
    cfg,
    dataset: List[Dict[str, Any]],
    output_path: str,
    max_concurrent: int,
    min_concurrent: int,
    target_latency_ms: int,
    target_queue_depth: int,
    metrics_type: str,
    save_intermediate: bool,
    show_progress: bool = True,
):
    """適応的並行性制御付きストリーミング版パイプライン実行（非同期）"""
    # 出力ディレクトリ作成
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # ファイル書き込み用ロック
    write_lock = asyncio.Lock()

    total = len(dataset)
    completed = 0
    errors = 0

    # 出力ファイルを開く（追記モードではなく新規作成）
    with open(output_path, "w", encoding="utf-8") as f:
        async for result in run_pipeline_streaming_adaptive(
            cfg,
            dataset,
            max_concurrent=max_concurrent,
            min_concurrent=min_concurrent,
            target_latency_ms=target_latency_ms,
            target_queue_depth=target_queue_depth,
            metrics_type=metrics_type,
            save_intermediate=save_intermediate,
        ):
            completed += 1

            if result.error:
                errors += 1
                if show_progress:
                    print(
                        f"\r[{completed}/{total}] Error in row {result.row_index}: {result.error}",
                        file=sys.stderr,
                    )
                # エラー時も空の結果を書き込む
                result_with_error = {
                    "_row_index": result.row_index,
                    "_error": str(result.error),
                    **result.data,
                }
                async with write_lock:
                    f.write(json.dumps(result_with_error, ensure_ascii=False) + "\n")
                    f.flush()
            else:
                result_data = {
                    "_row_index": result.row_index,
                    **result.data,
                }
                async with write_lock:
                    f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                    f.flush()

                if show_progress:
                    print(
                        f"\r[{completed}/{total}] Completed row {result.row_index}",
                        end="",
                        file=sys.stderr,
                    )

    if show_progress:
        print(file=sys.stderr)  # 最後に改行
        if errors > 0:
            print(
                f"Completed with {errors} errors out of {total} rows.", file=sys.stderr
            )
        else:
            print(f"Successfully processed all {total} rows.", file=sys.stderr)

    return completed, errors


def run_streaming_adaptive(
    yaml_path: str,
    input_path: str,
    output_path: str,
    max_concurrent: int = 64,
    min_concurrent: int = 1,
    target_latency_ms: int = 3000,
    target_queue_depth: int = 32,
    metrics_type: str = "none",
    save_intermediate: bool = False,
    show_progress: bool = True,
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

    Note:
        出力順序は処理完了順となるため、入力順序と異なる場合がある。
        元の順序が必要な場合は _row_index フィールドでソートすること。
    """
    cfg = load_config(yaml_path)

    # load data
    if input_path.endswith(".jsonl"):
        ds = read_jsonl(input_path)
    elif input_path.endswith(".csv"):
        ds = read_csv(input_path)
    else:
        raise ValueError("Unsupported input format. Use .jsonl or .csv")

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
        )
    )


def run_streaming(
    yaml_path: str,
    input_path: str,
    output_path: str,
    max_concurrent: int = 8,
    save_intermediate: bool = False,
    show_progress: bool = True,
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

    Note:
        出力順序は処理完了順となるため、入力順序と異なる場合がある。
        元の順序が必要な場合は _row_index フィールドでソートすること。
    """
    cfg = load_config(yaml_path)

    # load data
    if input_path.endswith(".jsonl"):
        ds = read_jsonl(input_path)
    elif input_path.endswith(".csv"):
        ds = read_csv(input_path)
    else:
        raise ValueError("Unsupported input format. Use .jsonl or .csv")

    # run
    asyncio.run(
        _run_streaming_async(
            cfg,
            ds,
            output_path,
            max_concurrent=max_concurrent,
            save_intermediate=save_intermediate,
            show_progress=show_progress,
        )
    )


def run(
    yaml_path: str,
    input_path: str,
    output_path: str,
    max_batch: int,
    min_batch: int,
    target_latency_ms: int,
    save_intermediate: bool,
):
    """
    従来のブロック単位一括処理パイプライン実行（後方互換性のため維持）
    """
    cfg = load_config(yaml_path)
    # load data
    if input_path.endswith(".jsonl"):
        ds = read_jsonl(input_path)
    elif input_path.endswith(".csv"):
        ds = read_csv(input_path)
    else:
        raise ValueError("Unsupported input format. Use .jsonl or .csv")
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
