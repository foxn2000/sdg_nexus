from __future__ import annotations
import asyncio
import csv
import json
import os
import sys
import time
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional

import aiofiles
import aiofiles.os

from .config import load_config
from .executors import (
    run_pipeline,
    run_pipeline_streaming,
    run_pipeline_streaming_adaptive,
    run_pipeline_streaming_adaptive_batched,
    StreamingResult,
)
from .logger import get_logger

# clean_jsonl_line is no longer used in AsyncBufferedWriter.write/write_many
# since Dict input with json.dumps guarantees valid JSON format


class AsyncBufferedWriter:
    """
    非同期バッファリングファイルライター。

    aiofilesを使用して非同期でファイルに書き込み、バッファリングによって
    I/O操作を最適化する。定期的なフラッシュと一定件数到達時のフラッシュを
    両方サポートし、堅牢なフォールバック処理を提供する。

    Attributes:
        DEFAULT_BUFFER_SIZE: デフォルトのバッファサイズ（件数）
        DEFAULT_FLUSH_INTERVAL: デフォルトのフラッシュ間隔（秒）
        DEFAULT_MAX_RETRIES: デフォルトの最大リトライ回数

    Example:
        async with AsyncBufferedWriter("output.jsonl") as writer:
            await writer.write({"key": "value"})
            # バッファが閾値に達するか、定期フラッシュ時に自動書き込み
    """

    DEFAULT_BUFFER_SIZE: ClassVar[int] = 100
    DEFAULT_FLUSH_INTERVAL: ClassVar[float] = 5.0
    DEFAULT_MAX_RETRIES: ClassVar[int] = 3

    def __init__(
        self,
        path: str,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        encoding: str = "utf-8",
        serializer: Optional[Callable[[Dict[str, Any]], str]] = None,
        clean_output: bool = True,
    ):
        """
        AsyncBufferedWriterを初期化する。

        Args:
            path: 出力ファイルパス
            buffer_size: バッファサイズ（件数、デフォルト: 100）
                        この件数に達するとバッファをフラッシュする
            flush_interval: 定期フラッシュ間隔（秒、デフォルト: 5.0）
                           この間隔ごとにバッファをフラッシュする
            max_retries: 書き込み失敗時の最大リトライ回数（デフォルト: 3）
            encoding: ファイルエンコーディング（デフォルト: utf-8）
            serializer: カスタムシリアライザ関数（デフォルト: JSON）
            clean_output: 出力をクリーニングするか（デフォルト: True）
        """
        self._path = path
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._encoding = encoding
        self._serializer = serializer or self._default_serializer
        self._clean_output = clean_output

        # 内部状態
        self._buffer: List[str] = []
        self._lock = asyncio.Lock()
        self._file: Optional[aiofiles.threadpool.binary.AsyncBufferedIOBase] = None
        self._last_flush_time: float = 0.0
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._total_written: int = 0
        self._total_errors: int = 0
        self._total_cleaned: int = 0
        self._fallback_buffer: List[str] = []  # フォールバック用バッファ

    @staticmethod
    def _default_serializer(data: Dict[str, Any]) -> str:
        """デフォルトのJSONシリアライザ。"""
        return json.dumps(data, ensure_ascii=False)

    async def __aenter__(self) -> "AsyncBufferedWriter":
        """非同期コンテキストマネージャーのエントリーポイント。"""
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """非同期コンテキストマネージャーの終了ポイント。"""
        await self.close()

    async def open(self) -> None:
        """
        ファイルを開き、定期フラッシュタスクを開始する。
        """
        # ディレクトリを作成
        dir_name = os.path.dirname(self._path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # ファイルを開く
        self._file = await aiofiles.open(
            self._path,
            mode="w",
            encoding=self._encoding,
        )
        self._last_flush_time = time.time()
        self._running = True

        # 定期フラッシュタスクを開始
        self._flush_task = asyncio.create_task(self._periodic_flush_loop())

    async def close(self) -> None:
        """
        残りのバッファをフラッシュし、ファイルを閉じる。
        """
        self._running = False

        # 定期フラッシュタスクを停止
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # 残りのバッファをフラッシュ
        await self._flush_buffer()

        # フォールバックバッファも書き込み
        await self._flush_fallback_buffer()

        # ファイルを閉じる
        if self._file is not None:
            await self._file.close()
            self._file = None

    async def write(self, data: Dict[str, Any]) -> bool:
        """
        データをバッファに追加する。

        バッファサイズに達した場合は自動的にフラッシュする。

        Args:
            data: 書き込むデータ（辞書）

        Returns:
            書き込みに成功した場合はTrue、失敗した場合はFalse

        Note:
            入力dataはDict型であり、json.dumpsでシリアライズした時点で
            有効なJSON形式が保証されるため、clean_jsonl_lineによる
            再パース・正規化処理はスキップされる（パフォーマンス最適化）。
        """
        try:
            line = self._serializer(data)
            # 入力がDictでシリアライズが成功した場合、有効なJSON形式であることが保証される
            # clean_jsonl_lineによる再パース・正規化は不要（CPUリソースの節約）
            line = line + "\n"
        except Exception as e:
            self._total_errors += 1
            print(
                f"Serialization error: {e}",
                file=sys.stderr,
            )
            return False

        async with self._lock:
            self._buffer.append(line)

            # バッファサイズチェック
            if len(self._buffer) >= self._buffer_size:
                await self._flush_buffer_unlocked()

        return True

    async def write_many(self, data_list: List[Dict[str, Any]]) -> int:
        """
        複数のデータをバッファに追加する。

        Args:
            data_list: 書き込むデータのリスト

        Returns:
            正常に追加されたデータの件数

        Note:
            入力dataはDict型であり、json.dumpsでシリアライズした時点で
            有効なJSON形式が保証されるため、clean_jsonl_lineによる
            再パース・正規化処理はスキップされる（パフォーマンス最適化）。
        """
        success_count = 0
        lines: List[str] = []

        for data in data_list:
            try:
                line = self._serializer(data)
                # 入力がDictでシリアライズが成功した場合、有効なJSON形式であることが保証される
                # clean_jsonl_lineによる再パース・正規化は不要（CPUリソースの節約）
                line = line + "\n"
                lines.append(line)
                success_count += 1
            except Exception as e:
                self._total_errors += 1
                print(
                    f"Serialization error: {e}",
                    file=sys.stderr,
                )

        if lines:
            async with self._lock:
                self._buffer.extend(lines)

                # バッファサイズチェック
                if len(self._buffer) >= self._buffer_size:
                    await self._flush_buffer_unlocked()

        return success_count

    async def flush(self) -> None:
        """
        バッファを強制的にフラッシュする。
        """
        async with self._lock:
            await self._flush_buffer_unlocked()

    async def _flush_buffer(self) -> None:
        """ロック付きでバッファをフラッシュする。"""
        async with self._lock:
            await self._flush_buffer_unlocked()

    async def _flush_buffer_unlocked(self) -> None:
        """
        バッファをファイルにフラッシュする（ロックなし）。

        呼び出し元で_lockを取得していることを前提とする。
        """
        if not self._buffer:
            return

        if self._file is None:
            # ファイルが開かれていない場合はフォールバックバッファに追加
            self._fallback_buffer.extend(self._buffer)
            self._buffer.clear()
            return

        # バッファの内容を結合
        content = "".join(self._buffer)
        buffer_count = len(self._buffer)

        # リトライ付きで書き込み
        success = False
        for attempt in range(self._max_retries):
            try:
                await self._file.write(content)
                await self._file.flush()
                success = True
                self._total_written += buffer_count
                break
            except Exception as e:
                if attempt < self._max_retries - 1:
                    # リトライ前に少し待機
                    await asyncio.sleep(0.1 * (attempt + 1))
                else:
                    # 最大リトライ回数に達した場合
                    self._total_errors += buffer_count
                    print(
                        f"Write error after {self._max_retries} attempts: {e}",
                        file=sys.stderr,
                    )
                    # フォールバックバッファに追加
                    self._fallback_buffer.extend(self._buffer)

        if success:
            self._buffer.clear()
            self._last_flush_time = time.time()
        else:
            self._buffer.clear()  # エラーでもバッファはクリア（フォールバックに移動済み）

    async def _flush_fallback_buffer(self) -> None:
        """
        フォールバックバッファをファイルに書き込む。

        通常の書き込みが失敗した場合のデータ回復用。
        """
        if not self._fallback_buffer or self._file is None:
            return

        try:
            content = "".join(self._fallback_buffer)
            await self._file.write(content)
            await self._file.flush()
            self._total_written += len(self._fallback_buffer)
            self._fallback_buffer.clear()
        except Exception as e:
            print(
                f"Fallback buffer write error: {e}",
                file=sys.stderr,
            )

    async def _periodic_flush_loop(self) -> None:
        """定期フラッシュのバックグラウンドループ。"""
        while self._running:
            await asyncio.sleep(self._flush_interval)

            if not self._running:
                break

            # 最後のフラッシュから十分な時間が経過している場合のみフラッシュ
            if time.time() - self._last_flush_time >= self._flush_interval:
                await self._flush_buffer()

    @property
    def total_written(self) -> int:
        """書き込み成功した総件数を返す。"""
        return self._total_written

    @property
    def total_errors(self) -> int:
        """エラーとなった総件数を返す。"""
        return self._total_errors

    @property
    def total_cleaned(self) -> int:
        """クリーニングされた総件数を返す。"""
        return self._total_cleaned

    @property
    def buffer_size(self) -> int:
        """現在のバッファ内の件数を返す。"""
        return len(self._buffer)

    @property
    def pending_count(self) -> int:
        """フラッシュ待ちの総件数（バッファ+フォールバック）を返す。"""
        return len(self._buffer) + len(self._fallback_buffer)


def read_jsonl(path: str, max_inputs: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    JSONLファイルを読み込む。

    Args:
        path: 入力ファイルパス
        max_inputs: 読み込む最大行数（Noneの場合は全件）

    Returns:
        各行をパースした辞書のリスト
    """
    with open(path, "r", encoding="utf-8") as f:
        if max_inputs is None:
            return [json.loads(line) for line in f if line.strip()]
        else:
            result = []
            for i, line in enumerate(f):
                if i >= max_inputs:
                    break
                if line.strip():
                    result.append(json.loads(line))
            return result


def read_csv(path: str, max_inputs: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    CSVファイルを読み込む。

    Args:
        path: 入力ファイルパス
        max_inputs: 読み込む最大行数（Noneの場合は全件）

    Returns:
        各行を辞書に変換したリスト
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if max_inputs is None:
            return list(reader)
        else:
            result = []
            for i, row in enumerate(reader):
                if i >= max_inputs:
                    break
                result.append(row)
            return result


def read_hf_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    max_inputs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Hugging Face Datasetを読み込む。

    Args:
        dataset_name: データセット名
        subset: サブセット名（オプション）
        split: スプリット名（デフォルト: train）
        max_inputs: 読み込む最大行数（Noneの場合は全件）

    Returns:
        辞書のリスト
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library is required to load Hugging Face datasets. "
            "Please install it with `pip install datasets`."
        )

    logger = get_logger()
    logger.info(
        f"Loading Hugging Face dataset: {dataset_name} (subset={subset}, split={split})..."
    )
    ds = load_dataset(dataset_name, name=subset, split=split)

    # Convert to list of dicts
    if max_inputs is None:
        return [item for item in ds]
    else:
        result = []
        for i, item in enumerate(ds):
            if i >= max_inputs:
                break
            result.append(item)
        return result


def apply_mapping(
    dataset: List[Dict[str, Any]], mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    データセットのキーをマッピングに従って変更する。

    Args:
        dataset: データセット（辞書のリスト）
        mapping: マッピング辞書 {old_key: new_key}

    Returns:
        キーが変更された新しいデータセット
    """
    if not mapping:
        return dataset

    mapped_dataset = []
    for item in dataset:
        new_item = item.copy()
        for old_key, new_key in mapping.items():
            if old_key in new_item:
                new_item[new_key] = new_item.pop(old_key)
        mapped_dataset.append(new_item)
    return mapped_dataset


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """
    JSONLファイルに書き込む（同期版）。

    Args:
        path: 出力ファイルパス
        rows: 書き込むデータのイテラブル
    """
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
):
    """
    ストリーミング版パイプライン実行（非同期）。

    AsyncBufferedWriterを使用して非同期でファイルに書き込み、
    バッファリングによるI/O最適化を行う。

    Args:
        cfg: パイプライン設定
        dataset: 入力データセット
        output_path: 出力ファイルパス
        max_concurrent: 最大並行処理数
        save_intermediate: 中間結果を保存するか
        show_progress: 進捗表示を行うか
        buffer_size: バッファサイズ（件数）
        flush_interval: 定期フラッシュ間隔（秒）
        clean_output: 出力をクリーニングするか

    Returns:
        (完了数, エラー数) のタプル
    """
    total = len(dataset)
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
                task = progress.add_task(f"[cyan]Processing {total} rows...", total=total)
                
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
            "total": total,
            "completed": completed,
            "errors": errors,
        }
        logger.print_stats(stats)

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
):
    """
    適応的並行性制御付きストリーミング版パイプライン実行（非同期）。

    AsyncBufferedWriterを使用して非同期でファイルに書き込み、
    バッファリングによるI/O最適化を行う。

    Args:
        cfg: パイプライン設定
        dataset: 入力データセット
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

    Returns:
        (完了数, エラー数) のタプル
    """
    total = len(dataset)
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
                task = progress.add_task(f"[cyan]Processing {total} rows (adaptive)...", total=total)
                
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
            "total": total,
            "completed": completed,
            "errors": errors,
        }
        logger.print_stats(stats)

    return completed, errors


async def _run_streaming_adaptive_batched_async(
    cfg,
    dataset: List[Dict[str, Any]],
    output_path: str,
    max_concurrent: int,
    min_concurrent: int,
    target_latency_ms: int,
    target_queue_depth: int,
    metrics_type: str,
    max_batch_size: int,
    max_wait_ms: int,
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
):
    """
    バッチング付き適応的並行性制御ストリーミング版パイプライン実行（非同期）。

    AsyncBufferedWriterを使用して非同期でファイルに書き込み、
    バッファリングによるI/O最適化を行う。

    Args:
        cfg: パイプライン設定
        dataset: 入力データセット
        output_path: 出力ファイルパス
        max_concurrent: 最大並行処理数
        min_concurrent: 最小並行処理数
        target_latency_ms: 目標レイテンシ（ミリ秒）
        target_queue_depth: 目標キュー深度
        metrics_type: メトリクスタイプ
        max_batch_size: 最大バッチサイズ
        max_wait_ms: バッチ形成の最大待機時間（ミリ秒）
        save_intermediate: 中間結果を保存するか
        show_progress: 進捗表示を行うか
        buffer_size: バッファサイズ（件数）
        flush_interval: 定期フラッシュ間隔（秒）
        clean_output: 出力をクリーニングするか

    Returns:
        (完了数, エラー数) のタプル
    """
    total = len(dataset)
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
                task = progress.add_task(f"[cyan]Processing {total} rows (batched)...", total=total)
                
                async for result in run_pipeline_streaming_adaptive_batched(
                    cfg,
                    dataset,
                    max_concurrent=max_concurrent,
                    min_concurrent=min_concurrent,
                    target_latency_ms=target_latency_ms,
                    target_queue_depth=target_queue_depth,
                    metrics_type=metrics_type,
                    max_batch_size=max_batch_size,
                    max_wait_ms=max_wait_ms,
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
            async for result in run_pipeline_streaming_adaptive_batched(
                cfg,
                dataset,
                max_concurrent=max_concurrent,
                min_concurrent=min_concurrent,
                target_latency_ms=target_latency_ms,
                target_queue_depth=target_queue_depth,
                metrics_type=metrics_type,
                max_batch_size=max_batch_size,
                max_wait_ms=max_wait_ms,
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
            "total": total,
            "completed": completed,
            "errors": errors,
        }
        logger.print_stats(stats)

    return completed, errors


def run_streaming_adaptive_batched(
    yaml_path: str,
    input_path: Optional[str],
    output_path: str,
    max_concurrent: int = 64,
    min_concurrent: int = 1,
    target_latency_ms: int = 3000,
    target_queue_depth: int = 32,
    metrics_type: str = "none",
    max_batch_size: int = 32,
    max_wait_ms: int = 50,
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
    バッチング付き適応的並行性制御ストリーミング版パイプライン実行

    レイテンシとオプションのバックエンドメトリクスに基づいて、
    並行処理数を動的に調整し、リクエストをバッチングしながら実行する。

    Args:
        yaml_path: YAMLブループリントのパス
        input_path: 入力データセット (.jsonl or .csv)
        output_path: 出力JSONLファイルのパス
        max_concurrent: 同時処理行数の上限 (デフォルト: 64)
        min_concurrent: 同時処理行数の下限 (デフォルト: 1)
        target_latency_ms: 目標レイテンシ (ミリ秒、デフォルト: 3000)
        target_queue_depth: 目標バックエンドキュー深度 (デフォルト: 32)
        metrics_type: メトリクスタイプ ("none", "vllm", "sglang")
        max_batch_size: 最大バッチサイズ (デフォルト: 32)
        max_wait_ms: バッチ形成の最大待機時間 (ミリ秒、デフォルト: 50)
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
    if show_progress:
        subtitle = "Adaptive concurrency control with request batching" if logger.locale == "en" else "リクエストバッチング付き適応的並行性制御"
        logger.header("SDG Pipeline - Adaptive Batched Mode", subtitle)
        
        if logger.locale == "ja":
            config_info = {
                "入力データ数": f"{len(ds)}" + (f" (--max-inputs {max_inputs}で制限)" if max_inputs else ""),
                "最大並行数": max_concurrent,
                "最小並行数": min_concurrent,
                "目標レイテンシ": f"{target_latency_ms}ms",
                "メトリクスタイプ": metrics_type,
                "最大バッチサイズ": max_batch_size,
                "最大待機時間": f"{max_wait_ms}ms",
            }
            logger.table("実行設定", config_info)
        else:
            config_info = {
                "Input Data Count": f"{len(ds)}" + (f" (limited by --max-inputs {max_inputs})" if max_inputs else ""),
                "Max Concurrency": max_concurrent,
                "Min Concurrency": min_concurrent,
                "Target Latency": f"{target_latency_ms}ms",
                "Metrics Type": metrics_type,
                "Max Batch Size": max_batch_size,
                "Max Wait Time": f"{max_wait_ms}ms",
            }
            logger.table("Execution Configuration", config_info)

    # run
    asyncio.run(
        _run_streaming_adaptive_batched_async(
            cfg,
            ds,
            output_path,
            max_concurrent=max_concurrent,
            min_concurrent=min_concurrent,
            target_latency_ms=target_latency_ms,
            target_queue_depth=target_queue_depth,
            metrics_type=metrics_type,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            save_intermediate=save_intermediate,
            show_progress=show_progress,
            clean_output=clean_output,
            enable_scheduling=enable_scheduling,
            max_pending_tasks=max_pending_tasks,
            chunk_size=chunk_size,
            enable_memory_optimization=enable_memory_optimization,
            max_cache_size=max_cache_size,
            enable_memory_monitoring=enable_memory_monitoring,
        )
    )


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
    if show_progress:
        subtitle = "Adaptive concurrency control" if logger.locale == "en" else "適応的並行性制御"
        logger.header("SDG Pipeline - Adaptive Mode", subtitle)
        
        if logger.locale == "ja":
            config_info = {
                "入力データ数": f"{len(ds)}" + (f" (--max-inputs {max_inputs}で制限)" if max_inputs else ""),
                "最大並行数": max_concurrent,
                "最小並行数": min_concurrent,
                "目標レイテンシ": f"{target_latency_ms}ms",
                "メトリクスタイプ": metrics_type,
            }
            logger.table("実行設定", config_info)
        else:
            config_info = {
                "Input Data Count": f"{len(ds)}" + (f" (limited by --max-inputs {max_inputs})" if max_inputs else ""),
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
        )
    )


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
    if show_progress:
        subtitle = "Fixed concurrency streaming processing" if logger.locale == "en" else "固定並行数ストリーミング処理"
        logger.header("SDG Pipeline - Streaming Mode", subtitle)
        
        if logger.locale == "ja":
            config_info = {
                "入力データ数": f"{len(ds)}" + (f" (--max-inputs {max_inputs}で制限)" if max_inputs else ""),
                "並行処理数": max_concurrent,
            }
            logger.table("実行設定", config_info)
        else:
            config_info = {
                "Input Data Count": f"{len(ds)}" + (f" (limited by --max-inputs {max_inputs})" if max_inputs else ""),
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
        )
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


async def _test_run_async(
    cfg,
    data: Dict[str, Any],
    logger,
):
    """
    テスト実行用の非同期関数。

    1件のデータに対してパイプラインを実行し、詳細なログを出力する。

    Args:
        cfg: パイプライン設定
        data: 入力データ（1件）
        logger: SDGLoggerインスタンス

    Returns:
        実行結果の辞書
    """
    from .executors import process_single_row
    from .executors.ai import _build_clients
    from .executors.core import ExecutionContext
    from .executors.python import _load_python_function
    from .config import PyBlock

    logger.info("Starting test run with single data item...")
    
    # 入力データを表示
    logger.table("Input Data", data)
    
    # パイプライン実行
    start_time = time.time()
    
    # モデルクライアント構築
    clients = _build_clients(cfg)
    
    # Python関数をプリロード
    python_functions = {}
    for block in cfg.blocks:
        if isinstance(block, PyBlock):
            fn_key = f"{block.exec}_{block.function or block.entrypoint}"
            python_functions[fn_key] = _load_python_function(block)
    
    # 実行コンテキストを作成
    exec_ctx = ExecutionContext(cfg)
    
    try:
        result = await process_single_row(
            row_index=0,
            initial_context=data,
            cfg=cfg,
            clients=clients,
            exec_ctx=exec_ctx,
            save_intermediate=True,  # 中間結果を保存
            python_functions=python_functions,
        )
        
        elapsed_time = time.time() - start_time
        
        # process_single_row returns a Dict, not StreamingResult
        logger.success(f"Pipeline execution completed in {elapsed_time:.2f}s")
        return {
            "_row_index": 0,
            "_elapsed_time_ms": int(elapsed_time * 1000),
            **result,
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Pipeline execution error: {e}")
        return {
            "_row_index": 0,
            "_error": str(e),
            "_elapsed_time_ms": int(elapsed_time * 1000),
        }


def test_run(
    yaml_path: str,
    input_path: Optional[str] = None,
    # HF Dataset options
    dataset_name: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "train",
    mapping: Optional[Dict[str, str]] = None,
    # UI options
    verbose: bool = True,
    locale: str = "en",
) -> Dict[str, Any]:
    """
    テスト実行: YAMLブループリントを1件のデータに対して実行し、動作確認を行う。

    開発者がエージェントの挙動を素早く検証できるようにするためのコマンド。
    詳細なログを出力し、各ステップの実行状況を可視化する。

    Args:
        yaml_path: YAMLブループリントのパス
        input_path: 入力データセット (.jsonl or .csv)
        dataset_name: Hugging Faceデータセット名
        subset: データセットサブセット
        split: データセットスプリット
        mapping: キーマッピング辞書
        verbose: 詳細ログを有効化（デフォルト: True）
        locale: UIロケール ('en' or 'ja')

    Returns:
        実行結果の辞書

    Example:
        >>> result = test_run("config.yaml", "data.jsonl")
        >>> print(result)
    """
    from .logger import init_logger
    
    # ロガーを初期化（test-runは常に詳細ログを有効化）
    logger = init_logger(
        verbose=verbose,
        quiet=False,
        use_rich=True,
        locale=locale,
    )
    
    # ヘッダーを表示
    if locale == "ja":
        subtitle = "AIエージェントの動作確認モード（1件のみ実行）"
    else:
        subtitle = "AI Agent verification mode (single item execution)"
    logger.header("SDG Test Run", subtitle)
    
    # 設定を読み込み
    cfg = load_config(yaml_path)
    
    # 最適化オプションを設定（テスト実行用）
    cfg.optimization = {
        "use_shared_transport": False,
        "http2": True,
        "retry_on_empty": True,
    }
    
    # 設定情報を表示
    if locale == "ja":
        config_info = {
            "YAMLファイル": yaml_path,
            "MABELバージョン": cfg.get_version(),
            "モデル数": len(cfg.models),
            "ブロック数": len(cfg.blocks),
        }
        logger.table("設定情報", config_info)
    else:
        config_info = {
            "YAML File": yaml_path,
            "MABEL Version": cfg.get_version(),
            "Model Count": len(cfg.models),
            "Block Count": len(cfg.blocks),
        }
        logger.table("Configuration", config_info)
    
    # モデル情報を表示
    if cfg.models:
        if locale == "ja":
            model_info = {}
            for i, m in enumerate(cfg.models):
                model_info[f"モデル {i+1}"] = f"{m.name} ({m.api_model})"
            logger.table("使用モデル", model_info)
        else:
            model_info = {}
            for i, m in enumerate(cfg.models):
                model_info[f"Model {i+1}"] = f"{m.name} ({m.api_model})"
            logger.table("Models", model_info)
    
    # ブロック情報を表示
    if cfg.blocks:
        if locale == "ja":
            block_info = {}
            for b in cfg.blocks:
                block_name = b.name or b.id or f"Block {b.exec}"
                block_info[block_name] = f"type={b.type}, exec={b.exec}"
            logger.table("ブロック構成", block_info)
        else:
            block_info = {}
            for b in cfg.blocks:
                block_name = b.name or b.id or f"Block {b.exec}"
                block_info[block_name] = f"type={b.type}, exec={b.exec}"
            logger.table("Block Structure", block_info)
    
    # データを読み込み（1件のみ）
    if input_path:
        if input_path.endswith(".jsonl"):
            ds = read_jsonl(input_path, max_inputs=1)
        elif input_path.endswith(".csv"):
            ds = read_csv(input_path, max_inputs=1)
        else:
            raise ValueError("Unsupported input format. Use .jsonl or .csv")
    elif dataset_name:
        ds = read_hf_dataset(dataset_name, subset, split, max_inputs=1)
    else:
        raise ValueError("Either input_path or dataset_name must be provided")
    
    if not ds:
        raise ValueError("No data found in input")
    
    # マッピングを適用
    if mapping:
        ds = apply_mapping(ds, mapping)
    
    # 1件目のデータを取得
    data = ds[0]
    
    # パイプラインを実行
    result = asyncio.run(_test_run_async(cfg, data, logger))
    
    # 結果を表示
    if locale == "ja":
        logger.info("実行結果:")
    else:
        logger.info("Execution Result:")
    
    # 結果をテーブル形式で表示
    result_display = {}
    for key, value in result.items():
        if key.startswith("_"):
            continue
        # 長い値は省略
        str_value = str(value)
        if len(str_value) > 100:
            str_value = str_value[:100] + "..."
        result_display[key] = str_value
    
    if result_display:
        if locale == "ja":
            logger.table("出力データ", result_display)
        else:
            logger.table("Output Data", result_display)
    
    # メタ情報を表示
    meta_info = {}
    if "_elapsed_time_ms" in result:
        if locale == "ja":
            meta_info["実行時間"] = f"{result['_elapsed_time_ms']}ms"
        else:
            meta_info["Elapsed Time"] = f"{result['_elapsed_time_ms']}ms"
    if "_error" in result:
        if locale == "ja":
            meta_info["エラー"] = result["_error"]
        else:
            meta_info["Error"] = result["_error"]
    
    if meta_info:
        if locale == "ja":
            logger.table("メタ情報", meta_info)
        else:
            logger.table("Meta Information", meta_info)
    
    # 統計情報を表示
    stats = {
        "total": 1,
        "completed": 1 if "_error" not in result else 0,
        "errors": 1 if "_error" in result else 0,
    }
    logger.print_stats(stats)
    
    return result
