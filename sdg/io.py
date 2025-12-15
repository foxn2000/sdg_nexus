from __future__ import annotations
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import aiofiles
import aiofiles.os

# orjsonによる高速JSONシリアライズ（フォールバック付き）
try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    _HAS_ORJSON = False

from .logger import get_logger

# clean_jsonl_line is no longer used in AsyncBufferedWriter.write/write_many
# since Dict input with json.dumps guarantees valid JSON format


def load_processed_indices(output_path: str) -> Tuple[Set[int], int]:
    """
    既存の出力ファイルから処理済み行インデックスを読み取る。

    出力ファイルの各行には `_row_index` フィールドが含まれていることを前提とする。
    このフィールドを使用して、どの入力行が既に処理されたかを特定する。

    Args:
        output_path: 出力ファイルパス

    Returns:
        (処理済み行インデックスのセット, 処理済み行数) のタプル
        ファイルが存在しない場合は (空のセット, 0) を返す

    Note:
        不正なJSONや _row_index がない行はスキップされる。
    """
    processed_indices: Set[int] = set()
    processed_count = 0

    if not os.path.exists(output_path):
        return processed_indices, processed_count

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "_row_index" in data:
                        row_index = data["_row_index"]
                        if isinstance(row_index, int):
                            processed_indices.add(row_index)
                            processed_count += 1
                except (json.JSONDecodeError, TypeError, KeyError):
                    # 不正な行はスキップ
                    continue
    except IOError:
        # ファイル読み取りエラーの場合は空を返す
        return set(), 0

    return processed_indices, processed_count


def skip_processed_rows(
    dataset: Iterable[Dict[str, Any]],
    processed_indices: Set[int],
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """
    処理済み行をスキップしながらデータセットをイテレートする。

    入力データセットの各行に対して、その行インデックスを付与しながら、
    処理済み行をスキップして未処理行のみを yield する。

    Args:
        dataset: 入力データセット（辞書のイテラブル）
        processed_indices: 処理済み行インデックスのセット

    Yields:
        (行インデックス, データ行) のタプル（未処理行のみ）
    """
    for row_index, row in enumerate(dataset):
        if row_index not in processed_indices:
            yield row_index, row


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
        append: bool = False,
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
            append: 追記モードで開くか（デフォルト: False）
                   True の場合、既存ファイルに追記する（処理再開時に使用）
        """
        self._path = path
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._encoding = encoding
        self._serializer = serializer or self._default_serializer
        self._clean_output = clean_output
        self._append = append

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
        """
        デフォルトのJSONシリアライザ。

        orjsonがインストールされている場合は高速なorjsonを使用し、
        インストールされていない場合は標準jsonにフォールバックする。
        """
        if _HAS_ORJSON:
            # orjson.dumpsはbytesを返すためデコードが必要
            return orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")
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

        追記モード（append=True）の場合は既存ファイルに追記し、
        そうでない場合は新規ファイルとして開く（既存内容は上書き）。
        """
        # ディレクトリを作成
        dir_name = os.path.dirname(self._path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # ファイルモードを決定（追記 or 上書き）
        file_mode = "a" if self._append else "w"

        # ファイルを開く
        self._file = await aiofiles.open(
            self._path,
            mode=file_mode,
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


def count_lines_fast(path: str) -> Optional[int]:
    """
    ファイルの行数を高速にカウントする。

    wcコマンドを使用して高速に行数をカウントする。
    テラバイト級のファイルでも効率的に行数を取得可能。
    wcコマンドが利用できない場合はNoneを返す。

    Args:
        path: ファイルパス

    Returns:
        行数（カウントできない場合はNone）
    """
    try:
        # wcコマンドで高速カウント
        result = subprocess.run(
            ["wc", "-l", path],
            capture_output=True,
            text=True,
            timeout=300,  # 5分タイムアウト
        )
        if result.returncode == 0:
            # 出力形式: "  12345 filename"
            count_str = result.stdout.strip().split()[0]
            return int(count_str)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        pass
    return None


def read_jsonl(path: str, max_inputs: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    JSONLファイルをジェネレータとして読み込む（省メモリ版）。

    テラバイト級のファイルでもメモリ効率的に処理可能。
    ファイルはジェネレータが消費されるまで開いたままになる。

    Args:
        path: 入力ファイルパス
        max_inputs: 読み込む最大行数（Noneの場合は全件）

    Yields:
        各行をパースした辞書
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_inputs is not None and count >= max_inputs:
                break
            if line.strip():
                yield json.loads(line)
                count += 1


def read_csv(path: str, max_inputs: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    CSVファイルをジェネレータとして読み込む（省メモリ版）。

    テラバイト級のファイルでもメモリ効率的に処理可能。
    ファイルはジェネレータが消費されるまで開いたままになる。

    Args:
        path: 入力ファイルパス
        max_inputs: 読み込む最大行数（Noneの場合は全件）

    Yields:
        各行を辞書に変換したもの
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_inputs is not None and count >= max_inputs:
                break
            yield dict(row)  # OrderedDictを通常のdictに変換
            count += 1


def read_hf_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    max_inputs: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Hugging Face Datasetをジェネレータとして読み込む（省メモリ版）。

    Args:
        dataset_name: データセット名
        subset: サブセット名（オプション）
        split: スプリット名（デフォルト: train）
        max_inputs: 読み込む最大行数（Noneの場合は全件）

    Yields:
        辞書
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
    ds = load_dataset(dataset_name, name=subset, split=split, streaming=True)

    count = 0
    for item in ds:
        if max_inputs is not None and count >= max_inputs:
            break
        yield dict(item)
        count += 1


def apply_mapping(
    dataset: Iterable[Dict[str, Any]], mapping: Dict[str, str]
) -> Iterator[Dict[str, Any]]:
    """
    データセットのキーをマッピングに従って変更する（ジェネレータ版）。

    Args:
        dataset: データセット（辞書のイテラブル）
        mapping: マッピング辞書 {old_key: new_key}

    Yields:
        キーが変更された辞書
    """
    if not mapping:
        yield from dataset
        return

    for item in dataset:
        new_item = item.copy()
        for old_key, new_key in mapping.items():
            if old_key in new_item:
                new_item[new_key] = new_item.pop(old_key)
        yield new_item


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
