"""
階層的タスクスケジューリングとメモリ効率化モジュール

このモジュールは大規模データセット処理のためのスケーラビリティ改善を提供します：
- HierarchicalTaskScheduler: チャンク単位での段階的タスク生成
- StreamingContextManager: LRUキャッシュ付きコンテキスト管理
- MemoryMonitor: メモリ使用状況の監視
"""

from __future__ import annotations
import asyncio
import gc
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

# psutilはオプション依存（メモリ監視用）
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class SchedulerConfig:
    """
    タスクスケジューラの設定

    Attributes:
        max_pending_tasks: 最大保留タスク数（メモリ制御用）
        chunk_size: データセットをチャンク化するサイズ
        enable_scheduling: 階層的スケジューリングを有効化するか（デフォルト: False）
    """

    max_pending_tasks: int = 1000
    chunk_size: int = 100
    enable_scheduling: bool = False

    def __post_init__(self):
        if self.max_pending_tasks < 1:
            raise ValueError("max_pending_tasks must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")


@dataclass
class MemoryConfig:
    """
    メモリ効率化の設定

    Attributes:
        max_cache_size: コンテキストキャッシュの最大サイズ
        enable_memory_optimization: メモリ最適化を有効化するか（デフォルト: False）
        enable_monitoring: メモリ使用状況監視を有効化するか（デフォルト: False）
        gc_interval: ガベージコレクション実行間隔（処理行数）
        memory_threshold_mb: メモリ使用量警告閾値（MB）
    """

    max_cache_size: int = 500
    enable_memory_optimization: bool = False
    enable_monitoring: bool = False
    gc_interval: int = 100
    memory_threshold_mb: int = 1024

    def __post_init__(self):
        if self.max_cache_size < 1:
            raise ValueError("max_cache_size must be >= 1")
        if self.gc_interval < 1:
            raise ValueError("gc_interval must be >= 1")


class HierarchicalTaskScheduler:
    """
    階層的タスクスケジューラ

    大規模データセットを効率的に処理するため、データをチャンクに分割し、
    段階的にタスクを生成・実行します。これによりメモリ使用量を制御し、
    処理開始の遅延を最小化します。

    Features:
        - チャンク単位でのデータセット分割
        - 最大保留タスク数の制限（Conditionによる協調制御）
        - タスク完了時の自動チャンク進行

    Example:
        ```python
        scheduler = HierarchicalTaskScheduler(
            config=SchedulerConfig(max_pending_tasks=100, chunk_size=50)
        )

        async for item in scheduler.schedule(dataset):
            # itemはIndexedDataItem
            task = asyncio.create_task(process(item.index, item.data))
            scheduler.mark_task_started()

        # タスク完了時
        scheduler.mark_task_completed()
        ```
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Args:
            config: スケジューラ設定（Noneの場合はデフォルト設定を使用）
        """
        self.config = config or SchedulerConfig()
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._pending_count = 0
        self._completed_count = 0
        self._total_items = 0
        self._started = False

    @property
    def pending_count(self) -> int:
        """現在の保留タスク数"""
        return self._pending_count

    @property
    def completed_count(self) -> int:
        """完了したタスク数"""
        return self._completed_count

    @property
    def total_items(self) -> int:
        """総アイテム数"""
        return self._total_items

    @property
    def is_enabled(self) -> bool:
        """スケジューリングが有効化されているか"""
        return self.config.enable_scheduling

    async def schedule(
        self, dataset: List[Dict[str, Any]]
    ) -> "AsyncIterator[IndexedDataItem]":
        """
        データセットを階層的にスケジューリングし、アイテムを順次yield

        スケジューリングが無効の場合は、従来通り全アイテムを一度にyield。
        有効の場合は、max_pending_tasksを超えないよう制御しながらyield。

        Args:
            dataset: 入力データセットのリスト

        Yields:
            IndexedDataItem: インデックスとデータのペア
        """
        self._total_items = len(dataset)
        self._pending_count = 0
        self._completed_count = 0
        self._started = True

        if not self.config.enable_scheduling:
            # スケジューリング無効時は従来の動作（全アイテム即座にyield）
            for i, item in enumerate(dataset):
                yield IndexedDataItem(index=i, data=item)
            return

        # チャンク単位でデータを処理
        chunk_size = self.config.chunk_size
        max_pending = self.config.max_pending_tasks

        for chunk_start in range(0, len(dataset), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(dataset))
            chunk = dataset[chunk_start:chunk_end]

            for i, item in enumerate(chunk):
                item_index = chunk_start + i

                # 保留タスク数が上限に達していたら待機
                async with self._condition:
                    while self._pending_count >= max_pending:
                        await self._condition.wait()

                    self._pending_count += 1

                yield IndexedDataItem(index=item_index, data=item)

    async def mark_task_completed(self):
        """
        タスク完了を通知

        保留カウンタを減らし、待機中のスケジューラに通知します。
        """
        async with self._condition:
            self._pending_count -= 1
            self._completed_count += 1
            self._condition.notify_all()

    def get_stats(self) -> Dict[str, Any]:
        """
        スケジューラの統計情報を取得

        Returns:
            統計情報の辞書（pending, completed, total, progress）
        """
        progress = (
            self._completed_count / self._total_items * 100
            if self._total_items > 0
            else 0
        )
        return {
            "pending": self._pending_count,
            "completed": self._completed_count,
            "total": self._total_items,
            "progress_percent": round(progress, 2),
            "scheduling_enabled": self.config.enable_scheduling,
        }


@dataclass
class IndexedDataItem:
    """インデックス付きデータアイテム"""

    index: int
    data: Dict[str, Any]


T = TypeVar("T")


class LRUCache(Generic[T]):
    """
    LRU（Least Recently Used）キャッシュ

    最大サイズを超えた場合、最も使用されていないアイテムを自動削除します。

    Example:
        ```python
        cache = LRUCache[MyContext](max_size=100)
        cache.put("key1", my_context)
        value = cache.get("key1")  # キャッシュヒット
        cache.remove("key1")  # 明示的な削除
        ```
    """

    def __init__(self, max_size: int = 500):
        """
        Args:
            max_size: キャッシュの最大サイズ
        """
        self._max_size = max_size
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """現在のキャッシュサイズ"""
        return len(self._cache)

    @property
    def max_size(self) -> int:
        """キャッシュの最大サイズ"""
        return self._max_size

    @property
    def hit_rate(self) -> float:
        """キャッシュヒット率"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get(self, key: str) -> Optional[T]:
        """
        キャッシュからアイテムを取得

        存在する場合はLRU順序を更新し、値を返します。

        Args:
            key: キャッシュキー

        Returns:
            キャッシュされた値、または存在しない場合はNone
        """
        if key in self._cache:
            # LRU: アクセスされたアイテムを末尾に移動
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: T) -> Optional[T]:
        """
        アイテムをキャッシュに追加

        キャッシュが満杯の場合、最も古いアイテムが削除されます。

        Args:
            key: キャッシュキー
            value: キャッシュする値

        Returns:
            削除されたアイテム（容量超過の場合）、またはNone
        """
        evicted = None

        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                # 最も古いアイテムを削除
                oldest_key, evicted = self._cache.popitem(last=False)

        self._cache[key] = value
        return evicted

    def remove(self, key: str) -> Optional[T]:
        """
        指定されたキーのアイテムを削除

        Args:
            key: 削除するキー

        Returns:
            削除されたアイテム、または存在しない場合はNone
        """
        if key in self._cache:
            return self._cache.pop(key)
        return None

    def clear(self):
        """キャッシュをクリア"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        キャッシュの統計情報を取得

        Returns:
            統計情報の辞書
        """
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }


class StreamingContextManager:
    """
    ストリーミング対応コンテキストマネージャ

    大規模データセット処理時のメモリ効率を向上させるため、
    アクティブなコンテキストを辞書で管理します。
    処理が完了したコンテキストは明示的に解放されます。

    Features:
        - アクティブセット方式によるコンテキスト保持（evictなし）
        - 処理完了時の明示的な解放
        - asyncio.Lockによるスレッドセーフなアクセス
        - オプションのメモリ監視

    Example:
        ```python
        ctx_manager = StreamingContextManager(
            config=MemoryConfig(enable_memory_optimization=True)
        )

        # コンテキストを取得（なければ作成）
        ctx = await ctx_manager.get_or_create(row_index, initial_data)

        # 処理完了時にマーク（コンテキストが解放される）
        await ctx_manager.mark_completed(row_index)
        ```
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Args:
            config: メモリ効率化設定（Noneの場合はデフォルト設定を使用）
        """
        self.config = config or MemoryConfig()
        # LRUCacheの代わりにアクティブなコンテキストを保持する辞書を使用
        self._active_contexts: Dict[str, Dict[str, Any]] = {}
        self._completed: set = set()
        self._lock = asyncio.Lock()
        self._processed_count = 0
        self._memory_monitor: Optional[MemoryMonitor] = None

        if self.config.enable_monitoring:
            self._memory_monitor = MemoryMonitor(
                threshold_mb=self.config.memory_threshold_mb
            )

    @property
    def is_enabled(self) -> bool:
        """メモリ最適化が有効化されているか"""
        return self.config.enable_memory_optimization

    async def get_or_create(
        self,
        row_index: int,
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        コンテキストを取得（存在しない場合は作成）

        メモリ最適化が無効の場合は常に新しい辞書を返します。
        アクティブセット方式のため、処理中のコンテキストは自動削除されません。

        Args:
            row_index: 行インデックス
            initial_data: 初期データ（新規作成時に使用）

        Returns:
            コンテキスト辞書
        """
        if not self.config.enable_memory_optimization:
            # 最適化無効時は単純に新規辞書を返す
            return dict(initial_data or {})

        key = str(row_index)

        async with self._lock:
            # アクティブコンテキストから取得を試みる
            if key in self._active_contexts:
                return self._active_contexts[key]

            # 新規作成してアクティブコンテキストに追加
            ctx = dict(initial_data or {})
            self._active_contexts[key] = ctx

            return ctx

    async def mark_completed(self, row_index: int):
        """
        行の処理完了をマーク

        メモリ最適化が有効な場合、完了した行のコンテキストを明示的に解放します。

        Args:
            row_index: 完了した行のインデックス
        """
        if not self.config.enable_memory_optimization:
            return

        key = str(row_index)

        async with self._lock:
            self._completed.add(row_index)
            self._processed_count += 1

            # アクティブコンテキストから削除して解放
            if key in self._active_contexts:
                removed = self._active_contexts[key]
                del self._active_contexts[key]
                removed.clear()

            # 定期的なガベージコレクション
            if self._processed_count % self.config.gc_interval == 0:
                gc.collect()

                # メモリ監視
                if self._memory_monitor:
                    self._memory_monitor.check_and_warn()

    async def release_all(self):
        """全てのコンテキストを解放"""
        async with self._lock:
            # 全てのアクティブコンテキストをクリア
            for ctx in self._active_contexts.values():
                ctx.clear()
            self._active_contexts.clear()
            self._completed.clear()
            gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """
        コンテキストマネージャの統計情報を取得

        Returns:
            統計情報の辞書
        """
        stats = {
            "processed_count": self._processed_count,
            "completed_count": len(self._completed),
            "active_contexts_count": len(self._active_contexts),
            "memory_optimization_enabled": self.config.enable_memory_optimization,
        }

        if self._memory_monitor:
            stats["memory"] = self._memory_monitor.get_current_usage()

        return stats


class MemoryMonitor:
    """
    メモリ使用状況の監視

    psutilを使用してプロセスのメモリ使用状況を監視します。
    psutilがインストールされていない場合は機能が制限されます。

    Example:
        ```python
        monitor = MemoryMonitor(threshold_mb=1024)
        usage = monitor.get_current_usage()
        monitor.check_and_warn()  # 警告閾値を超えた場合にログ出力
        ```
    """

    def __init__(self, threshold_mb: int = 1024):
        """
        Args:
            threshold_mb: メモリ使用量の警告閾値（MB）
        """
        self.threshold_mb = threshold_mb
        self._last_warning_time: float = 0
        self._warning_interval: float = 10.0  # 警告間隔（秒）

    def get_current_usage(self) -> Dict[str, Any]:
        """
        現在のメモリ使用状況を取得

        Returns:
            メモリ使用状況の辞書（rss_mb, vms_mb, percent）
        """
        if not PSUTIL_AVAILABLE:
            return {
                "available": False,
                "message": "psutil not installed",
            }

        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "available": True,
                "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
                "vms_mb": round(mem_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2),
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            }

    def check_and_warn(self) -> bool:
        """
        メモリ使用量をチェックし、閾値を超えていれば警告

        Returns:
            閾値を超えている場合はTrue
        """
        if not PSUTIL_AVAILABLE:
            return False

        try:
            current_time = time.time()
            if current_time - self._last_warning_time < self._warning_interval:
                return False

            process = psutil.Process()
            rss_mb = process.memory_info().rss / 1024 / 1024

            if rss_mb > self.threshold_mb:
                self._last_warning_time = current_time
                import logging

                logging.warning(
                    f"High memory usage: {rss_mb:.2f}MB (threshold: {self.threshold_mb}MB)"
                )
                return True

            return False
        except Exception:
            return False

    def get_object_sizes(self, objects: Dict[str, Any]) -> Dict[str, int]:
        """
        オブジェクトのサイズを取得（デバッグ用）

        Args:
            objects: 測定するオブジェクトの辞書

        Returns:
            各オブジェクトのサイズ（バイト）
        """
        return {name: sys.getsizeof(obj) for name, obj in objects.items()}


class BatchProgressiveRelease:
    """
    バッチ処理モードでの段階的メモリ解放

    run_pipeline（バッチモード）で使用するための段階的メモリ解放機能。
    コンテキストと結果リストを部分的に解放しながら処理を進めます。

    Example:
        ```python
        release = BatchProgressiveRelease(
            config=MemoryConfig(enable_memory_optimization=True),
            total_size=len(dataset)
        )

        for i, result in enumerate(results):
            # 結果が確定したら解放をマーク
            release.mark_row_done(i, contexts, results)
        ```
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        total_size: int = 0,
    ):
        """
        Args:
            config: メモリ効率化設定
            total_size: データセットの総サイズ
        """
        self.config = config or MemoryConfig()
        self.total_size = total_size
        self._completed_rows: set = set()
        self._gc_counter = 0
        self._memory_monitor: Optional[MemoryMonitor] = None

        if self.config.enable_monitoring:
            self._memory_monitor = MemoryMonitor(
                threshold_mb=self.config.memory_threshold_mb
            )

    @property
    def is_enabled(self) -> bool:
        """メモリ最適化が有効化されているか"""
        return self.config.enable_memory_optimization

    def mark_row_done(
        self,
        row_index: int,
        contexts: Optional[List[Dict[str, Any]]] = None,
        results: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        行の処理完了をマークし、可能であればメモリを解放

        メモリ最適化が無効の場合は何もしません。

        Args:
            row_index: 完了した行のインデックス
            contexts: コンテキストリスト（Noneの場合は解放しない）
            results: 結果リスト（参照用、通常は解放しない）
        """
        if not self.config.enable_memory_optimization:
            return

        self._completed_rows.add(row_index)
        self._gc_counter += 1

        # コンテキストを解放（結果の生成に不要になった場合）
        if contexts is not None and row_index < len(contexts):
            # コンテキストの内容をクリア（リスト要素は維持）
            if contexts[row_index]:
                contexts[row_index].clear()

        # 定期的なガベージコレクション
        if self._gc_counter % self.config.gc_interval == 0:
            gc.collect()

            if self._memory_monitor:
                self._memory_monitor.check_and_warn()

    def force_gc(self):
        """強制ガベージコレクション"""
        gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """
        統計情報を取得

        Returns:
            統計情報の辞書
        """
        stats = {
            "completed_rows": len(self._completed_rows),
            "total_size": self.total_size,
            "memory_optimization_enabled": self.config.enable_memory_optimization,
        }

        if self._memory_monitor:
            stats["memory"] = self._memory_monitor.get_current_usage()

        return stats
