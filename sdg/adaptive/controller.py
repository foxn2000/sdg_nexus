"""
Adaptive concurrency controller for maximizing inference throughput.

This module implements an AIMD (Additive Increase Multiplicative Decrease)
based controller that dynamically adjusts concurrency based on observed
latencies and optionally backend metrics from vLLM/SGLang.

Also includes DynamicSemaphore for real-time capacity adjustment.
"""

from __future__ import annotations
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from .metrics import BackendMetrics, MetricsCollector, MetricsType


class DynamicSemaphore:
    """
    動的に容量を調整可能なセマフォ。

    標準のasyncio.Semaphoreとは異なり、容量をリアルタイムに
    増減することができる。容量増加時は即座に反映し、
    容量減少時は即座に新規取得を停止する。

    Attributes:
        capacity: 現在の容量
        acquired_count: 現在取得中のパーミット数
        waiting_count: 待機中のコルーチン数

    Example:
        semaphore = DynamicSemaphore(initial_capacity=10)

        # 容量を増やす（即座に反映）
        semaphore.set_capacity(20)

        # 容量を減らす（即座に新規取得をブロック）
        semaphore.set_capacity(5)

        # 通常のセマフォとして使用
        async with semaphore:
            # クリティカルセクション
            ...
    """

    def __init__(self, initial_capacity: int = 1):
        """
        DynamicSemaphoreを初期化する。

        Args:
            initial_capacity: 初期容量（デフォルト: 1）

        Raises:
            ValueError: initial_capacityが1未満の場合
        """
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be at least 1")

        self._capacity = initial_capacity
        self._acquired = 0
        self._waiters: Deque[asyncio.Future[None]] = deque()
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> int:
        """現在の容量を返す。"""
        return self._capacity

    @property
    def acquired_count(self) -> int:
        """現在取得中のパーミット数を返す。"""
        return self._acquired

    @property
    def waiting_count(self) -> int:
        """待機中のコルーチン数を返す。"""
        return len(self._waiters)

    @property
    def available(self) -> int:
        """
        利用可能なパーミット数を返す。

        容量から取得中のパーミット数を引いた値。
        負の値になることはない。
        """
        return max(0, self._capacity - self._acquired)

    async def set_capacity(self, new_capacity: int) -> None:
        """
        容量を動的に変更する。

        容量増加時:
            - 待機中のコルーチンがあれば、新たに利用可能になった
              パーミット分だけウェイクアップする

        容量減少時:
            - 即座に新規の取得をブロック（取得中のパーミットは
              そのまま維持され、リリースされるまで有効）

        Args:
            new_capacity: 新しい容量（1以上）

        Raises:
            ValueError: new_capacityが1未満の場合
        """
        if new_capacity < 1:
            raise ValueError("capacity must be at least 1")

        async with self._lock:
            old_capacity = self._capacity
            self._capacity = new_capacity

            # 容量が増加した場合、待機者をウェイクアップ
            if new_capacity > old_capacity:
                available_slots = new_capacity - self._acquired
                while self._waiters and available_slots > 0:
                    waiter = self._waiters.popleft()
                    if not waiter.done():
                        waiter.set_result(None)
                        self._acquired += 1
                        available_slots -= 1

    def set_capacity_sync(self, new_capacity: int) -> None:
        """
        容量を同期的に変更する（即時変更のみ、待機者のウェイクアップなし）。

        非同期コンテキスト外から呼び出す場合に使用。
        容量変更のみを行い、待機者のウェイクアップは次回のacquire時に
        自動的に処理される。

        Args:
            new_capacity: 新しい容量（1以上）

        Raises:
            ValueError: new_capacityが1未満の場合
        """
        if new_capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = new_capacity

    async def acquire(self) -> None:
        """
        パーミットを取得する。

        利用可能なパーミットがない場合はブロックする。
        容量減少により取得可能数が減った場合も適切にブロックする。
        """
        async with self._lock:
            # パーミットが利用可能かチェック
            if self._acquired < self._capacity:
                self._acquired += 1
                return

            # 利用可能なパーミットがない場合は待機
            waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self._waiters.append(waiter)

        # ロックを解放して待機（デッドロック防止）
        try:
            await waiter
        except asyncio.CancelledError:
            # キャンセル時はウェイターリストから削除
            async with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                elif not waiter.done():
                    # 既にウェイクアップされていた場合は取得カウントを減らす
                    pass
            raise

    def release(self) -> None:
        """
        パーミットを解放する。

        解放後、待機中のコルーチンがあればウェイクアップする。

        Raises:
            ValueError: 取得していないパーミットを解放しようとした場合
        """
        if self._acquired <= 0:
            raise ValueError("Cannot release more permits than acquired")

        self._acquired -= 1

        # 容量に余裕があり、待機者がいればウェイクアップ
        if self._acquired < self._capacity and self._waiters:
            # ロックなしで直接操作（release()は通常await不可のため）
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                self._acquired += 1

    async def __aenter__(self) -> "DynamicSemaphore":
        """非同期コンテキストマネージャーのエントリーポイント。"""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """非同期コンテキストマネージャーの終了ポイント。"""
        self.release()

    def locked(self) -> bool:
        """
        セマフォがロック状態かどうかを返す。

        取得中のパーミット数が容量以上の場合にTrue。
        """
        return self._acquired >= self._capacity

    def get_stats(self) -> dict:
        """
        セマフォの統計情報を返す。

        Returns:
            容量、取得中数、待機中数、利用可能数を含む辞書
        """
        return {
            "capacity": self._capacity,
            "acquired": self._acquired,
            "waiting": len(self._waiters),
            "available": self.available,
        }


@dataclass
class LatencySample:
    """A single latency measurement."""

    latency_ms: float
    timestamp: float
    is_error: bool = False


class AdaptiveController:
    """
    AIMD-based adaptive concurrency controller with dynamic semaphore.

    This controller dynamically adjusts the concurrency level based on:
    - Client-side latencies (always available)
    - Backend metrics like queue depth (optional, for vLLM/SGLang)

    The algorithm uses Additive Increase Multiplicative Decrease (AIMD):
    - When latencies are low: gradually increase concurrency
    - When latencies spike or errors occur: quickly decrease concurrency

    Now uses DynamicSemaphore for real-time capacity adjustment:
    - Capacity increases are immediately reflected
    - Capacity decreases immediately block new acquisitions

    Usage:
        controller = AdaptiveController(
            min_concurrency=1,
            max_concurrency=64,
        )

        # Get current concurrency limit
        limit = controller.current_concurrency

        # Report completed requests
        controller.record_latency(latency_ms=150.0, is_error=False)

        # Optionally update with backend metrics
        controller.update_with_metrics(metrics)

        # Get adjusted concurrency
        new_limit = controller.current_concurrency

        # Use the dynamic semaphore for concurrency control
        async with controller.semaphore:
            # Execute request
            ...
    """

    def __init__(
        self,
        min_concurrency: int = 1,
        max_concurrency: int = 64,
        target_latency_ms: float = 2000.0,
        target_queue_depth: int = 32,
        # AIMD parameters
        increase_step: int = 2,  # Additive increase per adjustment
        decrease_factor: float = 0.5,  # Multiplicative decrease factor
        # Adjustment sensitivity
        latency_tolerance: float = 1.5,  # Trigger decrease if latency > target * tolerance
        error_rate_threshold: float = 0.05,  # 5% error rate triggers decrease
        # Timing
        adjustment_interval_ms: int = 1000,  # Minimum time between adjustments
        window_size: int = 50,  # Number of samples for averaging
        # Initial concurrency
        initial_concurrency: Optional[
            int
        ] = None,  # Initial concurrency level (default: max/2)
        # Use legacy semaphore for backward compatibility
        use_dynamic_semaphore: bool = True,
    ):
        """
        Initialize the adaptive controller.

        Args:
            min_concurrency: Minimum concurrency level (default: 1)
            max_concurrency: Maximum concurrency level (default: 64)
            target_latency_ms: Target P95 latency in milliseconds (default: 2000)
            target_queue_depth: Target backend queue depth (default: 32)
            increase_step: Additive increase per adjustment cycle (default: 2)
            decrease_factor: Multiplicative decrease factor (default: 0.5)
            latency_tolerance: Latency threshold multiplier for decrease (default: 1.5)
            error_rate_threshold: Error rate threshold for decrease (default: 0.05)
            adjustment_interval_ms: Minimum interval between adjustments (default: 1000)
            window_size: Number of samples to consider (default: 50)
            initial_concurrency: Initial concurrency level (default: max_concurrency / 2)
            use_dynamic_semaphore: Use DynamicSemaphore for real-time capacity changes
                                   (default: True, set to False for backward compatibility)
        """
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.target_latency = target_latency_ms
        self.target_queue_depth = target_queue_depth

        self.increase_step = increase_step
        self.decrease_factor = decrease_factor
        self.latency_tolerance = latency_tolerance
        self.error_rate_threshold = error_rate_threshold

        self.adjustment_interval = adjustment_interval_ms / 1000.0
        self.window_size = window_size
        self._use_dynamic_semaphore = use_dynamic_semaphore

        # State - 初期並行数を max から始める（最も積極的なスタート）
        if initial_concurrency is not None:
            self._current: int = max(
                min_concurrency, min(initial_concurrency, max_concurrency)
            )
        else:
            # デフォルト: max_concurrency から始める
            self._current: int = max_concurrency
        self._latency_window: Deque[LatencySample] = deque(maxlen=window_size)
        self._last_adjustment_time: float = 0.0
        self._last_metrics: Optional[BackendMetrics] = None

        # Semaphore for dynamic concurrency control
        # DynamicSemaphoreを使用する場合と従来のasyncio.Semaphoreを使用する場合を分ける
        self._dynamic_semaphore: Optional[DynamicSemaphore] = None
        self._legacy_semaphore: Optional[asyncio.Semaphore] = None

    @property
    def current_concurrency(self) -> int:
        """Get current concurrency limit."""
        return self._current

    @property
    def semaphore(self) -> DynamicSemaphore:
        """
        Get or create the concurrency-limiting semaphore.

        Returns DynamicSemaphore for real-time capacity adjustment.
        For backward compatibility, a legacy asyncio.Semaphore is also available
        via legacy_semaphore property.

        Returns:
            DynamicSemaphore instance
        """
        if self._dynamic_semaphore is None:
            self._dynamic_semaphore = DynamicSemaphore(self._current)
        return self._dynamic_semaphore

    @property
    def legacy_semaphore(self) -> asyncio.Semaphore:
        """
        Get or create the legacy asyncio.Semaphore (for backward compatibility).

        Note: This semaphore does not support dynamic capacity changes.
        Use the semaphore property for dynamic capacity support.

        Returns:
            asyncio.Semaphore instance
        """
        if self._legacy_semaphore is None:
            self._legacy_semaphore = asyncio.Semaphore(self._current)
        return self._legacy_semaphore

    @property
    def dynamic_semaphore(self) -> DynamicSemaphore:
        """
        Get or create the DynamicSemaphore.

        Alias for semaphore property for explicit access.

        Returns:
            DynamicSemaphore instance
        """
        return self.semaphore

    def record_latency(self, latency_ms: float, is_error: bool = False) -> None:
        """
        Record a latency measurement.

        Args:
            latency_ms: Request latency in milliseconds
            is_error: Whether this request resulted in an error
        """
        sample = LatencySample(
            latency_ms=latency_ms,
            timestamp=time.time(),
            is_error=is_error,
        )
        self._latency_window.append(sample)

        # Trigger adjustment if enough samples and interval passed
        self._maybe_adjust()

    def update_with_metrics(self, metrics: BackendMetrics) -> None:
        """
        Update controller with backend metrics.

        Args:
            metrics: Latest metrics from vLLM/SGLang backend
        """
        self._last_metrics = metrics
        self._maybe_adjust()

    def _maybe_adjust(self) -> None:
        """Check if adjustment is needed and apply it."""
        now = time.time()

        # Respect minimum adjustment interval
        if now - self._last_adjustment_time < self.adjustment_interval:
            return

        # Need minimum samples for adjustment
        if len(self._latency_window) < 10:
            return

        self._last_adjustment_time = now
        self._adjust_concurrency()

    def _adjust_concurrency(self) -> None:
        """Perform concurrency adjustment based on observations."""
        old_concurrency = self._current

        # Calculate metrics from latency window
        latencies = [s.latency_ms for s in self._latency_window if not s.is_error]
        errors = [s for s in self._latency_window if s.is_error]

        error_rate = (
            len(errors) / len(self._latency_window) if self._latency_window else 0
        )

        # Calculate P95 latency
        p95_latency = self._calculate_percentile(latencies, 95) if latencies else 0

        # Decision logic
        should_decrease = False
        should_increase = False

        # Check error rate
        if error_rate > self.error_rate_threshold:
            should_decrease = True

        # Check latency
        elif p95_latency > self.target_latency * self.latency_tolerance:
            should_decrease = True

        # Check backend queue depth if metrics available
        elif self._last_metrics and self._last_metrics.is_valid:
            queue_depth = self._last_metrics.queue_depth
            if queue_depth is not None:
                if queue_depth > self.target_queue_depth * 1.5:
                    should_decrease = True
                elif queue_depth < self.target_queue_depth * 0.8:
                    should_increase = True

            # Check if backend is overloaded
            if self._last_metrics.is_overloaded:
                should_decrease = True

        # Check if we have room to increase
        if (
            not should_decrease
            and p95_latency < self.target_latency * 0.85
            and error_rate < 0.02
        ):
            should_increase = True

        # Apply adjustment
        if should_decrease:
            # Multiplicative decrease
            new_concurrency = max(
                self.min_concurrency, int(self._current * self.decrease_factor)
            )
            self._update_semaphore(old_concurrency, new_concurrency)
            self._current = new_concurrency
        elif should_increase:
            # Additive increase
            new_concurrency = min(
                self.max_concurrency, self._current + self.increase_step
            )
            self._update_semaphore(old_concurrency, new_concurrency)
            self._current = new_concurrency

    def _update_semaphore(self, old_value: int, new_value: int) -> None:
        """
        Update semaphore capacity dynamically.

        With DynamicSemaphore:
        - Capacity increases are immediately reflected, waking up waiting coroutines
        - Capacity decreases immediately block new acquisitions

        With legacy asyncio.Semaphore:
        - Only capacity increases are supported (via release())
        - Capacity decreases happen naturally as permits are not released

        Args:
            old_value: Previous capacity
            new_value: New capacity
        """
        # DynamicSemaphoreの容量を更新
        if self._dynamic_semaphore is not None:
            # 同期的に容量を変更（即座に反映）
            self._dynamic_semaphore.set_capacity_sync(new_value)

        # レガシーセマフォの容量を更新（増加のみ対応）
        if self._legacy_semaphore is not None:
            diff = new_value - old_value
            if diff > 0:
                # Increase capacity by releasing extra permits
                for _ in range(diff):
                    self._legacy_semaphore.release()
            # Note: Decreasing capacity is not supported with legacy semaphore

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)

        fraction = index - lower
        return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction

    def get_stats(self) -> dict:
        """Get current controller statistics."""
        latencies = [s.latency_ms for s in self._latency_window if not s.is_error]
        errors = [s for s in self._latency_window if s.is_error]

        return {
            "current_concurrency": self._current,
            "min_concurrency": self.min_concurrency,
            "max_concurrency": self.max_concurrency,
            "target_latency_ms": self.target_latency,
            "sample_count": len(self._latency_window),
            "error_count": len(errors),
            "error_rate": (
                len(errors) / len(self._latency_window) if self._latency_window else 0
            ),
            "p50_latency_ms": (
                self._calculate_percentile(latencies, 50) if latencies else None
            ),
            "p95_latency_ms": (
                self._calculate_percentile(latencies, 95) if latencies else None
            ),
            "p99_latency_ms": (
                self._calculate_percentile(latencies, 99) if latencies else None
            ),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        }

    def reset(self) -> None:
        """Reset controller to initial state."""
        # 初期値を max_concurrency に戻す
        self._current = self.max_concurrency
        self._latency_window.clear()
        self._last_adjustment_time = 0.0
        self._last_metrics = None
        # セマフォをリセット
        self._dynamic_semaphore = None
        self._legacy_semaphore = None

    def get_semaphore_stats(self) -> Optional[dict]:
        """
        Get DynamicSemaphore statistics.

        Returns:
            Semaphore stats if DynamicSemaphore is initialized, None otherwise
        """
        if self._dynamic_semaphore is not None:
            return self._dynamic_semaphore.get_stats()
        return None


class AdaptiveConcurrencyManager:
    """
    High-level manager for adaptive concurrency with metrics collection.

    Combines AdaptiveController with optional MetricsCollector for
    integrated concurrency management.

    Usage:
        manager = AdaptiveConcurrencyManager(
            base_url="http://localhost:8000",
            metrics_type=MetricsType.VLLM,
            max_concurrency=64,
        )

        async with manager:
            async with manager.acquire():
                # Execute request
                ...
                manager.record_latency(latency_ms=150.0)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        metrics_type: MetricsType = MetricsType.NONE,
        min_concurrency: int = 1,
        max_concurrency: int = 64,
        target_latency_ms: float = 2000.0,
        target_queue_depth: int = 32,
        enabled: bool = True,
    ):
        """
        Initialize the concurrency manager.

        Args:
            base_url: Base URL of inference backend (required if metrics_type != NONE)
            metrics_type: Type of metrics to collect (VLLM, SGLANG, or NONE)
            min_concurrency: Minimum concurrency level
            max_concurrency: Maximum concurrency level
            target_latency_ms: Target P95 latency
            target_queue_depth: Target backend queue depth
            enabled: Whether adaptive control is enabled (False = fixed concurrency)
        """
        self.enabled = enabled
        self.metrics_type = metrics_type

        # Controller (always created, but may use fixed concurrency)
        self.controller = AdaptiveController(
            min_concurrency=min_concurrency if enabled else max_concurrency,
            max_concurrency=max_concurrency,
            target_latency_ms=target_latency_ms,
            target_queue_depth=target_queue_depth,
        )

        # Metrics collector (only if metrics enabled)
        self.metrics_collector: Optional[MetricsCollector] = None
        if metrics_type != MetricsType.NONE and base_url:
            self.metrics_collector = MetricsCollector(
                base_url=base_url,
                metrics_type=metrics_type,
            )

        self._metrics_update_task: Optional[asyncio.Task] = None
        self._running = False

    async def __aenter__(self):
        """Start the manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop the manager."""
        await self.stop()

    async def start(self) -> None:
        """Start metrics collection and updates."""
        if self._running:
            return

        self._running = True

        if self.metrics_collector:
            await self.metrics_collector.start()
            self._metrics_update_task = asyncio.create_task(self._metrics_update_loop())

    async def stop(self) -> None:
        """Stop metrics collection."""
        self._running = False

        if self._metrics_update_task:
            self._metrics_update_task.cancel()
            try:
                await self._metrics_update_task
            except asyncio.CancelledError:
                pass
            self._metrics_update_task = None

        if self.metrics_collector:
            await self.metrics_collector.stop()

    async def _metrics_update_loop(self) -> None:
        """Background loop to feed metrics to controller."""
        while self._running:
            await asyncio.sleep(0.5)  # Update every 500ms

            if self.metrics_collector:
                metrics = self.metrics_collector.get_latest()
                if metrics:
                    self.controller.update_with_metrics(metrics)

    @property
    def current_concurrency(self) -> int:
        """Get current concurrency limit."""
        return self.controller.current_concurrency

    @property
    def semaphore(self) -> DynamicSemaphore:
        """Get concurrency-limiting DynamicSemaphore."""
        return self.controller.semaphore

    @property
    def legacy_semaphore(self) -> asyncio.Semaphore:
        """Get legacy asyncio.Semaphore for backward compatibility."""
        return self.controller.legacy_semaphore

    def record_latency(self, latency_ms: float, is_error: bool = False) -> None:
        """Record a request latency."""
        if self.enabled:
            self.controller.record_latency(latency_ms, is_error)

    async def acquire(self):
        """Acquire a permit from the semaphore."""
        return self.semaphore

    def get_stats(self) -> dict:
        """Get current statistics."""
        stats = self.controller.get_stats()

        if self.metrics_collector:
            metrics = self.metrics_collector.get_latest()
            if metrics and metrics.is_valid:
                stats["backend_queue_depth"] = metrics.queue_depth
                stats["backend_cache_usage"] = metrics.cache_usage_percent

        return stats
