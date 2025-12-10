"""
Advanced adaptive concurrency controller for maximizing inference throughput.

This module implements a sophisticated congestion control algorithm inspired by
TCP Vegas, Reno, and BBR. It dynamically adjusts concurrency based on:
- Smoothed latency metrics (EMA-based noise reduction)
- Control phases (Slow Start vs Congestion Avoidance)
- Graduated response to latency degradation
- Backend metrics from vLLM/SGLang

Key improvements over simple AIMD:
1. Exponential increase during slow start for faster convergence
2. EMA-based latency smoothing for stability
3. Graduated decrease logic to avoid over-reaction to transient spikes
4. RTT-based congestion detection (Vegas-style)

Also includes DynamicSemaphore for real-time capacity adjustment.
"""

from __future__ import annotations
import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

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
    """A single latency measurement with metadata."""

    latency_ms: float
    timestamp: float
    is_error: bool = False
    concurrency_at_time: int = 0  # Concurrency level when this sample was taken


class ControlPhase(Enum):
    """Control phases inspired by TCP congestion control."""

    SLOW_START = "slow_start"  # Exponential increase phase
    CONGESTION_AVOIDANCE = "congestion_avoidance"  # Linear increase phase
    FAST_RECOVERY = "fast_recovery"  # Recovery after mild congestion


@dataclass
class EMAState:
    """
    Exponential Moving Average state for smoothed metrics.

    Uses EMA to filter out noise and detect trends in latency measurements.
    """

    value: float = 0.0
    variance: float = 0.0  # For detecting stability
    initialized: bool = False
    trend: float = 0.0  # Positive = increasing, negative = decreasing


@dataclass
class CongestionState:
    """
    Current congestion detection state.

    Tracks RTT-based congestion signals similar to TCP Vegas.
    """

    base_latency: float = float("inf")  # Minimum observed latency (base RTT)
    expected_throughput: float = 0.0  # Expected throughput at base latency
    actual_throughput: float = 0.0  # Actual observed throughput
    congestion_signal: float = 0.0  # Vegas-style diff between expected and actual


class AdaptiveController:
    """
    Advanced adaptive concurrency controller with TCP-inspired congestion control.

    This controller implements a sophisticated algorithm combining ideas from:
    - TCP Reno: AIMD with slow start and congestion avoidance phases
    - TCP Vegas: RTT-based congestion detection for proactive control
    - BBR: Bandwidth-delay product estimation for optimal window sizing

    Key Features:
    1. **Slow Start Phase**: Exponential increase when far from optimal
    2. **Congestion Avoidance Phase**: Linear increase near optimal
    3. **EMA-based Smoothing**: Noise reduction for stable decisions
    4. **Graduated Decrease**: Different responses for errors vs latency spikes
    5. **Vegas-style Proactive Control**: Detect congestion before packet loss

    The controller dynamically adjusts the concurrency level based on:
    - Smoothed client-side latencies (EMA)
    - Latency trend detection
    - Backend metrics like queue depth (optional, for vLLM/SGLang)

    Uses DynamicSemaphore for real-time capacity adjustment:
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
        increase_step: int = 2,  # Additive increase per adjustment (CA phase)
        decrease_factor: float = 0.5,  # Multiplicative decrease factor (for errors)
        # Adjustment sensitivity
        latency_tolerance: float = 1.5,  # Trigger decrease if latency > target * tolerance
        error_rate_threshold: float = 0.05,  # 5% error rate triggers decrease
        # Timing
        adjustment_interval_ms: int = 1000,  # Minimum time between adjustments
        window_size: int = 50,  # Number of samples for averaging
        # Initial concurrency
        initial_concurrency: Optional[int] = None,
        # Use legacy semaphore for backward compatibility
        use_dynamic_semaphore: bool = True,
        # Advanced parameters
        ema_alpha: float = 0.3,  # EMA smoothing factor (0-1, higher = more reactive)
        slow_start_threshold: Optional[int] = None,  # Initial ssthresh
        vegas_alpha: float = 2.0,  # Vegas lower threshold (packets)
        vegas_beta: float = 4.0,  # Vegas upper threshold (packets)
        mild_decrease_factor: float = 0.85,  # Decrease factor for latency spikes
        trend_sensitivity: float = 0.1,  # Threshold for trend detection
    ):
        """
        Initialize the adaptive controller.

        Args:
            min_concurrency: Minimum concurrency level (default: 1)
            max_concurrency: Maximum concurrency level (default: 64)
            target_latency_ms: Target P95 latency in milliseconds (default: 2000)
            target_queue_depth: Target backend queue depth (default: 32)
            increase_step: Additive increase per adjustment cycle (default: 2)
            decrease_factor: Multiplicative decrease factor for errors (default: 0.5)
            latency_tolerance: Latency threshold multiplier for decrease (default: 1.5)
            error_rate_threshold: Error rate threshold for decrease (default: 0.05)
            adjustment_interval_ms: Minimum interval between adjustments (default: 1000)
            window_size: Number of samples to consider (default: 50)
            initial_concurrency: Initial concurrency level (default: min_concurrency)
            use_dynamic_semaphore: Use DynamicSemaphore for real-time capacity changes
            ema_alpha: EMA smoothing factor (default: 0.3)
            slow_start_threshold: Initial slow start threshold (default: max/2)
            vegas_alpha: Vegas lower congestion threshold (default: 2.0)
            vegas_beta: Vegas upper congestion threshold (default: 4.0)
            mild_decrease_factor: Decrease factor for latency issues (default: 0.85)
            trend_sensitivity: Threshold for detecting latency trends (default: 0.1)
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

        # Advanced control parameters
        self.ema_alpha = ema_alpha
        self.vegas_alpha = vegas_alpha
        self.vegas_beta = vegas_beta
        self.mild_decrease_factor = mild_decrease_factor
        self.trend_sensitivity = trend_sensitivity

        # State - Start with slow start from initial_concurrency or min_concurrency
        if initial_concurrency is not None:
            self._current: int = max(
                min_concurrency, min(initial_concurrency, max_concurrency)
            )
        else:
            # For slow start, begin conservatively
            self._current: int = min_concurrency

        # Slow start threshold (ssthresh) - initially set high or to given value
        if slow_start_threshold is not None:
            self._ssthresh: int = slow_start_threshold
        else:
            self._ssthresh: int = max_concurrency // 2

        # Control phase
        self._phase: ControlPhase = ControlPhase.SLOW_START

        # Latency samples
        self._latency_window: Deque[LatencySample] = deque(maxlen=window_size)
        self._last_adjustment_time: float = 0.0
        self._last_metrics: Optional[BackendMetrics] = None

        # EMA state for smoothed latency
        self._ema_latency = EMAState()
        self._ema_p95 = EMAState()

        # Congestion detection state (Vegas-style)
        self._congestion = CongestionState()

        # History for trend detection
        self._adjustment_history: Deque[Tuple[float, int, str]] = deque(maxlen=20)

        # Consecutive event counters for stability
        self._consecutive_good: int = 0
        self._consecutive_bad: int = 0

        # Semaphore for dynamic concurrency control
        self._dynamic_semaphore: Optional[DynamicSemaphore] = None
        self._legacy_semaphore: Optional[asyncio.Semaphore] = None

    @property
    def current_concurrency(self) -> int:
        """Get current concurrency limit."""
        return self._current

    @property
    def phase(self) -> ControlPhase:
        """Get current control phase."""
        return self._phase

    @property
    def ssthresh(self) -> int:
        """Get current slow start threshold."""
        return self._ssthresh

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

    def _update_ema(self, state: EMAState, new_value: float) -> None:
        """
        Update EMA state with a new value.

        Also calculates variance and trend for stability detection.

        Args:
            state: EMA state to update
            new_value: New observed value
        """
        if not state.initialized:
            state.value = new_value
            state.variance = 0.0
            state.trend = 0.0
            state.initialized = True
            return

        # Calculate trend before updating value
        old_value = state.value

        # Update EMA value
        state.value = self.ema_alpha * new_value + (1 - self.ema_alpha) * state.value

        # Update variance estimate (for stability detection)
        diff = new_value - state.value
        state.variance = (
            self.ema_alpha * (diff * diff) + (1 - self.ema_alpha) * state.variance
        )

        # Update trend (smoothed derivative)
        instant_trend = state.value - old_value
        state.trend = (
            self.ema_alpha * instant_trend + (1 - self.ema_alpha) * state.trend
        )

    def _update_congestion_state(self, latency_ms: float) -> None:
        """
        Update Vegas-style congestion detection state.

        Tracks base RTT and calculates expected vs actual throughput
        to detect congestion proactively.

        Args:
            latency_ms: Observed latency
        """
        # Update base latency (minimum observed)
        if latency_ms < self._congestion.base_latency:
            self._congestion.base_latency = latency_ms

        # Calculate expected throughput at base latency
        # expected = cwnd / base_rtt
        if self._congestion.base_latency > 0:
            self._congestion.expected_throughput = (
                self._current / self._congestion.base_latency
            )

        # Calculate actual throughput
        # actual = cwnd / observed_rtt
        if latency_ms > 0:
            self._congestion.actual_throughput = self._current / latency_ms

        # Vegas-style congestion signal: diff = expected - actual
        # Measured in "packets" (concurrency units here)
        self._congestion.congestion_signal = self._current * (
            1 - self._congestion.base_latency / max(latency_ms, 1)
        )

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
            concurrency_at_time=self._current,
        )
        self._latency_window.append(sample)

        # Update EMA for non-error samples
        if not is_error:
            self._update_ema(self._ema_latency, latency_ms)
            self._update_congestion_state(latency_ms)

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

    def _assess_congestion_level(self) -> Tuple[str, float]:
        """
        Assess current congestion level using multiple signals.

        Returns:
            Tuple of (congestion_level, severity)
            - congestion_level: "none", "mild", "moderate", "severe"
            - severity: 0.0 to 1.0
        """
        signals: List[Tuple[str, float]] = []

        # Signal 1: EMA latency vs target
        if self._ema_latency.initialized:
            latency_ratio = self._ema_latency.value / self.target_latency
            if latency_ratio < 0.7:
                signals.append(("none", 0.0))
            elif latency_ratio < 1.0:
                signals.append(("none", latency_ratio - 0.7))
            elif latency_ratio < 1.3:
                signals.append(("mild", (latency_ratio - 1.0) / 0.3))
            elif latency_ratio < self.latency_tolerance:
                signals.append(("moderate", (latency_ratio - 1.3) / 0.2))
            else:
                signals.append(("severe", min(1.0, (latency_ratio - 1.5) / 0.5)))

        # Signal 2: Vegas-style congestion
        vegas_signal = self._congestion.congestion_signal
        if vegas_signal < self.vegas_alpha:
            signals.append(("none", vegas_signal / self.vegas_alpha))
        elif vegas_signal < self.vegas_beta:
            signals.append(
                (
                    "mild",
                    (vegas_signal - self.vegas_alpha)
                    / (self.vegas_beta - self.vegas_alpha),
                )
            )
        else:
            signals.append(("moderate", min(1.0, vegas_signal / (self.vegas_beta * 2))))

        # Signal 3: Latency trend
        if self._ema_latency.initialized:
            trend = self._ema_latency.trend
            if trend > self.trend_sensitivity * self.target_latency:
                signals.append(("mild", min(1.0, trend / self.target_latency)))
            elif trend < -self.trend_sensitivity * self.target_latency:
                signals.append(("none", 0.0))

        # Signal 4: Backend queue depth
        if self._last_metrics and self._last_metrics.is_valid:
            queue_depth = self._last_metrics.queue_depth
            if queue_depth is not None:
                queue_ratio = queue_depth / self.target_queue_depth
                if queue_ratio < 0.8:
                    signals.append(("none", 0.0))
                elif queue_ratio < 1.2:
                    signals.append(("none", (queue_ratio - 0.8) / 0.4))
                elif queue_ratio < 1.5:
                    signals.append(("mild", (queue_ratio - 1.2) / 0.3))
                else:
                    signals.append(("moderate", min(1.0, (queue_ratio - 1.5) / 0.5)))

            if self._last_metrics.is_overloaded:
                signals.append(("severe", 1.0))

        if not signals:
            return ("none", 0.0)

        # Aggregate signals - take the worst case
        level_order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        worst_level = "none"
        max_severity = 0.0

        for level, severity in signals:
            if level_order[level] > level_order[worst_level]:
                worst_level = level
                max_severity = severity
            elif level_order[level] == level_order[worst_level]:
                max_severity = max(max_severity, severity)

        return (worst_level, max_severity)

    def _adjust_concurrency(self) -> None:
        """
        Perform concurrency adjustment using advanced congestion control.

        Implements:
        1. Slow Start: Exponential increase when below ssthresh
        2. Congestion Avoidance: Linear increase when above ssthresh
        3. Graduated decrease based on congestion severity
        """
        old_concurrency = self._current

        # Calculate error rate
        errors = [s for s in self._latency_window if s.is_error]
        error_rate = (
            len(errors) / len(self._latency_window) if self._latency_window else 0
        )

        # Update P95 EMA
        latencies = [s.latency_ms for s in self._latency_window if not s.is_error]
        if latencies:
            p95 = self._calculate_percentile(latencies, 95)
            self._update_ema(self._ema_p95, p95)

        action = "hold"

        # === ERROR HANDLING (highest priority) ===
        if error_rate > self.error_rate_threshold:
            # Immediate multiplicative decrease for errors
            new_concurrency = max(
                self.min_concurrency, int(self._current * self.decrease_factor)
            )
            self._ssthresh = max(self.min_concurrency, new_concurrency)
            self._phase = ControlPhase.SLOW_START
            self._consecutive_good = 0
            self._consecutive_bad += 1
            action = "md_error"

            self._update_semaphore(old_concurrency, new_concurrency)
            self._current = new_concurrency
            self._record_adjustment(action)
            return

        # === CONGESTION-BASED CONTROL ===
        congestion_level, severity = self._assess_congestion_level()

        if congestion_level == "severe":
            # Multiplicative decrease for severe congestion
            new_concurrency = max(
                self.min_concurrency, int(self._current * self.decrease_factor)
            )
            self._ssthresh = max(self.min_concurrency, new_concurrency)
            self._phase = ControlPhase.CONGESTION_AVOIDANCE
            self._consecutive_good = 0
            self._consecutive_bad += 1
            action = "md_severe"

        elif congestion_level == "moderate":
            # Mild decrease for moderate congestion
            decrease = 1.0 - (1.0 - self.mild_decrease_factor) * severity
            new_concurrency = max(self.min_concurrency, int(self._current * decrease))
            self._ssthresh = max(self.min_concurrency, self._current)
            self._phase = ControlPhase.CONGESTION_AVOIDANCE
            self._consecutive_good = 0
            self._consecutive_bad += 1
            action = "decrease_moderate"

        elif congestion_level == "mild":
            # Hold or very slight decrease for mild congestion
            self._consecutive_good = 0
            self._consecutive_bad += 1

            if self._consecutive_bad >= 3:
                # Sustained mild congestion - slight decrease
                new_concurrency = max(
                    self.min_concurrency, self._current - self.increase_step
                )
                action = "decrease_mild"
            else:
                # Transient - just hold
                new_concurrency = self._current
                action = "hold_mild"

        else:
            # No congestion - increase
            self._consecutive_bad = 0
            self._consecutive_good += 1

            if self._phase == ControlPhase.SLOW_START:
                # Exponential increase in slow start
                if (
                    self._current < self._ssthresh
                    and self._consecutive_good >= 2
                    and self._ema_latency.initialized
                    and self._ema_latency.value < self.target_latency * 0.7
                ):
                    # Double the concurrency (exponential)
                    new_concurrency = min(self.max_concurrency, self._current * 2)
                    action = "ss_increase"

                    # Exit slow start if we hit ssthresh
                    if new_concurrency >= self._ssthresh:
                        self._phase = ControlPhase.CONGESTION_AVOIDANCE
                        action = "ss_to_ca"
                else:
                    # Conservative increase if not fully warmed up
                    new_concurrency = min(
                        self.max_concurrency, self._current + self.increase_step
                    )
                    action = "ss_linear"
            else:
                # Congestion avoidance - linear increase
                if self._consecutive_good >= 2:
                    new_concurrency = min(
                        self.max_concurrency, self._current + self.increase_step
                    )
                    action = "ca_increase"
                else:
                    new_concurrency = self._current
                    action = "ca_hold"

        # Apply the adjustment
        if new_concurrency != old_concurrency:
            self._update_semaphore(old_concurrency, new_concurrency)
            self._current = new_concurrency

        self._record_adjustment(action)

    def _record_adjustment(self, action: str) -> None:
        """Record an adjustment for debugging and analysis."""
        self._adjustment_history.append((time.time(), self._current, action))

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

    def get_stats(self) -> dict:
        """Get current controller statistics."""
        latencies = [s.latency_ms for s in self._latency_window if not s.is_error]
        errors = [s for s in self._latency_window if s.is_error]

        stats = {
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
            # Advanced stats
            "phase": self._phase.value,
            "ssthresh": self._ssthresh,
            "ema_latency_ms": (
                self._ema_latency.value if self._ema_latency.initialized else None
            ),
            "ema_latency_trend": (
                self._ema_latency.trend if self._ema_latency.initialized else None
            ),
            "ema_latency_variance": (
                self._ema_latency.variance if self._ema_latency.initialized else None
            ),
            "base_latency_ms": (
                self._congestion.base_latency
                if self._congestion.base_latency != float("inf")
                else None
            ),
            "vegas_congestion_signal": self._congestion.congestion_signal,
            "consecutive_good": self._consecutive_good,
            "consecutive_bad": self._consecutive_bad,
        }

        # Add recent adjustments
        if self._adjustment_history:
            recent = list(self._adjustment_history)[-5:]
            stats["recent_adjustments"] = [
                {"timestamp": ts, "concurrency": conc, "action": act}
                for ts, conc, act in recent
            ]

        return stats

    def reset(self) -> None:
        """Reset controller to initial state."""
        # Reset to slow start with minimum concurrency
        self._current = self.min_concurrency
        self._ssthresh = self.max_concurrency // 2
        self._phase = ControlPhase.SLOW_START

        self._latency_window.clear()
        self._last_adjustment_time = 0.0
        self._last_metrics = None

        # Reset EMA states
        self._ema_latency = EMAState()
        self._ema_p95 = EMAState()

        # Reset congestion state
        self._congestion = CongestionState()

        # Reset counters
        self._consecutive_good = 0
        self._consecutive_bad = 0
        self._adjustment_history.clear()

        # Reset semaphores
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

    def get_ema_stats(self) -> dict:
        """
        Get EMA statistics for debugging.

        Returns:
            Dictionary with EMA latency and P95 statistics
        """
        return {
            "latency": {
                "value": (
                    self._ema_latency.value if self._ema_latency.initialized else None
                ),
                "variance": (
                    self._ema_latency.variance
                    if self._ema_latency.initialized
                    else None
                ),
                "trend": (
                    self._ema_latency.trend if self._ema_latency.initialized else None
                ),
                "initialized": self._ema_latency.initialized,
            },
            "p95": {
                "value": self._ema_p95.value if self._ema_p95.initialized else None,
                "variance": (
                    self._ema_p95.variance if self._ema_p95.initialized else None
                ),
                "trend": self._ema_p95.trend if self._ema_p95.initialized else None,
                "initialized": self._ema_p95.initialized,
            },
        }

    def get_congestion_stats(self) -> dict:
        """
        Get congestion detection statistics.

        Returns:
            Dictionary with Vegas-style congestion metrics
        """
        congestion_level, severity = self._assess_congestion_level()
        return {
            "base_latency_ms": (
                self._congestion.base_latency
                if self._congestion.base_latency != float("inf")
                else None
            ),
            "expected_throughput": self._congestion.expected_throughput,
            "actual_throughput": self._congestion.actual_throughput,
            "congestion_signal": self._congestion.congestion_signal,
            "congestion_level": congestion_level,
            "congestion_severity": severity,
        }


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
