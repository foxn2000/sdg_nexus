"""
Metrics collection for vLLM and SGLang inference backends.

This module provides real-time monitoring of inference backend state
to enable adaptive concurrency control.
"""

from __future__ import annotations
import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import deque
import time


class MetricsType(Enum):
    """Supported metrics backend types."""

    NONE = "none"  # No metrics (fallback to latency-based)
    VLLM = "vllm"  # vLLM Prometheus metrics
    SGLANG = "sglang"  # SGLang Prometheus metrics


@dataclass
class BackendMetrics:
    """
    Metrics collected from inference backend.

    Note: GPU utilization is intentionally not included as it may not
    be available in all deployment scenarios.
    """

    # Queue and processing state
    num_requests_waiting: Optional[int] = None  # Requests in queue
    num_requests_running: Optional[int] = None  # Currently processing

    # Cache metrics (vLLM specific)
    cache_usage_percent: Optional[float] = None  # KV cache usage

    # Throughput metrics
    prompt_tokens_total: Optional[int] = None  # Total input tokens
    generation_tokens_total: Optional[int] = None  # Total output tokens

    # Latency metrics (from backend, not client-side)
    avg_time_to_first_token_ms: Optional[float] = None
    avg_generation_time_ms: Optional[float] = None

    # Collection metadata
    timestamp: float = field(default_factory=time.time)
    is_valid: bool = True
    error_message: Optional[str] = None

    @property
    def queue_depth(self) -> Optional[int]:
        """Total queue depth (waiting + running)."""
        if (
            self.num_requests_waiting is not None
            and self.num_requests_running is not None
        ):
            return self.num_requests_waiting + self.num_requests_running
        return self.num_requests_waiting or self.num_requests_running

    @property
    def is_overloaded(self) -> bool:
        """Quick check if backend appears overloaded."""
        if self.num_requests_waiting is not None and self.num_requests_waiting > 100:
            return True
        if self.cache_usage_percent is not None and self.cache_usage_percent > 0.95:
            return True
        return False


class MetricsCollector:
    """
    Collects metrics from vLLM or SGLang inference backends.

    Metrics are collected by polling the /metrics endpoint (Prometheus format).
    The collector runs in the background and maintains the latest metrics.

    Usage:
        collector = MetricsCollector(
            base_url="http://localhost:8000",
            metrics_type=MetricsType.VLLM,
        )
        await collector.start()

        # Get latest metrics
        metrics = collector.get_latest()

        # Stop when done
        await collector.stop()
    """

    def __init__(
        self,
        base_url: str,
        metrics_type: MetricsType = MetricsType.NONE,
        poll_interval_ms: int = 500,
        history_size: int = 100,
    ):
        """
        Initialize the metrics collector.

        Args:
            base_url: Base URL of the inference backend (e.g., "http://localhost:8000")
            metrics_type: Type of metrics to collect (VLLM, SGLANG, or NONE)
            poll_interval_ms: Polling interval in milliseconds (default: 500)
            history_size: Number of historical metrics to retain (default: 100)
        """
        # base_urlから末尾のスラッシュと /v1 を除去してメトリクスエンドポイントを構築
        # vLLMとSGLangのメトリクスは /metrics にあり、/v1/metrics ではない
        self.base_url = base_url.rstrip("/")
        # /v1 で終わる場合は除去
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]

        self.metrics_type = metrics_type
        self.poll_interval = poll_interval_ms / 1000.0
        self.history_size = history_size

        self._metrics_url = f"{self.base_url}/metrics"
        self._latest: Optional[BackendMetrics] = None
        self._history: deque[BackendMetrics] = deque(maxlen=history_size)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._http_session: Any = None  # aiohttp.ClientSession

    @property
    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self.metrics_type != MetricsType.NONE

    async def start(self) -> None:
        """Start background metrics collection."""
        if not self.is_enabled:
            return

        if self._running:
            return

        try:
            import aiohttp

            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0)
            )
        except ImportError:
            # aiohttp not available, disable metrics
            self.metrics_type = MetricsType.NONE
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop background metrics collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    def get_latest(self) -> Optional[BackendMetrics]:
        """Get the most recently collected metrics."""
        return self._latest

    def get_history(self) -> List[BackendMetrics]:
        """Get historical metrics."""
        return list(self._history)

    def get_avg_queue_depth(self, window: int = 10) -> Optional[float]:
        """Get average queue depth over recent samples."""
        if not self._history:
            return None

        samples = list(self._history)[-window:]
        depths = [m.queue_depth for m in samples if m.queue_depth is not None]

        if not depths:
            return None
        return sum(depths) / len(depths)

    async def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                metrics = await self._fetch_metrics()
                self._latest = metrics
                self._history.append(metrics)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Record error but continue polling
                error_metrics = BackendMetrics(
                    is_valid=False,
                    error_message=str(e),
                )
                self._latest = error_metrics
                self._history.append(error_metrics)

            await asyncio.sleep(self.poll_interval)

    async def _fetch_metrics(self) -> BackendMetrics:
        """Fetch and parse metrics from the backend."""
        if not self._http_session:
            return BackendMetrics(is_valid=False, error_message="No HTTP session")

        try:
            async with self._http_session.get(self._metrics_url) as resp:
                if resp.status != 200:
                    return BackendMetrics(
                        is_valid=False,
                        error_message=f"HTTP {resp.status}",
                    )

                text = await resp.text()
                return self._parse_prometheus_metrics(text)
        except Exception as e:
            return BackendMetrics(
                is_valid=False,
                error_message=str(e),
            )

    def _parse_prometheus_metrics(self, text: str) -> BackendMetrics:
        """Parse Prometheus format metrics text."""
        metrics = BackendMetrics()

        if self.metrics_type == MetricsType.VLLM:
            metrics = self._parse_vllm_metrics(text)
        elif self.metrics_type == MetricsType.SGLANG:
            metrics = self._parse_sglang_metrics(text)

        return metrics

    def _parse_vllm_metrics(self, text: str) -> BackendMetrics:
        """
        Parse vLLM Prometheus metrics.

        Expected metrics format:
            vllm:num_requests_running 5
            vllm:num_requests_waiting 10
            vllm:gpu_cache_usage_perc 0.75
            vllm:prompt_tokens_total 12345
            vllm:generation_tokens_total 67890
        """
        metrics = BackendMetrics()

        patterns = {
            # Queue metrics
            r"vllm:num_requests_running\s+(\d+(?:\.\d+)?)": "num_requests_running",
            r"vllm:num_requests_waiting\s+(\d+(?:\.\d+)?)": "num_requests_waiting",
            r"vllm:num_requests_swapped\s+(\d+(?:\.\d+)?)": None,  # Ignored for now
            # Cache metrics
            r"vllm:gpu_cache_usage_perc\s+(\d+(?:\.\d+)?)": "cache_usage_percent",
            r"vllm:cpu_cache_usage_perc\s+(\d+(?:\.\d+)?)": None,  # Ignored
            # Token counters
            r"vllm:prompt_tokens_total\s+(\d+(?:\.\d+)?)": "prompt_tokens_total",
            r"vllm:generation_tokens_total\s+(\d+(?:\.\d+)?)": "generation_tokens_total",
            # Latency metrics (histogram buckets are more complex, skip for now)
            r"vllm:time_to_first_token_seconds_sum\s+(\d+(?:\.\d+)?)": None,
            r"vllm:time_per_output_token_seconds_sum\s+(\d+(?:\.\d+)?)": None,
        }

        for pattern, attr in patterns.items():
            if attr is None:
                continue
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                if attr in (
                    "num_requests_running",
                    "num_requests_waiting",
                    "prompt_tokens_total",
                    "generation_tokens_total",
                ):
                    value = int(value)
                setattr(metrics, attr, value)

        return metrics

    def _parse_sglang_metrics(self, text: str) -> BackendMetrics:
        """
        Parse SGLang Prometheus metrics.

        SGLang uses similar format to vLLM but with different prefixes.
        Expected metrics:
            sglang_num_requests_running 5
            sglang_num_requests_waiting 10
            sglang_token_usage 0.75
        """
        metrics = BackendMetrics()

        patterns = {
            # Queue metrics (SGLang format)
            r"sglang_num_requests_running\s+(\d+(?:\.\d+)?)": "num_requests_running",
            r"sglang_num_requests_waiting\s+(\d+(?:\.\d+)?)": "num_requests_waiting",
            r"sglang_running_req_count\s+(\d+(?:\.\d+)?)": "num_requests_running",
            r"sglang_waiting_req_count\s+(\d+(?:\.\d+)?)": "num_requests_waiting",
            # Cache/memory metrics
            r"sglang_token_usage\s+(\d+(?:\.\d+)?)": "cache_usage_percent",
            r"sglang_cache_hit_rate\s+(\d+(?:\.\d+)?)": None,
            # Token counters
            r"sglang_prompt_tokens_total\s+(\d+(?:\.\d+)?)": "prompt_tokens_total",
            r"sglang_gen_tokens_total\s+(\d+(?:\.\d+)?)": "generation_tokens_total",
            r"sglang_generation_tokens_total\s+(\d+(?:\.\d+)?)": "generation_tokens_total",
        }

        for pattern, attr in patterns.items():
            if attr is None:
                continue
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                if attr in (
                    "num_requests_running",
                    "num_requests_waiting",
                    "prompt_tokens_total",
                    "generation_tokens_total",
                ):
                    value = int(value)
                setattr(metrics, attr, value)

        return metrics


class MultiBackendMetricsCollector:
    """
    Collect metrics from multiple backend instances.

    Useful for load-balanced deployments with multiple vLLM/SGLang servers.
    """

    def __init__(
        self,
        backends: Dict[str, MetricsType],
        poll_interval_ms: int = 500,
    ):
        """
        Initialize multi-backend collector.

        Args:
            backends: Dict mapping base_url -> MetricsType
            poll_interval_ms: Polling interval in milliseconds
        """
        self.collectors: Dict[str, MetricsCollector] = {}

        for url, metrics_type in backends.items():
            self.collectors[url] = MetricsCollector(
                base_url=url,
                metrics_type=metrics_type,
                poll_interval_ms=poll_interval_ms,
            )

    async def start(self) -> None:
        """Start all collectors."""
        for collector in self.collectors.values():
            await collector.start()

    async def stop(self) -> None:
        """Stop all collectors."""
        for collector in self.collectors.values():
            await collector.stop()

    def get_aggregated_metrics(self) -> BackendMetrics:
        """Get aggregated metrics across all backends."""
        total_waiting = 0
        total_running = 0
        max_cache_usage = 0.0
        valid_count = 0

        for collector in self.collectors.values():
            metrics = collector.get_latest()
            if metrics and metrics.is_valid:
                valid_count += 1
                if metrics.num_requests_waiting is not None:
                    total_waiting += metrics.num_requests_waiting
                if metrics.num_requests_running is not None:
                    total_running += metrics.num_requests_running
                if metrics.cache_usage_percent is not None:
                    max_cache_usage = max(max_cache_usage, metrics.cache_usage_percent)

        if valid_count == 0:
            return BackendMetrics(is_valid=False, error_message="No valid backends")

        return BackendMetrics(
            num_requests_waiting=total_waiting,
            num_requests_running=total_running,
            cache_usage_percent=max_cache_usage if max_cache_usage > 0 else None,
        )

    def get_total_queue_depth(self) -> int:
        """Get total queue depth across all backends."""
        aggregated = self.get_aggregated_metrics()
        return aggregated.queue_depth or 0
