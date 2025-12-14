"""
Request batcher for optimizing throughput with vLLM/SGLang backends.

This module implements dynamic request batching to maximize the benefits
of continuous batching in inference backends by grouping requests together
before submission.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")  # Result type


@dataclass
class PendingRequest(Generic[T]):
    """A request waiting to be batched and processed."""

    request_id: int
    payload: Dict[str, Any]
    future: asyncio.Future[T]
    submit_time: float = field(default_factory=time.time)
    priority: int = 1  # Lower = higher priority


@dataclass
class BatchResult(Generic[T]):
    """Result of a batch execution."""

    results: List[T]
    latencies_ms: List[float]
    error_count: int


class RequestBatcher(Generic[T]):
    """
    Dynamic request batcher for optimizing inference throughput.

    Groups multiple requests together before submission to maximize
    the benefits of continuous batching in vLLM/SGLang backends.

    Key features:
    - Dynamic batch sizing based on queue state
    - Maximum wait time to ensure latency bounds
    - Token-aware batching (optional)
    - Priority queue support

    Usage:
        async def batch_processor(payloads: List[Dict]) -> List[str]:
            # Send batch to inference backend
            return await client.batch_chat(payloads)

        batcher = RequestBatcher(
            batch_processor=batch_processor,
            max_batch_size=64,
            max_wait_ms=50,
        )

        async with batcher:
            # Submit requests
            result = await batcher.submit({"messages": [...]})
    """

    def __init__(
        self,
        batch_processor: Callable[[List[Dict[str, Any]]], Coroutine[Any, Any, List[T]]],
        max_batch_size: int = 64,
        # デフォルト0ms: 標準API（OpenAI等）では即時送信が最適
        # vLLM/SGLang等の真のバッチ処理バックエンドを使用する場合のみ増加を検討
        max_wait_ms: int = 0,
        max_tokens_per_batch: Optional[int] = None,
        token_estimator: Optional[Callable[[Dict[str, Any]], int]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the request batcher.

        Args:
            batch_processor: Async function that processes a list of payloads
                            and returns a list of results
            max_batch_size: Maximum number of requests per batch (default: 64)
            max_wait_ms: Maximum time to wait for batch formation (default: 0ms)
                         Set to higher values (e.g., 50ms) only when using backends
                         that support true batch processing (vLLM, SGLang, etc.)
            max_tokens_per_batch: Optional maximum tokens per batch
            token_estimator: Optional function to estimate tokens for a request
            enabled: Whether batching is enabled (False = process immediately)
        """
        self.batch_processor = batch_processor
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.max_tokens = max_tokens_per_batch
        self.token_estimator = token_estimator or self._default_token_estimator
        self.enabled = enabled

        # State
        self._pending: asyncio.Queue[PendingRequest[T]] = asyncio.Queue()
        self._request_counter = 0
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        # Metrics
        self._total_batches = 0
        self._total_requests = 0
        self._total_latency_ms = 0.0

    @property
    def is_running(self) -> bool:
        """Check if batcher is running."""
        return self._running

    @property
    def pending_count(self) -> int:
        """Get number of pending requests."""
        return self._pending.qsize()

    async def __aenter__(self) -> "RequestBatcher[T]":
        """Start the batcher."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the batcher."""
        await self.stop()

    async def start(self) -> None:
        """Start the batch processing loop."""
        if self._running:
            return

        self._running = True
        if self.enabled:
            self._processor_task = asyncio.create_task(self._batch_processor_loop())

    async def stop(self) -> None:
        """Stop the batch processing loop and process remaining requests."""
        self._running = False

        if self._processor_task is not None:
            # Process any remaining requests
            await self._flush_pending()

            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

    async def submit(
        self,
        payload: Dict[str, Any],
        priority: int = 1,
    ) -> T:
        """
        Submit a request for batched processing.

        Args:
            payload: Request payload (messages, parameters, etc.)
            priority: Request priority (lower = higher priority)

        Returns:
            Result from the batch processor
        """
        if not self.enabled:
            # Bypass batching, process immediately
            results = await self.batch_processor([payload])
            return results[0]

        # Create future for result
        loop = asyncio.get_event_loop()
        future: asyncio.Future[T] = loop.create_future()

        # Create pending request
        self._request_counter += 1
        pending = PendingRequest(
            request_id=self._request_counter,
            payload=payload,
            future=future,
            submit_time=time.time(),
            priority=priority,
        )

        # Add to queue
        await self._pending.put(pending)

        # Wait for result
        return await future

    async def _batch_processor_loop(self) -> None:
        """Main loop that forms and processes batches."""
        while self._running:
            try:
                batch = await self._collect_batch()

                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue processing
                import sys

                print(f"Batcher error: {e}", file=sys.stderr)

    async def _collect_batch(self) -> List[PendingRequest[T]]:
        """
        Collect requests into a batch.

        最適化: asyncio.wait_for によるタスクラップのオーバーヘッドを回避するため、
        get_nowait() + asyncio.sleep によるポーリング方式を採用。
        """
        batch: List[PendingRequest[T]] = []
        batch_tokens = 0
        deadline = time.time() + self.max_wait_ms / 1000

        # Wait for at least one request (check running flag every 100ms)
        # asyncio.wait_for の代わりに get_nowait + sleep で実装
        wait_deadline = time.time() + 0.1
        while time.time() < wait_deadline:
            try:
                first = self._pending.get_nowait()
                batch.append(first)
                if self.max_tokens:
                    batch_tokens += self.token_estimator(first.payload)
                break
            except asyncio.QueueEmpty:
                # キューが空の場合、短い間隔でリトライ
                remaining = wait_deadline - time.time()
                if remaining > 0:
                    await asyncio.sleep(min(0.01, remaining))  # 10ms間隔でポーリング
                else:
                    return []

        # 最初のリクエストが取得できなかった場合
        if not batch:
            return []

        # Collect more requests until batch is full or deadline reached
        # asyncio.wait_for の代わりに get_nowait + 短い sleep で実装
        while len(batch) < self.max_batch_size:
            now = time.time()
            if now >= deadline:
                break

            try:
                pending = self._pending.get_nowait()

                # Check token limit if enabled
                if self.max_tokens:
                    estimated_tokens = self.token_estimator(pending.payload)
                    if batch_tokens + estimated_tokens > self.max_tokens:
                        # Put back and break
                        await self._pending.put(pending)
                        break
                    batch_tokens += estimated_tokens

                batch.append(pending)

            except asyncio.QueueEmpty:
                # キューが空の場合、deadline まで短い間隔で待機
                remaining = deadline - time.time()
                if remaining > 0:
                    # 最大10ms待機し、その後再度チェック
                    await asyncio.sleep(min(0.01, remaining))
                else:
                    break

        # Sort by priority if mixed priorities
        if len(batch) > 1:
            batch.sort(key=lambda x: (x.priority, x.submit_time))

        return batch

    async def _process_batch(self, batch: List[PendingRequest[T]]) -> None:
        """Process a batch of requests."""
        start_time = time.time()

        try:
            # Extract payloads
            payloads = [req.payload for req in batch]

            # Call batch processor
            results = await self.batch_processor(payloads)

            # Distribute results to futures
            for pending, result in zip(batch, results):
                if not pending.future.done():
                    pending.future.set_result(result)

            # Update metrics
            self._total_batches += 1
            self._total_requests += len(batch)
            latency = (time.time() - start_time) * 1000
            self._total_latency_ms += latency

        except Exception as e:
            # Set exception for all pending futures
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(e)

    async def _flush_pending(self) -> None:
        """Process all remaining pending requests."""
        while not self._pending.empty():
            batch: List[PendingRequest[T]] = []

            # Collect all remaining requests
            while not self._pending.empty() and len(batch) < self.max_batch_size:
                try:
                    pending = self._pending.get_nowait()
                    batch.append(pending)
                except asyncio.QueueEmpty:
                    break

            if batch:
                await self._process_batch(batch)

    def _default_token_estimator(self, payload: Dict[str, Any]) -> int:
        """Default token estimator based on character count."""
        total_chars = 0

        messages = payload.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Multimodal content
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_chars += len(part.get("text", ""))

        # Rough estimate: 1 token ≈ 4 characters
        return max(1, total_chars // 4)

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        return {
            "enabled": self.enabled,
            "running": self._running,
            "pending_count": self.pending_count,
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": (
                self._total_requests / self._total_batches
                if self._total_batches > 0
                else 0
            ),
            "avg_batch_latency_ms": (
                self._total_latency_ms / self._total_batches
                if self._total_batches > 0
                else 0
            ),
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
        }


class AdaptiveRequestBatcher(RequestBatcher[T]):
    """
    Request batcher with adaptive batch sizing based on controller metrics.

    Integrates with AdaptiveController to dynamically adjust batch size
    based on observed latencies and backend load.
    """

    def __init__(
        self,
        batch_processor: Callable[[List[Dict[str, Any]]], Coroutine[Any, Any, List[T]]],
        controller: Optional[Any] = None,  # AdaptiveController
        max_batch_size: int = 64,
        min_batch_size: int = 1,
        # デフォルト0ms: 標準API（OpenAI等）では即時送信が最適
        # vLLM/SGLang等の真のバッチ処理バックエンドを使用する場合のみ増加を検討
        max_wait_ms: int = 0,
        max_tokens_per_batch: Optional[int] = None,
        token_estimator: Optional[Callable[[Dict[str, Any]], int]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the adaptive request batcher.

        Args:
            batch_processor: Async function that processes batches
            controller: Optional AdaptiveController for dynamic sizing
            max_batch_size: Maximum batch size (default: 64)
            min_batch_size: Minimum batch size (default: 1)
            max_wait_ms: Maximum wait time in ms (default: 0 for immediate dispatch)
                         Set to higher values (e.g., 50ms) only when using backends
                         that support true batch processing (vLLM, SGLang, etc.)
            max_tokens_per_batch: Optional token limit per batch
            token_estimator: Optional token estimation function
            enabled: Whether batching is enabled
        """
        super().__init__(
            batch_processor=batch_processor,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            max_tokens_per_batch=max_tokens_per_batch,
            token_estimator=token_estimator,
            enabled=enabled,
        )

        self.controller = controller
        self.min_batch_size = min_batch_size
        self._current_batch_size = max_batch_size

    @property
    def current_batch_size(self) -> int:
        """Get current dynamic batch size."""
        # バッチサイズは並行数とは独立して max_batch_size を使用
        # バッチングの目的は複数リクエストを集約して送信することであり、
        # 並行数制御とは別の概念
        return self.max_batch_size

    async def _collect_batch(self) -> List[PendingRequest[T]]:
        """
        Collect requests with adaptive batch size.

        最適化: asyncio.wait_for によるタスクラップのオーバーヘッドを回避するため、
        get_nowait() + asyncio.sleep によるポーリング方式を採用。
        """
        batch: List[PendingRequest[T]] = []
        batch_tokens = 0
        deadline = time.time() + self.max_wait_ms / 1000
        current_max = self.current_batch_size

        # Wait for at least one request (check running flag every 100ms)
        # asyncio.wait_for の代わりに get_nowait + sleep で実装
        wait_deadline = time.time() + 0.1
        while time.time() < wait_deadline:
            try:
                first = self._pending.get_nowait()
                batch.append(first)
                if self.max_tokens:
                    batch_tokens += self.token_estimator(first.payload)
                break
            except asyncio.QueueEmpty:
                # キューが空の場合、短い間隔でリトライ
                remaining = wait_deadline - time.time()
                if remaining > 0:
                    await asyncio.sleep(min(0.01, remaining))  # 10ms間隔でポーリング
                else:
                    return []

        # 最初のリクエストが取得できなかった場合
        if not batch:
            return []

        # Collect more requests until batch is full or deadline reached
        # asyncio.wait_for の代わりに get_nowait + 短い sleep で実装
        while len(batch) < current_max:
            now = time.time()
            if now >= deadline:
                break

            try:
                pending = self._pending.get_nowait()

                # Check token limit if enabled
                if self.max_tokens:
                    estimated_tokens = self.token_estimator(pending.payload)
                    if batch_tokens + estimated_tokens > self.max_tokens:
                        await self._pending.put(pending)
                        break
                    batch_tokens += estimated_tokens

                batch.append(pending)

            except asyncio.QueueEmpty:
                # キューが空の場合、deadline まで短い間隔で待機
                remaining = deadline - time.time()
                if remaining > 0:
                    # 最大10ms待機し、その後再度チェック
                    await asyncio.sleep(min(0.01, remaining))
                else:
                    break

        # Sort by priority
        if len(batch) > 1:
            batch.sort(key=lambda x: (x.priority, x.submit_time))

        return batch

    async def _process_batch(self, batch: List[PendingRequest[T]]) -> None:
        """Process batch and report metrics to controller."""
        start_time = time.time()

        try:
            payloads = [req.payload for req in batch]
            results = await self.batch_processor(payloads)

            for pending, result in zip(batch, results):
                if not pending.future.done():
                    pending.future.set_result(result)

            # Update metrics
            self._total_batches += 1
            self._total_requests += len(batch)
            latency = (time.time() - start_time) * 1000
            self._total_latency_ms += latency

            # Report to controller
            if self.controller is not None:
                # Report per-request latency estimate
                per_request_latency = latency / len(batch)
                for _ in batch:
                    self.controller.record_latency(per_request_latency, is_error=False)

        except Exception as e:
            # Set exception for all pending futures
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(e)

            # Report errors to controller
            if self.controller is not None:
                latency = (time.time() - start_time) * 1000
                for _ in batch:
                    self.controller.record_latency(latency, is_error=True)
