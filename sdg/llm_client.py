from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from .utils import now_ms


class LLMError(RuntimeError):
    pass


class BatchOptimizer:
    """Simple adaptive concurrency controller based on latency and error rate."""

    def __init__(self, min_batch=1, max_batch=8, target_latency_ms=3000):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_latency_ms = target_latency_ms
        self._current = max_batch  # Start with max batch size for better throughput

    def current(self) -> int:
        return self._current

    def update(self, latencies_ms: List[int], errors: int):
        if not latencies_ms:
            return

        # Calculate per-request latency instead of total batch latency
        # Since we process multiple requests in parallel, we should measure
        # the average time per individual request, not the batch as a whole
        total_latency = sum(latencies_ms)
        num_requests = len(latencies_ms)
        per_request_latency = total_latency / num_requests

        # Adjust batch size based on per-request latency and error rate
        error_rate = errors / num_requests if num_requests > 0 else 0

        if error_rate > 0.05:  # More than 5% errors
            # Decrease aggressively on errors
            self._current = max(self.min_batch, self._current - 2)
        elif per_request_latency > self.target_latency_ms:
            # Decrease gradually if latency is too high
            self._current = max(self.min_batch, self._current - 1)
        elif per_request_latency < self.target_latency_ms * 0.7:
            # Increase gradually if we have headroom
            self._current = min(self.max_batch, self._current + 1)
        # else: keep current batch size (latency is in acceptable range)


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        organization: Optional[str],
        headers: Dict[str, str],
        timeout_sec: Optional[float] = None,
    ):
        base = (base_url or "https://api.openai.com").rstrip("/")
        if base.endswith("/v1"):
            self.api_root = base
        else:
            self.api_root = base + "/v1"
        self.api_key = api_key
        self.organization = organization

        # Keep user-provided headers, but avoid duplicating standard headers that the SDK manages.
        custom_headers = dict(headers or {})
        for h in ["Authorization", "Content-Type", "OpenAI-Organization"]:
            custom_headers.pop(h, None)
        self.extra_headers = custom_headers

        self.timeout = timeout_sec or 60.0

        # Initialize AsyncOpenAI client configured for OpenAI-compatible servers
        self.client = AsyncOpenAI(
            base_url=self.api_root,
            api_key=self.api_key,
            organization=self.organization,
            timeout=self.timeout,
        )

    async def _one_chat(
        self,
        payload: Dict[str, Any],
        retry_cfg: Dict[str, Any] | None = None,
    ) -> Tuple[Optional[str], Optional[Exception], int]:
        t0 = now_ms()
        attempts = int((retry_cfg or {}).get("max_attempts", 1))
        backoff = (retry_cfg or {}).get("backoff", {})
        delay_ms = int(backoff.get("initial_ms", 250))
        factor = float(backoff.get("factor", 2.0))

        # Remove internal keys not supported by the SDK API
        req = {k: v for k, v in payload.items() if k not in ("retry", "timeout_sec")}
        per_req_timeout = payload.get("timeout_sec", None)

        for i in range(max(1, attempts)):
            try:
                resp = await self.client.chat.completions.create(
                    **req,
                    # pass through any additional vendor-specific headers if needed (e.g., OpenRouter, etc.)
                    extra_headers=self.extra_headers if self.extra_headers else None,
                    timeout=per_req_timeout or self.timeout,
                )
                content = resp.choices[0].message.content
                return content, None, now_ms() - t0
            except Exception as e:
                # Try to classify retryable errors similar to original logic
                status = getattr(e, "status_code", None)
                retryable_status = {408, 409, 429, 500, 502, 503, 504}
                is_retryable = status in retryable_status

                # If status unknown, heuristic on error type/name for transient issues
                if not is_retryable and status is None:
                    name = e.__class__.__name__.lower()
                    msg = str(e).lower()
                    if any(
                        s in name for s in ["timeout", "rate", "connection", "server"]
                    ) or any(
                        s in msg
                        for s in [
                            "timeout",
                            "rate limit",
                            "temporarily",
                            "retry",
                            "connection",
                            "server error",
                        ]
                    ):
                        is_retryable = True

                if is_retryable and i < attempts - 1:
                    await asyncio.sleep(delay_ms / 1000.0)
                    delay_ms = int(delay_ms * factor)
                    continue

                return None, LLMError(str(e)), now_ms() - t0

        return None, LLMError("Retry attempts exhausted"), now_ms() - t0

    async def batched_chat(
        self,
        *,
        model: str,
        messages_list: List[
            List[Dict[str, Any]]
        ],  # content can be string or list (multimodal)
        request_params: Dict[str, Any],
        batch_size: int,
    ) -> Tuple[List[Optional[str]], List[int], int]:
        """Run many chats concurrently with bounded concurrency = batch_size.

        Supports multimodal messages where content is a list of text/image parts:
        [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]

        Returns (results, latencies_ms_per_task, error_count)
        """
        limit = asyncio.Semaphore(batch_size)

        tasks = []
        latencies: List[int] = []
        results: List[Optional[str]] = [None] * len(messages_list)
        errors = 0

        async def runner(idx: int, msgs: List[Dict[str, Any]]):
            nonlocal errors
            async with limit:
                retry_cfg = (request_params or {}).get("retry")
                payload = {"model": model, "messages": msgs, **(request_params or {})}
                out, err, latency = await self._one_chat(payload, retry_cfg)
                latencies.append(latency)
                if err:
                    errors += 1
                results[idx] = out

        for i, msgs in enumerate(messages_list):
            tasks.append(asyncio.create_task(runner(i, msgs)))
        await asyncio.gather(*tasks)

        return results, latencies, errors
