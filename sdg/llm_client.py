from __future__ import annotations
import asyncio, json, os, random
from typing import Any, Dict, List, Optional, Tuple
import httpx
from .utils import now_ms

class LLMError(RuntimeError):
    pass

class BatchOptimizer:
    """Simple adaptive concurrency controller based on latency and error rate."""
    def __init__(self, min_batch=1, max_batch=8, target_latency_ms=3000):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_latency_ms = target_latency_ms
        self._current = min_batch

    def current(self) -> int:
        return self._current

    def update(self, latencies_ms: List[int], errors: int):
        if not latencies_ms:
            return
        avg = sum(latencies_ms) / len(latencies_ms)
        if errors > 0 or avg > self.target_latency_ms:
            self._current = max(self.min_batch, self._current - 1)
        else:
            self._current = min(self.max_batch, self._current + 1)

class LLMClient:
    def __init__(self, *, base_url: str, api_key: str, organization: Optional[str], headers: Dict[str,str], timeout_sec: Optional[float] = None):
        base = (base_url or 'https://api.openai.com').rstrip('/')
        if base.endswith('/v1'):
            self.api_root = base
        else:
            self.api_root = base + '/v1'
        self.api_key = api_key
        self.organization = organization
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **(headers or {})
        }
        if organization:
            self.headers["OpenAI-Organization"] = organization
        self.timeout = timeout_sec or 60.0

    async def _one_chat(self, client: httpx.AsyncClient, payload: Dict[str, Any], retry_cfg: Dict[str, Any]|None=None) -> Tuple[Optional[str], Optional[Exception], int]:
        t0 = now_ms()
        try:
            attempts = int((retry_cfg or {}).get('max_attempts', 1))
            backoff = (retry_cfg or {}).get('backoff', {})
            delay_ms = int(backoff.get('initial_ms', 250))
            factor = float(backoff.get('factor', 2.0))
            for i in range(max(1, attempts)):
                r = await client.post(f"{self.api_root}/chat/completions", json={k:v for k,v in payload.items() if k!='retry'}, timeout=self.timeout)
                if r.status_code < 400:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    return content, None, now_ms()-t0
                # retryable?
                if r.status_code in (408, 409, 429, 500, 502, 503, 504) and i < attempts-1:
                    await asyncio.sleep(delay_ms/1000.0)
                    delay_ms = int(delay_ms * factor)
                    continue
                return None, LLMError(f"HTTP {r.status_code}: {r.text[:500]}"), now_ms()-t0
            return None, LLMError("Retry attempts exhausted"), now_ms()-t0
        except Exception as e:
            return None, e, now_ms()-t0

    async def batched_chat(self, *, model: str, messages_list: List[List[Dict[str,str]]], request_params: Dict[str,Any], batch_size: int) -> Tuple[List[Optional[str]], List[int], int]:
        """Run many chats concurrently with bounded concurrency = batch_size.
        Returns (results, latencies_ms_per_task, error_count)
        """
        limit = asyncio.Semaphore(batch_size)

        async with httpx.AsyncClient() as client:
            tasks = []
            latencies: List[int] = []
            results: List[Optional[str]] = [None]*len(messages_list)
            errors = 0

            async def runner(idx: int, msgs: List[Dict[str,str]]):
                nonlocal errors
                async with limit:
                    retry_cfg = (request_params or {}).get('retry')
                    payload = {"model": model, "messages": msgs, **request_params}
                    out, err, latency = await self._one_chat(client, payload, retry_cfg)
                    latencies.append(latency)
                    if err:
                        errors += 1
                    results[idx] = out

            for i, msgs in enumerate(messages_list):
                tasks.append(asyncio.create_task(runner(i, msgs)))
            await asyncio.gather(*tasks)

        return results, latencies, errors
