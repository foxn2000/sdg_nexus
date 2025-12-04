from __future__ import annotations
import asyncio
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI
from .utils import now_ms


class LLMError(RuntimeError):
    """LLM APIリクエスト中に発生したエラーを表すカスタム例外クラス。"""

    pass


class SharedHttpTransport:
    """
    共有HTTPトランスポート層を管理するシングルトンクラス。

    全てのLLMClientインスタンス間でHTTPコネクションプールを共有し、
    HTTP/2多重化を利用してオーバーヘッドを低減する。

    Attributes:
        DEFAULT_MAX_CONNECTIONS: デフォルトの最大接続数
        DEFAULT_MAX_KEEPALIVE_CONNECTIONS: デフォルトのKeep-Alive接続数
        DEFAULT_KEEPALIVE_EXPIRY: デフォルトのKeep-Alive有効期限（秒）
        DEFAULT_CONNECT_TIMEOUT: デフォルトの接続タイムアウト（秒）
        DEFAULT_READ_TIMEOUT: デフォルトの読み取りタイムアウト（秒）
        DEFAULT_WRITE_TIMEOUT: デフォルトの書き込みタイムアウト（秒）
        DEFAULT_POOL_TIMEOUT: デフォルトのプールタイムアウト（秒）
    """

    # デフォルト設定値
    DEFAULT_MAX_CONNECTIONS: ClassVar[int] = 100
    DEFAULT_MAX_KEEPALIVE_CONNECTIONS: ClassVar[int] = 50
    DEFAULT_KEEPALIVE_EXPIRY: ClassVar[float] = 30.0
    DEFAULT_CONNECT_TIMEOUT: ClassVar[float] = 10.0
    DEFAULT_READ_TIMEOUT: ClassVar[float] = 60.0
    DEFAULT_WRITE_TIMEOUT: ClassVar[float] = 30.0
    DEFAULT_POOL_TIMEOUT: ClassVar[float] = 10.0

    _instance: ClassVar[Optional["SharedHttpTransport"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(
        self,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections: int = DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry: float = DEFAULT_KEEPALIVE_EXPIRY,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        pool_timeout: float = DEFAULT_POOL_TIMEOUT,
        http2: bool = True,
    ):
        """
        SharedHttpTransportを初期化する。

        Args:
            max_connections: 最大接続数（デフォルト: 100）
            max_keepalive_connections: Keep-Alive接続の最大数（デフォルト: 50）
            keepalive_expiry: Keep-Alive接続の有効期限（秒、デフォルト: 30.0）
            connect_timeout: 接続タイムアウト（秒、デフォルト: 10.0）
            read_timeout: 読み取りタイムアウト（秒、デフォルト: 60.0）
            write_timeout: 書き込みタイムアウト（秒、デフォルト: 30.0）
            pool_timeout: プールからの接続取得タイムアウト（秒、デフォルト: 10.0）
            http2: HTTP/2を有効にするかどうか（デフォルト: True）
        """
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        self._keepalive_expiry = keepalive_expiry
        self._http2 = http2

        # httpxのLimits設定
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
        )

        # タイムアウト設定
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )

        # 遅延初期化のためNoneで初期化
        self._transport: Optional[httpx.AsyncHTTPTransport] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    @classmethod
    async def get_instance(
        cls,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections: int = DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry: float = DEFAULT_KEEPALIVE_EXPIRY,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        pool_timeout: float = DEFAULT_POOL_TIMEOUT,
        http2: bool = True,
    ) -> "SharedHttpTransport":
        """
        SharedHttpTransportのシングルトンインスタンスを取得する。

        Args:
            max_connections: 最大接続数
            max_keepalive_connections: Keep-Alive接続の最大数
            keepalive_expiry: Keep-Alive接続の有効期限（秒）
            connect_timeout: 接続タイムアウト（秒）
            read_timeout: 読み取りタイムアウト（秒）
            write_timeout: 書き込みタイムアウト（秒）
            pool_timeout: プールからの接続取得タイムアウト（秒）
            http2: HTTP/2を有効にするかどうか

        Returns:
            SharedHttpTransportのシングルトンインスタンス
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        max_connections=max_connections,
                        max_keepalive_connections=max_keepalive_connections,
                        keepalive_expiry=keepalive_expiry,
                        connect_timeout=connect_timeout,
                        read_timeout=read_timeout,
                        write_timeout=write_timeout,
                        pool_timeout=pool_timeout,
                        http2=http2,
                    )
        return cls._instance

    @classmethod
    def get_instance_sync(
        cls,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections: int = DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry: float = DEFAULT_KEEPALIVE_EXPIRY,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        pool_timeout: float = DEFAULT_POOL_TIMEOUT,
        http2: bool = True,
    ) -> "SharedHttpTransport":
        """
        SharedHttpTransportのシングルトンインスタンスを同期的に取得する。

        非同期コンテキスト外から呼び出す場合に使用。

        Args:
            max_connections: 最大接続数
            max_keepalive_connections: Keep-Alive接続の最大数
            keepalive_expiry: Keep-Alive接続の有効期限（秒）
            connect_timeout: 接続タイムアウト（秒）
            read_timeout: 読み取りタイムアウト（秒）
            write_timeout: 書き込みタイムアウト（秒）
            pool_timeout: プールからの接続取得タイムアウト（秒）
            http2: HTTP/2を有効にするかどうか

        Returns:
            SharedHttpTransportのシングルトンインスタンス
        """
        if cls._instance is None:
            cls._instance = cls(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                keepalive_expiry=keepalive_expiry,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                write_timeout=write_timeout,
                pool_timeout=pool_timeout,
                http2=http2,
            )
        return cls._instance

    def get_transport(self) -> httpx.AsyncHTTPTransport:
        """
        共有HTTPトランスポートを取得する。

        Returns:
            HTTP/2対応のAsyncHTTPTransportインスタンス
        """
        if self._transport is None:
            self._transport = httpx.AsyncHTTPTransport(
                http2=self._http2,
                limits=self._limits,
            )
        return self._transport

    def get_http_client(self) -> httpx.AsyncClient:
        """
        共有HTTPクライアントを取得する。

        Returns:
            設定済みのAsyncClientインスタンス
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                transport=self.get_transport(),
                timeout=self._timeout,
                http2=self._http2,
            )
        return self._http_client

    async def close(self) -> None:
        """
        HTTPクライアントとトランスポートをクローズする。

        アプリケーション終了時に呼び出すことを推奨。
        """
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._transport is not None:
            await self._transport.aclose()
            self._transport = None

    @classmethod
    async def close_instance(cls) -> None:
        """
        シングルトンインスタンスをクローズしてクリアする。
        """
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None

    @property
    def is_http2_enabled(self) -> bool:
        """HTTP/2が有効かどうかを返す。"""
        return self._http2

    @property
    def limits(self) -> httpx.Limits:
        """現在のLimits設定を返す。"""
        return self._limits

    @property
    def timeout(self) -> httpx.Timeout:
        """現在のTimeout設定を返す。"""
        return self._timeout


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
    """
    LLM APIクライアント。

    OpenAI互換のAPIサーバー（vLLM、SGLang、OpenAI等）と通信するための
    非同期HTTPクライアント。共有HTTPトランスポートを使用することで、
    コネクションプールを効率的に再利用し、HTTP/2多重化を活用できる。

    Attributes:
        api_root: APIのベースURL
        api_key: API認証キー
        organization: 組織ID（オプション）
        extra_headers: 追加のHTTPヘッダー
        timeout: リクエストタイムアウト（秒）
        use_shared_transport: 共有トランスポートを使用するかどうか

    Example:
        # 標準的な使用方法（AsyncOpenAI SDK使用）
        client = LLMClient(
            base_url="http://localhost:8000",
            api_key="your-api-key",
            organization=None,
            headers={},
        )

        # 共有HTTPトランスポートを使用（推奨：大規模並列処理時）
        client = LLMClient(
            base_url="http://localhost:8000",
            api_key="your-api-key",
            organization=None,
            headers={},
            use_shared_transport=True,
        )
    """

    # クラスレベルの共有トランスポート参照
    _shared_transport: ClassVar[Optional[SharedHttpTransport]] = None

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        organization: Optional[str],
        headers: Dict[str, str],
        timeout_sec: Optional[float] = None,
        use_shared_transport: bool = False,
        http2: bool = True,
    ):
        """
        LLMClientを初期化する。

        Args:
            base_url: APIのベースURL（例：http://localhost:8000）
            api_key: API認証キー
            organization: 組織ID（オプション）
            headers: 追加のHTTPヘッダー
            timeout_sec: リクエストタイムアウト（秒、デフォルト: 60.0）
            use_shared_transport: 共有HTTPトランスポートを使用するかどうか
                                  （デフォルト: False、後方互換性のため）
            http2: HTTP/2を有効にするかどうか（use_shared_transport=True時のみ有効）
        """
        base = (base_url or "https://api.openai.com").rstrip("/")
        if base.endswith("/v1"):
            self.api_root = base
        else:
            self.api_root = base + "/v1"
        self.api_key = api_key
        self.organization = organization
        self.use_shared_transport = use_shared_transport
        self._http2 = http2

        # Keep user-provided headers, but avoid duplicating standard headers that the SDK manages.
        custom_headers = dict(headers or {})
        for h in ["Authorization", "Content-Type", "OpenAI-Organization"]:
            custom_headers.pop(h, None)
        self.extra_headers = custom_headers

        self.timeout = timeout_sec or 60.0

        # 共有トランスポートの初期化または取得
        if use_shared_transport:
            self._init_shared_transport()

        # AsyncOpenAI クライアントを初期化
        # 共有トランスポート使用時はhttpxクライアントを注入
        if use_shared_transport and LLMClient._shared_transport is not None:
            # 共有httpxクライアントを使用するAsyncOpenAIクライアントを作成
            self.client = AsyncOpenAI(
                base_url=self.api_root,
                api_key=self.api_key,
                organization=self.organization,
                timeout=self.timeout,
                http_client=LLMClient._shared_transport.get_http_client(),
            )
        else:
            # 標準のAsyncOpenAIクライアント
            self.client = AsyncOpenAI(
                base_url=self.api_root,
                api_key=self.api_key,
                organization=self.organization,
                timeout=self.timeout,
            )

    def _init_shared_transport(self) -> None:
        """
        共有HTTPトランスポートを初期化または取得する。

        クラスレベルでシングルトンとして管理される。
        """
        if LLMClient._shared_transport is None:
            LLMClient._shared_transport = SharedHttpTransport.get_instance_sync(
                http2=self._http2,
                read_timeout=self.timeout,
            )

    @classmethod
    async def close_shared_transport(cls) -> None:
        """
        共有HTTPトランスポートをクローズする。

        アプリケーション終了時に呼び出すことを推奨。
        全てのLLMClientインスタンスで共有されているため、
        全てのリクエストが完了してから呼び出すこと。
        """
        if cls._shared_transport is not None:
            await cls._shared_transport.close()
            cls._shared_transport = None
        # SharedHttpTransportのシングルトンもクリア
        await SharedHttpTransport.close_instance()

    @classmethod
    def get_shared_transport(cls) -> Optional[SharedHttpTransport]:
        """
        共有HTTPトランスポートを取得する。

        Returns:
            共有トランスポートが存在する場合はそのインスタンス、
            存在しない場合はNone
        """
        return cls._shared_transport

    async def _one_chat(
        self,
        payload: Dict[str, Any],
        retry_cfg: Dict[str, Any] | None = None,
    ) -> Tuple[Optional[str], Optional[Exception], int]:
        t0 = now_ms()
        attempts = int((retry_cfg or {}).get("max_attempts", 1))
        backoff = (retry_cfg or {}).get("backoff", {})
        initial_delay_ms = int(backoff.get("initial_ms", 250))
        factor = float(backoff.get("factor", 2.0))
        # 空返答リトライ機能（デフォルト: 有効）
        retry_on_empty = (retry_cfg or {}).get("retry_on_empty", True)
        max_empty_retries = int((retry_cfg or {}).get("max_empty_retries", 3))

        # Remove internal keys not supported by the SDK API
        req = {k: v for k, v in payload.items() if k not in ("retry", "timeout_sec")}
        per_req_timeout = payload.get("timeout_sec", None)

        # 空返答リトライ用の外側ループ
        # 空返答リトライはエラーリトライとは独立して処理
        for empty_retry_count in range(
            max(1, max_empty_retries) if retry_on_empty else 1
        ):
            delay_ms = initial_delay_ms

            # エラーリトライ用の内側ループ
            for i in range(max(1, attempts)):
                try:
                    resp = await self.client.chat.completions.create(
                        **req,
                        # pass through any additional vendor-specific headers if needed (e.g., OpenRouter, etc.)
                        extra_headers=(
                            self.extra_headers if self.extra_headers else None
                        ),
                        timeout=per_req_timeout or self.timeout,
                    )
                    content = resp.choices[0].message.content

                    # 空返答チェック: contentがNone、空文字列、またはwhitespaceのみの場合
                    if retry_on_empty and (content is None or not content.strip()):
                        # まだ空返答リトライ回数が残っている場合はリトライ
                        if empty_retry_count < max_empty_retries - 1:
                            await asyncio.sleep(delay_ms / 1000.0)
                            delay_ms = int(delay_ms * factor)
                            break  # 内側ループを抜けて外側ループで再試行
                        # max_empty_retriesを超えた場合はそのまま返す（エラーにはしない）
                        return content, None, now_ms() - t0

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
                            s in name
                            for s in ["timeout", "rate", "connection", "server"]
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
            else:
                # 内側ループが正常に完了した場合（breakで抜けなかった場合）
                # これはエラーリトライが尽きた場合
                continue
            # breakで抜けた場合（空返答リトライ）は外側ループを続行

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
