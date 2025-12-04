from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, List, Optional

from ..config import (
    SDGConfig,
    AIBlock,
    LogicBlock,
    PyBlock,
    EndBlock,
    OutputDef,
)
from ..llm_client import BatchOptimizer, LLMClient
from ..utils import render_template, has_image_placeholders

from .core import (
    ExecutionContext,
    _eval_cond,
    _execute_end_block_single,
    _apply_outputs,
    StreamingResult,
)
from .logic import _apply_logic_block
from .python import _load_python_function, _execute_python_block_single
from .ai import _execute_ai_block_single, _build_clients, _build_multimodal_content
from .scheduling import (
    HierarchicalTaskScheduler,
    StreamingContextManager,
    BatchProgressiveRelease,
    SchedulerConfig,
    MemoryConfig,
)

# Import adaptive components (optional dependency)
try:
    from ..adaptive import (
        AdaptiveController,
        MetricsCollector,
        MetricsType,
        AdaptiveRequestBatcher,
    )

    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


async def process_single_row(
    row_index: int,
    initial_context: Dict[str, Any],
    cfg: SDGConfig,
    clients: Dict[str, Any],
    exec_ctx: ExecutionContext,
    *,
    save_intermediate: bool = False,
    python_functions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    1つのデータ行に対して全ブロックを順次実行し、最終結果を返す。

    Args:
        row_index: 行インデックス（デバッグ用）
        initial_context: 初期コンテキスト（入力データの1行分）
        cfg: SDG設定
        clients: LLMクライアント辞書
        exec_ctx: 実行コンテキスト（行ごとに独立したコピーを渡すこと）
        save_intermediate: 中間結果を保存するか
        python_functions: プリロードされたPython関数（キャッシュ用）

    Returns:
        処理結果の辞書
    """
    result: Dict[str, Any] = {}
    ctx = dict(initial_context)

    # Python関数キャッシュ
    py_funcs = python_functions or {}

    for block in cfg.blocks:
        # run_if評価
        run_ok = True
        if block.run_if:
            run_ok = _eval_cond(ctx, block.run_if, exec_ctx)

        if not run_ok:
            continue

        try:
            if isinstance(block, AIBlock):
                out_map = await _execute_ai_block_single(
                    block, ctx, cfg, clients, exec_ctx
                )
                if save_intermediate:
                    result.update({f"_{block.exec}_{k}": v for k, v in out_map.items()})
                ctx.update(out_map)

            elif isinstance(block, LogicBlock):
                out_map = _apply_logic_block(block, ctx, exec_ctx)
                if save_intermediate:
                    result.update({f"_{block.exec}_{k}": v for k, v in out_map.items()})
                ctx.update(out_map)

            elif isinstance(block, PyBlock):
                # 関数をキャッシュから取得またはロード
                fn_key = f"{block.exec}_{block.function or block.entrypoint}"
                if fn_key not in py_funcs:
                    py_funcs[fn_key] = _load_python_function(block)
                fn = py_funcs[fn_key]

                out_map = _execute_python_block_single(block, ctx, cfg, exec_ctx, fn)
                if save_intermediate:
                    result.update({f"_{block.exec}_{k}": v for k, v in out_map.items()})
                ctx.update(out_map)

            elif isinstance(block, EndBlock):
                out_map = _execute_end_block_single(block, ctx, exec_ctx)
                result.update(out_map)

            else:
                raise ValueError(f"Unknown block class: {type(block)}")

        except Exception as e:
            if block.on_error != "continue":
                raise
            # continue on error
            ctx[f"error_block_{block.exec}"] = str(e)

    return result


async def run_pipeline_streaming(
    cfg: SDGConfig,
    dataset: List[Dict[str, Any]],
    *,
    max_concurrent: int = 8,
    save_intermediate: bool = False,
    # Phase 2: スケジューリングオプション（デフォルト無効）
    max_pending_tasks: int = 1000,
    chunk_size: int = 100,
    enable_scheduling: bool = False,
    # Phase 2: メモリ最適化オプション（デフォルト無効）
    max_cache_size: int = 500,
    enable_memory_optimization: bool = False,
    enable_memory_monitoring: bool = False,
):
    """
    ストリーミング版パイプライン - 完了した行から順次yield

    Args:
        cfg: SDG設定
        dataset: 入力データセット
        max_concurrent: 同時処理行数の上限
        save_intermediate: 中間結果を保存するか
        max_pending_tasks: 最大保留タスク数（スケジューリング有効時）
        chunk_size: データセット分割サイズ（スケジューリング有効時）
        enable_scheduling: 階層的タスクスケジューリングを有効化
        max_cache_size: コンテキストキャッシュの最大サイズ
        enable_memory_optimization: メモリ最適化を有効化
        enable_memory_monitoring: メモリ使用状況監視を有効化

    Yields:
        StreamingResult: 各行の処理結果
    """
    # モデルクライアント構築
    clients = _build_clients(cfg)

    # Python関数をプリロード
    python_functions: Dict[str, Any] = {}
    for block in cfg.blocks:
        if isinstance(block, PyBlock):
            fn_key = f"{block.exec}_{block.function or block.entrypoint}"
            python_functions[fn_key] = _load_python_function(block)

    # 同時実行数制御用セマフォ
    semaphore = asyncio.Semaphore(max_concurrent)

    # 結果キュー
    result_queue: asyncio.Queue[StreamingResult] = asyncio.Queue()

    # 処理完了カウンター
    completed = 0
    total = len(dataset)

    # Phase 2: 階層的タスクスケジューラの初期化
    scheduler_config = SchedulerConfig(
        max_pending_tasks=max_pending_tasks,
        chunk_size=chunk_size,
        enable_scheduling=enable_scheduling,
    )
    scheduler = HierarchicalTaskScheduler(config=scheduler_config)

    # Phase 2: ストリーミングコンテキストマネージャの初期化
    memory_config = MemoryConfig(
        max_cache_size=max_cache_size,
        enable_memory_optimization=enable_memory_optimization,
        enable_monitoring=enable_memory_monitoring,
    )
    ctx_manager = StreamingContextManager(config=memory_config)

    async def process_row(row_index: int, row_data: Dict[str, Any]):
        """1行を処理してキューに結果を入れる"""
        async with semaphore:
            # 行ごとに独立したExecutionContextを作成
            row_exec_ctx = ExecutionContext(cfg)

            try:
                result_data = await process_single_row(
                    row_index=row_index,
                    initial_context=row_data,
                    cfg=cfg,
                    clients=clients,
                    exec_ctx=row_exec_ctx,
                    save_intermediate=save_intermediate,
                    python_functions=python_functions,
                )
                await result_queue.put(
                    StreamingResult(
                        row_index=row_index,
                        data=result_data,
                        error=None,
                    )
                )
            except Exception as e:
                await result_queue.put(
                    StreamingResult(
                        row_index=row_index,
                        data={},
                        error=e,
                    )
                )
            finally:
                # Phase 2: スケジューラに完了を通知
                if scheduler.is_enabled:
                    await scheduler.mark_task_completed()
                # Phase 2: コンテキストマネージャに完了を通知してメモリ解放
                if ctx_manager.is_enabled:
                    await ctx_manager.mark_completed(row_index)

    try:
        if enable_scheduling:
            # Phase 2: 階層的スケジューリングでタスクを段階的に起動
            tasks = []
            async for item in scheduler.schedule(dataset):
                task = asyncio.create_task(process_row(item.index, item.data))
                tasks.append(task)

            # 完了した結果を順次yield
            while completed < total:
                result = await result_queue.get()
                completed += 1
                yield result

            # すべてのタスク完了を待機
            await asyncio.gather(*tasks)
        else:
            # 従来の動作: 全行のタスクを一度に起動
            tasks = [
                asyncio.create_task(process_row(i, row))
                for i, row in enumerate(dataset)
            ]

            # 完了した結果を順次yield
            while completed < total:
                result = await result_queue.get()
                completed += 1
                yield result

            # すべてのタスク完了を待機（エラー発生時の例外を伝播）
            await asyncio.gather(*tasks)

    finally:
        # Phase 2: リソース解放
        if ctx_manager.is_enabled:
            await ctx_manager.release_all()


async def run_pipeline_streaming_adaptive(
    cfg: SDGConfig,
    dataset: List[Dict[str, Any]],
    *,
    max_concurrent: int = 64,
    min_concurrent: int = 1,
    target_latency_ms: int = 3000,
    target_queue_depth: int = 32,
    metrics_type: str = "none",
    save_intermediate: bool = False,
    # Phase 2: スケジューリングオプション（デフォルト無効）
    max_pending_tasks: int = 1000,
    chunk_size: int = 100,
    enable_scheduling: bool = False,
    # Phase 2: メモリ最適化オプション（デフォルト無効）
    max_cache_size: int = 500,
    enable_memory_optimization: bool = False,
    enable_memory_monitoring: bool = False,
):
    """
    適応的並行性制御付きストリーミング版パイプライン

    レイテンシとオプションのバックエンドメトリクス（vLLM/SGLang）に基づいて、
    並行処理数を動的に調整しながら処理を行う。

    Args:
        cfg: SDG設定
        dataset: 入力データセット
        max_concurrent: 同時処理行数の上限 (デフォルト: 64)
        min_concurrent: 同時処理行数の下限 (デフォルト: 1)
        target_latency_ms: 目標P95レイテンシ (ミリ秒、デフォルト: 3000)
        target_queue_depth: 目標バックエンドキュー深度 (デフォルト: 32)
        metrics_type: メトリクスタイプ ("none", "vllm", "sglang")
        save_intermediate: 中間結果を保存するか
        max_pending_tasks: 最大保留タスク数（スケジューリング有効時）
        chunk_size: データセット分割サイズ（スケジューリング有効時）
        enable_scheduling: 階層的タスクスケジューリングを有効化
        max_cache_size: コンテキストキャッシュの最大サイズ
        enable_memory_optimization: メモリ最適化を有効化
        enable_memory_monitoring: メモリ使用状況監視を有効化

    Yields:
        StreamingResult: 各行の処理結果
    """
    if not ADAPTIVE_AVAILABLE:
        # Fall back to standard streaming if adaptive module not available
        async for result in run_pipeline_streaming(
            cfg,
            dataset,
            max_concurrent=max_concurrent,
            save_intermediate=save_intermediate,
            max_pending_tasks=max_pending_tasks,
            chunk_size=chunk_size,
            enable_scheduling=enable_scheduling,
            max_cache_size=max_cache_size,
            enable_memory_optimization=enable_memory_optimization,
            enable_memory_monitoring=enable_memory_monitoring,
        ):
            yield result
        return

    # モデルクライアント構築
    clients = _build_clients(cfg)

    # Python関数をプリロード
    python_functions: Dict[str, Any] = {}
    for block in cfg.blocks:
        if isinstance(block, PyBlock):
            fn_key = f"{block.exec}_{block.function or block.entrypoint}"
            python_functions[fn_key] = _load_python_function(block)

    # AdaptiveController設定
    controller = AdaptiveController(
        min_concurrency=min_concurrent,
        max_concurrency=max_concurrent,
        target_latency_ms=float(target_latency_ms),
        target_queue_depth=target_queue_depth,
    )

    # MetricsCollector設定（メトリクスタイプに応じて）
    metrics_collector: Optional[MetricsCollector] = None
    if metrics_type != "none":
        # モデル設定から最初のbase_urlを取得
        base_url = None
        for model in cfg.models:
            if model.base_url:
                base_url = model.base_url
                break

        if base_url:
            if metrics_type == "vllm":
                metrics_collector = MetricsCollector(
                    base_url=base_url,
                    metrics_type=MetricsType.VLLM,
                )
            elif metrics_type == "sglang":
                metrics_collector = MetricsCollector(
                    base_url=base_url,
                    metrics_type=MetricsType.SGLANG,
                )

    # 結果キュー
    result_queue: asyncio.Queue[StreamingResult] = asyncio.Queue()

    # 処理完了カウンター
    completed = 0
    total = len(dataset)

    # メトリクス更新タスク
    metrics_update_task: Optional[asyncio.Task] = None

    # Phase 2: 階層的タスクスケジューラの初期化
    scheduler_config = SchedulerConfig(
        max_pending_tasks=max_pending_tasks,
        chunk_size=chunk_size,
        enable_scheduling=enable_scheduling,
    )
    scheduler = HierarchicalTaskScheduler(config=scheduler_config)

    # Phase 2: ストリーミングコンテキストマネージャの初期化
    memory_config = MemoryConfig(
        max_cache_size=max_cache_size,
        enable_memory_optimization=enable_memory_optimization,
        enable_monitoring=enable_memory_monitoring,
    )
    ctx_manager = StreamingContextManager(config=memory_config)

    async def update_metrics_loop():
        """バックエンドメトリクスを定期的にコントローラーに反映"""
        if metrics_collector is None:
            return

        await metrics_collector.start()
        try:
            while True:
                await asyncio.sleep(0.5)
                metrics = metrics_collector.get_latest()
                if metrics and metrics.is_valid:
                    controller.update_with_metrics(metrics)
        except asyncio.CancelledError:
            pass
        finally:
            await metrics_collector.stop()

    async def process_row(row_index: int, row_data: Dict[str, Any]):
        """1行を処理してキューに結果を入れる（適応的並行性制御付き）"""
        # 動的セマフォで制御
        async with controller.semaphore:
            start_time = time.time()

            # 行ごとに独立したExecutionContextを作成
            row_exec_ctx = ExecutionContext(cfg)

            try:
                result_data = await process_single_row(
                    row_index=row_index,
                    initial_context=row_data,
                    cfg=cfg,
                    clients=clients,
                    exec_ctx=row_exec_ctx,
                    save_intermediate=save_intermediate,
                    python_functions=python_functions,
                )

                # レイテンシを記録
                latency_ms = (time.time() - start_time) * 1000
                controller.record_latency(latency_ms, is_error=False)

                await result_queue.put(
                    StreamingResult(
                        row_index=row_index,
                        data=result_data,
                        error=None,
                    )
                )
            except Exception as e:
                # エラーもレイテンシとして記録
                latency_ms = (time.time() - start_time) * 1000
                controller.record_latency(latency_ms, is_error=True)

                await result_queue.put(
                    StreamingResult(
                        row_index=row_index,
                        data={},
                        error=e,
                    )
                )
            finally:
                # Phase 2: スケジューラに完了を通知
                if scheduler.is_enabled:
                    await scheduler.mark_task_completed()
                # Phase 2: コンテキストマネージャに完了を通知してメモリ解放
                if ctx_manager.is_enabled:
                    await ctx_manager.mark_completed(row_index)

    try:
        # メトリクス収集を開始（有効な場合）
        if metrics_collector is not None:
            metrics_update_task = asyncio.create_task(update_metrics_loop())

        if enable_scheduling:
            # Phase 2: 階層的スケジューリングでタスクを段階的に起動
            tasks = []
            async for item in scheduler.schedule(dataset):
                task = asyncio.create_task(process_row(item.index, item.data))
                tasks.append(task)

            # 完了した結果を順次yield
            while completed < total:
                result = await result_queue.get()
                completed += 1
                yield result

            # すべてのタスク完了を待機
            await asyncio.gather(*tasks)
        else:
            # 従来の動作: 全行のタスクを一度に起動
            tasks = [
                asyncio.create_task(process_row(i, row))
                for i, row in enumerate(dataset)
            ]

            # 完了した結果を順次yield
            while completed < total:
                result = await result_queue.get()
                completed += 1
                yield result

            # すべてのタスク完了を待機
            await asyncio.gather(*tasks)

    finally:
        # Phase 2: リソース解放
        if ctx_manager.is_enabled:
            await ctx_manager.release_all()

        # メトリクス収集を停止
        if metrics_update_task is not None:
            metrics_update_task.cancel()
            try:
                await metrics_update_task
            except asyncio.CancelledError:
                pass


async def run_pipeline_streaming_adaptive_batched(
    cfg: SDGConfig,
    dataset: List[Dict[str, Any]],
    *,
    max_concurrent: int = 64,
    min_concurrent: int = 1,
    target_latency_ms: int = 3000,
    target_queue_depth: int = 32,
    metrics_type: str = "none",
    max_batch_size: int = 32,
    max_wait_ms: int = 50,
    save_intermediate: bool = False,
    # Phase 2: スケジューリングオプション（デフォルト無効）
    max_pending_tasks: int = 1000,
    chunk_size: int = 100,
    enable_scheduling: bool = False,
    # Phase 2: メモリ最適化オプション（デフォルト無効）
    max_cache_size: int = 500,
    enable_memory_optimization: bool = False,
    enable_memory_monitoring: bool = False,
):
    """
    適応的並行性制御付きストリーミング版パイプライン（バッチング有効）

    レイテンシとオプションのバックエンドメトリクス（vLLM/SGLang）に基づいて、
    並行処理数を動的に調整し、リクエストをバッチングして処理を行う。

    Args:
        cfg: SDG設定
        dataset: 入力データセット
        max_concurrent: 同時処理行数の上限 (デフォルト: 64)
        min_concurrent: 同時処理行数の下限 (デフォルト: 1)
        target_latency_ms: 目標P95レイテンシ (ミリ秒、デフォルト: 3000)
        target_queue_depth: 目標バックエンドキュー深度 (デフォルト: 32)
        metrics_type: メトリクスタイプ ("none", "vllm", "sglang")
        max_batch_size: 最大バッチサイズ (デフォルト: 32)
        max_wait_ms: バッチ形成の最大待機時間 (ミリ秒、デフォルト: 50)
        save_intermediate: 中間結果を保存するか
        max_pending_tasks: 最大保留タスク数（スケジューリング有効時）
        chunk_size: データセット分割サイズ（スケジューリング有効時）
        enable_scheduling: 階層的タスクスケジューリングを有効化
        max_cache_size: コンテキストキャッシュの最大サイズ
        enable_memory_optimization: メモリ最適化を有効化
        enable_memory_monitoring: メモリ使用状況監視を有効化

    Yields:
        StreamingResult: 各行の処理結果
    """
    if not ADAPTIVE_AVAILABLE:
        # Fall back to standard adaptive streaming if adaptive module not available
        async for result in run_pipeline_streaming_adaptive(
            cfg,
            dataset,
            max_concurrent=max_concurrent,
            min_concurrent=min_concurrent,
            target_latency_ms=target_latency_ms,
            target_queue_depth=target_queue_depth,
            metrics_type=metrics_type,
            save_intermediate=save_intermediate,
            max_pending_tasks=max_pending_tasks,
            chunk_size=chunk_size,
            enable_scheduling=enable_scheduling,
            max_cache_size=max_cache_size,
            enable_memory_optimization=enable_memory_optimization,
            enable_memory_monitoring=enable_memory_monitoring,
        ):
            yield result
        return

    # モデルクライアント構築
    clients = _build_clients(cfg)

    # Python関数をプリロード
    python_functions: Dict[str, Any] = {}
    for block in cfg.blocks:
        if isinstance(block, PyBlock):
            fn_key = f"{block.exec}_{block.function or block.entrypoint}"
            python_functions[fn_key] = _load_python_function(block)

    # AdaptiveController設定
    controller = AdaptiveController(
        min_concurrency=min_concurrent,
        max_concurrency=max_concurrent,
        target_latency_ms=float(target_latency_ms),
        target_queue_depth=target_queue_depth,
    )

    # MetricsCollector設定（メトリクスタイプに応じて）
    metrics_collector: Optional[MetricsCollector] = None
    base_url = None
    if metrics_type != "none":
        for model in cfg.models:
            if model.base_url:
                base_url = model.base_url
                break

        if base_url:
            if metrics_type == "vllm":
                metrics_collector = MetricsCollector(
                    base_url=base_url,
                    metrics_type=MetricsType.VLLM,
                )
            elif metrics_type == "sglang":
                metrics_collector = MetricsCollector(
                    base_url=base_url,
                    metrics_type=MetricsType.SGLANG,
                )

    # モデルごとにバッチャーを作成
    batchers: Dict[str, AdaptiveRequestBatcher] = {}

    # Phase 2: 階層的タスクスケジューラの初期化
    scheduler_config = SchedulerConfig(
        max_pending_tasks=max_pending_tasks,
        chunk_size=chunk_size,
        enable_scheduling=enable_scheduling,
    )
    scheduler = HierarchicalTaskScheduler(config=scheduler_config)

    # Phase 2: ストリーミングコンテキストマネージャの初期化
    memory_config = MemoryConfig(
        max_cache_size=max_cache_size,
        enable_memory_optimization=enable_memory_optimization,
        enable_monitoring=enable_memory_monitoring,
    )
    ctx_manager = StreamingContextManager(config=memory_config)

    async def create_batch_processor(
        client: LLMClient, model_api_name: str, req_params: Dict[str, Any]
    ):
        """バッチプロセッサを作成"""

        async def batch_processor(
            payloads: List[Dict[str, Any]],
        ) -> List[Optional[str]]:
            messages_list = [p["messages"] for p in payloads]
            results, latencies, errors = await client.batched_chat(
                model=model_api_name,
                messages_list=messages_list,
                request_params=req_params,
                batch_size=len(messages_list),  # Already batched
            )
            return results

        return batch_processor

    # 結果キュー
    result_queue: asyncio.Queue[StreamingResult] = asyncio.Queue()

    # 処理完了カウンター
    completed = 0
    total = len(dataset)

    # メトリクス更新タスク
    metrics_update_task: Optional[asyncio.Task] = None

    async def update_metrics_loop():
        """バックエンドメトリクスを定期的にコントローラーに反映"""
        if metrics_collector is None:
            return

        await metrics_collector.start()
        try:
            while True:
                await asyncio.sleep(0.5)
                metrics = metrics_collector.get_latest()
                if metrics and metrics.is_valid:
                    controller.update_with_metrics(metrics)
        except asyncio.CancelledError:
            pass
        finally:
            await metrics_collector.stop()

    async def process_row_batched(row_index: int, row_data: Dict[str, Any]):
        """1行を処理してキューに結果を入れる（バッチング対応）"""
        async with controller.semaphore:
            start_time = time.time()
            row_exec_ctx = ExecutionContext(cfg)

            try:
                result: Dict[str, Any] = {}
                ctx = dict(row_data)

                for block in cfg.blocks:
                    run_ok = True
                    if block.run_if:
                        run_ok = _eval_cond(ctx, block.run_if, row_exec_ctx)

                    if not run_ok:
                        continue

                    try:
                        if isinstance(block, AIBlock):
                            # バッチャー経由でAIブロックを実行
                            model_key = block.model
                            if model_key not in batchers:
                                client = clients[model_key]
                                model_def = cfg.model_by_name(model_key)
                                req_params = dict((model_def.request_defaults or {}))
                                req_params.update(block.params or {})

                                if block.mode == "json":
                                    req_params["response_format"] = {
                                        "type": "json_object"
                                    }

                                processor = await create_batch_processor(
                                    client, model_def.api_model, req_params
                                )
                                batchers[model_key] = AdaptiveRequestBatcher(
                                    batch_processor=processor,
                                    controller=controller,
                                    max_batch_size=max_batch_size,
                                    min_batch_size=1,
                                    max_wait_ms=max_wait_ms,
                                    enabled=True,
                                )
                                await batchers[model_key].start()

                            # メッセージ構築
                            msgs = []
                            if block.system_prompt:
                                msgs.append(
                                    {
                                        "role": "system",
                                        "content": render_template(
                                            block.system_prompt, ctx
                                        ),
                                    }
                                )

                            raw_user_content = "\n\n".join(
                                [render_template(p, ctx) for p in (block.prompts or [])]
                            )

                            if has_image_placeholders(raw_user_content):
                                multimodal_content = _build_multimodal_content(
                                    raw_user_content, ctx, cfg, None
                                )
                                msgs.append(
                                    {"role": "user", "content": multimodal_content}
                                )
                            else:
                                msgs.append(
                                    {"role": "user", "content": raw_user_content}
                                )

                            # バッチャー経由で送信
                            text = await batchers[model_key].submit({"messages": msgs})
                            text = text or ""

                            out_map = _apply_outputs(
                                text,
                                block.outputs
                                or [OutputDef(name="full", select="full")],
                            )

                            if block.save_to and "vars" in block.save_to:
                                for var_name, out_name in block.save_to["vars"].items():
                                    if out_name in out_map:
                                        row_exec_ctx.set_global(
                                            var_name, out_map[out_name]
                                        )

                            if save_intermediate:
                                result.update(
                                    {
                                        f"_{block.exec}_{k}": v
                                        for k, v in out_map.items()
                                    }
                                )
                            ctx.update(out_map)

                        elif isinstance(block, LogicBlock):
                            out_map = _apply_logic_block(block, ctx, row_exec_ctx)
                            if save_intermediate:
                                result.update(
                                    {
                                        f"_{block.exec}_{k}": v
                                        for k, v in out_map.items()
                                    }
                                )
                            ctx.update(out_map)

                        elif isinstance(block, PyBlock):
                            fn_key = (
                                f"{block.exec}_{block.function or block.entrypoint}"
                            )
                            if fn_key not in python_functions:
                                python_functions[fn_key] = _load_python_function(block)
                            fn = python_functions[fn_key]

                            out_map = _execute_python_block_single(
                                block, ctx, cfg, row_exec_ctx, fn
                            )
                            if save_intermediate:
                                result.update(
                                    {
                                        f"_{block.exec}_{k}": v
                                        for k, v in out_map.items()
                                    }
                                )
                            ctx.update(out_map)

                        elif isinstance(block, EndBlock):
                            out_map = _execute_end_block_single(
                                block, ctx, row_exec_ctx
                            )
                            result.update(out_map)

                        else:
                            raise ValueError(f"Unknown block class: {type(block)}")

                    except Exception as e:
                        if block.on_error != "continue":
                            raise
                        ctx[f"error_block_{block.exec}"] = str(e)

                latency_ms = (time.time() - start_time) * 1000
                controller.record_latency(latency_ms, is_error=False)

                await result_queue.put(
                    StreamingResult(
                        row_index=row_index,
                        data=result,
                        error=None,
                    )
                )
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                controller.record_latency(latency_ms, is_error=True)

                await result_queue.put(
                    StreamingResult(
                        row_index=row_index,
                        data={},
                        error=e,
                    )
                )
            finally:
                # Phase 2: スケジューラに完了を通知
                if scheduler.is_enabled:
                    await scheduler.mark_task_completed()
                # Phase 2: コンテキストマネージャに完了を通知してメモリ解放
                if ctx_manager.is_enabled:
                    await ctx_manager.mark_completed(row_index)

    try:
        # メトリクス収集を開始
        if metrics_collector is not None:
            metrics_update_task = asyncio.create_task(update_metrics_loop())

        if enable_scheduling:
            # Phase 2: 階層的スケジューリングでタスクを段階的に起動
            tasks = []
            async for item in scheduler.schedule(dataset):
                task = asyncio.create_task(process_row_batched(item.index, item.data))
                tasks.append(task)

            # 完了した結果を順次yield
            while completed < total:
                result = await result_queue.get()
                completed += 1
                yield result

            # すべてのタスク完了を待機
            await asyncio.gather(*tasks)
        else:
            # 従来の動作: 全行のタスクを一度に起動
            tasks = [
                asyncio.create_task(process_row_batched(i, row))
                for i, row in enumerate(dataset)
            ]

            # 完了した結果を順次yield
            while completed < total:
                result = await result_queue.get()
                completed += 1
                yield result

            # すべてのタスク完了を待機
            await asyncio.gather(*tasks)

    finally:
        # Phase 2: リソース解放
        if ctx_manager.is_enabled:
            await ctx_manager.release_all()

        # バッチャーを停止
        for batcher in batchers.values():
            await batcher.stop()

        # メトリクス収集を停止
        if metrics_update_task is not None:
            metrics_update_task.cancel()
            try:
                await metrics_update_task
            except asyncio.CancelledError:
                pass


async def run_pipeline(
    cfg: SDGConfig,
    dataset: List[Dict[str, Any]],
    *,
    max_batch: int = 8,
    min_batch: int = 1,
    target_latency_ms: int = 3000,
    save_intermediate: bool = False,
    # Phase 2: メモリ最適化オプション（デフォルト無効）
    enable_memory_optimization: bool = False,
    enable_memory_monitoring: bool = False,
    gc_interval: int = 100,
    memory_threshold_mb: int = 1024,
) -> List[Dict[str, Any]]:
    """
    パイプライン実行（従来のブロック単位一括処理 - 後方互換性のため維持）

    Args:
        cfg: SDG設定
        dataset: 入力データセット
        max_batch: 最大バッチサイズ
        min_batch: 最小バッチサイズ
        target_latency_ms: 目標レイテンシ（ミリ秒）
        save_intermediate: 中間結果を保存するか
        enable_memory_optimization: メモリ最適化を有効化（デフォルト: False）
        enable_memory_monitoring: メモリ使用状況監視を有効化（デフォルト: False）
        gc_interval: ガベージコレクション実行間隔（処理行数）
        memory_threshold_mb: メモリ使用量警告閾値（MB）

    Returns:
        処理結果のリスト
    """

    # 実行コンテキスト
    exec_ctx = ExecutionContext(cfg)

    # モデルクライアント構築
    clients = _build_clients(cfg)

    results: List[Dict[str, Any]] = [{} for _ in dataset]
    contexts: List[Dict[str, Any]] = [dict(rec) for rec in dataset]

    optimizer = BatchOptimizer(
        min_batch=min_batch, max_batch=max_batch, target_latency_ms=target_latency_ms
    )

    # Phase 2: バッチ処理用段階的メモリ解放
    memory_config = MemoryConfig(
        enable_memory_optimization=enable_memory_optimization,
        enable_monitoring=enable_memory_monitoring,
        gc_interval=gc_interval,
        memory_threshold_mb=memory_threshold_mb,
    )
    progressive_release = BatchProgressiveRelease(
        config=memory_config,
        total_size=len(dataset),
    )

    for block in cfg.blocks:
        # run_if評価
        run_flags = []
        for ctx in contexts:
            ok = True
            if block.run_if:
                ok = _eval_cond(ctx, block.run_if, exec_ctx)
            run_flags.append(ok)

        try:
            if isinstance(block, AIBlock):
                # メッセージ構築
                messages_list = []
                rec_indices = []
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue
                    msgs = []
                    if block.system_prompt:
                        msgs.append(
                            {
                                "role": "system",
                                "content": render_template(block.system_prompt, ctx),
                            }
                        )

                    # プロンプト内に画像があるかチェック
                    raw_user_content = "\n\n".join(
                        [render_template(p, ctx) for p in (block.prompts or [])]
                    )

                    # 画像プレースホルダーがある場合はマルチモーダルコンテンツを構築
                    if has_image_placeholders(raw_user_content):
                        multimodal_content = _build_multimodal_content(
                            raw_user_content, ctx, cfg, None
                        )
                        msgs.append({"role": "user", "content": multimodal_content})
                    else:
                        msgs.append({"role": "user", "content": raw_user_content})

                    messages_list.append(msgs)
                    rec_indices.append(i)

                if messages_list:
                    client = clients[block.model]
                    model_def = cfg.model_by_name(block.model)
                    req_params = dict((model_def.request_defaults or {}))
                    req_params.update(block.params or {})

                    # v2: JSONモード
                    if block.mode == "json":
                        req_params["response_format"] = {"type": "json_object"}

                    # バッチ呼び出し
                    res, lats, errs = await client.batched_chat(
                        model=model_def.api_model,
                        messages_list=messages_list,
                        request_params=req_params,
                        batch_size=optimizer.current(),
                    )
                    optimizer.update(lats, errs)

                    # 出力適用
                    pos = 0
                    for idx in rec_indices:
                        text = res[pos] or ""
                        out_map = _apply_outputs(
                            text,
                            block.outputs or [OutputDef(name="full", select="full")],
                        )

                        # v2: save_to
                        if block.save_to and "vars" in block.save_to:
                            for var_name, out_name in block.save_to["vars"].items():
                                if out_name in out_map:
                                    exec_ctx.set_global(var_name, out_map[out_name])

                        if save_intermediate:
                            results[idx].update(
                                {f"_{block.exec}_{k}": v for k, v in out_map.items()}
                            )
                        contexts[idx].update(out_map)
                        pos += 1

            elif isinstance(block, LogicBlock):
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue
                    out_map = _apply_logic_block(block, ctx, exec_ctx)
                    if save_intermediate:
                        results[i].update(
                            {f"_{block.exec}_{k}": v for k, v in out_map.items()}
                        )
                    contexts[i].update(out_map)

            elif isinstance(block, PyBlock):
                # 関数名決定
                fn = _load_python_function(block)

                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue

                    out_map = _execute_python_block_single(
                        block, ctx, cfg, exec_ctx, fn
                    )

                    if save_intermediate:
                        results[i].update(
                            {f"_{block.exec}_{k}": v for k, v in out_map.items()}
                        )
                    contexts[i].update(out_map)

            elif isinstance(block, EndBlock):
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue

                    out_map = _execute_end_block_single(block, ctx, exec_ctx)
                    results[i].update(out_map)

                    # Phase 2: Endブロック処理後にメモリを解放
                    if progressive_release.is_enabled:
                        progressive_release.mark_row_done(i, contexts)

            else:
                raise ValueError(f"Unknown block class: {type(block)}")

        except Exception as e:
            if block.on_error != "continue":
                raise
            # continue on error
            for i, ok in enumerate(run_flags):
                if ok:
                    contexts[i][f"error_block_{block.exec}"] = str(e)

    # Phase 2: 最終的なガベージコレクション
    if progressive_release.is_enabled:
        progressive_release.force_gc()

    return results
