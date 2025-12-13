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
from ..llm_client import LLMClient
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
from .ai import _build_clients, _build_multimodal_content
from .scheduling import (
    HierarchicalTaskScheduler,
    StreamingContextManager,
    SchedulerConfig,
    MemoryConfig,
)
from .pipeline_adaptive import run_pipeline_streaming_adaptive

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
    # デフォルト0ms: OpenAI API等の標準エンドポイントでは即時送信が最適
    # vLLM/SGLang等の真のバッチ処理バックエンドを使用する場合のみ、
    # max_wait_ms を適切な値（例: 50ms）に設定してください
    max_wait_ms: int = 0,
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
        max_wait_ms: バッチ形成の最大待機時間 (ミリ秒、デフォルト: 0)
                     標準APIでは0（即時送信）が最適。vLLM/SGLang等の
                     真のバッチ処理バックエンドを使用する場合のみ増加を検討
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

    # AdaptiveController設定 - 最大並行数から開始し、エラー時に下げる
    controller = AdaptiveController(
        min_concurrency=min_concurrent,
        max_concurrency=max_concurrent,
        target_latency_ms=float(target_latency_ms),
        target_queue_depth=target_queue_depth,
        initial_concurrency=max_concurrent,  # 最大並行数から開始
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
    batchers_lock = asyncio.Lock()  # 競合状態防止用のロック

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
                    # グローバル変数をコンテキストに追加（run_if評価用）
                    extended_ctx = {
                        **row_exec_ctx.globals_const,
                        **row_exec_ctx.globals_vars,
                        **ctx,
                    }

                    run_ok = True
                    if block.run_if:
                        run_ok = _eval_cond(extended_ctx, block.run_if, row_exec_ctx)

                    if not run_ok:
                        continue

                    try:
                        if isinstance(block, AIBlock):
                            # バッチャー経由でAIブロックを実行
                            model_key = block.model
                            # 競合状態を防ぐためダブルチェックロッキングを使用
                            if model_key not in batchers:
                                async with batchers_lock:
                                    # ロック取得後に再度チェック（ダブルチェック）
                                    if model_key not in batchers:
                                        client = clients[model_key]
                                        model_def = cfg.model_by_name(model_key)
                                        req_params = dict(
                                            (model_def.request_defaults or {})
                                        )
                                        req_params.update(block.params or {})

                                        if block.mode == "json":
                                            req_params["response_format"] = {
                                                "type": "json_object"
                                            }

                                        # 最適化オプションからretry_on_empty設定を取得してrequest_paramsに反映
                                        if (
                                            hasattr(cfg, "optimization")
                                            and cfg.optimization
                                        ):
                                            retry_on_empty = cfg.optimization.get(
                                                "retry_on_empty", True
                                            )
                                            retry_cfg = dict(
                                                req_params.get("retry") or {}
                                            )
                                            retry_cfg["retry_on_empty"] = retry_on_empty
                                            req_params["retry"] = retry_cfg

                                        processor = await create_batch_processor(
                                            client, model_def.api_model, req_params
                                        )
                                        # max_wait_ms=0 で即時送信を優先
                                        # 標準API（OpenAI等）では待機によるバッチ化は
                                        # スループット向上に寄与せずレイテンシが悪化するため
                                        batchers[model_key] = AdaptiveRequestBatcher(
                                            batch_processor=processor,
                                            controller=controller,
                                            max_batch_size=max_batch_size,
                                            min_batch_size=1,
                                            max_wait_ms=max_wait_ms,
                                            enabled=True,
                                        )
                                        await batchers[model_key].start()

                            # メッセージ構築（グローバル変数も参照可能）
                            msgs = []
                            if block.system_prompt:
                                msgs.append(
                                    {
                                        "role": "system",
                                        "content": render_template(
                                            block.system_prompt, extended_ctx
                                        ),
                                    }
                                )

                            raw_user_content = "\n\n".join(
                                [
                                    render_template(p, extended_ctx)
                                    for p in (block.prompts or [])
                                ]
                            )

                            if has_image_placeholders(raw_user_content):
                                multimodal_content = _build_multimodal_content(
                                    raw_user_content, extended_ctx, cfg, None
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

        async def progressive_task_launcher_batched():
            """
            Progressive Task Launcher (Batched版) - セマフォ容量に応じて段階的にタスクを起動
            """
            tasks: List[asyncio.Task] = []
            next_index = 0
            active_count = 0

            while next_index < total or active_count > 0:
                # 現在のセマフォ容量を取得
                current_capacity = controller.current_concurrency
                available_slots = controller.get_available_slots()

                # 新しいタスクを起動できるスロット数を計算
                pending_tasks = active_count
                slots_to_fill = max(
                    0,
                    min(
                        current_capacity - pending_tasks,
                        available_slots,
                        total - next_index,
                    ),
                )

                # 新しいタスクを起動
                for _ in range(slots_to_fill):
                    if next_index < total:
                        row_index = next_index
                        row_data = dataset[row_index]
                        task = asyncio.create_task(
                            process_row_batched(row_index, row_data)
                        )
                        tasks.append(task)
                        next_index += 1
                        active_count += 1

                # 結果を待つ
                if active_count > 0:
                    try:
                        result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                        active_count -= 1
                        yield result
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(0.01)

            # 残りの結果を取得
            while not result_queue.empty():
                result = await result_queue.get()
                yield result

            # 全タスクの完了を待機
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

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
            # Progressive Task Launcher を使用してタスクを段階的に起動
            async for result in progressive_task_launcher_batched():
                completed += 1
                yield result

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
