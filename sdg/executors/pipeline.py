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
from ..llm_client import BatchOptimizer
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

# Import adaptive components (optional dependency)
try:
    from ..adaptive import AdaptiveController, MetricsCollector, MetricsType

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
):
    """
    ストリーミング版パイプライン - 完了した行から順次yield

    Args:
        cfg: SDG設定
        dataset: 入力データセット
        max_concurrent: 同時処理行数の上限
        save_intermediate: 中間結果を保存するか

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

    # 全行のタスクを起動
    tasks = [asyncio.create_task(process_row(i, row)) for i, row in enumerate(dataset)]

    # 完了した結果を順次yield
    while completed < total:
        result = await result_queue.get()
        completed += 1
        yield result

    # すべてのタスク完了を待機（エラー発生時の例外を伝播）
    await asyncio.gather(*tasks)


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

    try:
        # メトリクス収集を開始（有効な場合）
        if metrics_collector is not None:
            metrics_update_task = asyncio.create_task(update_metrics_loop())

        # 全行のタスクを起動
        tasks = [
            asyncio.create_task(process_row(i, row)) for i, row in enumerate(dataset)
        ]

        # 完了した結果を順次yield
        while completed < total:
            result = await result_queue.get()
            completed += 1
            yield result

        # すべてのタスク完了を待機
        await asyncio.gather(*tasks)

    finally:
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
) -> List[Dict[str, Any]]:
    """パイプライン実行（従来のブロック単位一括処理 - 後方互換性のため維持）"""

    # 実行コンテキスト
    exec_ctx = ExecutionContext(cfg)

    # モデルクライアント構築
    clients = _build_clients(cfg)

    results: List[Dict[str, Any]] = [{} for _ in dataset]
    contexts: List[Dict[str, Any]] = [dict(rec) for rec in dataset]

    optimizer = BatchOptimizer(
        min_batch=min_batch, max_batch=max_batch, target_latency_ms=target_latency_ms
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

            else:
                raise ValueError(f"Unknown block class: {type(block)}")

        except Exception as e:
            if block.on_error != "continue":
                raise
            # continue on error
            for i, ok in enumerate(run_flags):
                if ok:
                    contexts[i][f"error_block_{block.exec}"] = str(e)

    return results
