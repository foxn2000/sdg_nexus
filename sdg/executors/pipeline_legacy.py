from __future__ import annotations
from typing import Any, Dict, List

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
)
from .logic import _apply_logic_block
from .python import _load_python_function, _execute_python_block_single
from .ai import _build_clients, _build_multimodal_content
from .scheduling import (
    BatchProgressiveRelease,
    MemoryConfig,
)


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
        # run_if評価（グローバル変数も参照可能）
        run_flags = []
        for ctx in contexts:
            extended_ctx = {
                **exec_ctx.globals_const,
                **exec_ctx.globals_vars,
                **ctx,
            }
            ok = True
            if block.run_if:
                ok = _eval_cond(extended_ctx, block.run_if, exec_ctx)
            run_flags.append(ok)

        try:
            if isinstance(block, AIBlock):
                # メッセージ構築
                messages_list = []
                rec_indices = []
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue

                    # グローバル変数をコンテキストに追加
                    extended_ctx = {
                        **exec_ctx.globals_const,
                        **exec_ctx.globals_vars,
                        **ctx,
                    }

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

                    # プロンプト内に画像があるかチェック
                    raw_user_content = "\n\n".join(
                        [
                            render_template(p, extended_ctx)
                            for p in (block.prompts or [])
                        ]
                    )

                    # 画像プレースホルダーがある場合はマルチモーダルコンテンツを構築
                    if has_image_placeholders(raw_user_content):
                        multimodal_content = _build_multimodal_content(
                            raw_user_content, extended_ctx, cfg, None
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

                    # 最適化オプションからretry_on_empty設定を取得してrequest_paramsに反映
                    if hasattr(cfg, "optimization") and cfg.optimization:
                        retry_on_empty = cfg.optimization.get("retry_on_empty", True)
                        retry_cfg = dict(req_params.get("retry") or {})
                        retry_cfg["retry_on_empty"] = retry_on_empty
                        req_params["retry"] = retry_cfg

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
