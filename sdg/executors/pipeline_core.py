from __future__ import annotations
from typing import Any, Dict, Optional

from ..config import (
    SDGConfig,
    AIBlock,
    LogicBlock,
    PyBlock,
    EndBlock,
)
from .core import (
    ExecutionContext,
    _eval_cond,
    _execute_end_block_single,
)
from .logic import _apply_logic_block
from .python import _load_python_function, _execute_python_block_single
from .ai import _execute_ai_block_single


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
        # グローバル変数をコンテキストに追加（run_if評価用）
        extended_ctx = {
            **exec_ctx.globals_const,
            **exec_ctx.globals_vars,
            **ctx,
        }

        # run_if評価（グローバル変数も参照可能）
        run_ok = True
        if block.run_if:
            run_ok = _eval_cond(extended_ctx, block.run_if, exec_ctx)

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
