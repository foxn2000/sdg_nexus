from __future__ import annotations
import json
from collections import ChainMap
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import re

from ..config import (
    SDGConfig,
    OutputDef,
    EndBlock,
)
from ..mex import eval_mex
from ..utils import (
    render_template,
    extract_by_regex,
    extract_by_tag,
)


class BudgetExceeded(Exception):
    """予算超過例外"""

    pass


class ExecutionContext:
    """実行コンテキスト - グローバル変数とカウンタを管理"""

    def __init__(self, cfg: SDGConfig):
        self.cfg = cfg
        self.globals_vars = dict(cfg.globals_.vars)  # 書き込み可能
        self.globals_const = dict(cfg.globals_.const)  # 読み取り専用
        self.loop_counter = 0
        self.recursion_depth = 0
        self.ai_call_counter = 0
        self.ai_token_counter = 0

    def get_global(self, name: str) -> Any:
        """グローバル変数取得（定数優先）"""
        if name in self.globals_const:
            return self.globals_const[name]
        return self.globals_vars.get(name)

    def set_global(self, name: str, value: Any):
        """グローバル変数設定"""
        if name in self.globals_const:
            raise ValueError(f"Cannot modify constant: {name}")
        self.globals_vars[name] = value

    def check_loop_budget(self, block_budget: Optional[Dict[str, Any]] = None):
        """ループ予算チェック"""
        budget = block_budget or {}
        loops_cfg = budget.get("loops") or self.cfg.budgets.loops
        max_iters = loops_cfg.get("max_iters", 10000)

        if self.loop_counter >= max_iters:
            on_exceed = loops_cfg.get("on_exceed", "error")
            if on_exceed == "error":
                raise BudgetExceeded(f"Loop budget exceeded: {max_iters}")
            elif on_exceed == "truncate":
                return False  # 中断信号

        self.loop_counter += 1
        return True

    def check_recursion_budget(self, block_budget: Optional[Dict[str, Any]] = None):
        """再帰予算チェック"""
        budget = block_budget or {}
        rec_cfg = budget.get("recursion") or self.cfg.budgets.recursion
        max_depth = rec_cfg.get("max_depth", 256)

        if self.recursion_depth >= max_depth:
            on_exceed = rec_cfg.get("on_exceed", "error")
            if on_exceed == "error":
                raise BudgetExceeded(f"Recursion budget exceeded: {max_depth}")

        self.recursion_depth += 1


def _truthy(v: Any) -> bool:
    """真偽値判定"""
    if v is None or v is False:
        return False
    if v == "" or v == 0 or v == [] or v == {}:
        return False
    return True


def _apply_outputs(text: str, outs: List[OutputDef]) -> Dict[str, Any]:
    """AI出力を抽出"""
    d: Dict[str, Any] = {}
    for out in outs:
        vals: List[str] = []

        if out.select == "full":
            vals = [text]
        elif out.select == "tag":
            if not out.tag:
                raise ValueError("outputs.select=tag requires 'tag'")
            vals = extract_by_tag(text, out.tag)
        elif out.select == "regex":
            if not out.regex:
                raise ValueError("outputs.select=regex requires 'regex'")
            vals = extract_by_regex(text, out.regex)
        elif out.select == "jsonpath":
            # v2: JSONPathサポート（簡易実装）
            if not out.path:
                raise ValueError("outputs.select=jsonpath requires 'path'")
            try:
                import jsonpath_ng

                parsed = json.loads(text)
                expr = jsonpath_ng.parse(out.path)
                matches = [match.value for match in expr.find(parsed)]
                vals = matches
            except Exception:
                # jsonpath_ngがない場合は簡易実装
                try:
                    parsed = json.loads(text)
                    path_parts = out.path.replace("$.", "").split(".")
                    val = parsed
                    for part in path_parts:
                        if isinstance(val, dict):
                            val = val.get(part)
                        else:
                            val = None
                            break
                    vals = [val] if val is not None else []
                except Exception:
                    vals = []
        else:
            raise ValueError(f"Unknown select: {out.select}")

        # 型変換（v2）
        if out.type_hint and vals:
            if out.type_hint == "number":
                vals = [float(v) if v else 0 for v in vals]
            elif out.type_hint == "boolean":
                vals = [_truthy(v) for v in vals]
            elif out.type_hint == "json":
                vals = [json.loads(str(v)) if v else None for v in vals]

        # 結合
        if out.join_with is not None:
            d[out.name] = out.join_with.join(str(v) for v in vals)
        else:
            # 常に最初の要素を文字列として返す（型の一貫性を保証）
            # 複数の要素がある場合も最初の1つのみ使用
            # これによりHuggingFaceなどでのスキーマ不整合を防ぐ
            d[out.name] = str(vals[0]) if vals else ""

    return d


def _eval_cond(
    ctx: Dict[str, Any],
    cond: Dict[str, Any],
    exec_ctx: Optional[ExecutionContext] = None,
) -> bool:
    """条件式を評価（v1互換 + v2 MEX）"""
    if not cond:
        return True

    # v2: MEXエンジンで評価
    if exec_ctx:
        try:
            result = eval_mex(cond, ctx, exec_ctx.globals_vars)
            return _truthy(result)
        except Exception:
            pass  # フォールバック

    # v1互換の条件評価
    if "and" in cond:
        return all(_eval_cond(ctx, c, exec_ctx) for c in cond["and"])
    if "or" in cond:
        return any(_eval_cond(ctx, c, exec_ctx) for c in cond["or"])
    if "not" in cond:
        return not _eval_cond(ctx, cond["not"], exec_ctx)
    if "equals" in cond:
        a, b = cond["equals"]
        return str(render_template(str(a), ctx)) == str(render_template(str(b), ctx))
    if "not_equals" in cond:
        a, b = cond["not_equals"]
        return str(render_template(str(a), ctx)) != str(render_template(str(b), ctx))
    if "contains" in cond:
        a, b = cond["contains"]
        return (
            str(render_template(str(a), ctx)).find(str(render_template(str(b), ctx)))
            != -1
        )
    if "is_empty" in cond:
        a = cond["is_empty"]
        return str(render_template(str(a), ctx)).strip() == ""
    if "gt" in cond or "lt" in cond or "gte" in cond or "lte" in cond:
        key = (
            "gt"
            if "gt" in cond
            else "lt" if "lt" in cond else "gte" if "gte" in cond else "lte"
        )
        a, b = cond[key]
        try:
            av = float(render_template(str(a), ctx))
            bv = float(render_template(str(b), ctx))
        except Exception:
            av = len(str(render_template(str(a), ctx)))
            bv = len(str(render_template(str(b), ctx)))
        if key == "gt":
            return av > bv
        if key == "lt":
            return av < bv
        if key == "gte":
            return av >= bv
        if key == "lte":
            return av <= bv

    return _truthy(render_template(json.dumps(cond), ctx))


def _execute_end_block_single(
    block: EndBlock,
    ctx: Dict[str, Any],
    exec_ctx: ExecutionContext,
) -> Dict[str, Any]:
    """単一行のEndブロック実行"""
    # グローバル定数と変数をコンテキストに追加
    # ローカルコンテキスト(ctx)が最優先されるようChainMapで論理結合
    # 検索順序: ctx -> globals_vars -> globals_const（ゼロコピー）
    extended_ctx = ChainMap(ctx, exec_ctx.globals_vars, exec_ctx.globals_const)

    out_map = {}
    for f in block.final or []:
        name = f.get("name")
        value_tmpl = f.get("value", "")
        out_map[name] = render_template(value_tmpl, extended_ctx)

    # v2: include_vars
    if block.include_vars:
        for var_name in block.include_vars:
            out_map[var_name] = exec_ctx.get_global(var_name)

    return out_map


@dataclass
class StreamingResult:
    """ストリーミング結果"""

    row_index: int
    data: Dict[str, Any]
    error: Optional[Exception] = None
