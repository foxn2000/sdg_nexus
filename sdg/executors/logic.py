from __future__ import annotations
import sys, os, re, json
from typing import Any, Dict

from ..config import LogicBlock
from ..mex import eval_mex
from ..utils import render_template, ensure_json_obj
from .core import ExecutionContext, _eval_cond, _truthy


def _execute_logic_step(
    step: Dict[str, Any], ctx: Dict[str, Any], exec_ctx: ExecutionContext
) -> Dict[str, Any]:
    """ロジックステップを実行（while/recurse用）"""
    op = step.get("op")

    if op == "set":
        var_name = step.get("var", "result")
        # MEX評価時にローカルコンテキストをマージ
        # グローバル変数とローカル変数を統合したコンテキストを作成
        eval_ctx = {**exec_ctx.globals_vars, **ctx}
        # 統合されたコンテキストを第2引数（context）として渡す
        value = eval_mex(step.get("value"), eval_ctx, exec_ctx.globals_vars)
        exec_ctx.set_global(var_name, value)
        # コンテキストも更新する（再帰関数内で必要）
        ctx[var_name] = value
        return {var_name: value}

    if op == "emit":
        value = eval_mex(step.get("value"), ctx, exec_ctx.globals_vars)
        return {"_emitted": value}

    return {}


def _apply_logic_block(
    block: LogicBlock, ctx: Dict[str, Any], exec_ctx: ExecutionContext
) -> Dict[str, Any]:
    """ロジックブロックを実行"""

    # v2: set - 変数代入
    if block.op == "set":
        var_name = block.var or "result"
        value = eval_mex(block.value, ctx, exec_ctx.globals_vars)
        exec_ctx.set_global(var_name, value)
        out_map = {}
        for o in block.outputs or []:
            name = o.get("name")
            out_map[name] = value
        return out_map

    # v2: let - ローカル束縛
    if block.op == "let":
        # ローカルコンテキストを作成
        local_ctx = dict(ctx)
        if block.bindings:
            for var_name, value_expr in block.bindings.items():
                value = eval_mex(value_expr, ctx, exec_ctx.globals_vars)
                local_ctx[var_name] = value
                # ローカル束縛はグローバルにも一時的に設定
                exec_ctx.set_global(var_name, value)

        # bodyを実行
        result_values = {}
        if block.body:
            for step in block.body:
                step_result = _execute_logic_step(step, local_ctx, exec_ctx)
                if step_result:
                    result_values.update(step_result)

        # 出力
        out_map = {}
        for o in block.outputs or []:
            name = o.get("name")
            from_ = o.get("from", "value")
            if from_ == "var":
                var_name = o.get("var", "result")
                # グローバル変数から取得（setで書き込まれている）
                out_map[name] = exec_ctx.get_global(var_name)
            else:
                # 最後のステップ結果を使用
                out_map[name] = (
                    result_values.get(name) or list(result_values.values())[-1]
                    if result_values
                    else None
                )
        return out_map

    # v2: reduce - リスト畳み込み
    if block.op == "reduce":
        src_name = block.list
        items = ctx.get(src_name, [])
        if isinstance(items, str):
            items = [s.strip() for s in items.split("\n") if s.strip()]

        # 初期値を設定
        if block.value is not None:
            accumulator = eval_mex(block.value, ctx, exec_ctx.globals_vars)
        else:
            # 初期値が未指定の場合、最初の要素を使用
            accumulator = items[0] if items else None
            items = items[1:] if items else []

        # accumulatorをグローバル変数に設定
        exec_ctx.set_global("accumulator", accumulator)

        # 各要素を畳み込み
        var_name = block.var or "item"

        for item in items:
            # 予算チェック
            if not exec_ctx.check_loop_budget(block.budget):
                break

            # ローカルコンテキスト
            local_ctx = {
                **ctx,
                var_name: item,
                "accumulator": exec_ctx.get_global("accumulator"),
            }

            # bodyを実行
            if block.body:
                for step in block.body:
                    _execute_logic_step(step, local_ctx, exec_ctx)
                # 更新された accumulator を取得
                accumulator = exec_ctx.get_global("accumulator")

        # 出力
        out_map = {}
        for o in block.outputs or []:
            name = o.get("name")
            from_ = o.get("from", "accumulator")
            if from_ == "accumulator":
                out_map[name] = accumulator
            elif from_ == "var":
                var_name = o.get("var", "result")
                out_map[name] = exec_ctx.get_global(var_name)
            else:
                out_map[name] = accumulator
        return out_map

    # v2: call - ユーザ定義ロジック関数呼び出し
    if block.op == "call":
        func_name = block.function or block.name
        if not func_name:
            raise ValueError("call requires 'function' or 'name'")

        # 関数定義を取得
        func_def = None
        if exec_ctx.cfg.functions.get("logic"):
            for f in exec_ctx.cfg.functions["logic"]:
                if f.name == func_name:
                    func_def = f
                    break

        if not func_def:
            raise ValueError(f"Logic function not found: {func_name}")

        # 引数を評価
        call_ctx = dict(ctx)
        if block.with_:
            for arg_name, value_expr in block.with_.items():
                # 文字列の場合はテンプレート展開してから評価
                if isinstance(value_expr, str):
                    value_str = render_template(value_expr, ctx)
                    # 数値に変換できれば変換
                    try:
                        value = float(value_str)
                    except:
                        value = value_str
                else:
                    value = eval_mex(value_expr, ctx, exec_ctx.globals_vars)
                call_ctx[arg_name] = value
                exec_ctx.set_global(arg_name, value)

        # 関数本体を実行
        if func_def.body:
            for step in func_def.body:
                _execute_logic_step(step, call_ctx, exec_ctx)

        # 出力を処理
        out_map = {}
        for o in block.outputs or []:
            name = o.get("name")
            from_ = o.get("from", "var")
            if from_ == "var":
                var_name = o.get("var", "result")
                out_map[name] = exec_ctx.get_global(var_name)
            else:
                # returns から取得（後方互換性）
                returns = block.returns or func_def.returns or []
                if returns:
                    out_map[name] = exec_ctx.get_global(returns[0])

        return out_map

    # v2: recurse - 再帰関数
    if block.op == "recurse":
        func_name = block.name
        if not func_name:
            raise ValueError("recurse requires 'name'")

        function = block.function
        if not function:
            raise ValueError("recurse requires 'function' definition")

        # デバッグフラグ（環境変数で制御）
        debug = os.environ.get("DEBUG_RECURSE") == "1"

        # 再帰関数を定義
        def recursive_call(args_dict: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
            if debug:
                print(
                    f"[DEBUG] recurse depth={depth}, args={args_dict}", file=sys.stderr
                )
            # 予算チェック
            if depth > 0:  # 最初の呼び出しではカウントしない
                exec_ctx.check_recursion_budget(block.budget)

            # 引数をコンテキストとグローバル変数に設定
            call_ctx = dict(ctx)
            for arg_name, arg_value in args_dict.items():
                call_ctx[arg_name] = arg_value
                exec_ctx.set_global(arg_name, arg_value)

            # ベースケースチェック
            base_case = function.get("base_case", {})
            if base_case:
                base_cond = base_case.get("cond", {})
                if _eval_cond(call_ctx, base_cond, exec_ctx):
                    # ベースケースの値を返す
                    base_value = base_case.get("value", [])
                    returns = function.get("returns", [])
                    result = {}
                    for i, ret_name in enumerate(returns):
                        if i < len(base_value):
                            val = base_value[i]
                            # 値が整数・浮動小数の場合はそのまま、式の場合は評価
                            if isinstance(val, (int, float)):
                                result[ret_name] = val
                            elif isinstance(val, dict):
                                result[ret_name] = eval_mex(
                                    val, call_ctx, exec_ctx.globals_vars
                                )
                            else:
                                result[ret_name] = val
                    if debug:
                        print(
                            f"[DEBUG] base case reached, returning {result}",
                            file=sys.stderr,
                        )
                    if depth > 0:
                        exec_ctx.recursion_depth -= 1
                    return result

            # 本体を実行
            body = function.get("body", [])
            for step in body:
                # call ステップを特別処理（再帰呼び出し）
                if step.get("op") == "call" and step.get("name") == func_name:
                    # 再帰呼び出しの引数を評価
                    nested_args = {}
                    if step.get("with"):
                        for arg_name, value_expr in step["with"].items():
                            nested_args[arg_name] = eval_mex(
                                value_expr, call_ctx, exec_ctx.globals_vars
                            )

                    # 再帰呼び出し
                    nested_result = recursive_call(nested_args, depth + 1)
                    if debug:
                        print(
                            f"[DEBUG] recursive call returned: {nested_result}",
                            file=sys.stderr,
                        )

                    # 戻り値をコンテキストに格納
                    # returnsで指定された名前と、関数のreturnsの対応を取る
                    nested_returns = step.get("returns", [])
                    func_returns = function.get("returns", [])
                    for i, ret_name in enumerate(nested_returns):
                        # 関数の戻り値の順序に従って値を取得
                        if i < len(func_returns):
                            func_ret_name = func_returns[i]
                            if func_ret_name in nested_result:
                                call_ctx[ret_name] = nested_result[func_ret_name]
                                if debug:
                                    print(
                                        f"[DEBUG] stored {ret_name}={nested_result[func_ret_name]} (from {func_ret_name}) in context",
                                        file=sys.stderr,
                                    )
                                    print(
                                        f"[DEBUG] current context vars: {list(call_ctx.keys())}",
                                        file=sys.stderr,
                                    )
                else:
                    # 通常のステップを実行
                    if debug:
                        print(f"[DEBUG] executing step: {step}", file=sys.stderr)
                    step_result = _execute_logic_step(step, call_ctx, exec_ctx)
                    # ステップ結果をコンテキストに反映
                    if step_result:
                        call_ctx.update(step_result)
                        if debug:
                            print(
                                f"[DEBUG] step result: {step_result}", file=sys.stderr
                            )

            # 戻り値を収集
            returns = function.get("returns", [])
            result = {}
            for ret_name in returns:
                # コンテキストから取得（グローバル変数ではなく）
                result[ret_name] = call_ctx.get(ret_name, exec_ctx.get_global(ret_name))

            if debug:
                print(
                    f"[DEBUG] returning from depth={depth}: {result}", file=sys.stderr
                )

            if depth > 0:
                exec_ctx.recursion_depth -= 1
            return result

        # 初期引数を準備
        initial_args = {}
        if block.with_:
            for arg_name, value_expr in block.with_.items():
                if isinstance(value_expr, (int, float)):
                    initial_args[arg_name] = value_expr
                elif isinstance(value_expr, dict):
                    initial_args[arg_name] = eval_mex(
                        value_expr, ctx, exec_ctx.globals_vars
                    )
                else:
                    initial_args[arg_name] = value_expr

        # 再帰関数を実行
        result = recursive_call(initial_args, 0)

        if debug:
            print(f"[DEBUG] final result: {result}", file=sys.stderr)

        # 出力
        out_map = {}
        for o in block.outputs or []:
            name = o.get("name")
            from_ = o.get("from", "value")
            if from_ == "value":
                # 最初の戻り値を使用
                returns = function.get("returns", [])
                if returns:
                    out_map[name] = result.get(returns[0])
                    if debug:
                        print(
                            f"[DEBUG] output {name} = {out_map[name]}", file=sys.stderr
                        )
            elif from_ == "var":
                var_name = o.get("var")
                out_map[name] = result.get(var_name) if var_name else None
            else:
                out_map[name] = result

        return out_map

    # v2: while - 反復
    if block.op == "while":
        # init
        if block.init:
            for step in block.init:
                _execute_logic_step(step, ctx, exec_ctx)

        collected = []
        while True:
            # 予算チェック
            if not exec_ctx.check_loop_budget(block.budget):
                break

            # 条件評価
            if not _eval_cond(ctx, block.cond or {}, exec_ctx):
                break

            # ステップ実行
            if block.step:
                for step in block.step:
                    result = _execute_logic_step(step, ctx, exec_ctx)
                    # emit収集
                    if step.get("op") == "emit":
                        collected.append(result.get("_emitted"))

        # 出力
        out_map = {}
        for o in block.outputs or []:
            name = o.get("name")
            from_ = o.get("from", "list")
            if from_ == "list":
                out_map[name] = collected
            elif from_ == "count":
                out_map[name] = len(collected)
        return out_map

    # v1: if
    if block.op == "if":
        cond = block.cond or {}
        ok = _eval_cond(ctx, cond, exec_ctx)
        choice = block.then if ok else block.else_

        out_map = {}
        for o in block.outputs or []:
            name = o.get("name")
            source = o.get("from", "boolean")
            if source == "boolean":
                out_map[name] = ok
            elif source == "text" or source == "value":
                # choice がテンプレート文字列の場合は展開する
                if isinstance(choice, str):
                    out_map[name] = render_template(choice, ctx)
                else:
                    out_map[name] = choice
            elif source == "source":
                src = o.get("source")
                out_map[name] = ctx.get(src, "")
            else:
                # デフォルトケースでもテンプレート展開を適用
                if isinstance(choice, str):
                    out_map[name] = render_template(choice, ctx)
                else:
                    out_map[name] = choice
        return out_map

    # v1: and/or/not
    if block.op in ("and", "or", "not"):
        vals = [_eval_cond(ctx, c, exec_ctx) for c in (block.operands or [])]
        res = (
            all(vals)
            if block.op == "and"
            else (any(vals) if block.op == "or" else (not vals[0] if vals else False))
        )
        out_map = {}
        for o in block.outputs or []:
            name = o.get("name")
            out_map[name] = res
        return out_map

    # v1: for
    if block.op == "for":
        src_name = block.list
        items = ctx.get(src_name, [])
        if isinstance(items, str):
            if (block.parse or "") == "regex" and block.regex_pattern:
                items = re.findall(block.regex_pattern, items, re.DOTALL)
            elif (block.parse or "") == "lines":
                items = [s.strip() for s in items.split("\n") if s.strip()]
            else:
                items = [s.strip() for s in items.split(",")]

        if block.drop_empty:
            items = [x for x in items if str(x).strip() != ""]

        # where
        where = ensure_json_obj(block.where)

        def where_ok(item):
            subctx = {**ctx, (block.var or "item"): item}
            return _eval_cond(subctx, where, exec_ctx) if where else True

        items = [x for x in items if where_ok(x)]

        # map
        def map_one(item):
            subctx = {**ctx, (block.var or "item"): item}
            return render_template(
                block.map or "{" + (block.var or "item") + "}", subctx
            )

        mapped = [map_one(x) for x in items]

        out_map = {}
        for o in block.outputs or []:
            name = o.get("name")
            joiner = o.get("join_with")
            limit = o.get("limit")
            offset = o.get("offset", 0)
            seq = mapped[offset : offset + limit if limit else None]
            out_map[name] = (joiner or "\n").join(seq) if joiner is not None else seq
        return out_map

    raise ValueError(f"Unsupported logic op: {block.op}")
