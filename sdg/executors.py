from __future__ import annotations
import asyncio, importlib.util, os, re, json, sys, tempfile
from typing import Any, Dict, List, Tuple, Optional
from .config import SDGConfig, AIBlock, LogicBlock, PyBlock, EndBlock, OutputDef, BudgetConfig
from .llm_client import LLMClient, BatchOptimizer
from .utils import render_template, extract_by_regex, extract_by_tag, ensure_json_obj
from .mex import eval_mex, MEXEvaluator

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
            d[out.name] = vals[0] if len(vals) == 1 else vals
    
    return d

def _truthy(v: Any) -> bool:
    """真偽値判定"""
    if v is None or v is False:
        return False
    if v == "" or v == 0 or v == [] or v == {}:
        return False
    return True

def _eval_cond(ctx: Dict[str, Any], cond: Dict[str, Any], exec_ctx: Optional[ExecutionContext] = None) -> bool:
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
        return str(render_template(str(a), ctx)).find(str(render_template(str(b), ctx))) != -1
    if "is_empty" in cond:
        a = cond["is_empty"]
        return str(render_template(str(a), ctx)).strip() == ""
    if "gt" in cond or "lt" in cond or "gte" in cond or "lte" in cond:
        key = "gt" if "gt" in cond else "lt" if "lt" in cond else "gte" if "gte" in cond else "lte"
        a, b = cond[key]
        try:
            av = float(render_template(str(a), ctx))
            bv = float(render_template(str(b), ctx))
        except Exception:
            av = len(str(render_template(str(a), ctx)))
            bv = len(str(render_template(str(b), ctx)))
        if key == "gt": return av > bv
        if key == "lt": return av < bv
        if key == "gte": return av >= bv
        if key == "lte": return av <= bv
    
    return _truthy(render_template(json.dumps(cond), ctx))

def _apply_logic_block(block: LogicBlock, ctx: Dict[str, Any], exec_ctx: ExecutionContext) -> Dict[str, Any]:
    """ロジックブロックを実行"""
    
    # v2: set - 変数代入
    if block.op == "set":
        var_name = block.var or "result"
        value = eval_mex(block.value, ctx, exec_ctx.globals_vars)
        exec_ctx.set_global(var_name, value)
        out_map = {}
        for o in (block.outputs or []):
            name = o.get("name")
            out_map[name] = value
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
        for o in (block.outputs or []):
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
        for o in (block.outputs or []):
            name = o.get("name")
            source = o.get("from", "boolean")
            if source == "boolean":
                out_map[name] = ok
            elif source == "text" or source == "value":
                out_map[name] = choice
            elif source == "source":
                src = o.get("source")
                out_map[name] = ctx.get(src, "")
            else:
                out_map[name] = choice
        return out_map
    
    # v1: and/or/not
    if block.op in ("and", "or", "not"):
        vals = [_eval_cond(ctx, c, exec_ctx) for c in (block.operands or [])]
        res = (all(vals) if block.op == "and" else 
               (any(vals) if block.op == "or" else 
                (not vals[0] if vals else False)))
        out_map = {}
        for o in (block.outputs or []):
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
            return render_template(block.map or "{" + (block.var or "item") + "}", subctx)
        mapped = [map_one(x) for x in items]
        
        out_map = {}
        for o in (block.outputs or []):
            name = o.get("name")
            joiner = o.get("join_with")
            limit = o.get("limit")
            offset = o.get("offset", 0)
            seq = mapped[offset: offset + limit if limit else None]
            out_map[name] = (joiner or "\n").join(seq) if joiner is not None else seq
        return out_map
    
    raise ValueError(f"Unsupported logic op: {block.op}")

def _execute_logic_step(step: Dict[str, Any], ctx: Dict[str, Any], exec_ctx: ExecutionContext) -> Dict[str, Any]:
    """ロジックステップを実行（while/recurse用）"""
    op = step.get("op")
    
    if op == "set":
        var_name = step.get("var", "result")
        value = eval_mex(step.get("value"), ctx, exec_ctx.globals_vars)
        exec_ctx.set_global(var_name, value)
        return {var_name: value}
    
    if op == "emit":
        value = eval_mex(step.get("value"), ctx, exec_ctx.globals_vars)
        return {"_emitted": value}
    
    return {}

class PythonContext:
    """Python関数用のコンテキスト"""
    def __init__(self, exec_ctx: ExecutionContext, local_ctx: Dict[str, Any]):
        self.exec_ctx = exec_ctx
        self.local_ctx = local_ctx
        self.vars = exec_ctx.globals_vars
    
    def get(self, path: str) -> Any:
        """パス指定で値を取得"""
        parts = path.split(".")
        current = self.local_ctx
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current
    
    def set(self, path: str, value: Any):
        """パス指定で値を設定"""
        self.exec_ctx.set_global(path, value)
    
    def emit(self, name: str, value: Any):
        """値を収集"""
        pass  # 簡易実装では未サポート
    
    def log(self, level: str, message: str):
        """ログ出力"""
        print(f"[{level.upper()}] {message}", file=sys.stderr)

async def run_pipeline(
    cfg: SDGConfig,
    dataset: List[Dict[str, Any]],
    *,
    max_batch: int = 8,
    min_batch: int = 1,
    target_latency_ms: int = 3000,
    save_intermediate: bool = False
) -> List[Dict[str, Any]]:
    """パイプライン実行"""
    
    # 実行コンテキスト
    exec_ctx = ExecutionContext(cfg)
    
    # モデルクライアント構築
    clients: Dict[str, LLMClient] = {}
    for m in cfg.models:
        # 環境変数注入
        api_key = m.api_key
        if api_key.startswith("${ENV."):
            env_name = api_key[6:-1]  # ${ENV.NAME} -> NAME
            api_key = os.environ.get(env_name, "")
        
        base_url = m.base_url or "https://api.openai.com"
        timeout = (m.request_defaults or {}).get("timeout_sec")
        clients[m.name] = LLMClient(
            base_url=base_url,
            api_key=api_key,
            organization=m.organization,
            headers=m.headers,
            timeout_sec=timeout
        )
    
    results: List[Dict[str, Any]] = [{} for _ in dataset]
    contexts: List[Dict[str, Any]] = [dict(rec) for rec in dataset]
    
    optimizer = BatchOptimizer(
        min_batch=min_batch,
        max_batch=max_batch,
        target_latency_ms=target_latency_ms
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
                        msgs.append({
                            "role": "system",
                            "content": render_template(block.system_prompt, ctx)
                        })
                    user_content = "\n\n".join([
                        render_template(p, ctx) for p in (block.prompts or [])
                    ])
                    msgs.append({"role": "user", "content": user_content})
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
                        batch_size=optimizer.current()
                    )
                    optimizer.update(lats, errs)
                    
                    # 出力適用
                    pos = 0
                    for idx in rec_indices:
                        text = res[pos] or ""
                        out_map = _apply_outputs(
                            text,
                            block.outputs or [OutputDef(name="full", select="full")]
                        )
                        
                        # v2: save_to
                        if block.save_to and "vars" in block.save_to:
                            for var_name, out_name in block.save_to["vars"].items():
                                if out_name in out_map:
                                    exec_ctx.set_global(var_name, out_map[out_name])
                        
                        if save_intermediate:
                            results[idx].update({f"_{block.exec}_{k}": v for k, v in out_map.items()})
                        contexts[idx].update(out_map)
                        pos += 1
            
            elif isinstance(block, LogicBlock):
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue
                    out_map = _apply_logic_block(block, ctx, exec_ctx)
                    if save_intermediate:
                        results[i].update({f"_{block.exec}_{k}": v for k, v in out_map.items()})
                    contexts[i].update(out_map)
            
            elif isinstance(block, PyBlock):
                # 関数名決定
                fn_name = block.entrypoint or block.function
                if not fn_name:
                    raise ValueError("python block requires 'function' or 'entrypoint'")
                
                # コードロード
                if block.function_code:
                    # v2: インラインコード
                    namespace = {}
                    exec(block.function_code, namespace)
                    fn = namespace.get(fn_name)
                elif block.code_path:
                    # v1: 外部ファイル
                    module_path = os.path.abspath(block.code_path)
                    spec = importlib.util.spec_from_file_location("sdg_user_module", module_path)
                    if spec is None or spec.loader is None:
                        raise FileNotFoundError(f"Cannot load python module: {module_path}")
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    fn = getattr(mod, fn_name, None)
                else:
                    raise ValueError("python block requires 'code_path' or 'function_code'")
                
                if not fn:
                    raise AttributeError(f"Function not found: {fn_name}")
                
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue
                    
                    # v2: ctxオブジェクト
                    py_ctx = PythonContext(exec_ctx, ctx)
                    
                    # 引数準備
                    if isinstance(block.inputs, dict):
                        # v2: キーワード引数
                        kwargs = {k: ctx.get(v) for k, v in block.inputs.items()}
                        out = fn(py_ctx, **kwargs)
                    else:
                        # v1: 位置引数
                        args = [ctx.get(name) for name in (block.inputs or [])]
                        out = fn(py_ctx, *args) if cfg.is_v2() else fn(*args)
                    
                    # 出力処理
                    if isinstance(out, dict):
                        out_map = {k: out.get(k) for k in block.outputs}
                    else:
                        out_map = {
                            name: val
                            for name, val in zip(
                                block.outputs,
                                out if isinstance(out, (list, tuple)) else [out]
                            )
                        }
                    
                    if save_intermediate:
                        results[i].update({f"_{block.exec}_{k}": v for k, v in out_map.items()})
                    contexts[i].update(out_map)
            
            elif isinstance(block, EndBlock):
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue
                    
                    out_map = {}
                    for f in (block.final or []):
                        name = f.get("name")
                        value_tmpl = f.get("value", "")
                        out_map[name] = render_template(value_tmpl, ctx)
                    
                    # v2: include_vars
                    if block.include_vars:
                        for var_name in block.include_vars:
                            out_map[var_name] = exec_ctx.get_global(var_name)
                    
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
