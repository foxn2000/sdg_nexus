from __future__ import annotations
import asyncio, importlib.util, os, re, json
from typing import Any, Dict, List, Tuple
from .config import SDGConfig, AIBlock, LogicBlock, PyBlock, EndBlock, OutputDef
from .llm_client import LLMClient, BatchOptimizer
from .utils import render_template, extract_by_regex, extract_by_tag, ensure_json_obj

def _apply_outputs(text: str, outs: List[OutputDef]) -> Dict[str, Any]:
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
        else:
            raise ValueError(f"Unknown select: {out.select}")
        if out.join_with is not None:
            d[out.name] = out.join_with.join(vals)
        else:
            d[out.name] = vals[0] if len(vals) == 1 else vals
    return d

def _truthy(v: Any) -> bool:
    return bool(v) and v not in ("", "0", "false", "False", "null", None)

def _eval_cond(ctx: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    # Supported ops: equals, not_equals, contains, gt, lt, gte, lte, and, or, not, is_empty
    if not cond:
        return True
    if "and" in cond:
        return all(_eval_cond(ctx, c) for c in cond["and"])
    if "or" in cond:
        return any(_eval_cond(ctx, c) for c in cond["or"])
    if "not" in cond:
        return not _eval_cond(ctx, cond["not"])
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
    # fallback: any non-empty
    return _truthy(render_template(json.dumps(cond), ctx))

def _apply_logic_block(block: LogicBlock, ctx: Dict[str, Any]) -> Dict[str, Any]:
    if block.op == "if":
        cond = block.cond or {}
        ok = _eval_cond(ctx, cond)
        choice = block.then if ok else block.else_
        out_map = {}
        for o in (block.outputs or []):
            name = o.get("name")
            source = o.get("from", "boolean")
            if source == "boolean":
                out_map[name] = ok
            elif source == "text":
                out_map[name] = choice
            elif source == "source":
                src = o.get("source")
                out_map[name] = ctx.get(src, "")
            else:
                out_map[name] = choice
        return out_map
    if block.op in ("and","or","not"):
        vals = [ _eval_cond(ctx, c) for c in (block.operands or []) ]
        res = (all(vals) if block.op=="and" else (any(vals) if block.op=="or" else (not vals[0] if vals else False)))
        out_map = {}
        for o in (block.outputs or []):
            name = o.get("name")
            out_map[name] = res
        return out_map
    if block.op == "for":
        src_name = block.list
        items = ctx.get(src_name, [])
        if isinstance(items, str):
            if (block.parse or "") == "regex" and block.regex_pattern:
                items = re.findall(block.regex_pattern, items, re.DOTALL)
            else:
                items = [s.strip() for s in items.split(",")]
        if block.drop_empty:
            items = [x for x in items if str(x).strip() != ""]
        # where
        where = ensure_json_obj(block.where)
        def where_ok(item):
            subctx = {**ctx, (block.var or "item"): item}
            return _eval_cond(subctx, where) if where else True
        items = [x for x in items if where_ok(x)]
        # map
        def map_one(item):
            subctx = {**ctx, (block.var or "item"): item}
            return render_template(block.map or "{"+(block.var or "item")+"}", subctx)
        mapped = [map_one(x) for x in items]
        out_map = {}
        for o in (block.outputs or []):
            name = o.get("name")
            joiner = o.get("join_with")
            limit = o.get("limit")
            offset = o.get("offset", 0)
            seq = mapped[offset: offset+limit if limit else None]
            out_map[name] = (joiner or "\n").join(seq) if joiner is not None else seq
        return out_map
    raise ValueError(f"Unsupported logic op: {block.op}")

async def run_pipeline(cfg: SDGConfig, dataset: List[Dict[str,Any]], *, max_batch: int=8, min_batch: int=1, target_latency_ms: int=3000, save_intermediate: bool=False) -> List[Dict[str,Any]]:
    # Build model clients
    clients: Dict[str, LLMClient] = {}
    for m in cfg.models:
        api_key = os.environ.get("OPENAI_API_KEY") if m.api_key.startswith("${ENV.") else m.api_key
        base_url = m.base_url or "https://api.openai.com"
        timeout = (m.request_defaults or {}).get("timeout_sec")
        clients[m.name] = LLMClient(base_url=base_url, api_key=api_key, organization=m.organization, headers=m.headers, timeout_sec=timeout)

    results: List[Dict[str,Any]] = [ {} for _ in dataset ]
    contexts: List[Dict[str,Any]] = [ dict(rec) for rec in dataset ]

    optimizer = BatchOptimizer(min_batch=min_batch, max_batch=max_batch, target_latency_ms=target_latency_ms)

    for block in cfg.blocks:
        # Evaluate run_if per record
        run_flags = []
        for ctx in contexts:
            ok = True
            if block.run_if:
                ok = _eval_cond(ctx, block.run_if)
            run_flags.append(ok)

        try:
            if isinstance(block, AIBlock):
                # Build messages per record
                messages_list = []
                rec_indices = []
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue
                    msgs = []
                    if block.system_prompt:
                        msgs.append({ "role": "system", "content": render_template(block.system_prompt, ctx) })
                    user_content = "\n\n".join([ render_template(p, ctx) for p in (block.prompts or []) ])
                    msgs.append({ "role": "user", "content": user_content })
                    messages_list.append(msgs)
                    rec_indices.append(i)
                if messages_list:
                    client = clients[block.model]
                    model_def = cfg.model_by_name(block.model)
                    req_params = dict((model_def.request_defaults or {}))
                    # merge request_defaults + block.params, latter has priority
                    req_params.update(block.params or {})
                    # call in batches with adaptive concurrency
                    res, lats, errs = await client.batched_chat(model=cfg.model_by_name(block.model).api_model, messages_list=messages_list, request_params=req_params, batch_size=optimizer.current())
                    optimizer.update(lats, errs)
                    # apply outputs
                    pos = 0
                    for idx in rec_indices:
                        text = res[pos] or ""
                        out_map = _apply_outputs(text, block.outputs or [OutputDef(name="full", select="full")])
                        results[idx].update({f"_{block.exec}_{k}": v for k,v in out_map.items()}) if save_intermediate else None
                        contexts[idx].update(out_map)
                        pos += 1

            elif isinstance(block, LogicBlock):
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue
                    out_map = _apply_logic_block(block, ctx)
                    results[i].update({f"_{block.exec}_{k}": v for k,v in out_map.items()}) if save_intermediate else None
                    contexts[i].update(out_map)

            elif isinstance(block, PyBlock):
                # load module
                module_path = os.path.abspath(block.code_path)
                spec = importlib.util.spec_from_file_location("sdg_user_module", module_path)
                if spec is None or spec.loader is None:
                    raise FileNotFoundError(f"Cannot load python module: {module_path}")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                fn = getattr(mod, block.function, None)
                if not fn:
                    raise AttributeError(f"Function not found: {block.function} in {module_path}")

                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue
                    args = [ctx.get(name) for name in (block.inputs or [])]
                    out = fn(*args)
                    if isinstance(out, dict):
                        out_map = {k: out.get(k) for k in block.outputs}
                    else:
                        out_map = {name: val for name, val in zip(block.outputs, out if isinstance(out, (list,tuple)) else [out])}
                    results[i].update({f"_{block.exec}_{k}": v for k,v in out_map.items()}) if save_intermediate else None
                    contexts[i].update(out_map)

            elif isinstance(block, EndBlock):
                # gather finals
                for i, (run_ok, ctx) in enumerate(zip(run_flags, contexts)):
                    if not run_ok:
                        continue
                    finals = []
                    out_map = {}
                    for f in (block.final or []):
                        name = f.get("name")
                        value_tmpl = f.get("value","")
                        out_map[name] = render_template(value_tmpl, ctx)
                        finals.append((name, out_map[name]))
                    results[i].update(out_map)
            else:
                raise ValueError(f"Unknown block class: {type(block)}")
        except Exception as e:
            if block.on_error != "continue":
                raise
            # continue on error: write error message in context
            for i, ok in enumerate(run_flags):
                if ok:
                    contexts[i][f"error_block_{block.exec}"] = str(e)
    return results
