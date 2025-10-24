from __future__ import annotations
import os, yaml, json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .utils import ensure_json_obj

@dataclass
class ModelDef:
    name: str
    api_model: str
    api_key: str
    base_url: Optional[str] = None
    organization: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_defaults: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OutputDef:
    name: str
    select: str = "full"  # full | tag | regex
    tag: Optional[str] = None
    regex: Optional[str] = None
    join_with: Optional[str] = None

@dataclass
class Block:
    type: str
    exec: int
    run_if: Any = None
    on_error: str = "fail"  # fail | continue

@dataclass
class AIBlock(Block):
    model: str = ""
    system_prompt: Optional[str] = None
    prompts: List[str] = field(default_factory=list)
    outputs: List[OutputDef] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LogicBlock(Block):
    name: Optional[str] = None
    op: str = "if"
    cond: Any = None
    then: Optional[str] = None
    else_: Optional[str] = None
    operands: Optional[List[Any]] = None
    # for-loop family
    list: Optional[str] = None
    parse: Optional[str] = None
    regex_pattern: Optional[str] = None
    var: Optional[str] = None
    drop_empty: Optional[bool] = None
    where: Any = None
    map: Optional[str] = None
    outputs: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PyBlock(Block):
    name: str = ""
    function: str = ""
    inputs: List[str] = field(default_factory=list)
    code_path: str = "./script.py"
    venv_path: Optional[str] = None
    outputs: List[str] = field(default_factory=list)

@dataclass
class EndBlock(Block):
    reason: Optional[str] = None
    exit_code: Optional[str] = None
    final: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class SDGConfig:
    mabel: Dict[str, Any] = field(default_factory=dict)
    models: List[ModelDef] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)

    def model_by_name(self, name: str) -> ModelDef:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"Model not found: {name}")

def _normalize_block(d: Dict[str, Any]) -> Block:
    typ = d.get("type")
    common = {
        "type": typ,
        "exec": int(d.get("exec", 0)),
        "run_if": ensure_json_obj(d.get("run_if")),
        "on_error": d.get("on_error", "fail"),
    }
    if typ == "ai":
        outs = [OutputDef(**o) for o in d.get("outputs", [])]
        return AIBlock(outputs=outs, model=d.get("model",""), system_prompt=d.get("system_prompt"),
                       prompts=list(d.get("prompts", [])), params=d.get("params", {}), **common)
    if typ == "logic":
        return LogicBlock(
            name=d.get("name"), op=d.get("op","if"), cond=ensure_json_obj(d.get("cond")),
            then=d.get("then"), else_=d.get("else"), operands=d.get("operands"),
            list=d.get("list"), parse=d.get("parse"), regex_pattern=d.get("regex_pattern"),
            var=d.get("var"), drop_empty=d.get("drop_empty"), where=ensure_json_obj(d.get("where")),
            map=d.get("map"), outputs=d.get("outputs", []), **common
        )
    if typ == "python":
        return PyBlock(name=d.get("name",""), function=d.get("function",""),
                       inputs=d.get("inputs", []), code_path=d.get("code_path","./script.py"),
                       venv_path=d.get("venv_path"), outputs=d.get("outputs", []), **common)
    if typ == "end":
        return EndBlock(reason=d.get("reason"), exit_code=d.get("exit_code"),
                        final=d.get("final", []), **common)
    raise ValueError(f"Unsupported block type: {typ}")

def load_config(path: str) -> SDGConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    models = [ModelDef(**m) for m in data.get("models", [])]
    blocks = [_normalize_block(b) for b in data.get("blocks", [])]
    conns = data.get("connections", [])
    cfg = SDGConfig(mabel=data.get("mabel", {}), models=models, blocks=sorted(blocks, key=lambda b: b.exec), connections=conns)
    # basic validation
    for b in cfg.blocks:
        if b.type == "ai" and not isinstance(b, AIBlock):
            raise ValueError("Block casting failed for ai")
        if b.type == "ai" and not getattr(b, "model"):
            raise ValueError("ai block requires 'model'")
        if b.type == "python" and not getattr(b, "function"):
            raise ValueError("python block requires 'function'")
        if b.type == "end" and not getattr(b, "final", None):
            # allow empty but warn
            pass
    return cfg
