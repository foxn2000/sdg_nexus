from __future__ import annotations
import os, yaml, json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .utils import ensure_json_obj


@dataclass
class RuntimeConfig:
    """v2: 実行時環境設定"""

    python: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RuntimeConfig:
        return cls(python=d.get("python", {}))


@dataclass
class BudgetConfig:
    """v2: 予算（安全停止）設定"""

    loops: Dict[str, Any] = field(default_factory=dict)
    recursion: Dict[str, Any] = field(default_factory=dict)
    wall_time_ms: Optional[int] = None
    ai: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BudgetConfig:
        return cls(
            loops=d.get("loops", {"max_iters": 10000, "on_exceed": "error"}),
            recursion=d.get("recursion", {"max_depth": 256, "on_exceed": "error"}),
            wall_time_ms=d.get("wall_time_ms"),
            ai=d.get("ai", {}),
        )


@dataclass
class GlobalsConfig:
    """v2: グローバル変数/定数"""

    const: Dict[str, Any] = field(default_factory=dict)
    vars: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GlobalsConfig:
        return cls(const=d.get("const", {}), vars=d.get("vars", {}))


@dataclass
class FunctionDef:
    """v2: ユーザ定義関数"""

    name: str
    args: List[str] = field(default_factory=list)
    returns: List[str] = field(default_factory=list)
    body: Any = None  # logic関数の場合はステップリスト、python関数の場合はコード

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FunctionDef:
        return cls(
            name=d["name"],
            args=d.get("args", []),
            returns=d.get("returns", []),
            body=d.get("body"),
        )


@dataclass
class ModelDef:
    name: str
    api_model: str
    api_key: str
    base_url: Optional[str] = None
    organization: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_defaults: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)  # v2
    safety: Dict[str, Any] = field(default_factory=dict)  # v2


@dataclass
class OutputDef:
    name: str
    select: str = "full"  # full | tag | regex | jsonpath (v2)
    tag: Optional[str] = None
    regex: Optional[str] = None
    path: Optional[str] = None  # v2: jsonpath
    join_with: Optional[str] = None
    type_hint: Optional[str] = None  # v2: string|number|boolean|json
    # logic出力用
    from_: Optional[str] = (
        None  # boolean|value|join|count|any|all|first|last|list|var|accumulator
    )
    var: Optional[str] = None
    source: Optional[str] = None


@dataclass
class Block:
    type: str
    exec: int
    id: Optional[str] = None  # v2: 明示ID
    name: Optional[str] = None
    run_if: Any = None
    on_error: str = "fail"  # fail | continue | retry (v2)
    retry: Optional[Dict[str, Any]] = None  # v2
    budget: Optional[Dict[str, Any]] = None  # v2: ブロック局所予算


@dataclass
class AIBlock(Block):
    model: str = ""
    system_prompt: Optional[str] = None
    prompts: List[str] = field(default_factory=list)
    outputs: List[OutputDef] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)  # v2
    mode: str = "text"  # text | json (v2)
    save_to: Optional[Dict[str, Any]] = None  # v2: 変数保存


@dataclass
class LogicBlock(Block):
    op: str = "if"  # if|and|or|not|for|while|recurse|set|let|reduce|call|emit (v2拡張)
    # if/and/or/not
    cond: Any = None
    then: Optional[str] = None
    else_: Optional[str] = None
    operands: Optional[List[Any]] = None
    # for
    list: Optional[str] = None
    parse: Optional[str] = None  # lines|csv|json|regex
    regex_pattern: Optional[str] = None
    var: Optional[str] = None
    drop_empty: Optional[bool] = None
    where: Any = None
    map: Optional[str] = None
    # while (v2)
    init: Optional[List[Dict[str, Any]]] = None
    step: Optional[List[Dict[str, Any]]] = None
    # recurse (v2)
    function: Optional[Dict[str, Any]] = None
    with_: Optional[Dict[str, Any]] = None  # 'with'は予約語なのでwith_
    returns: Optional[List[str]] = None
    # set/let (v2)
    value: Any = None
    bindings: Optional[Dict[str, Any]] = None
    body: Optional[List[Dict[str, Any]]] = None
    # 共通
    outputs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PyBlock(Block):
    function: str = ""  # v1互換
    entrypoint: Optional[str] = None  # v2: functionと同義
    inputs: Any = field(default_factory=list)  # v2: list or dict
    code_path: Optional[str] = None
    function_code: Optional[str] = None  # v2: インラインコード
    venv_path: Optional[str] = None  # v1互換（非推奨）
    outputs: List[str] = field(default_factory=list)
    # v2拡張
    use_env: str = "global"  # global | override
    override_env: Optional[Dict[str, Any]] = None
    timeout_ms: Optional[int] = None
    ctx_access: List[str] = field(default_factory=list)


@dataclass
class EndBlock(Block):
    reason: Optional[str] = None
    exit_code: Optional[str] = None
    final: List[Dict[str, str]] = field(default_factory=list)
    final_mode: str = "map"  # v2: map | list
    include_vars: List[str] = field(default_factory=list)  # v2


@dataclass
class TemplateDef:
    """v2: 文字列テンプレート"""

    name: str
    text: str


@dataclass
class FileDef:
    """v2: 埋め込みファイル"""

    name: str
    mime: str
    content: str


@dataclass
class ImageDef:
    """v2.1: 画像定義"""

    name: str
    path: Optional[str] = None  # ローカルファイルパス
    url: Optional[str] = None  # URL
    base64: Optional[str] = None  # Base64エンコード済み
    media_type: str = "image/png"  # MIMEタイプ

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageDef":
        return cls(
            name=d["name"],
            path=d.get("path"),
            url=d.get("url"),
            base64=d.get("base64"),
            media_type=d.get("media_type", "image/png"),
        )


@dataclass
class Connection:
    """明示配線"""

    from_: str  # ブロックID
    output: str
    to: str  # ブロックID
    input: str


@dataclass
class SDGConfig:
    mabel: Dict[str, Any] = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)  # v2
    budgets: BudgetConfig = field(default_factory=BudgetConfig)  # v2
    globals_: GlobalsConfig = field(default_factory=GlobalsConfig)  # v2
    functions: Dict[str, List[FunctionDef]] = field(default_factory=dict)  # v2
    models: List[ModelDef] = field(default_factory=list)
    templates: List[TemplateDef] = field(default_factory=list)  # v2
    files: List[FileDef] = field(default_factory=list)  # v2
    images: List[ImageDef] = field(default_factory=list)  # v2.1: 画像定義
    blocks: List[Block] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)

    def model_by_name(self, name: str) -> ModelDef:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"Model not found: {name}")

    def image_by_name(self, name: str) -> Optional[ImageDef]:
        """画像定義を名前で取得"""
        for img in self.images:
            if img.name == name:
                return img
        return None

    def get_version(self) -> str:
        """MABELバージョンを取得"""
        return self.mabel.get("version", "1.0")

    def is_v2(self) -> bool:
        """v2仕様かどうか（v2.1も含む）"""
        version = self.get_version()
        return version.startswith("2.")


def _normalize_output(d: Dict[str, Any]) -> OutputDef:
    """出力定義を正規化"""
    return OutputDef(
        name=d["name"],
        select=d.get("select", "full"),
        tag=d.get("tag"),
        regex=d.get("regex"),
        path=d.get("path"),
        join_with=d.get("join_with"),
        type_hint=d.get("type_hint"),
        from_=d.get("from"),
        var=d.get("var"),
        source=d.get("source"),
    )


def _normalize_block(d: Dict[str, Any]) -> Block:
    typ = d.get("type")
    common = {
        "type": typ,
        "exec": int(d.get("exec", 0)),
        "id": d.get("id"),
        "name": d.get("name"),
        "run_if": ensure_json_obj(d.get("run_if")),
        "on_error": d.get("on_error", "fail"),
        "retry": d.get("retry"),
        "budget": d.get("budget"),
    }

    if typ == "ai":
        outs = [_normalize_output(o) for o in d.get("outputs", [])]
        return AIBlock(
            outputs=outs,
            model=d.get("model", ""),
            system_prompt=d.get("system_prompt"),
            prompts=list(d.get("prompts", [])),
            params=d.get("params", {}),
            attachments=d.get("attachments", []),
            mode=d.get("mode", "text"),
            save_to=d.get("save_to"),
            **common,
        )

    if typ == "logic":
        return LogicBlock(
            op=d.get("op", "if"),
            cond=ensure_json_obj(d.get("cond")),
            then=d.get("then"),
            else_=d.get("else"),
            operands=d.get("operands"),
            list=d.get("list"),
            parse=d.get("parse"),
            regex_pattern=d.get("regex_pattern"),
            var=d.get("var"),
            drop_empty=d.get("drop_empty"),
            where=ensure_json_obj(d.get("where")),
            map=d.get("map"),
            init=d.get("init"),
            step=d.get("step"),
            function=d.get("function"),
            with_=d.get("with"),
            returns=d.get("returns"),
            value=d.get("value"),
            bindings=d.get("bindings"),
            body=d.get("body"),
            outputs=d.get("outputs", []),
            **common,
        )

    if typ == "python":
        return PyBlock(
            function=d.get("function", ""),
            entrypoint=d.get("entrypoint"),
            inputs=d.get("inputs", []),
            code_path=d.get("code_path"),
            function_code=d.get("function_code"),
            venv_path=d.get("venv_path"),
            outputs=d.get("outputs", []),
            use_env=d.get("use_env", "global"),
            override_env=d.get("override_env"),
            timeout_ms=d.get("timeout_ms"),
            ctx_access=d.get("ctx_access", []),
            **common,
        )

    if typ == "end":
        return EndBlock(
            reason=d.get("reason"),
            exit_code=d.get("exit_code"),
            final=d.get("final", []),
            final_mode=d.get("final_mode", "map"),
            include_vars=d.get("include_vars", []),
            **common,
        )

    raise ValueError(f"Unsupported block type: {typ}")


def load_config(path: str) -> SDGConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # MABEL情報
    mabel = data.get("mabel", {})
    version = mabel.get("version", "1.0")

    # v2拡張フィールド
    runtime = RuntimeConfig.from_dict(data.get("runtime", {}))
    budgets = BudgetConfig.from_dict(data.get("budgets", {}))
    globals_ = GlobalsConfig.from_dict(data.get("globals", {}))

    # 関数定義
    functions = {}
    if "functions" in data:
        funcs_data = data["functions"]
        if "logic" in funcs_data:
            functions["logic"] = [FunctionDef.from_dict(f) for f in funcs_data["logic"]]
        if "python" in funcs_data:
            functions["python"] = [
                FunctionDef.from_dict(f) for f in funcs_data["python"]
            ]

    # モデル
    models = [ModelDef(**m) for m in data.get("models", [])]

    # テンプレート
    templates = [TemplateDef(**t) for t in data.get("templates", [])]

    # ファイル
    files = [FileDef(**f) for f in data.get("files", [])]

    # 画像（v2.1）
    images = [ImageDef.from_dict(img) for img in data.get("images", [])]

    # ブロック
    blocks = [_normalize_block(b) for b in data.get("blocks", [])]
    blocks = sorted(blocks, key=lambda b: b.exec)

    # 接続
    connections = []
    for c in data.get("connections", []):
        connections.append(
            Connection(
                from_=c["from"], output=c["output"], to=c["to"], input=c["input"]
            )
        )

    cfg = SDGConfig(
        mabel=mabel,
        runtime=runtime,
        budgets=budgets,
        globals_=globals_,
        functions=functions,
        models=models,
        templates=templates,
        files=files,
        images=images,
        blocks=blocks,
        connections=connections,
    )

    # 基本検証
    for b in cfg.blocks:
        if b.type == "ai" and not isinstance(b, AIBlock):
            raise ValueError("Block casting failed for ai")
        if b.type == "ai" and not getattr(b, "model"):
            raise ValueError("ai block requires 'model'")
        if b.type == "python":
            if not (getattr(b, "function") or getattr(b, "entrypoint")):
                raise ValueError("python block requires 'function' or 'entrypoint'")

    return cfg
