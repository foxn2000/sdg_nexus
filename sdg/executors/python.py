from __future__ import annotations
import sys, os, importlib.util
from typing import Any, Dict

from ..config import PyBlock, SDGConfig
from ..utils import render_template
from .core import ExecutionContext


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


def _load_python_function(block: PyBlock) -> Any:
    """Pythonブロックの関数をロード"""
    fn_name = block.entrypoint or block.function
    if not fn_name:
        raise ValueError("python block requires 'function' or 'entrypoint'")

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

    return fn


def _execute_python_block_single(
    block: PyBlock,
    ctx: Dict[str, Any],
    cfg: SDGConfig,
    exec_ctx: ExecutionContext,
    fn: Any,
) -> Dict[str, Any]:
    """単一行のPythonブロック実行"""
    # v2: ctxオブジェクト
    py_ctx = PythonContext(exec_ctx, ctx)

    # 引数準備
    if isinstance(block.inputs, dict):
        # v2: キーワード引数（テンプレート展開をサポート）
        kwargs = {}
        for k, v in block.inputs.items():
            if isinstance(v, str):
                # テンプレート形式の場合は展開
                kwargs[k] = render_template(v, ctx)
            else:
                kwargs[k] = v
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
                out if isinstance(out, (list, tuple)) else [out],
            )
        }

    return out_map
