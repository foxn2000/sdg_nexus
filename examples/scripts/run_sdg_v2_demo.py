#!/usr/bin/env python3
"""
SDG Nexus - MABEL v2.0 機能デモ

このスクリプトはMABEL v2.0の新機能（MEX式、グローバル変数、While、インラインPython等）を
Python APIから使用する方法を示します。

使用方法:
    python examples/scripts/run_sdg_v2_demo.py
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sdg.config import load_config, SDGConfig
from sdg.executors import run_pipeline, ExecutionContext
from sdg.mex import eval_mex


async def demo_comprehensive_v2():
    """
    sdg_comprehensive_v2.yaml を使用した包括的なデモ
    """
    examples_dir = Path(__file__).parent.parent
    yaml_path = examples_dir / "sdg_comprehensive_v2.yaml"

    print("\n" + "=" * 60)
    print("sdg_comprehensive_v2.yaml デモ")
    print("=" * 60)

    if not yaml_path.exists():
        print(f"ファイルが見つかりません: {yaml_path}")
        return

    cfg = load_config(str(yaml_path))

    print(f"バージョン: {cfg.get_version()}")
    print(f"ID: {cfg.mabel.get('id', 'N/A')}")
    print(f"名前: {cfg.mabel.get('name', 'N/A')}")

    # グローバル変数の確認
    print("\nグローバル定数:")
    for k, v in cfg.globals_.const.items():
        print(f"  {k}: {v}")

    print("\nグローバル変数:")
    for k, v in cfg.globals_.vars.items():
        print(f"  {k}: {v}")

    # 予算設定の確認
    print("\n予算設定:")
    print(f"  ループ上限: {cfg.budgets.loops.get('max_iters', 'N/A')}")
    print(f"  再帰深度上限: {cfg.budgets.recursion.get('max_depth', 'N/A')}")
    print(f"  実行時間上限: {cfg.budgets.wall_time_ms or 'N/A'}ms")

    # ユーザー定義関数の確認
    if cfg.functions:
        print("\nユーザー定義関数:")
        for func_type, funcs in cfg.functions.items():
            for f in funcs:
                print(f"  [{func_type}] {f.name}({', '.join(f.args)}) -> {f.returns}")

    # ブロック構成の確認
    print("\nブロック構成:")
    for b in cfg.blocks:
        info = f"  exec={b.exec:3d} | type={b.type:8s}"
        if hasattr(b, "op"):
            info += f" | op={b.op}"
        if b.name:
            info += f" | name={b.name}"
        print(info)


def demo_mex_expressions():
    """
    MEX式の評価デモ
    """
    print("\n" + "=" * 60)
    print("MEX式 評価デモ")
    print("=" * 60)

    # コンテキスト（ローカル変数）
    context = {
        "x": 10,
        "y": 5,
        "name": "World",
        "items": [1, 2, 3, 4, 5],
        "Status": "ok",
    }

    # グローバル変数
    global_vars = {"counter": 100, "multiplier": 2}

    # 算術演算
    print("\n【算術演算】")
    expressions = [
        ({"add": [1, 2, 3]}, "1 + 2 + 3"),
        ({"mul": [{"var": "x"}, {"var": "y"}]}, "x * y"),
        ({"sub": [{"var": "x"}, 3]}, "x - 3"),
        ({"div": [20, 4]}, "20 / 4"),
        ({"mod": [{"var": "x"}, 3]}, "x % 3"),
    ]

    for expr, desc in expressions:
        try:
            result = eval_mex(expr, context, global_vars)
            print(f"  {desc} = {result}  (expr: {expr})")
        except Exception as e:
            print(f"  {desc} -> エラー: {e}")

    # 比較演算
    print("\n【比較演算】")
    comparisons = [
        ({"gt": [{"var": "x"}, 5]}, "x > 5"),
        ({"lt": [{"var": "y"}, 10]}, "y < 10"),
        ({"eq": ["{Status}", "ok"]}, "Status == 'ok'"),
        ({"ne": ["{Status}", "error"]}, "Status != 'error'"),
        ({"gte": [{"var": "x"}, 10]}, "x >= 10"),
        ({"lte": [{"var": "y"}, 5]}, "y <= 5"),
    ]

    for expr, desc in comparisons:
        try:
            result = eval_mex(expr, context, global_vars)
            print(f"  {desc} = {result}")
        except Exception as e:
            print(f"  {desc} -> エラー: {e}")

    # 論理演算
    print("\n【論理演算】")
    logic_exprs = [
        (
            {"and": [{"gt": [{"var": "x"}, 5]}, {"lt": [{"var": "y"}, 10]}]},
            "x > 5 AND y < 10",
        ),
        (
            {"or": [{"eq": ["{Status}", "ok"]}, {"eq": ["{Status}", "pending"]}]},
            "Status == 'ok' OR Status == 'pending'",
        ),
        ({"not": {"eq": ["{Status}", "error"]}}, "NOT Status == 'error'"),
    ]

    for expr, desc in logic_exprs:
        try:
            result = eval_mex(expr, context, global_vars)
            print(f"  {desc} = {result}")
        except Exception as e:
            print(f"  {desc} -> エラー: {e}")

    # 文字列操作
    print("\n【文字列操作】")
    string_exprs = [
        ({"concat": ["Hello, ", {"var": "name"}, "!"]}, "concat('Hello, ', name, '!')"),
        ({"upper": ["hello"]}, "upper('hello')"),
        ({"lower": ["HELLO"]}, "lower('HELLO')"),
        ({"length": ["Hello World"]}, "length('Hello World')"),
    ]

    for expr, desc in string_exprs:
        try:
            result = eval_mex(expr, context, global_vars)
            print(f"  {desc} = {result}")
        except Exception as e:
            print(f"  {desc} -> エラー: {e}")

    # コレクション操作
    print("\n【コレクション操作】")
    collection_exprs = [
        ({"length": [{"var": "items"}]}, "length(items)"),
        ({"get": {"list": {"var": "items"}, "index": 0}}, "items[0]"),
    ]

    for expr, desc in collection_exprs:
        try:
            result = eval_mex(expr, context, global_vars)
            print(f"  {desc} = {result}")
        except Exception as e:
            print(f"  {desc} -> エラー: {e}")

    # 条件分岐
    print("\n【条件分岐】")
    if_expr = {
        "if": {"cond": {"gt": [{"var": "x"}, 5]}, "then": "大きい", "else": "小さい"}
    }
    try:
        result = eval_mex(if_expr, context, global_vars)
        print(f"  if x > 5 then '大きい' else '小さい' = {result}")
    except Exception as e:
        print(f"  条件分岐 -> エラー: {e}")


async def demo_simple_pipeline():
    """
    シンプルなパイプライン実行デモ（LLM不要）
    """
    print("\n" + "=" * 60)
    print("シンプルパイプライン デモ（LLM不要）")
    print("=" * 60)

    examples_dir = Path(__file__).parent.parent
    yaml_path = examples_dir / "sdg_demo_v2.yaml"

    if not yaml_path.exists():
        print(f"ファイルが見つかりません: {yaml_path}")
        return

    cfg = load_config(str(yaml_path))

    # 実行コンテキストを作成して直接操作
    exec_ctx = ExecutionContext(cfg)

    print("\n初期状態:")
    print(f"  グローバル変数: {exec_ctx.globals_vars}")
    print(f"  グローバル定数: {exec_ctx.globals_const}")

    # 変数を設定
    exec_ctx.set_global("counter", 42)
    print(f"\n変数を設定後:")
    print(f"  counter = {exec_ctx.get_global('counter')}")

    # 定数にアクセス
    print(f"  APP_NAME = {exec_ctx.get_global('APP_NAME')}")
    print(f"  MAX_ITERATIONS = {exec_ctx.get_global('MAX_ITERATIONS')}")

    # MEX式を評価
    from sdg.mex import eval_mex

    expr = {"add": [{"var": "counter"}, {"var": "MAX_ITERATIONS"}]}
    result = eval_mex(expr, {}, exec_ctx.globals_vars)
    print(f"\nMEX評価: counter + MAX_ITERATIONS = {result}")


async def main():
    """メイン関数"""
    print("=" * 60)
    print("SDG Nexus - MABEL v2.0 機能デモ")
    print("=" * 60)

    # MEX式のデモ
    demo_mex_expressions()

    # シンプルなパイプラインデモ
    await demo_simple_pipeline()

    # 包括的なv2デモ（設定確認のみ）
    await demo_comprehensive_v2()

    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())