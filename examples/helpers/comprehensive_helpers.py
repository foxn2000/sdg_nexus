"""
SDG Nexus - 包括的デモ用ヘルパー関数

対応YAML: examples/sdg_comprehensive_v2.yaml

このファイルには、Comprehensive Demoで使用される全てのヘルパー関数が含まれています。
"""


def format_comprehensive_result(ctx, **inputs) -> dict:
    """
    sdg_comprehensive_v2用の整形関数

    Args:
        ctx: 実行コンテキスト
        inputs: 以下のキーを含む辞書
            - total: 合計値
            - fib: フィボナッチ数
            - steps: ステップのリスト
            - category: カテゴリ
            - stats: 統計情報

    Returns:
        dict: FormattedOutput キーを含む辞書
    """
    total = inputs.get("total", 0)
    fib = inputs.get("fib", 0)
    steps = inputs.get("steps", [])
    category = inputs.get("category", "unknown")
    stats = inputs.get("stats", {})

    result = f"""
=== Comprehensive Result ===
Total: {total}
Fibonacci: {fib}
Steps: {len(steps) if isinstance(steps, list) else 0}
Category: {category}
Statistics: {stats}
"""
    return {"FormattedOutput": result.strip()}


def format_output(answer: str, flag: bool):
    """
    基本的な出力フォーマット関数

    Args:
        answer: 回答テキスト
        flag: フラグ

    Returns:
        dict: Formatted キーを含む辞書
    """
    short = (answer or "").split("\n", 1)[0]
    return {"Formatted": f"[SHORT]\n{short}\n\n[LONG]\n{answer}\n\n[FLAG]\n{flag}"}
