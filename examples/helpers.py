def format_output(answer: str, flag: bool):
    short = (answer or "").split("\n", 1)[0]
    return {"Formatted": f"[SHORT]\n{short}\n\n[LONG]\n{answer}\n\n[FLAG]\n{flag}"}


def format_comprehensive_result(ctx, **inputs) -> dict:
    """sdg_comprehensive_v2用の整形関数"""
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


def format_prompt_evolution(ctx, **inputs) -> dict:
    """プロンプト進化エージェント用の整形関数"""
    original_input = inputs.get("original_input", "")
    generated_prompt = inputs.get("generated_prompt", "")
    enhanced_prompt = inputs.get("enhanced_prompt", "")
    is_valid = inputs.get("is_valid", "")
    final_answer = inputs.get("final_answer", "")

    # 思考プロセスの抽出（存在する場合）
    generate_thinking = inputs.get("generate_thinking", "")
    enhance_thinking = inputs.get("enhance_thinking", "")
    validate_thinking = inputs.get("validate_thinking", "")
    answer_thinking = inputs.get("answer_thinking", "")

    def truncate(text, max_len=200):
        text = str(text) if text else ""
        return text[:max_len] + "..." if len(text) > max_len else text

    result = f"""
================================================================================
                        プロンプト進化エージェント - 実行結果
================================================================================

【元の入力】
{original_input}

--------------------------------------------------------------------------------
【ステップ1: プロンプト生成】
思考プロセス: {truncate(generate_thinking)}

生成されたプロンプト:
{generated_prompt}

--------------------------------------------------------------------------------
【ステップ2: 難易度向上】
思考プロセス: {truncate(enhance_thinking)}

難易度を高めたプロンプト:
{enhanced_prompt}

--------------------------------------------------------------------------------
【ステップ3: 妥当性チェック】
思考プロセス: {truncate(validate_thinking)}

妥当性: {is_valid}

--------------------------------------------------------------------------------
【ステップ4: 最終回答】
思考プロセス: {truncate(answer_thinking)}

最終回答:
{final_answer}

================================================================================
"""

    ctx.log("info", "フォーマット処理完了")

    return {"FormattedResult": result.strip()}
