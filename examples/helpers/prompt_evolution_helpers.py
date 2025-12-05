"""
SDG Nexus - プロンプト進化エージェント用ヘルパー関数

対応YAML: examples/prompt_evolution_agent.yaml

このファイルには、Prompt Evolution Agentで使用される全てのヘルパー関数が含まれています。
"""


def format_prompt_evolution(ctx, **inputs) -> dict:
    """
    プロンプト進化エージェント用の整形関数（リトライ機能対応）

    Args:
        ctx: 実行コンテキスト
        inputs: 以下のキーを含む辞書
            - original_input: 元の入力
            - generated_prompt: 生成されたプロンプト
            - enhanced_prompt: 難易度を高めたプロンプト
            - is_valid: 妥当性チェック結果
            - final_answer: 最終回答
            - retry_count: リトライ回数（オプション）
            - thinking_logs: 各ステップの思考ログ

    Returns:
        dict: FormattedResult キーを含む辞書
    """
    original_input = inputs.get("original_input", "")
    generated_prompt = inputs.get("generated_prompt", "")
    enhanced_prompt = inputs.get("enhanced_prompt", "")
    is_valid = inputs.get("is_valid", "")
    final_answer = inputs.get("final_answer", "")

    # リトライ回数
    retry_count = inputs.get("retry_count", "1")
    try:
        retry_count_int = int(retry_count) if retry_count else 1
    except (ValueError, TypeError):
        retry_count_int = 1

    # 思考プロセスの抽出（存在する場合）
    generate_thinking = inputs.get("generate_thinking", "")
    enhance_thinking = inputs.get("enhance_thinking", "")
    validate_thinking = inputs.get("validate_thinking", "")
    answer_thinking = inputs.get("answer_thinking", "")

    def truncate(text, max_len=200):
        text = str(text) if text else ""
        return text[:max_len] + "..." if len(text) > max_len else text

    # リトライ情報の表示
    retry_info = ""
    if retry_count_int > 1:
        retry_info = f"\n※ 妥当性チェックに失敗したため、{retry_count_int}回目の試行で成功しました。\n"
    elif retry_count_int == 1:
        retry_info = "\n※ 1回目の試行で妥当性チェックに成功しました。\n"

    result = f"""
================================================================================
                        プロンプト進化エージェント - 実行結果
================================================================================
{retry_info}
【元の入力】
{original_input}

--------------------------------------------------------------------------------
【ステップ1: プロンプト生成】（実行回数: {retry_count_int}）
思考プロセス: {truncate(generate_thinking)}

生成されたプロンプト:
{generated_prompt}

--------------------------------------------------------------------------------
【ステップ2: 難易度向上】（実行回数: {retry_count_int}）
思考プロセス: {truncate(enhance_thinking)}

難易度を高めたプロンプト:
{enhanced_prompt}

--------------------------------------------------------------------------------
【ステップ3: 妥当性チェック】（実行回数: {retry_count_int}）
思考プロセス: {truncate(validate_thinking)}

妥当性: {is_valid}

--------------------------------------------------------------------------------
【ステップ4: 最終回答】
思考プロセス: {truncate(answer_thinking)}

最終回答:
{final_answer}

================================================================================
                           実行統計
================================================================================
- 総試行回数: {retry_count_int}
- 最大リトライ回数: 2
- 最終妥当性: {is_valid}
================================================================================
"""

    ctx.log("info", f"フォーマット処理完了 (試行回数: {retry_count_int})")

    return {"FormattedResult": result.strip()}
