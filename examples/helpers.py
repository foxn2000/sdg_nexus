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
    """プロンプト進化エージェント用の整形関数（リトライ機能対応）"""
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


def format_question_generator(ctx, **inputs) -> dict:
    """
    LLMベンチマーク用質問生成エージェントの整形関数

    Args:
        ctx: 実行コンテキスト
        inputs: 以下のキーを含む辞書
            - source_text: 元の資料テキスト
            - original_question: 原型質問
            - diversified_questions: 多角化された質問リスト（4つ）
            - evolved_questions: 進化させた質問リスト（4つ）
            - selected_question: 最終選定された質問
            - selection_reason: 選定理由
            - thinking_logs: 各ステップの思考ログ

    Returns:
        dict: FormattedResult キーを含む辞書
    """
    source_text = inputs.get("source_text", "")
    original_question = inputs.get("original_question", "")
    diversified_questions = inputs.get("diversified_questions", "")
    evolved_questions = inputs.get("evolved_questions", "")
    selected_question = inputs.get("selected_question", "")
    selection_reason = inputs.get("selection_reason", "")

    # 思考ログ
    step1_thinking = inputs.get("step1_thinking", "")
    step2_thinking = inputs.get("step2_thinking", "")
    step3_thinking = inputs.get("step3_thinking", "")
    step4_thinking = inputs.get("step4_thinking", "")

    def truncate(text, max_len=300):
        """テキストを指定長で切り詰め"""
        text = str(text) if text else ""
        return text[:max_len] + "..." if len(text) > max_len else text

    # ソーステキストのプレビュー
    source_preview = truncate(source_text, 200)

    result = f"""
================================================================================
                    LLMベンチマーク用質問生成 - 実行結果
================================================================================

【入力資料（抜粋）】
{source_preview}

--------------------------------------------------------------------------------
【ステップ1: 質問の原型作成】
思考プロセス: {truncate(step1_thinking)}

原型質問:
{original_question}

--------------------------------------------------------------------------------
【ステップ2: 質問の多角化】
思考プロセス: {truncate(step2_thinking)}

多角化された質問:
{diversified_questions}

--------------------------------------------------------------------------------
【ステップ3: 質問の進化（難易度調整・人間らしさ付与）】
思考プロセス: {truncate(step3_thinking)}

進化後の質問:
{evolved_questions}

--------------------------------------------------------------------------------
【ステップ4: 検証と選定】
思考プロセス: {truncate(step4_thinking)}

選定理由:
{selection_reason}

================================================================================
                           最終選定質問
================================================================================
{selected_question}

================================================================================
"""

    ctx.log("info", f"質問生成完了: {truncate(selected_question, 50)}")

    return {"FormattedResult": result.strip()}


def format_question_output_jsonl(ctx, **inputs) -> dict:
    """
    質問生成結果をJSONL出力形式に整形する関数

    Args:
        ctx: 実行コンテキスト
        inputs: 生成された質問と関連データ

    Returns:
        dict: JSONL出力用のデータ
    """
    import json

    source_text = inputs.get("source_text", "")
    original_question = inputs.get("original_question", "")
    diversified_questions_raw = inputs.get("diversified_questions", "")
    evolved_questions_raw = inputs.get("evolved_questions", "")
    selected_question = inputs.get("selected_question", "")
    selection_reason = inputs.get("selection_reason", "")

    # 質問リストをパース（文字列の場合）
    def parse_questions(q_str):
        if isinstance(q_str, list):
            return q_str
        if isinstance(q_str, str):
            # 番号付きリストを解析
            lines = q_str.strip().split("\n")
            questions = []
            for line in lines:
                line = line.strip()
                # "1. ", "2. ", etc. または "- " で始まる行を抽出
                if line and (line[0].isdigit() or line.startswith("-")):
                    # 番号やダッシュを除去
                    cleaned = line.lstrip("0123456789.-) ").strip()
                    if cleaned:
                        questions.append(cleaned)
            return questions if questions else [q_str]
        return []

    diversified_list = parse_questions(diversified_questions_raw)
    evolved_list = parse_questions(evolved_questions_raw)

    output_data = {
        "final_question": selected_question.strip() if selected_question else "",
        "original_question": original_question.strip() if original_question else "",
        "diversified_questions": diversified_list,
        "evolved_questions": evolved_list,
        "selection_reason": selection_reason.strip() if selection_reason else "",
        "source_text_preview": (
            (source_text[:500] + "...")
            if len(str(source_text)) > 500
            else str(source_text)
        ),
    }

    ctx.log("info", f"JSONL出力データを生成しました")

    return {
        "OutputData": output_data,
        "FinalQuestion": selected_question.strip() if selected_question else "",
    }
