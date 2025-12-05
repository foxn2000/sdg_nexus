"""
SDG Nexus - 質問生成エージェント用ヘルパー関数

対応YAML: examples/question_generator_agent.yaml

このファイルには、Question Generator Agentで使用される全てのヘルパー関数が含まれています。
"""


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

    ctx.log("info", "JSONL出力データを生成しました")

    return {
        "OutputData": output_data,
        "FinalQuestion": selected_question.strip() if selected_question else "",
    }
