"""
SDG Nexus - フォーマットヘルパー関数

YAMLパイプラインから呼び出されるPythonヘルパー関数群。
各関数はctxコンテキストと**inputsを受け取り、出力辞書を返す。
"""


def format_output(answer: str, flag: bool):
    """基本的な出力フォーマット関数"""
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


def split_diversified_questions(ctx, **inputs) -> dict:
    """
    多角化された質問を4つの個別質問に分離する関数

    Args:
        ctx: 実行コンテキスト
        inputs:
            - diversified_questions: 多角化された4つの質問（番号付きリスト形式）

    Returns:
        dict: Question1, Question2, Question3, Question4 を含む辞書
    """
    diversified_questions = inputs.get("diversified_questions", "")

    def parse_numbered_list(text):
        """番号付きリストを解析して質問を抽出"""
        if not text:
            return ["", "", "", ""]

        lines = str(text).strip().split("\n")
        questions = []
        current_question = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 新しい番号付き項目の検出 (1., 2., 3., 4. など)
            if (
                line
                and len(line) >= 2
                and line[0].isdigit()
                and (line[1] == "." or (len(line) >= 3 and line[1].isdigit() and line[2] == "."))
            ):
                if current_question:
                    questions.append(" ".join(current_question))
                # 番号と視点ラベルを除去
                cleaned = line.lstrip("0123456789. ")
                # [視点A] などのラベルを除去
                if cleaned.startswith("["):
                    bracket_end = cleaned.find("]")
                    if bracket_end != -1:
                        cleaned = cleaned[bracket_end + 1 :].strip()
                current_question = [cleaned] if cleaned else []
            elif current_question:
                current_question.append(line)

        if current_question:
            questions.append(" ".join(current_question))

        # 4つに満たない場合は空文字で埋める
        while len(questions) < 4:
            questions.append("")

        return questions[:4]

    questions = parse_numbered_list(diversified_questions)

    ctx.log("info", f"質問を4つに分離しました: {[q[:30] + '...' if len(q) > 30 else q for q in questions]}")

    return {
        "Question1": questions[0],
        "Question2": questions[1],
        "Question3": questions[2],
        "Question4": questions[3],
    }


def format_question_generator_v2(ctx, **inputs) -> dict:
    """
    LLMベンチマーク用質問生成エージェントv2の整形関数
    4つの質問を個別に評価した結果を整形

    Args:
        ctx: 実行コンテキスト
        inputs:
            - source_text: 元の資料テキスト
            - original_question: 原型質問
            - diversified_questions: 多角化された質問リスト
            - evolved_question1-4: 進化させた4つの質問
            - is_valid1-4: 各質問の使用可否判定
            - evaluation1-4: 各質問の評価詳細
            - thinking_logs: 各ステップの思考ログ

    Returns:
        dict: FormattedResult キーを含む辞書
    """
    source_text = inputs.get("source_text", "")
    original_question = inputs.get("original_question", "")
    diversified_questions = inputs.get("diversified_questions", "")

    # 4つの進化した質問と評価
    evolved1 = inputs.get("evolved_question1", "")
    evolved2 = inputs.get("evolved_question2", "")
    evolved3 = inputs.get("evolved_question3", "")
    evolved4 = inputs.get("evolved_question4", "")

    is_valid1 = inputs.get("is_valid1", "")
    is_valid2 = inputs.get("is_valid2", "")
    is_valid3 = inputs.get("is_valid3", "")
    is_valid4 = inputs.get("is_valid4", "")

    eval1 = inputs.get("evaluation1", "")
    eval2 = inputs.get("evaluation2", "")
    eval3 = inputs.get("evaluation3", "")
    eval4 = inputs.get("evaluation4", "")

    # 思考ログ
    step1_thinking = inputs.get("step1_thinking", "")
    step2_thinking = inputs.get("step2_thinking", "")

    def truncate(text, max_len=300):
        text = str(text) if text else ""
        return text[:max_len] + "..." if len(text) > max_len else text

    def format_validity(is_valid):
        """使用可否を日本語表記に変換"""
        valid_str = str(is_valid).lower().strip()
        if "true" in valid_str or valid_str == "使用可能":
            return "✅ 使用可能"
        else:
            return "❌ 使用不可"

    # 使用可能な質問を収集
    valid_questions = []
    for i, (evolved, is_valid) in enumerate(
        [
            (evolved1, is_valid1),
            (evolved2, is_valid2),
            (evolved3, is_valid3),
            (evolved4, is_valid4),
        ],
        1,
    ):
        valid_str = str(is_valid).lower().strip()
        if "true" in valid_str or valid_str == "使用可能":
            valid_questions.append((i, evolved))

    valid_count = len(valid_questions)
    source_preview = truncate(source_text, 200)

    result = f"""
================================================================================
              LLMベンチマーク用質問生成 v2 - 実行結果
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
【ステップ3 & 4: 個別進化・評価結果】

▼ 質問1（原因・理由の視点）
進化後: {truncate(evolved1)}
判定: {format_validity(is_valid1)}
評価: {truncate(eval1, 150)}

▼ 質問2（方法・プロセスの視点）
進化後: {truncate(evolved2)}
判定: {format_validity(is_valid2)}
評価: {truncate(eval2, 150)}

▼ 質問3（比較・対照の視点）
進化後: {truncate(evolved3)}
判定: {format_validity(is_valid3)}
評価: {truncate(eval3, 150)}

▼ 質問4（影響・結果の視点）
進化後: {truncate(evolved4)}
判定: {format_validity(is_valid4)}
評価: {truncate(eval4, 150)}

================================================================================
                         使用可能な質問一覧（{valid_count}/4）
================================================================================
"""

    if valid_questions:
        for i, q in valid_questions:
            result += f"\n【質問{i}】\n{q}\n"
    else:
        result += "\n※ 使用可能な質問はありませんでした。\n"

    result += "\n================================================================================\n"

    ctx.log("info", f"質問生成v2完了: 使用可能 {valid_count}/4")

    return {"FormattedResult": result.strip()}


def format_question_output_jsonl_v2(ctx, **inputs) -> dict:
    """
    v2: 質問生成結果をJSONL出力形式に整形する関数
    各質問の個別評価結果を含む

    Args:
        ctx: 実行コンテキスト
        inputs: 生成された質問と関連データ

    Returns:
        dict: JSONL出力用のデータ
    """
    source_text = inputs.get("source_text", "")
    original_question = inputs.get("original_question", "")
    diversified_questions = inputs.get("diversified_questions", "")

    # 4つの進化した質問と評価
    evolved_questions = [
        {
            "id": 1,
            "perspective": "原因・理由",
            "evolved_question": inputs.get("evolved_question1", ""),
            "is_valid": "true" in str(inputs.get("is_valid1", "")).lower(),
            "evaluation": inputs.get("evaluation1", ""),
        },
        {
            "id": 2,
            "perspective": "方法・プロセス",
            "evolved_question": inputs.get("evolved_question2", ""),
            "is_valid": "true" in str(inputs.get("is_valid2", "")).lower(),
            "evaluation": inputs.get("evaluation2", ""),
        },
        {
            "id": 3,
            "perspective": "比較・対照",
            "evolved_question": inputs.get("evolved_question3", ""),
            "is_valid": "true" in str(inputs.get("is_valid3", "")).lower(),
            "evaluation": inputs.get("evaluation3", ""),
        },
        {
            "id": 4,
            "perspective": "影響・結果",
            "evolved_question": inputs.get("evolved_question4", ""),
            "is_valid": "true" in str(inputs.get("is_valid4", "")).lower(),
            "evaluation": inputs.get("evaluation4", ""),
        },
    ]

    # 使用可能な質問のみ抽出
    valid_questions = [q for q in evolved_questions if q["is_valid"]]

    output_data = {
        "original_question": original_question.strip() if original_question else "",
        "diversified_questions_raw": diversified_questions,
        "evolved_questions": evolved_questions,
        "valid_questions": valid_questions,
        "valid_count": len(valid_questions),
        "total_count": 4,
        "source_text_preview": (
            (source_text[:500] + "...") if len(str(source_text)) > 500 else str(source_text)
        ),
    }

    # 使用可能な質問のリスト（テキストのみ）
    valid_question_texts = [q["evolved_question"] for q in valid_questions]

    ctx.log("info", f"JSONL出力データを生成しました（使用可能: {len(valid_questions)}/4）")

    return {
        "OutputData": output_data,
        "ValidQuestions": valid_question_texts,
        "ValidCount": len(valid_questions),
    }