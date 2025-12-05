"""
SDG Nexus - 三次元スケーリング推論エージェント用ヘルパー関数

対応YAML: examples/3d_scaling_agent.yaml

このファイルには、3D Scaling Agentで使用される全てのヘルパー関数が含まれています。
"""


def split_3d_perspectives(ctx, **inputs) -> dict:
    """
    三次元スケーリング: 視点を個別のプロンプトに分離する関数

    Args:
        ctx: 実行コンテキスト
        inputs:
            - perspectives: 生成された視点リスト(番号付き)
            - parallel_count: 並列数
            - original_question: 元の質問
            - reasoning_effort: 推論エフォートレベル

    Returns:
        dict: Prompt1-Prompt10とActualParallelCountを含む辞書
    """
    perspectives = inputs.get("perspectives", "")
    parallel_count_str = inputs.get("parallel_count", "4")
    original_question = inputs.get("original_question", "")
    reasoning_effort = inputs.get("reasoning_effort", "medium")

    # 並列数を整数に変換
    try:
        parallel_count = int(parallel_count_str)
        parallel_count = max(1, min(10, parallel_count))  # 1-10の範囲に制限
    except (ValueError, TypeError):
        parallel_count = 4  # デフォルト

    def parse_perspectives(text):
        """番号付き視点リストをパース"""
        if not text:
            return []

        lines = str(text).strip().split("\n")
        prompts = []
        current_prompt = []
        current_perspective = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 新しい番号付き項目の検出 (1., 2., etc.)
            if line and len(line) >= 2 and line[0].isdigit() and line[1] in ".)":
                if current_prompt:
                    # 前のプロンプトを保存
                    prompt_text = " ".join(current_prompt)
                    prompts.append((current_perspective, prompt_text))

                # 番号を除去し、視点名を抽出
                cleaned = line.lstrip("0123456789.) ")

                # [視点名]: プロンプト の形式を処理
                if "[" in cleaned and "]" in cleaned:
                    bracket_start = cleaned.find("[")
                    bracket_end = cleaned.find("]")
                    current_perspective = cleaned[bracket_start + 1 : bracket_end]
                    prompt_part = cleaned[bracket_end + 1 :].lstrip(":").strip()
                    current_prompt = [prompt_part] if prompt_part else []
                else:
                    # [視点名]がない場合
                    parts = cleaned.split(":", 1)
                    if len(parts) == 2:
                        current_perspective = parts[0].strip()
                        current_prompt = [parts[1].strip()]
                    else:
                        current_perspective = f"視点{len(prompts)+1}"
                        current_prompt = [cleaned]
            elif current_prompt is not None:
                current_prompt.append(line)

        # 最後のプロンプトを追加
        if current_prompt:
            prompt_text = " ".join(current_prompt)
            prompts.append((current_perspective, prompt_text))

        return prompts

    parsed = parse_perspectives(perspectives)

    # プロンプトを構築
    result = {}
    actual_count = min(len(parsed), parallel_count)

    # エフォートレベルに応じた指示を追加
    effort_instructions = {
        "low": "簡潔に要点のみを回答してください（100-200字程度）。",
        "medium": "バランスの取れた詳細度で回答してください（200-400字程度）。",
        "high": "詳細な分析と根拠を示して回答してください（400-800字程度）。",
    }
    effort_instruction = effort_instructions.get(
        reasoning_effort.lower(), effort_instructions["medium"]
    )

    for i in range(1, 11):
        if i <= len(parsed) and i <= parallel_count:
            perspective_name, prompt_content = parsed[i - 1]
            # 完全なプロンプトを構築
            full_prompt = f"""【元の質問】
{original_question}

【視点】{perspective_name}

【指示】
{prompt_content}

{effort_instruction}"""
            result[f"Prompt{i}"] = full_prompt
        else:
            result[f"Prompt{i}"] = ""

    result["ActualParallelCount"] = actual_count

    ctx.log("info", f"視点を{actual_count}個のプロンプトに分離しました")

    return result


def collect_3d_answers(ctx, **inputs) -> dict:
    """
    三次元スケーリング: 各視点からの回答を収集・整理する関数

    Args:
        ctx: 実行コンテキスト
        inputs:
            - parallel_count: 実際の並列数
            - answer_1 ~ answer_10: 各視点からの回答
            - perspectives: 元の視点リスト

    Returns:
        dict: CollectedAnswersとAnswerSummaryを含む辞書
    """
    parallel_count_str = inputs.get("parallel_count", "4")
    perspectives = inputs.get("perspectives", "")

    try:
        parallel_count = int(parallel_count_str)
    except (ValueError, TypeError):
        parallel_count = 4

    # 回答を収集
    answers = []
    for i in range(1, 11):
        answer = inputs.get(f"answer_{i}", "")
        if answer and i <= parallel_count:
            answers.append((i, answer))

    # 視点名を抽出（可能であれば）
    def extract_perspective_names(text):
        names = []
        if not text:
            return names

        lines = str(text).strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and len(line) >= 2 and line[0].isdigit() and line[1] in ".)":
                cleaned = line.lstrip("0123456789.) ")
                if "[" in cleaned and "]" in cleaned:
                    bracket_start = cleaned.find("[")
                    bracket_end = cleaned.find("]")
                    names.append(cleaned[bracket_start + 1 : bracket_end])
                else:
                    parts = cleaned.split(":", 1)
                    if len(parts) >= 1:
                        names.append(parts[0].strip())
        return names

    perspective_names = extract_perspective_names(perspectives)

    # フォーマット済み回答を構築
    collected_lines = []
    for idx, (i, answer) in enumerate(answers):
        perspective_name = (
            perspective_names[idx] if idx < len(perspective_names) else f"視点{i}"
        )
        collected_lines.append(f"【{perspective_name}】")
        collected_lines.append(answer.strip())
        collected_lines.append("")

    collected_answers = "\n".join(collected_lines)

    # サマリーを作成
    summary = {
        "total_perspectives": parallel_count,
        "collected_answers": len(answers),
        "perspective_names": (
            perspective_names[:parallel_count]
            if perspective_names
            else [f"視点{i}" for i in range(1, parallel_count + 1)]
        ),
        "answer_lengths": [len(a[1]) for a in answers],
    }

    ctx.log("info", f"回答を{len(answers)}個収集しました")

    return {"CollectedAnswers": collected_answers, "AnswerSummary": str(summary)}


def format_3d_scaling_result(ctx, **inputs) -> dict:
    """
    三次元スケーリング: 最終結果をフォーマットする関数

    Args:
        ctx: 実行コンテキスト
        inputs: 全ての処理結果データ

    Returns:
        dict: FormattedResultとExecutionMetricsを含む辞書
    """
    original_question = inputs.get("original_question", "")
    question_type = inputs.get("question_type", "")
    difficulty_level = inputs.get("difficulty_level", "")
    reasoning_effort = inputs.get("reasoning_effort", "")
    parallel_count = inputs.get("parallel_count", "")
    iteration_count = inputs.get("iteration_count", "")
    quality_score = inputs.get("quality_score", "")
    perspectives = inputs.get("perspectives", "")
    collected_answers = inputs.get("collected_answers", "")
    integrated_answer = inputs.get("integrated_answer", "")
    synthesized_answer = inputs.get("synthesized_answer", "")
    final_answer = inputs.get("final_answer", "")
    quality_assessment = inputs.get("quality_assessment", "")

    # 思考ログ
    analysis_thinking = inputs.get("analysis_thinking", "")
    perspective_thinking = inputs.get("perspective_thinking", "")
    integration_thinking = inputs.get("integration_thinking", "")
    synthesis_thinking = inputs.get("synthesis_thinking", "")
    proofread_thinking = inputs.get("proofread_thinking", "")

    def truncate(text, max_len=300):
        text = str(text) if text else ""
        return text[:max_len] + "..." if len(text) > max_len else text

    # 質問タイプの日本語表記
    question_type_ja = {
        "broad_shallow": "広く浅い質問",
        "deep_narrow": "深く狭い質問",
        "composite": "複合的な質問",
    }.get(str(question_type).lower().strip(), question_type)

    # 推論エフォートの日本語表記
    effort_ja = {
        "low": "低（簡潔）",
        "medium": "中（バランス）",
        "high": "高（詳細）",
    }.get(str(reasoning_effort).lower().strip(), reasoning_effort)

    result = f"""
================================================================================
              三次元スケーリング推論エージェント - 実行結果
================================================================================

【元の質問】
{original_question}

================================================================================
                         三次元スケーリングのパラメータ
================================================================================

┌─────────────────────┬──────────────────────────────────────────┐
│ パラメータ          │ 値                                       │
├─────────────────────┼──────────────────────────────────────────┤
│ 質問タイプ          │ {question_type_ja}                       │
│ 難易度レベル        │ {difficulty_level}/5                     │
│ 推論エフォート      │ {effort_ja}                              │
│ 並列数（視点数）    │ {parallel_count}                         │
│ 反復回数            │ {iteration_count}/5                      │
│ 品質スコア          │ {quality_score}/100                      │
└─────────────────────┴──────────────────────────────────────────┘

--------------------------------------------------------------------------------
【ステップ1: 質問分析】
思考プロセス: {truncate(analysis_thinking)}

--------------------------------------------------------------------------------
【ステップ2: 視点生成】
思考プロセス: {truncate(perspective_thinking)}

生成された視点:
{truncate(perspectives, 500)}

--------------------------------------------------------------------------------
【ステップ3: 並列推論】
{truncate(collected_answers, 800)}

--------------------------------------------------------------------------------
【ステップ4: 統合・評価】
思考プロセス: {truncate(integration_thinking)}

統合回答:
{truncate(integrated_answer, 500)}

--------------------------------------------------------------------------------
【ステップ5: 総合化】
思考プロセス: {truncate(synthesis_thinking)}

総合回答:
{truncate(synthesized_answer, 500)}

--------------------------------------------------------------------------------
【ステップ6: 校正・最終化】
思考プロセス: {truncate(proofread_thinking)}

品質評価:
{quality_assessment}

================================================================================
                              最終回答
================================================================================

{final_answer}

================================================================================
"""

    # 実行メトリクスを構築
    metrics = {
        "question_type": question_type,
        "difficulty_level": difficulty_level,
        "reasoning_effort": reasoning_effort,
        "parallel_count": parallel_count,
        "iteration_count": iteration_count,
        "quality_score": quality_score,
        "final_answer_length": len(str(final_answer)) if final_answer else 0,
        "total_perspectives_used": parallel_count,
    }

    ctx.log(
        "info",
        f"三次元スケーリング結果をフォーマットしました（品質スコア: {quality_score}）",
    )

    return {"FormattedResult": result.strip(), "ExecutionMetrics": str(metrics)}
