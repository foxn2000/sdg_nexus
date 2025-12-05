"""
SDG Nexus - ヘルパー関数モジュール

YAMLパイプラインから呼び出されるPythonヘルパー関数群をエクスポートします。

各YAMLファイルに対応するヘルパーファイル:
- scaling_3d_helpers.py: examples/3d_scaling_agent.yaml
- question_generator_helpers.py: examples/question_generator_agent.yaml
- question_generator_v2_helpers.py: examples/question_generator_agent_v2.yaml
- prompt_evolution_helpers.py: examples/prompt_evolution_agent.yaml
- comprehensive_helpers.py: examples/sdg_comprehensive_v2.yaml
"""

# 3D Scaling Agent用
from .scaling_3d_helpers import (
    split_3d_perspectives,
    collect_3d_answers,
    format_3d_scaling_result,
)

# Question Generator Agent用（v1）
from .question_generator_helpers import (
    format_question_generator,
    format_question_output_jsonl,
)

# Question Generator Agent用（v2）
from .question_generator_v2_helpers import (
    split_diversified_questions,
    format_question_generator_v2,
    format_question_output_jsonl_v2,
)

# Prompt Evolution Agent用
from .prompt_evolution_helpers import (
    format_prompt_evolution,
)

# Comprehensive Demo用
from .comprehensive_helpers import (
    format_comprehensive_result,
    format_output,
)

__all__ = [
    # 3D Scaling Agent
    "split_3d_perspectives",
    "collect_3d_answers",
    "format_3d_scaling_result",
    # Question Generator Agent (v1)
    "format_question_generator",
    "format_question_output_jsonl",
    # Question Generator Agent (v2)
    "split_diversified_questions",
    "format_question_generator_v2",
    "format_question_output_jsonl_v2",
    # Prompt Evolution Agent
    "format_prompt_evolution",
    # Comprehensive Demo
    "format_comprehensive_result",
    "format_output",
]
