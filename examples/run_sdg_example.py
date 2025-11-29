#!/usr/bin/env python3
"""
SDG Nexus - Python API サンプルコード

このスクリプトはPython APIを使用してSDGパイプラインを実行する方法を示します。

使用方法:
    python examples/run_sdg_example.py

環境変数:
    OPENAI_API_KEY: OpenAI APIキー（オプション、YAMLで直接指定している場合は不要）
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sdg.config import load_config
from sdg.executors import run_pipeline


def load_jsonl(file_path: str) -> list:
    """JSONLファイルを読み込む"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, file_path: str) -> None:
    """JSONLファイルに保存する"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def run_example():
    """
    SDGパイプライン実行のサンプル
    """
    # 設定ファイルのパス
    examples_dir = Path(__file__).parent

    # v1.0 の例: sdg_demo.yaml
    yaml_path_v1 = examples_dir / "sdg_demo.yaml"

    # v2.0 の例: sdg_demo_v2.yaml
    yaml_path_v2 = examples_dir / "sdg_demo_v2.yaml"

    # 入力データ
    input_path = examples_dir / "data" / "input.jsonl"

    # 出力パス
    output_path = project_root / "output" / "python_api_result.jsonl"

    print("=" * 60)
    print("SDG Nexus - Python API サンプル")
    print("=" * 60)

    # ===== 方法1: JSONLファイルからデータを読み込んで実行 =====
    print("\n【方法1】JSONLファイルからデータを読み込んで実行")
    print("-" * 40)

    if input_path.exists() and yaml_path_v2.exists():
        # 設定を読み込み
        print(f"設定ファイル: {yaml_path_v2}")
        cfg = load_config(str(yaml_path_v2))
        print(f"  - MABELバージョン: {cfg.get_version()}")
        print(f"  - ブロック数: {len(cfg.blocks)}")
        print(f"  - モデル数: {len(cfg.models)}")

        # データセットを読み込み
        print(f"\n入力ファイル: {input_path}")
        dataset = load_jsonl(str(input_path))
        print(f"  - レコード数: {len(dataset)}")

        # パイプライン実行
        print("\nパイプラインを実行中...")
        try:
            results = await run_pipeline(
                cfg,
                dataset,
                max_batch=4,  # 最大バッチサイズ
                min_batch=1,  # 最小バッチサイズ
                target_latency_ms=3000,  # 目標レイテンシ
                save_intermediate=False,  # 中間結果を保存しない
            )

            # 結果を表示
            print("\n実行結果:")
            for i, result in enumerate(results):
                print(f"\n  レコード {i + 1}:")
                for key, value in result.items():
                    # 長い値は省略して表示
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    print(f"    {key}: {value_str}")

            # 結果を保存
            save_jsonl(results, str(output_path))
            print(f"\n結果を保存しました: {output_path}")

        except Exception as e:
            print(f"\nエラー: {e}")
            print("（注: LLM APIへの接続が必要です）")
    else:
        print(f"ファイルが見つかりません: {yaml_path_v2} または {input_path}")

    # ===== 方法2: インラインデータで実行 =====
    print("\n\n【方法2】インラインデータで実行")
    print("-" * 40)

    if yaml_path_v2.exists():
        # 設定を読み込み
        cfg = load_config(str(yaml_path_v2))

        # 直接データセットを定義
        dataset = [
            {"UserInput": "AIとは何ですか？"},
            {"UserInput": "機械学習について説明してください"},
        ]

        print(f"データセット: {len(dataset)} レコード")
        for i, record in enumerate(dataset):
            print(f"  {i + 1}. {record}")

        # パイプライン実行
        print("\nパイプラインを実行中...")
        try:
            results = await run_pipeline(cfg, dataset)

            print("\n実行結果:")
            for i, result in enumerate(results):
                print(f"\n  レコード {i + 1}: {list(result.keys())}")

        except Exception as e:
            print(f"\nエラー: {e}")
            print("（注: LLM APIへの接続が必要です）")

    # ===== 設定情報の確認 =====
    print("\n\n【参考】設定情報の確認方法")
    print("-" * 40)

    if yaml_path_v2.exists():
        cfg = load_config(str(yaml_path_v2))

        print(f"\n1. バージョン確認:")
        print(f"   cfg.get_version() = '{cfg.get_version()}'")
        print(f"   cfg.is_v2() = {cfg.is_v2()}")

        print(f"\n2. グローバル変数:")
        print(f"   定数 (const): {cfg.globals_.const}")
        print(f"   変数 (vars): {cfg.globals_.vars}")

        print(f"\n3. モデル情報:")
        for m in cfg.models:
            print(f"   - {m.name}: {m.api_model}")

        print(f"\n4. ブロック一覧:")
        for b in cfg.blocks:
            print(
                f"   - exec={b.exec}, type={b.type}, name={b.name or b.id or '(unnamed)'}"
            )


def main():
    """エントリーポイント"""
    asyncio.run(run_example())


if __name__ == "__main__":
    main()
