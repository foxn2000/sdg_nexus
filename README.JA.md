# SDG Nexus

## 概要

**SDG-Nexus（Scalable Data Generator Nexus）** は、LLM（大規模言語モデル）向けの合成データセット生成、およびAIエージェントによる大規模データ解析を効率的に行うためのフレームワークです。特に大量のAIエージェントを並列的に運用するタスクや、高速なバッチ処理が必要となるユースケースを想定して設計されており、従来手法に比べて処理能力および柔軟性の大幅な向上を実現しています。

最新バージョンの**MABEL (Model And Blocks Expansion Language) v2.0**を採用し、記述性と柔軟性が極めて高い構造化されたエージェントプログラムを実現できます。また、異なるLLMモデルを同時に稼働させ、負荷分散・パフォーマンス最適化を容易に行うことが可能であるため、LLMを用いた大規模なデータ分析、データ拡張、リアルタイム推論、合成データ生成といったタスクで高い効果を発揮します。

さらに、内部で適応型のバッチ処理とエラー処理機構を搭載することで、リクエスト量が変動する状況でも安定した稼働が可能となっています。特に、自然言語処理（NLP）、生成AIアプリケーション、AIエージェントベースの自動化システムといった、高頻度かつ大量の推論を伴うワークロードに最適化されています。

本フレームワークは、AIエージェントを大規模・高速・安定的に活用することを重視した設計になっており、LLMを活用した高度なタスクを効率的にスケールアップする必要があるユーザーに最適なツールとなっています。

---

## 特徴

* **MABEL v2.0 サポート**

  * チューリング完全な式言語（MEX）
  * 高度な制御構造（`while`, `recurse`, `reduce`, `call`, `let`）
  * インラインPython関数
  * グローバル変数サポート
* **MABEL v1.x 後方互換**

  * 自動バージョン検出機能搭載
* **高度な並行処理**

  * TCP輻輳制御（Vegas/Reno/BBR）にインスパイアされた適応型並行制御
    * Slow Start（指数増加）とCongestion Avoidance（線形増加）の2フェーズ制御
    * EMA（指数移動平均）によるノイズ除去とトレンド検出
    * Vegas-styleプロアクティブ輻輳検出
    * 段階的減少ロジック（軽度の輻輳は無視、深刻な輻輳には即座に対応）
  * vLLM/SGLangバックエンドからのリアルタイムメトリクス収集
  * 最適なスループットのための動的リクエストバッチング
  * レイテンシベースの自動最適化
* **マルチモデル対応**

  * 同時に複数のLLMモデルを定義・運用可能
* **柔軟なI/Oサポート**

  * ストリーミング・バッチモードでのJSONL・CSVフォーマット対応
  * Hugging Face Datasetsの直接読み込み対応
  * キーマッピング機能によるデータセット互換性の向上
* **堅牢なエラーハンドリング**

  * リトライ機構付きで柔軟なエラー処理設定が可能
* **パフォーマンス最適化**

  * 共有HTTPトランスポートによるコネクションプーリング
  * HTTP/2サポートによるスループット向上
  * 非同期バッファI/Oによる効率的なファイル操作
  * Phase 2: 階層的タスクスケジューリングとメモリ最適化（[Phase 2最適化ガイド](docs/phase2_optimization.md)参照）

---

## 必要要件

* Python `>= 3.10`
* PyYAML `>= 6.0.1`
* openai `>= 1.40.0`
* tqdm `>= 4.66.0`

---

## インストール方法

複数の環境管理方法を用いたインストール例を紹介します。

### 通常のpipでインストール

```bash
pip install -e .
```

### pyenvを使用したインストール方法

```bash
# Pythonのバージョン管理
pyenv install 3.12.0
pyenv local 3.12.0

# venvを設定
python -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install -e .
```

### condaを使用したインストール方法

```bash
# 環境作成と有効化
conda create -n sdg python=3.12
conda activate sdg

# インストール
pip install -e .
```

### uvを使用した高速インストール方法（推奨）

[uv](https://github.com/astral-sh/uv) はPythonの高速パッケージマネージャーです。

```bash
# uvのインストール (まだの場合)
pip install uv

# 仮想環境作成と依存関係インストール
uv venv
source .venv/bin/activate

uv pip install -e .
```

---

## クイックスタート

最小限の設定例:

```yaml
mabel:
  version: "2.0"

models:
  - name: gpt4
    api_model: gpt-4o-mini
    api_key: ${ENV.OPENAI_API_KEY}

blocks:
  - type: ai
    exec: 1
    model: gpt4
    prompts:
      - "要約: {UserInput}"
    outputs:
      - name: Summary
        select: full
  
  - type: end
    exec: 2
    final:
      - name: answer
        value: "{Summary}"
```

詳細な仕様は以下を参照してください：

* **[MABEL v2 仕様書](docs/mabel/mabel_v2.md)** - 詳細な機能説明、サンプル、仕様

---

## 使用方法

### コマンドライン(CLI)での実行

基本的なJSONL処理:

```bash
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input examples/data/input.jsonl \
  --output output/result.jsonl
```

カスタムバッチ設定による実行:

```bash
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --max-batch 16 \
  --min-batch 2 \
  --target-latency 2000
```

### Python APIによる利用

```python
from sdg.config import load_config
from sdg.executors import run_pipeline
import asyncio

# 設定をロード
cfg = load_config("pipeline.yaml")

# データセットを準備
dataset = [
    {"UserInput": "AIとは何ですか？"},
    {"UserInput": "機械学習を説明してください"}
]

# 非同期処理でパイプライン実行
results = asyncio.run(run_pipeline(cfg, dataset))

for result in results:
    print(result)
```

---

## 詳細ドキュメント 📖

* **[使用ガイド](docs/usage.ja.md)** - CLI・Python APIの詳細な使用方法
* **[MABEL v2 完全仕様](docs/mabel/mabel_v2.md)** - MABELの文法・機能詳細

---

## MABEL エディター 🎨

MABELファイルのビジュアル編集用に、専用のGUIツールを提供しています：

* **[SDG UI](https://github.com/foxn2000/sdg_ui)** - MABEL設定ファイルを作成・編集するためのグラフィカルユーザーインターフェース

このツールを使用すると、YAMLファイルを手動で編集することなく、直感的にMABELパイプラインを設計・管理できます。

---

## サンプル集

以下のディレクトリでサンプルコード・データを提供しています。

* **`examples/`**

  * `sdg_demo.yaml` : 基本的な使用例
  * `sdg_demo_v2.yaml` : 高度なMABEL v2サンプル
  * `sdg_comprehensive_v2.yaml` : v2全機能網羅的サンプル
  * `helpers.py` : 外部Python関数の活用例
  * `data/` : サンプル入力・出力データセット

---

## ライセンス 📝

本プロジェクトは **MITライセンス** のもとで提供されます。
詳しくは [LICENSE](LICENSE) ファイルをご覧ください。

---

## コントリビューション 🤝

SDG-Nexusへの貢献を歓迎しています！
プルリクエスト提出時は以下を確認してください：

* MABEL v1互換性を維持していること
* MABEL v2機能が最新仕様に準拠していること
* すべての既存サンプルでテストがパスすること
* 適切なドキュメンテーションがされていること

---

## サポート 🛠️

問題報告や機能リクエストは [GitHub Issues](https://github.com/your-repository/issues) をご利用ください。

---

以上のように構造化と詳細な情報を追加することで、より包括的かつ実用的なREADMEになっています。
