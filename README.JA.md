# SDG Nexus

**v2.0仕様**をサポートしたMABEL（Model And Blocks Expansion Language）ベースのAIエージェントシステム

## 特徴

- **MABEL v2.0 サポート**: チューリング完全な式言語（MEX）、高度な制御構造（`while`、`recurse`、`reduce`、`call`、`let`）、インラインPython関数、グローバル変数
- **MABEL v1.x 互換性**: 自動バージョン検出による完全な後方互換性
- **バッチ処理**: 適応型バッチングによる最適化された並行AI API呼び出し
- **マルチモデルサポート**: 複数のLLMモデルを定義・使用
- **柔軟なI/O**: ストリーミングとバッチモードをサポートするJSONLとCSV
- **エラーハンドリング**: リトライ機構を持つ設定可能なエラー処理

## インストール

```bash
pip install -e .
```

## 必要要件

- Python >= 3.10
- PyYAML >= 6.0.1
- openai >= 1.40.0
- tqdm >= 4.66.0

## クイックスタート

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

詳細なMABEL構文と高度な機能については、以下のドキュメントを参照してください：
- **[MABEL v2 仕様書](docs/mabel/mabel_v2.md)** - 全機能、サンプル、実装状況を含む完全な仕様書

## 使用方法

### コマンドライン

```bash
# JSONL入力の処理
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input examples/data/input.jsonl \
  --output output/result.jsonl

# カスタムバッチ設定を使用
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --max-batch 16 \
  --min-batch 2 \
  --target-latency 2000
```

### Python API

```python
from sdg.config import load_config
from sdg.executors import run_pipeline
import asyncio

# 設定の読み込み
cfg = load_config("pipeline.yaml")

# データセットの準備
dataset = [
    {"UserInput": "AIとは何ですか？"},
    {"UserInput": "機械学習を説明してください"}
]

# パイプラインの実行
results = asyncio.run(run_pipeline(cfg, dataset))

for result in results:
    print(result)
```

📖 **ドキュメント:**
- **[使用ガイド](docs/usage.ja.md)** - SDGパイプラインの実行方法（CLIとPython API）
- **[MABEL v2 仕様書](docs/mabel/mabel_v2.md)** - 全機能とサンプルを含む完全なMABEL仕様

## サンプル

サンプルYAMLファイルとデータについては`examples/`ディレクトリを参照:
- `sdg_demo.yaml` / `sdg_demo_v2.yaml` - 基本および高度なサンプル
- `sdg_comprehensive_v2.yaml` - 全機能を含む包括的なv2.0サンプル
- `helpers.py` - 外部Python関数のサンプル
- `data/` - サンプル入出力データファイル

## ライセンス

MITライセンス - LICENSEファイルを参照

## コントリビューション

コントリビューション歓迎！以下を確認してください:
- v1互換性が維持されていること
- v2機能がMABEL 2.0仕様に従っていること
- v1とv2両方のサンプルでテストが通ること
- コードが適切にドキュメント化されていること

## サポート

問題や機能リクエストについては、GitHubのissue trackerをご利用ください。
