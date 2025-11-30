# SDG (Scalable Data Generator) 使用ガイド

このドキュメントでは、既存のYAMLファイル（MABEL形式）を使用してSDGパイプラインを実行する方法について説明します。

## 目次

1. [概要](#概要)
2. [CLIからの使用方法](#cliからの使用方法)
3. [Python APIからの使用方法](#python-apiからの使用方法)
4. [パーサーについて](#パーサーについて)
5. [入出力データ形式](#入出力データ形式)

---

## 概要

SDGは、YAMLで定義されたパイプライン（MABEL形式）に従って、入力データセットを処理し、LLM呼び出しやPythonコード実行を組み合わせて出力データを生成するツールです。

基本的なワークフロー:
1. YAMLブループリントファイルを用意する
2. 入力データセット（JSONL または CSV）を用意する
3. CLIまたはPython APIでパイプラインを実行する
4. 出力データ（JSONL）を取得する

---

## CLIからの使用方法

### 基本コマンド

```bash
# 基本形式
sdg run --yaml <YAMLファイル> --input <入力ファイル> --output <出力ファイル>

# 例
sdg run --yaml examples/sdg_demo.yaml --input examples/data/input.jsonl --output output/result.jsonl
```

### 実行モード

SDGには2つの実行モードがあります:

#### 1. ストリーミングモード（デフォルト）

各データ行を並列処理し、完了した行から順次出力ファイルへ書き込みます。

```bash
# ストリーミングモード（デフォルト）
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl

# 同時処理数を指定
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --max-concurrent 16

# 進捗表示を無効化
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --no-progress
```

**オプション:**
| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--max-concurrent` | 8 | 同時処理行数の上限 |
| `--no-progress` | false | 進捗表示を無効化 |

**特徴:**
- 途中結果が失われにくい（リアルタイム書き込み）
- メモリ効率が良い
- 出力順序は処理完了順（入力順序と異なる場合あり）

> **Note:** 元の順序が必要な場合は、出力の `_row_index` フィールドでソートしてください。

#### 2. バッチモード

ブロック単位で一括処理を行います。`--batch-mode` フラグで有効化します。

```bash
# バッチモードを有効化
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --batch-mode

# バッチサイズを指定
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --batch-mode --max-batch 16 --min-batch 2 --target-latency-ms 5000
```

**オプション:**
| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--batch-mode` | false | バッチモードを有効化 |
| `--max-batch` | 8 | ブロックあたりの最大同時リクエスト数 |
| `--min-batch` | 1 | ブロックあたりの最小同時リクエスト数 |
| `--target-latency-ms` | 3000 | 目標平均レイテンシ（ミリ秒） |

### 共通オプション

| オプション | 説明 |
|-----------|------|
| `--save-intermediate` | 中間結果を保存する |

### レガシーモード

後方互換性のため、サブコマンドなしでの実行もサポートしています:

```bash
# レガシー形式（後方互換）
sdg --yaml pipeline.yaml --input data.jsonl --output result.jsonl
```

---

## Python APIからの使用方法

### 基本的な使用方法

```python
import asyncio
from sdg.config import load_config
from sdg.executors import run_pipeline

async def main():
    # 1. 設定を読み込み
    cfg = load_config("examples/sdg_demo.yaml")
    
    # 2. データセットを準備
    dataset = [
        {"UserInput": "AIとは何ですか？"},
        {"UserInput": "機械学習について説明してください"},
    ]
    
    # 3. パイプラインを実行
    results = await run_pipeline(
        cfg,
        dataset,
        max_batch=4,
        min_batch=1,
        target_latency_ms=3000,
        save_intermediate=False,
    )
    
    # 4. 結果を処理
    for result in results:
        print(result)

asyncio.run(main())
```

### ストリーミング実行

```python
from sdg.runner import run_streaming

# ストリーミングモードで実行（同期関数）
run_streaming(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_concurrent=8,
    save_intermediate=False,
    show_progress=True,
)
```

### バッチ実行

```python
from sdg.runner import run

# バッチモードで実行（同期関数）
run(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_batch=8,
    min_batch=1,
    target_latency_ms=3000,
    save_intermediate=False,
)
```

### JSONLファイルからデータを読み込んで実行

```python
import asyncio
import json
from sdg.config import load_config
from sdg.executors import run_pipeline

def load_jsonl(file_path: str) -> list:
    """JSONLファイルを読み込む"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: list, file_path: str) -> None:
    """JSONLファイルに保存する"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

async def main():
    cfg = load_config("examples/sdg_demo.yaml")
    dataset = load_jsonl("examples/data/input.jsonl")
    
    results = await run_pipeline(cfg, dataset)
    
    save_jsonl(results, "output/result.jsonl")

asyncio.run(main())
```

### 設定情報の確認

```python
from sdg.config import load_config

cfg = load_config("examples/sdg_demo_v2.yaml")

# バージョン確認
print(f"MABELバージョン: {cfg.get_version()}")  # "1.0" or "2.0"
print(f"v2仕様か: {cfg.is_v2()}")  # True or False

# グローバル変数（v2のみ）
print(f"定数: {cfg.globals_.const}")
print(f"変数: {cfg.globals_.vars}")

# モデル情報
for m in cfg.models:
    print(f"モデル: {m.name} -> {m.api_model}")

# ブロック一覧
for b in cfg.blocks:
    print(f"exec={b.exec}, type={b.type}, name={b.name or '(unnamed)'}")
```

---

## パーサーについて

SDGには、LLMの出力や入力データを処理するためのパーサーが用意されています。

### AI出力パーサー（`select` オプション）

AIブロックの `outputs` で使用します。LLMからの応答テキストを解析して必要な部分を抽出します。

```yaml
outputs:
  - name: FullResponse
    select: full          # 応答全体を取得

  - name: ExtractedTag
    select: tag           # 特定のタグ内容を抽出
    tag: answer           # <answer>...</answer> の内容

  - name: FirstLine
    select: regex         # 正規表現でマッチした部分を抽出
    regex: "^(.+?)$"      # 正規表現パターン

  - name: JsonField
    select: jsonpath      # JSONパス式で値を抽出（v2）
    path: "$.result.value"
```

| select値 | 説明 | 必須オプション |
|----------|------|---------------|
| `full` | 応答全体を取得（デフォルト） | なし |
| `tag` | XMLタグ内の内容を抽出 | `tag` |
| `regex` | 正規表現でマッチした部分を抽出 | `regex` |
| `jsonpath` | JSONパス式で値を抽出（v2） | `path` |

### ロジックパーサー（`parse` オプション）

`logic` ブロックの `for` ループで使用します。テキストデータをリストに変換します。

```yaml
- type: logic
  exec: 2
  op: for
  list: "{TextData}"
  parse: lines           # パース方法を指定
  var: item
  outputs:
    - name: ProcessedItems
      from: list
```

| parse値 | 説明 | 例 |
|---------|------|-----|
| `lines` | 改行で分割 | `"a\nb\nc"` → `["a", "b", "c"]` |
| `csv` | CSVとしてパース | `"a,b,c"` → `["a", "b", "c"]` |
| `json` | JSONとしてパース | `"[1,2,3]"` → `[1, 2, 3]` |
| `regex` | 正規表現でマッチを抽出 | パターンに応じた結果 |

```yaml
# 正規表現パースの例
- type: logic
  op: for
  list: "{TextData}"
  parse: regex
  regex_pattern: "\\d+"   # 数字をすべて抽出
  var: number
  outputs:
    - name: Numbers
      from: list
```

### 追加オプション

```yaml
- type: logic
  op: for
  list: "{TextData}"
  parse: lines
  drop_empty: true        # 空行を除外
  where:                  # フィルタ条件（MEX式）
    ne: ["{item}", ""]
  map: "{item} processed" # マッピング（変換）
  var: item
```

---

## 入出力データ形式

### 入力データ形式

SDGは以下の入力形式をサポートしています:

#### JSONL形式（推奨）

```jsonl
{"UserInput": "AIとは何ですか？", "Category": "tech"}
{"UserInput": "天気について教えて", "Category": "general"}
```

#### CSV形式

```csv
UserInput,Category
AIとは何ですか？,tech
天気について教えて,general
```

### 出力データ形式

出力は常にJSONL形式です。パイプラインの `end` ブロックで定義した `final` フィールドが出力されます。

```jsonl
{"answer": "AIは...", "status": "success", "_row_index": 0}
{"answer": "天気は...", "status": "success", "_row_index": 1}
```

> **Note:** `_row_index` は入力データの元の行番号を示します。ストリーミングモードでは出力順序が入力順序と異なる場合があるため、このフィールドで元の順序を復元できます。

---

## 実行例

### 例1: シンプルなQ&Aパイプライン

```bash
# examples/sdg_demo.yaml を使用
sdg run \
  --yaml examples/sdg_demo.yaml \
  --input examples/data/input.jsonl \
  --output output/qa_result.jsonl
```

### 例2: v2機能を使用したパイプライン

```bash
# examples/sdg_demo_v2.yaml を使用（グローバル変数、MEX式、Whileループなど）
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input examples/data/input.jsonl \
  --output output/v2_result.jsonl \
  --max-concurrent 4
```

### 例3: 大量データの処理

```bash
# 大量データをストリーミング処理
sdg run \
  --yaml pipeline.yaml \
  --input large_dataset.jsonl \
  --output output/large_result.jsonl \
  --max-concurrent 16
```

---

## トラブルシューティング

### よくある問題

1. **APIキーエラー**
   - YAMLファイルの `api_key` フィールドを確認してください
   - 環境変数を使用する場合は `api_key: "${OPENAI_API_KEY}"` のように設定します

2. **入力ファイルが見つからない**
   - ファイルパスが正しいことを確認してください
   - 相対パスは実行ディレクトリからの相対パスです

3. **出力順序が入力と異なる**
   - ストリーミングモードでは処理完了順に出力されます
   - `_row_index` フィールドでソートして元の順序を復元してください

4. **メモリ不足**
   - `--max-concurrent` の値を小さくしてください
   - ストリーミングモード（デフォルト）を使用してください