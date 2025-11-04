# SDG Nexus

**v2.0仕様**をサポートしたMABEL（Model And Blocks Expansion Language）ベースのAIエージェントシステム

## 特徴

### MABEL v2.0 サポート
- **MEX式言語**: チューリング完全な式評価
- **グローバル変数**: `globals.const`と`globals.vars`による定数とミュータブル変数
- **高度な論理演算子**:
  - `set`: MEX式による変数代入
  - `while`: バジェット制御付き条件ループ
  - `emit`: ループ内での値収集
  - 完全なMEX演算子: 算術、比較、文字列、コレクション、正規表現など
- **インラインPython関数**: YAML内でPythonコードを直接定義（`function_code`）
- **強化されたPython統合**: `vars`、`get`、`set`、`log`を持つコンテキストオブジェクト（`ctx`）
- **バジェット制御**: `budgets`設定によるループ/再帰の制限
- **強化されたAI出力**:
  - JSONPathサポート（`select: jsonpath`）
  - 型ヒント（`type_hint: number|boolean|json`）
  - 変数への保存（`save_to.vars`）

### MABEL v1.x 互換性
- 完全な後方互換性を維持
- `mabel.version`からの自動バージョン検出
- v1.0 YAMLファイルは変更なしで動作

### コア機能
- **バッチ処理**: 最適化された並行AI API呼び出し
- **適応型バッチング**: レイテンシに基づく動的バッチサイズ調整
- **マルチモデルサポート**: 複数のLLMモデルを定義・使用
- **柔軟なI/O**: JSONLとCSVのサポート
- **エラーハンドリング**: 設定可能なエラー処理（`fail`、`continue`、`retry`）

## インストール

```bash
pip install -e .
```

## クイックスタート

### v2.0 の例

```yaml
mabel:
  version: "2.0"

globals:
  const:
    APP_NAME: "My Agent"
  vars:
    counter: 0

budgets:
  loops:
    max_iters: 100
    on_exceed: "error"

models:
  - name: gpt4
    api_model: gpt-4o-mini
    api_key: ${ENV.OPENAI_API_KEY}
    request_defaults:
      temperature: 0.0
      max_tokens: 500

blocks:
  # MEX式で変数を設定
  - type: logic
    exec: 1
    op: set
    var: counter
    value: {"add": [{"var": "counter"}, 1]}
  
  # whileループ
  - type: logic
    exec: 2
    op: while
    init:
      - op: set
        var: i
        value: 0
    cond:
      lt:
        - {"var": "i"}
        - 5
    step:
      - op: set
        var: i
        value: {"add": [{"var": "i"}, 1]}
      - op: emit
        value: {"var": "i"}
    outputs:
      - name: Numbers
        from: list
  
  # インラインPython
  - type: python
    exec: 3
    entrypoint: process
    function_code: |
      def process(ctx, numbers: list) -> dict:
          ctx.log("info", f"Processing {len(numbers)} numbers")
          total = sum(numbers)
          return {"Sum": total}
    inputs:
      numbers: "{Numbers}"
    outputs: [Sum]
  
  - type: end
    exec: 100
    final:
      - name: result
        value: "{Sum}"
    include_vars:
      - counter
```

### v1.0 の例（引き続きサポート）

```yaml
mabel:
  version: "1.0"

models:
  - name: planner
    api_model: gpt-4o-mini
    api_key: ${ENV.OPENAI_API_KEY}
    request_defaults:
      temperature: 0.2

blocks:
  - type: ai
    exec: 1
    model: planner
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

## MABEL v2 アーキテクチャ

### MEX式言語

MEXは安全でチューリング完全な式言語を提供します:

```yaml
# 算術
{"add": [1, 2, 3]}  # 6
{"mul": [{"var": "x"}, 2]}  # x * 2

# 比較
{"gt": [{"var": "count"}, 10]}  # count > 10
{"eq": ["{Status}", "ok"]}     # Status == "ok"

# 論理
{"and": [
  {"gt": [{"var": "score"}, 80]},
  {"lt": [{"var": "errors"}, 5]}
]}

# 文字列操作
{"concat": ["Hello, ", {"var": "name"}]}
{"replace": ["{text}", "old", "new"]}

# コレクション
{"map": {"list": [1,2,3], "fn": {"mul": [{"var": "item"}, 2]}}}
{"filter": {"list": [1,2,3,4], "fn": {"gt": [{"var": "item"}, 2]}}}

# 制御フロー
{"if": {
  "cond": {"gt": [{"var": "x"}, 0]},
  "then": "positive",
  "else": "non-positive"
}}
```

### ブロックタイプ

#### AIブロック
```yaml
- type: ai
  exec: 1
  model: gpt4
  system_prompt: "あなたは親切なアシスタントです。"
  prompts:
    - "質問: {UserInput}"
  mode: json  # v2: jsonモード
  outputs:
    - name: Answer
      select: jsonpath  # v2: JSONPath
      path: "$.response.text"
      type_hint: string
  save_to:  # v2: グローバル変数に保存
    vars:
      last_answer: Answer
```

#### Logicブロック
```yaml
# v2: set
- type: logic
  exec: 1
  op: set
  var: total
  value: {"add": [{"var": "total"}, 10]}

# v2: while
- type: logic
  exec: 2
  op: while
  init:
    - op: set
      var: i
      value: 0
  cond:
    lt: [{"var": "i"}, 10]
  step:
    - op: set
      var: i
      value: {"add": [{"var": "i"}, 1]}
    - op: emit
      value: {"var": "i"}
  outputs:
    - name: Numbers
      from: list

# v1: for (引き続きサポート)
- type: logic
  exec: 3
  op: for
  list: "{Lines}"
  parse: lines
  var: line
  map: "- {line}"
  outputs:
    - name: Formatted
      from: join
      join_with: "\n"
```

#### Pythonブロック
```yaml
# v2: インライン関数
- type: python
  exec: 1
  entrypoint: process
  function_code: |
    def process(ctx, data: dict) -> dict:
        # ctx.vars: グローバル変数
        # ctx.get(path): ネストされた値を取得
        # ctx.set(path, val): グローバル変数を設定
        # ctx.log(level, msg): ロギング
        
        ctx.log("info", f"Processing {len(data)} items")
        result = {"processed": len(data)}
        return result
  inputs:
    data: "{InputData}"
  outputs: [processed]

# v1: 外部ファイル（引き続きサポート）
- type: python
  exec: 2
  function: my_function
  code_path: ./helper.py
  inputs: [Input1, Input2]
  outputs: [Output1]
```

#### Endブロック
```yaml
- type: end
  exec: 100
  final:
    - name: answer
      value: "{Result}"
    - name: metadata
      value: "{Meta}"
  include_vars:  # v2: グローバル変数を含める
    - counter
    - timestamp
```

## 設定

### ランタイム (v2)
```yaml
runtime:
  python:
    interpreter: "python>=3.11,<3.13"
    venv: ".venv"
    requirements:
      - "numpy>=1.24"
      - "pandas>=2.0"
    allow_network: false
```

### バジェット (v2)
```yaml
budgets:
  loops:
    max_iters: 1000
    on_exceed: "error"  # error | truncate | continue
  recursion:
    max_depth: 128
    on_exceed: "error"
  wall_time_ms: 300000
  ai:
    max_calls: 100
    max_tokens: 500000
```

### グローバル変数 (v2)
```yaml
globals:
  const:  # 読み取り専用
    APP_VERSION: "1.0"
    MAX_RETRIES: 3
  vars:   # ミュータブル
    counter: 0
    state: "init"
    results: []
```

## v1からv2への移行

v1 YAMLファイルは変更なしで動作します。v2機能を活用するには:

1. バージョンを更新:
```yaml
mabel:
  version: "2.0"  # 以前は "1.0"
```

2. グローバル変数を追加（オプション）:
```yaml
globals:
  vars:
    my_var: 0
```

3. 条件でMEX式を使用:
```yaml
# v1 (JSON文字列、引き続き動作)
run_if: "{\"equals\":[\"{ Status}\",\"ok\"]}"

# v2 (ネイティブMEX、推奨)
run_if:
  eq: ["{Status}", "ok"]
```

4. シンプルな関数にはインラインPythonを使用:
```yaml
# v1 (外部ファイル)
- type: python
  function: helper
  code_path: ./helper.py

# v2 (インライン)
- type: python
  entrypoint: helper
  function_code: |
    def helper(ctx, x):
        return {"result": x * 2}
```

## サンプル

`examples/`ディレクトリを参照:
- `sdg_demo.yaml` - v1.0互換サンプル
- `sdg_demo_v2.yaml` - v2.0機能のショーケース
- `helpers.py` - 外部Python関数のサンプル

## ライセンス

MITライセンス - LICENSEファイルを参照

## コントリビューション

コントリビューション歓迎！以下を確認してください:
- v1互換性が維持されていること
- v2機能がMABEL 2.0仕様に従っていること
- v1とv2両方のサンプルでテストが通ること
