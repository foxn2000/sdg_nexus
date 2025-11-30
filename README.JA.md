# SDG Nexus

**v2.0仕様**をサポートしたMABEL（Model And Blocks Expansion Language）ベースのAIエージェントシステム

## 特徴

### MABEL v2.0 サポート
- **MEX式言語**: チューリング完全な式評価エンジン
- **グローバル変数**: `globals.const`と`globals.vars`による定数とミュータブル変数
- **高度な論理演算子**:
  - `set`: MEX式による変数代入
  - `let`: スコープ付きローカル変数束縛
  - `while`: バジェット制御付き条件ループ
  - `emit`: ループ内での値収集
  - `reduce`: アキュムレータによるリスト畳み込み
  - `call`: ユーザ定義ロジック関数呼び出し
  - `recurse`: ベースケース付き再帰関数実行
  - 完全なMEX演算子: 算術、比較、文字列、コレクション、正規表現、論理
- **インラインPython関数**: YAML内でPythonコードを直接定義（`function_code`）
- **強化されたPython統合**: `vars`、`get`、`set`、`log`、`emit`を持つコンテキストオブジェクト（`ctx`）
- **バジェット制御**: `budgets`設定によるループ/再帰/AI呼び出しの制限
- **強化されたAI出力**:
  - JSONPathサポート（`select: jsonpath`）
  - 型ヒント（`type_hint: number|boolean|json|string`）
  - 変数への保存（`save_to.vars`）
  - JSONモード（`mode: json`）
- **ユーザ定義関数**: 再利用可能なロジック関数とPython関数の定義

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

## 必要要件

- Python >= 3.10
- PyYAML >= 6.0.1
- openai >= 1.40.0
- tqdm >= 4.66.0

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
  
  # emitを使ったwhileループ
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
          ctx.set("total_sum", total)
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
      - total_sum
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

📖 **詳細な使用ガイドについては [docs/usage.ja.md](docs/usage.ja.md) を参照してください**

## MABEL v2 アーキテクチャ

### MEX式言語

MEXは安全でチューリング完全な式言語を提供します:

```yaml
# 算術
{"add": [1, 2, 3]}  # 6
{"mul": [{"var": "x"}, 2]}  # x * 2
{"sub": [10, 3]}  # 7
{"div": [10, 2]}  # 5
{"mod": [10, 3]}  # 1

# 比較
{"gt": [{"var": "count"}, 10]}  # count > 10
{"lt": [{"var": "score"}, 50]}  # score < 50
{"gte": [{"var": "x"}, 0]}  # x >= 0
{"lte": [{"var": "y"}, 100]}  # y <= 100
{"eq": ["{Status}", "ok"]}  # Status == "ok"
{"ne": ["{Status}", "error"]}  # Status != "error"

# 論理
{"and": [
  {"gt": [{"var": "score"}, 80]},
  {"lt": [{"var": "errors"}, 5]}
]}
{"or": [
  {"eq": ["{Status}", "ok"]},
  {"eq": ["{Status}", "pending"]}
]}
{"not": {"eq": ["{Status}", "failed"]}}

# 文字列操作
{"concat": ["Hello, ", {"var": "name"}]}
{"replace": ["{text}", "old", "new"]}
{"length": ["{message}"]}
{"upper": ["{text}"]}
{"lower": ["{TEXT}"]}
{"trim": ["  spaced  "]}
{"split": ["{csv}", ","]}
{"join": [["a", "b", "c"], "_"]}

# コレクション
{"map": {"list": [1,2,3], "fn": {"mul": [{"var": "item"}, 2]}}}
{"filter": {"list": [1,2,3,4], "fn": {"gt": [{"var": "item"}, 2]}}}
{"reduce": {"list": [1,2,3], "fn": {"add": [{"var": "acc"}, {"var": "item"}]}, "init": 0}}
{"get": {"dict": {"a": 1, "b": 2}, "key": "a"}}
{"keys": [{"a": 1, "b": 2}]}
{"values": [{"a": 1, "b": 2}]}
{"length": [[1, 2, 3]]}

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
  mode: json  # v2: 構造化出力用のjsonモード
  outputs:
    - name: Answer
      select: jsonpath  # v2: JSONPath抽出
      path: "$.response.text"
      type_hint: string  # v2: 型変換
  save_to:  # v2: グローバル変数に保存
    vars:
      last_answer: Answer
  on_error: continue  # v2: エラーハンドリング
  retry:  # v2: リトライ設定
    max_attempts: 3
    backoff_ms: 1000
```

#### Logicブロック

##### set - 変数代入
```yaml
- type: logic
  exec: 1
  op: set
  var: total
  value: {"add": [{"var": "total"}, 10]}
```

##### let - ローカル束縛
```yaml
- type: logic
  exec: 1
  op: let
  bindings:
    x: 10
    y: {"mul": [{"var": "x"}, 2]}
  body:
    - op: set
      var: result
      value: {"add": [{"var": "x"}, {"var": "y"}]}
  outputs:
    - name: Result
      from: var
      var: result
```

##### while - 条件ループ
```yaml
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
```

##### reduce - リスト畳み込み
```yaml
- type: logic
  exec: 1
  op: reduce
  list: "{Items}"
  var: item
  value: 0  # 初期アキュムレータ値
  body:
    - op: set
      var: accumulator
      value: {"add": [{"var": "accumulator"}, {"var": "item"}]}
  outputs:
    - name: Total
      from: accumulator
```

##### call - ユーザ定義関数
```yaml
# 関数定義
functions:
  logic:
    - name: double
      args: [x]
      returns: [result]
      body:
        - op: set
          var: result
          value: {"mul": [{"var": "x"}, 2]}

# 関数呼び出し
blocks:
  - type: logic
    exec: 1
    op: call
    function: double
    with:
      x: 5
    outputs:
      - name: Doubled
        from: var
        var: result
```

##### recurse - 再帰関数
```yaml
# 再帰を使った階乗関数
- type: logic
  exec: 1
  op: recurse
  name: factorial
  function:
    args: [n]
    returns: [result]
    base_case:
      cond:
        lte: [{"var": "n"}, 1]
      value: [1]
    body:
      - op: set
        var: n_minus_1
        value: {"sub": [{"var": "n"}, 1]}
      - op: call
        name: factorial
        with:
          n: {"var": "n_minus_1"}
        returns: [sub_result]
      - op: set
        var: result
        value: {"mul": [{"var": "n"}, {"var": "sub_result"}]}
  with:
    n: 5
  outputs:
    - name: Factorial
      from: value
```

##### for - リスト反復（v1互換）
```yaml
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

##### v2: インライン関数
```yaml
- type: python
  exec: 1
  entrypoint: process
  function_code: |
    def process(ctx, data: dict) -> dict:
        # ctx.vars: グローバル変数辞書
        # ctx.get(path): コンテキストからネストされた値を取得
        # ctx.set(path, val): グローバル変数を設定
        # ctx.log(level, msg): メッセージをログ出力 (info, warning, error)
        # ctx.emit(name, value): コレクタに値を送出
        
        ctx.log("info", f"Processing {len(data)} items")
        
        # グローバル変数へのアクセス
        counter = ctx.vars.get("counter", 0)
        ctx.set("counter", counter + 1)
        
        result = {"processed": len(data)}
        return result
  inputs:
    data: "{InputData}"
  outputs: [processed]
  timeout_ms: 5000  # v2: 実行タイムアウト
```

##### v1: 外部ファイル（引き続きサポート）
```yaml
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
      value: '{"status": "complete", "count": {counter}}'
  include_vars:  # v2: 出力にグローバル変数を含める
    - counter
    - timestamp
  final_mode: map  # v2: 出力モード (map | list)
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
  const:  # 読み取り専用定数
    APP_VERSION: "1.0"
    MAX_RETRIES: 3
    API_ENDPOINT: "https://api.example.com"
  vars:   # ミュータブル変数
    counter: 0
    state: "init"
    results: []
```

### ユーザ定義関数 (v2)
```yaml
functions:
  logic:
    - name: calculate_sum
      args: [a, b]
      returns: [sum]
      body:
        - op: set
          var: sum
          value: {"add": [{"var": "a"}, {"var": "b"}]}
  
  python:
    - name: custom_transform
      args: [data]
      returns: [transformed]
      # 実装詳細...
```

## v1からv2への移行

v1 YAMLファイルは変更なしで動作します。v2機能を活用するには:

### 1. バージョンを更新
```yaml
mabel:
  version: "2.0"  # 以前は "1.0"
```

### 2. グローバル変数を追加（オプション）
```yaml
globals:
  vars:
    my_var: 0
```

### 3. MEX式を使用
```yaml
# v1 (JSON文字列、引き続き動作)
run_if: "{\"equals\":[\"{ Status}\",\"ok\"]}"

# v2 (ネイティブMEX、推奨)
run_if:
  eq: ["{Status}", "ok"]
```

### 4. インラインPythonを使用
```yaml
# v1 (外部ファイル)
- type: python
  function: helper
  code_path: ./helper.py

# v2 (インライン、シンプルな関数に推奨)
- type: python
  entrypoint: helper
  function_code: |
    def helper(ctx, x):
        return {"result": x * 2}
```

### 5. 新しいロジック演算子を活用
```yaml
# set, let, while, reduce, call, recurseを使用
- type: logic
  exec: 1
  op: set
  var: counter
  value: {"add": [{"var": "counter"}, 1]}
```

## サンプル

`examples/`ディレクトリを参照:
- `sdg_demo.yaml` - v1.0互換サンプル
- `sdg_demo_v2.yaml` - v2.0機能のショーケース
- `sdg_comprehensive_v2.yaml` - 全機能を含む包括的なv2.0サンプル
- `helpers.py` - 外部Python関数のサンプル
- `data/` - サンプル入出力データファイル

## 高度な機能

### 再帰関数
```yaml
# 再帰を使ったフィボナッチ数列
- type: logic
  exec: 1
  op: recurse
  name: fib
  function:
    args: [n]
    returns: [result]
    base_case:
      cond:
        lte: [{"var": "n"}, 1]
      value: [{"var": "n"}]
    body:
      - op: set
        var: n1
        value: {"sub": [{"var": "n"}, 1]}
      - op: set
        var: n2
        value: {"sub": [{"var": "n"}, 2]}
      - op: call
        name: fib
        with: {n: {"var": "n1"}}
        returns: [fib1]
      - op: call
        name: fib
        with: {n: {"var": "n2"}}
        returns: [fib2]
      - op: set
        var: result
        value: {"add": [{"var": "fib1"}, {"var": "fib2"}]}
  with: {n: 10}
  outputs:
    - name: Result
      from: value
```

### 複雑なMEX式
```yaml
# ネストされた条件と演算
- type: logic
  exec: 1
  op: set
  var: grade
  value:
    if:
      cond: {"gte": [{"var": "score"}, 90]}
      then: "A"
      else:
        if:
          cond: {"gte": [{"var": "score"}, 80]}
          then: "B"
          else:
            if:
              cond: {"gte": [{"var": "score"}, 70]}
              then: "C"
              else: "F"
```

### JSONPathを使ったAI
```yaml
- type: ai
  exec: 1
  model: gpt4
  mode: json
  prompts:
    - "以下のフィールドを含むJSONオブジェクトを生成: name, age, email"
  outputs:
    - name: Name
      select: jsonpath
      path: "$.name"
      type_hint: string
    - name: Age
      select: jsonpath
      path: "$.age"
      type_hint: number
    - name: Email
      select: jsonpath
      path: "$.email"
      type_hint: string
```

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
