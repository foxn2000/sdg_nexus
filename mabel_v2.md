# MABEL 2.0 完全仕様（Model And Blocks Expansion Language）
**— 新版：フルスタック仕様（v1系の機能も含めた全記述）—**

発行日: 2025-11-05

---

## 0. 目的 / 適用範囲
本書は、AI エージェントの処理フローを YAML で定義する **MABEL (Model And Blocks Expansion Language)** の完全仕様である。**v1 系で利用可能だったすべての要素**（`mabel` ヘッダ、`models`、`blocks`、`connections`、ブロック型 `ai`/`logic`/`python`/`end`、`ai.outputs` の抽出モード、`logic` の `if/and/or/not/for`、`python` の外部コード連携、`end.final` による最終出力 等）を**本書に収載**し、さらに v2 で新規追加された **統合仮想環境・インライン Python・Turing 完全な制御構造**を包括する。

> 本仕様だけで MABEL の文書を作成・検証・実行できることを目標とする。

---

## 1. YAML 全体構造（トップレベル）
MABEL ドキュメントは 1 つの YAML ファイルで表現され、**トップレベル**は原則として以下のキーを持つ。

```yaml
mabel:            # 言語メタ情報
  version: "2.0" # 本仕様のバージョン。文字列固定
  dialect: "mabel-2"   # 将来の派生方言識別子（任意）
  id: "com.example.agent.demo"  # 文書ID（任意）
  name: "Demo Agent"            # 表示名（任意）
  description: "Demo pipeline with AI/Logic/Python/End"  # 概要（任意）

runtime:          # 実行時環境（v2 で新設/拡張）
  python:
    interpreter: "python>=3.11,<3.13"   # PEP 440 互換
    venv: ".venv"                        # ワークフロー全体の仮想環境
    requirements_file: "requirements.txt" # 任意：requirements ファイル
    requirements:                         # 任意：追加/上書きの配列
      - "numpy==2.*"
      - "httpx>=0.27"
    allow_network: false                  # 既定は外部ネットワーク遮断
    env:                                  # 環境変数（必要なら）
      OPENAI_API_KEY: ${ENV.OPENAI_API_KEY}
    setup:                                # 任意のセットアップフック
      pre_install: []
      post_install: []

budgets:          # グローバル予算（安全停止・制限）
  loops:
    max_iters: 10000
    on_exceed: "error"     # "error" | "truncate" | "continue"
  recursion:
    max_depth: 256
    on_exceed: "error"
  wall_time_ms: 300000      # 全体のウォールタイム上限（例：5分）
  ai:
    max_calls: 64
    max_tokens: 100000

models:           # AIモデル定義の配列（v1 継承／完全定義）
  - name: "planner"                   # ブロックから参照する識別名
    api_model: "gpt-4o-mini"          # API 上のモデル名
    api_key: ${ENV.OPENAI_API_KEY}
    base_url: "https://api.openai.com/v1"  # 任意
    organization: null                     # 任意（プロバイダー固有）
    headers: {}                            # 任意の追加 HTTP ヘッダ
    request_defaults:                      # 呼び出し既定値
      temperature: 0.0
      top_p: 1.0
      max_tokens: 800
      timeout_sec: 120
      retry:
        max_attempts: 2
        backoff: { type: "exponential", base_ms: 500 }
    # 任意ヒント
    capabilities: ["json_mode", "tool_calling"]
    safety: {}

globals:          # グローバル変数/定数（v2 で整理）
  const:          # 読み取り専用（上書き不可）
    APP_NAME: "NEXUS"
  vars:           # 実行中に読み書き可能
    counter: 0
    memo: {}

functions:        # ユーザ関数群（v2）
  logic: []       # MEX/ロジック関数（§6.2.6）
  python: []      # インラインPython関数（§6.3.2）

templates:        # 文字列テンプレート（任意）
  - name: "report"
    text: |
      App: {APP_NAME}\nAnswer: {Answer}

files:            # 任意：組み込みテキスト/バイナリ（base64等）
  - name: "terms.txt"
    mime: "text/plain"
    content: "..."

blocks: []        # 実行ブロック群（§6）

connections: []   # 明示配線（任意、§8）
```

> **互換注意**: v1 文書の `mabel.version` は "1.0" であった。v2 では 2.0 を必須とする。`runtime` は v2 で導入されたが、v1 の動作互換のために **存在しない場合は実装が既定値を補う**（例：`.venv` を暗黙作成、ネットワーク遮断）。

---

## 2. データモデル / 型 / パス参照

### 2.1 基本型
- `null` / `boolean` / `number` / `string` / `list` / `object`（YAML に準じる）

### 2.2 出力名と変数参照
- **出力名参照**: `{OutputName}` でブロック出力を参照。
- **変数参照**: `{VarName}` または `{a.b[0]}` のようなパス。`globals.vars` を起点とする。
- **テンプレート**: `templates[].text` 内で `{...}` 展開可。

### 2.3 環境変数注入
- `${ENV.NAME}` 記法を値に埋め込むと、実行時に環境変数から展開される。

---

## 3. 実行モデル（共通規約）
1. ブロックは `exec` の昇順で評価。
2. `run_if` が **真** のときのみ実行。
3. 各ブロックは `outputs` に従って**名前付き出力**を公開。
4. 例外・予算超過は `on_error` または `budget.on_exceed` に従って処理。
5. `end` ブロック実行でフロー終了、`final` に基づき応答ペイロードを組み立てる。

### 3.1 ブロック共通フィールド

| フィールド | 必須 | 型/既定 | 説明 |
|---|:--:|---|---|
| `type` | ✓ | `string` | `ai` / `logic` / `python` / `end` |
| `exec` | ✓ | `integer` | 実行順序 |
| `id` |  | `string` | 明示 ID。`connections` で参照可能 |
| `name` |  | `string` | ラベル |
| `run_if` |  | `string` or `object` | 条件式。v1 互換として **JSON 文字列**表記を許容。v2 では **MEX 式**（§6.2.2）を推奨 |
| `on_error` |  | `string` | `"fail"`（既定）/`"continue"`/`"retry"` |
| `retry` |  | `object` | `on_error: "retry"` の詳細（`max_attempts`, `backoff`）|
| `budget` |  | `object` | このブロックに限定した予算上書き（`loops`, `recursion`, `wall_time_ms`, `ai`）|
| `outputs` |  | `array` | ブロック固有（§6 各節）|

---

## 4. モデル定義（`models`）
`models` は AI モデル接続の宣言一覧である。**各要素**のフィールドは以下。

| フィールド | 必須 | 型/例 | 説明 |
|---|:--:|---|---|
| `name` | ✓ | `"planner"` | ブロックから参照するモデル識別子 |
| `api_model` | ✓ | `"gpt-4o-mini"` | 実際の API 上モデル名 |
| `api_key` | ✓ | `${ENV.OPENAI_API_KEY}` | 認証キー |
| `base_url` |  | `"https://api.openai.com/v1"` | エンドポイント |
| `organization` |  | `string` | 任意の組織 ID |
| `headers` |  | `object` | 追加ヘッダ（`{"User-Agent":"Mabel"}` など） |
| `request_defaults` |  | `object` | `temperature`, `top_p`, `max_tokens`, `timeout_sec`, `retry` 等 |
| `capabilities` |  | `list` | 実装ヒント：`json_mode`, `tool_calling` 等 |
| `safety` |  | `object` | セーフティポリシー |

**推奨**: セキュアな運用のため `api_key` は環境変数注入を用いる。

---

## 5. 文字列テンプレート（`templates`）
任意。`name` と `text` を持ち、`{...}` 展開が行える。テンプレートは `ai.prompts` や `end.final.value` 等から挿入可能。

---

## 6. ブロック仕様（`blocks[]`）

### 6.1 AI ブロック（`type: ai`）
**機能**: モデルにプロンプトを送信し、応答を取得して出力に変換する。

```yaml
- type: ai
  exec: 1
  id: "ask"
  model: planner
  system_prompt: |
    You are a concise planner.
  prompts:
    - |
      Summarize:
      {UserInput}
  params:                 # 任意：呼び出し時上書き
    temperature: 0.1
    max_tokens: 400
    stop: ["\nEND"]
  attachments:            # 任意：補助テキスト/ファイル
    - name: "spec"
      mime: "text/plain"
      content: "..."
  mode: "text"            # text | json（JSONモード）
  outputs:
    - name: Answer
      select: full        # full | tag | regex | jsonpath
    - name: Title
      select: regex
      regex: "(?s)^(.*?)\n"  # 先頭行
    - name: FirstCode
      select: tag
      tag: "code"
      join_with: "\n\n"
    - name: JsonField
      select: jsonpath
      path: "$.data.value"
      type_hint: json     # string|number|boolean|json
  save_to:
    vars:                 # 応答をグローバル変数に保存（任意）
      last_answer: Answer
```

**抽出規則**
- `select: full` — 応答全文。
- `select: tag` — タグ名で抽出（Markdown/HTML 解析を実装依存でサポート）。
- `select: regex` — 正規表現で抽出。複数ヒット時はリスト。
- `select: jsonpath` — JSON モードのとき JSONPath で抽出。
- `type_hint` — 文字列を型変換。

**エラー/再試行**
- ブロック内 `on_error: "retry"` 時、`retry` 設定に従う。グローバル `models[].request_defaults.retry` より優先。

---

### 6.2 Logic ブロック（`type: logic`）
**機能**: 条件分岐、反復、集合処理、代入、再帰などのロジックを記述する。

#### 6.2.1 v1 の基本演算
- `op: if` — 条件分岐
- `op: and` / `op: or` / `op: not` — 論理演算
- `op: for` — 反復/フィルタ/マップ

**v1 互換の `run_if`/条件式**: JSON 文字列で表す。
```yaml
run_if: "{\"equals\":[\"{Flag}\",\"on\"]}"
```

**`op: for` の詳細**
```yaml
- type: logic
  exec: 10
  name: "loop_lines"
  op: for
  list: "{Answer}"            # 反復対象
  parse: lines                 # lines|csv|json|regex（任意）
  regex_pattern: "^(.+)$"      # parse: regex のとき
  var: item                    # ループ変数名（既定: item）
  drop_empty: true
  where: { "ne": ["{item}", ""] }  # 条件式(JSON)
  map: "Line: {item}"          # テンプレート
  outputs:
    - name: Joined
      from: join               # boolean|value|join|count|any|all|first|last|list
      source: mapped           # raw|filtered|mapped
      join_with: "\n"
```

#### 6.2.2 v2 の式言語 MEX（MABEL EXPR）
**MEX** は JSON 風の式で、`run_if`、`logic` 本文、`value` 計算などで用いる。例：
```yaml
{"add": [1, {"mul": [{"var": "x"}, 2]}]}
{"if": {"cond": {"gt":[{"var":"n"}, 0]}, "then": "pos", "else": "non-pos"}}
{"and": [ {"eq":[{"var":"a"}, 1]}, {"not":{"lt":[{"var":"b"}, 3]}} ]}
```

**主な演算子**
- 論理: `and`, `or`, `not`
- 比較: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- 算術: `add`, `sub`, `mul`, `div`, `mod`, `pow`, `neg`
- 文字列: `concat`, `split`, `replace`, `lower`, `upper`, `trim`, `len`
- コレクション: `map`, `filter`, `reduce`, `any`, `all`, `unique`, `sort`, `slice`
- 正規表現: `regex_match`, `regex_extract`, `regex_replace`
- 制御: `if`, `case`（`when:` 配列）
- 参照: `var`（変数）, `ref`（出力名）, `get`（パス参照）
- 代入: `set`（`var` と `value`）, `let`（ローカル束縛）
- 時間/乱数: `now`, `rand`
- 変換: `to_number`, `to_string`, `to_boolean`, `parse_json`, `stringify`

> v1 の JSON 条件式は、そのまま MEX として解釈可能。

#### 6.2.3 代入/束縛（`op: set` / `op: let`）
```yaml
- type: logic
  exec: 20
  op: set
  var: total
  value: {"add": [{"var":"total"}, 10]}
```

```yaml
- type: logic
  exec: 21
  op: let
  bindings: { x: 2, y: 3 }
  body:
    - op: set
      var: tmp
      value: {"mul": [{"var":"x"}, {"var":"y"}]}
  outputs:
    - name: Product
      from: var
      var: tmp
```

#### 6.2.4 反復（`op: while`）
v2 で追加。条件が真の間、`step` を反復する。
```yaml
- type: logic
  exec: 30
  op: while
  init:
    - op: set
      var: i
      value: 0
  cond: {"lt":[{"var":"i"}, 10]}
  step:
    - op: set
      var: i
      value: {"add":[{"var":"i"}, 1]}
    - op: emit
      value: {"var":"i"}        # 収集
  budget:
    loops: { max_iters: 1000, on_exceed: "error" }
  outputs:
    - name: Iters
      from: list                    # emit の収集結果
```

#### 6.2.5 再帰（`op: recurse`）
自己/相互再帰を記述できる。Turing 完全性を担保。
```yaml
- type: logic
  exec: 31
  op: recurse
  name: "fib"               # 関数名（自己参照用）
  function:
    args: [n]
    returns: [f]
    base_case:
      cond: {"le":[{"var":"n"}, 1]}
      value: [1]
    body:
      - op: call
        name: "fib"
        with: { n: {"sub":[{"var":"n"}, 1]} }
        returns: [a]
      - op: call
        name: "fib"
        with: { n: {"sub":[{"var":"n"}, 2]} }
        returns: [b]
      - op: set
        var: f
        value: {"add":[{"var":"a"}, {"var":"b"}]}
  with: { n: 10 }
  budget:
    recursion: { max_depth: 64, on_exceed: "error" }
  outputs:
    - name: Fib10
      from: value                 # 最終 f
```

#### 6.2.6 ロジック関数呼び出し（`op: call` / `functions.logic`）
ロジック関数を定義して再利用できる。
```yaml
functions:
  logic:
    - name: "inc"
      args: [x]
      returns: [y]
      body:
        - op: set
          var: y
          value: {"add": [{"var":"x"}, 1]}

blocks:
  - type: logic
    exec: 40
    op: call
    name: "use_inc"
    with: { x: 41 }
    returns: [Answer]
```

#### 6.2.7 Logic 出力定義
`logic.outputs[].from` は以下を取る：
- `boolean` / `value` / `join` / `count` / `any` / `all` / `first` / `last` / `list` / `var` / `accumulator`

---

### 6.3 Python ブロック（`type: python`）
**機能**: Python コード/関数を実行し、出力を返す。

#### 6.3.1 v1 互換フィールド
- `name`（必須）: ブロック名
- `function`（必須）: 呼び出す関数名
- `inputs`（任意）: 引数名の**配列**（例: `[Answer, Plan]`）
- `code_path`（任意）: 実行モジュールのパス（例: `./script.py`）
- `venv_path`（任意, 互換）: 旧フィールド。**v2 では非推奨**（`runtime.python` を使用）
- `outputs`（必須）: 返す出力名の配列

#### 6.3.2 v2 拡張（インライン関数・統合環境）
追加フィールド：
- `function_code`（任意）: **インライン Python** のソースコード。
- `entrypoint`（任意）: 関数名（既定: `main`）。`function` と同義で、どちらかを使用。
- `inputs` マップ対応: 位置引数配列に加え、`{name: value}` 形式の**キーワード引数**も可。
- `use_env`（任意）: `"global"`（既定。`runtime.python.venv` を使う）/`"override"`（個別環境）。
- `override_env`（任意）: `use_env: "override"` の場合に `venv`, `requirements`, `requirements_file`, `allow_network`, `env` などを指定。
- `timeout_ms`（任意）: ブロック実行の時間制限。
- `ctx_access`（任意）: `vars.read`/`vars.write`/`files.read`/`files.write`/`net` 等の**最小権限宣言**。

**関数シグネチャ規約**
```python
def main(ctx, **inputs) -> dict:
    """
    ctx.vars: グローバル変数(dict)
    ctx.get(path), ctx.set(path, value)
    ctx.emit(name, value)   # logic の emit 相当
    ctx.call_ai(model, system, prompts, params) -> raw
    ctx.log(level, message) # "debug"|"info"|"warn"|"error"
    return { "Out1": value, ... }  # `outputs` で宣言したキー
    """
```

**例：インライン関数**
```yaml
- type: python
  exec: 50
  name: "normalize"
  entrypoint: "normalize_text"
  inputs:
    text: "{Answer}"
  function_code: |
    def normalize_text(ctx, text: str) -> dict:
        return {"Normalized": " ".join(text.split())}
  outputs: [Normalized]
  use_env: "global"
  timeout_ms: 5000
  ctx_access: ["vars.write"]
```

---

### 6.4 End ブロック（`type: end`）
**機能**: フローを終了し、最終応答を構築。

```yaml
- type: end
  exec: 999
  reason: "completed"
  exit_code: "success"
  final:
    - name: answer
      value: "{Answer}"
    - name: meta
      value: "{Plan}"
  final_mode: "map"           # map|list（既定: map）
  include_vars: ["counter"]   # 任意：グローバル変数を返す
```

---

## 7. 予算（Budgets）
**目的**: 無限ループや過度な再帰を防止し安全に停止する。

- **グローバル**: `budgets.*`（§1）
- **ブロック局所**: `blocks[].budget` で上書き可

```yaml
budget:
  loops: { max_iters: 1000, on_exceed: "truncate" }
  recursion: { max_depth: 64, on_exceed: "error" }
  wall_time_ms: 20000
  ai: { max_calls: 8, max_tokens: 16000 }
```

---

## 8. 明示配線（`connections`）
自動配線（**同名**の入出力を自動結線）に加え、明示的に接続を記述できる。

```yaml
connections:
  - from: block_id_1
    output: Answer
    to: block_id_2
    input: Plan
  - from: block_id_2
    output: Plan
    to: block_id_3
    input: response
```

各ブロックには `id` を付与して参照する。`output`/`input` はブロック内で宣言した名前。

---

## 9. セキュリティ / サンドボックス
- 既定で `runtime.python.allow_network: false`（外部通信禁止）。
- `ctx_access` により権限を最小化。
- 機密値は `${ENV.*}` を用いて注入。YAMLに生埋めしない。

---

## 10. エラー処理 / 再試行 / ログ
- `on_error: "fail"|"continue"|"retry"`。`retry` は `max_attempts` と `backoff`（`type: exponential|fixed`, `base_ms`）を取る。
- 予算超過は `on_exceed` の方針で処理。
- ログ API（実装依存）: `ctx.log(level, message)`、実行トレース保存。

---

## 11. 形式仕様（Schema 概観）

### 11.1 トップレベル（概観）
```yaml
mabel:
  version: { type: string, const: "2.0" }
runtime:
  python:
    interpreter: string
    venv: string
    requirements_file: string?
    requirements: list<string>?
    allow_network: boolean?
    env: object?
    setup: { pre_install?: list<string>, post_install?: list<string> }
budgets:
  loops: { max_iters: int, on_exceed?: enum[error,truncate,continue] }
  recursion: { max_depth: int, on_exceed?: enum[error,truncate,continue] }
  wall_time_ms?: int
  ai?: { max_calls?: int, max_tokens?: int }
models: list<Model>
globals: { const?: object, vars?: object }
functions: { logic?: list<LogicFn>, python?: list<PythonFn> }
templates: list<{name:string,text:string}>
files: list<{name:string,mime:string,content:string}>
blocks: list<Block>
connections: list<Connection>
```

### 11.2 `Model`
```yaml
name: string
api_model: string
api_key: string
base_url?: string
organization?: string
headers?: object
request_defaults?: { temperature?: number, top_p?: number, max_tokens?: int, timeout_sec?: int, retry?: { max_attempts?: int, backoff?: { type: string, base_ms?: int } } }
capabilities?: list<string>
safety?: object
```

### 11.3 `Block`（共通）
```yaml
type: enum[ai,logic,python,end]
exec: int
id?: string
name?: string
run_if?: string|object  # JSON 文字列 or MEX
on_error?: enum[fail,continue,retry]
retry?: { max_attempts?: int, backoff?: { type: string, base_ms?: int } }
budget?: { loops?: {max_iters:int,on_exceed?:string}, recursion?:{max_depth:int,on_exceed?:string}, wall_time_ms?:int, ai?:{max_calls?:int,max_tokens?:int} }
outputs?: list<Output>
```

### 11.4 `ai` ブロック専用
```yaml
model: string
system_prompt?: string
prompts: list<string>
params?: object
attachments?: list<{name:string,mime:string,content:string}>
mode?: enum[text,json]
outputs: list<AiOutput>
save_to?: { vars?: object }
```

### 11.5 `logic` ブロック専用
```yaml
op: enum[if,and,or,not,for,while,recurse,set,let,reduce,call,emit]
# for
list?: any
parse?: enum[lines,csv,json,regex]
regex_pattern?: string
var?: string
drop_empty?: boolean
where?: object   # JSON/MEX
map?: string
# while
init?: list<Step>
cond?: object    # MEX
step?: list<Step>
# recurse/call
name?: string
function?: { args:list<string>, returns:list<string>, base_case:{cond:object,value:list<any>}, body:list<Step> }
with?: object
returns?: list<string>
```

### 11.6 `python` ブロック専用
```yaml
function?: string
entrypoint?: string
inputs?: list<string>|object
code_path?: string
function_code?: string
use_env?: enum[global,override]
override_env?: { venv?: string, requirements?: list<string>, requirements_file?: string, allow_network?: boolean, env?: object }
timeout_ms?: int
ctx_access?: list<string>
outputs: list<string>
venv_path?: string  # 互換（非推奨）
```

### 11.7 `end` ブロック専用
```yaml
reason?: string
exit_code?: string
final?: list<{name:string,value:any}>
final_mode?: enum[map,list]
include_vars?: list<string>
```

### 11.8 `Output`/`AiOutput`
```yaml
# Output (logic 共通)
name: string
from: enum[boolean,value,join,count,any,all,first,last,list,var,accumulator]
var?: string
join_with?: string

# AiOutput
name: string
select: enum[full,tag,regex,jsonpath]
tag?: string
regex?: string
path?: string
join_with?: string
type_hint?: enum[string,number,boolean,json]
```

### 11.9 `Connection`
```yaml
from: string   # 出力側ブロック ID
output: string # 出力名
to: string     # 入力側ブロック ID
input: string  # 入力名
```

---

## 12. ベストプラクティス
- 仮想環境は**原則1つ**（`runtime.python.venv`）。例外のみ `override_env`。
- ループ/再帰には**明示予算**を付与。
- `ai` の抽出は `json` モード＋`jsonpath` を優先（構造化）。
- Python 関数は可能な限り**純粋関数**で実装し、副作用は `ctx_access` で明示。
- 出力/入力名は**一貫した命名**（`snake_case` 推奨）。

---

## 13. マイグレーション指針（v1 → v2）
1. `mabel.version` を `"2.0"` に更新。
2. 旧 `venv_path` は削除し、`runtime.python.venv` を使用。必要なら `use_env: "override"` + `override_env` を指定。
3. `run_if` の JSON 文字列はそのまま利用可能。可能なら MEX へ正規化。
4. `logic.for` の `parse/where/map` は同名で継続。`while/recurse/set/let/reduce/call/emit` が追加可能。
5. 共通関数は `functions.logic` / `functions.python` に切り出し、再利用性を高める。

---

## 14. 例題集

### 14.1 最小（Hello）
```yaml
mabel:
  version: "2.0"
blocks:
  - type: logic
    exec: 1
    op: set
    var: greeting
    value: "Hello, World"
  - type: end
    exec: 2
    final:
      - name: message
        value: "{greeting}"
```

### 14.2 v1 風：AI→AI→logic→python→end
```yaml
mabel:
  version: "2.0"
models:
  - name: questioner
    api_model: gpt-4o-mini
    api_key: ${ENV.OPENAI_API_KEY}
    request_defaults: { temperature: 0.2, max_tokens: 300 }
  - name: responder
    api_model: gpt-4.1
    api_key: ${ENV.OPENAI_API_KEY}
    request_defaults: { temperature: 0.5, max_tokens: 800 }
blocks:
  - type: ai
    exec: 1
    id: q
    model: questioner
    system_prompt: |
      You formulate concise questions.
    prompts:
      - |
        Summarize the key question from:
        {UserInput}
    outputs:
      - name: Question
        select: full
  - type: ai
    exec: 2
    id: a
    model: responder
    system_prompt: |
      You answer clearly and accurately.
    prompts:
      - |
        Provide a detailed answer:
        {Question}
    outputs:
      - name: Answer
        select: full
      - name: ShortAnswer
        select: regex
        regex: "(?s)^(.*?)\\n"
  - type: logic
    exec: 3
    id: c
    name: Check
    op: if
    cond: {"equals":["{ShortAnswer}",""]}
    then: "No short answer."
    else: "Short answer available."
    outputs:
      - name: Flag
        from: boolean
  - type: python
    exec: 4
    id: p
    name: format
    entrypoint: format_output
    inputs: [Answer, Flag]
    code_path: ./helpers.py
    outputs: [Formatted]
  - type: end
    exec: 5
    final:
      - name: answer
        value: "{Formatted}"
      - name: status
        value: "{Flag}"
```

### 14.3 統合仮想環境＋インライン Python
```yaml
mabel:
  version: "2.0"
runtime:
  python:
    interpreter: "python>=3.11,<3.13"
    venv: ".venv"
    requirements: ["numpy==2.*"]
blocks:
  - type: python
    exec: 1
    name: stats
    function_code: |
      import numpy as np
      def main(ctx, **inputs):
          arr = np.array([1,2,3,4,5], dtype=float)
          return {"Mean": float(arr.mean())}
    outputs: [Mean]
  - type: end
    exec: 2
    final:
      - name: mean
        value: "{Mean}"
```

### 14.4 `while`：ユークリッド互除法
```yaml
mabel:
  version: "2.0"
globals:
  vars: { a: 1071, b: 462 }
blocks:
  - type: logic
    exec: 1
    op: while
    cond: {"ne":[{"var":"b"},0]}
    step:
      - op: set
        var: tmp
        value: {"mod":[{"var":"a"},{"var":"b"}]}
      - op: set
        var: a
        value: {"var":"b"}
      - op: set
        var: b
        value: {"var":"tmp"}
    budget: { loops: { max_iters: 1000 } }
    outputs:
      - name: GCD
        from: var
        var: a
  - type: end
    exec: 2
    final:
      - name: gcd
        value: "{GCD}"
```

### 14.5 `recurse`：フィボナッチ（メモ化）
```yaml
mabel:
  version: "2.0"
globals:
  vars: { memo: {"0":0, "1":1} }
blocks:
  - type: logic
    exec: 1
    op: recurse
    name: "fib"
    function:
      args: [n]
      returns: [f]
      base_case:
        cond: {"or":[{"le":[{"var":"n"},1]}, {"get":[{"var":"memo"},{"path":"{n}"}]}]}
        value:
          - {"get":[{"var":"memo"},{"path":"{n}"}], "default": {"var":"n"}}
      body:
        - op: call
          name: "fib"
          with: { n: {"sub":[{"var":"n"},1]} }
          returns: [a]
        - op: call
          name: "fib"
          with: { n: {"sub":[{"var":"n"},2]} }
          returns: [b]
        - op: set
          var: f
          value: {"add":[{"var":"a"},{"var":"b"}]}
        - op: set
          var: memo
          value: {"set":{"in":{"var":"memo"}, "path":"{n}", "value":{"var":"f"}}}
    with: { n: 20 }
    budget: { recursion: { max_depth: 128 } }
    outputs:
      - name: Fib20
        from: value
  - type: end
    exec: 2
    final:
      - name: fib
        value: "{Fib20}"
```

### 14.6 AI→Python→`while` の複合
```yaml
mabel:
  version: "2.0"
runtime:
  python:
    interpreter: "python>=3.11,<3.13"
    venv: ".venv"
models:
  - name: planner
    api_model: gpt-4o-mini
    api_key: ${ENV.OPENAI_API_KEY}
    request_defaults: { temperature: 0.0, max_tokens: 400 }
globals:
  vars: { done: false, iteration: 0, plan: "" }
blocks:
  - type: ai
    exec: 1
    model: planner
    system_prompt: |
      You are a concise planner. Improve the plan until DONE.
    prompts:
      - |
        Iteration: {iteration}
        Current Plan:\n{plan}
        Improve the plan and say "DONE" on the last line when complete.
    outputs: [{ name: Draft, select: full }]
  - type: python
    exec: 2
    name: check_done
    function_code: |
      def main(ctx, Draft: str) -> dict:
          lines = Draft.strip().splitlines()
          done = (lines[-1].strip() == "DONE") if lines else False
          new_plan = Draft if not done else "\n".join(lines[:-1]).strip()
          ctx.vars["done"] = done
          ctx.vars["plan"] = new_plan
          ctx.vars["iteration"] = ctx.vars.get("iteration", 0) + 1
          return {"Done": done, "Plan": new_plan}
    inputs: { Draft: "{Draft}" }
    outputs: [Done, Plan]
    ctx_access: ["vars.write"]
  - type: logic
    exec: 3
    op: while
    cond: {"not":{"var":"done"}}
    step:
      - op: emit
        value: {"var":"plan"}
      - op: set
        var: iteration
        value: {"add":[{"var":"iteration"},1]}
    budget: { loops: { max_iters: 10, on_exceed: "truncate" } }
    outputs:
      - name: PlanHistory
        from: list
  - type: end
    exec: 100
    final:
      - { name: final_plan, value: "{plan}" }
      - { name: iterations, value: "{iteration}" }
      - { name: history, value: "{PlanHistory}" }
```

---

## 15. 実装ノート（参考）
- 実装は **`exec` の安定ソート**で評価すると副作用順序が予測可能。
- `while` ステップの**増分評価**（ループ本体へのブロック再実行）は、実装側のスケジューラで扱う。
- MEX は安全のため **関数呼出し/属性アクセスを禁止**し、演算子ホワイトリストを維持する。

---

## 16. まとめ
- **統合仮想環境**（`runtime.python`）で再現性を確保。
- **インライン Python** と `functions.python` で迅速な拡張。
- **`while`/`recurse`/`set`/`let`/`reduce`/`call`/`emit`** を備え**Turing 完全**なロジックを安全な予算付きで実現。
- v1 機能は本仕様に**完全内包**され、単独で利用可能。