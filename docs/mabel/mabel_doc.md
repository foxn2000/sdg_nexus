# MABEL 記法 v2.1 ― 実装準拠ドキュメント（完全版）

**対象実装**: `/sdg_nexus/sdg` ディレクトリ配下（`config.py / executors.py / llm_client.py / utils.py / mex.py / runner.py / cli.py`）
**準拠バージョン**: `mabel.version: "2.0"` または `"2.1"`（v2.1 で画像入力サポートを追加）

> 本書は、同梱のバックエンド実装が読み込む YAML スキーマと実行モデルを、**コードで確認できるフィールド名／振る舞い**に基づいて、**読めばそのまま MABEL の YAML が書ける**形でまとめたものです。
> 以降、各節末の「実装ポイント」では、対応するクラスやフィールド（例: `AIBlock.model`, `LogicBlock.op` など）を明示して、**仕様⇄実装**の対応関係を確定させます。

---

## 1. YAML 全体構造（トップレベル）

MABEL ドキュメントは 1 ファイルの YAML です。**トップレベルに置くキー**と典型値は以下。

```yaml
mabel:
  version: "2.0"            # ← 必須。実装は v2 を前提
  dialect: "mabel-2"        # 任意。将来の方言識別に予約
  id: "com.example.agent"   # 任意の文書ID
  name: "Demo Agent"        # 任意の表示名
  description: "AI/Logic/Python/End のパイプライン"

runtime:                    # v2: 実行環境
  python:
    interpreter: "python>=3.11,<3.13"  # 目安（実装は文字列を保持）
    venv: ".venv"                      # プロジェクト用仮想環境パス
    requirements_file: "requirements.txt"   # 任意
    allow_network: false               # 既定で外部通信を禁止（サンドボックス方針）
    env:                               # 子プロセスや関数に渡す環境変数
      OPENAI_API_KEY: "${ENV.OPENAI_API_KEY}"

globals:                    # v2: グローバルの定数と変数
  const:                    # 読み取り専用
    APP_NAME: "NEXUS"
    MAX_TURNS: 3
  vars:                     # 実行中に更新可能
    counter: 0
    memo: ""

budgets:                    # v2: 安全停止のしきい値
  loops:    { max_iters: 1000, on_exceed: "truncate" }  # while など
  recursion:{ max_depth: 64,   on_exceed: "error"     }  # recurse
  wall_time_ms: 20000
  ai: { max_calls: 8, max_tokens: 16000 }

functions:                  # v2: ユーザ定義関数群
  logic:   []               # MEX/ロジック関数（§6.2.6）
  python:  []               # インライン Python（§6.3.2）

models:                     # 接続先 LLM 群（§4）
  - name: planner
    api_model: "gpt-4o-mini"
    api_key: "${ENV.OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    request_defaults:
      temperature: 0.2
      top_p: 1.0
      max_tokens: 800
      timeout_sec: 120
      retry:
        max_attempts: 2
        backoff: { type: "exponential", base_ms: 500 }
    capabilities: ["json_mode", "tool_calling"]
    safety: {}

templates:                  # 任意: 文字列テンプレート（§5）
  - name: "report"
    text: |
      App: {APP_NAME}
      Answer: {Answer}

files:                       # 任意: 同梱ファイル（テキスト/バイナリ）
  - name: "terms.txt"
    mime: "text/plain"
    content: "..."

images:                      # v2.1: 画像定義（§4.5）
  - name: "logo"
    path: "./assets/logo.png"           # ローカルファイル
  - name: "reference"
    url: "https://example.com/ref.png"  # URL
  - name: "inline_img"
    base64: "iVBORw0KGgoAAAANSUhEUg..."  # Base64埋め込み
    media_type: "image/png"

blocks: []                  # 実行ブロック（§6）

connections: []             # 明示配線（§8、**現実装では解析のみ**）
```

**実装ポイント**

* トップレベルは `SDGConfig` のフィールドに読み込まれます（`config.py`）。`models / blocks` は必須、`runtime / budgets / globals / functions / templates / files / connections` は存在すれば取り込み。
* 実行は `runner.run()` → `executors.run_pipeline()` で駆動。ブロックは `exec` の昇順で処理（§3）。

---

## 2. データモデル / 型 / パス参照

### 2.1 型

YAML 標準: `null / boolean / number / string / list / object`。

### 2.2 出力名・変数参照・テンプレート

* **ブロック出力参照**: `{OutputName}`
* **グローバル変数参照**: `{VarName}` や `{a.b[0]}` など（起点は `globals.vars`）
* **テンプレート展開**: 文字列テンプレート（`templates[].text`、`ai.prompts` など）に `{...}` を埋め込むと、実行時に `utils.render_template()` で置換されます。

### 2.3 環境変数注入

`${ENV.NAME}` の形で YAML 値に書くと、実行時に OS の環境変数へ解決（例: `${ENV.OPENAI_API_KEY}`）。

**実装ポイント**

* 文字列置換は `utils.PLACEHOLDER_RE` による `{name}` 走査で実行。
* 参照可能なのは **先行ブロックの出力**および `globals.vars`。未定義は空文字として展開。

---

## 3. 実行モデル（共通規約）

1. ブロックは `exec` の昇順で実行。
2. `run_if` が **真** のときのみ実行（v1 の JSON 条件、v2 は MEX 式も可）。
3. 各ブロックは `outputs` に従って**名前付き出力**を作る（下流ブロックから `{Name}` で参照）。
4. 例外や予算超過は `on_error` / `budget.on_exceed` に従って処理。
5. `end` ブロックが最終応答を `final` に従って構築し、レコード出力を確定。

**共通フィールド**

| フィールド      |  必須 | 型 / 既定               | 説明                                  |
| ---------- | :-: | -------------------- | ----------------------------------- |
| `type`     |  ✓  | `string`             | `ai` / `logic` / `python` / `end`   |
| `exec`     |  ✓  | `integer`            | 実行順序                                |
| `id`       |     | `string`             | 任意 ID（`connections` 用の参照名）          |
| `name`     |     | `string`             | ラベル                                 |
| `run_if`   |     | `object` or `string` | v1 条件（JSON）または v2 MEX（§6.2.2）       |
| `on_error` |     | `"fail"`             | `"fail"` / `"continue"` / `"retry"` |
| `retry`    |     | `object`             | `on_error: "retry"` 時の再試行構成         |
| `budget`   |     | `object`             | ブロック局所の予算上書き（§7）                    |
| `outputs`  |     | `array`              | 各ブロック固有の出力記述                        |

**実装ポイント**

* 逐次評価と結果の束縛は `executors.run_pipeline()`。`on_error: continue` はブロック例外を握り潰し、当該レコードに `error_block_{exec}` を残します。
* 予算超過は `BudgetExceeded` を送出（`executors.py`）。

---

## 4. モデル定義（`models`）

| フィールド              |  必須 | 例                             | 説明                                                  |
| ------------------ | :-: | ----------------------------- | --------------------------------------------------- |
| `name`             |  ✓  | `"planner"`                   | ブロックから参照する識別子                                       |
| `api_model`        |  ✓  | `"gpt-4o-mini"`               | 実 API 上のモデル名                                        |
| `api_key`          |  ✓  | `"${ENV.OPENAI_API_KEY}"`     | 認証キー                                                |
| `base_url`         |     | `"https://api.openai.com/v1"` | Chat Completions 互換エンドポイント                          |
| `organization`     |     | `string`                      | 任意の組織 ID                                            |
| `headers`          |     | `object`                      | 追加ヘッダ                                               |
| `request_defaults` |     | `object`                      | `temperature/top_p/max_tokens/timeout_sec/retry` など |
| `capabilities`     |     | `list`                        | `"json_mode"`, `"tool_calling"` 等のヒント               |
| `safety`           |     | `object`                      | 任意の安全設定                                             |

**実装ポイント**

* OpenAI 互換の `/v1/chat/completions` で実行（`llm_client.AsyncOpenAI`）。
* ブロック側 `params` は `models[].request_defaults` を**上書き**します（`executors.py` がマージ）。

```yaml
models:
  - name: writer
    api_model: "gemma-3-1b-it"
    api_key: "${ENV.OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    request_defaults:
      temperature: 0.3
      max_tokens: 400

  # v2.1: Vision対応モデル
  - name: vision_model
    api_model: "gpt-4o"
    api_key: "${ENV.OPENAI_API_KEY}"
    capabilities: ["vision"]  # Vision対応を明示
```

---

## 4.5 画像定義（`images`）— v2.1

v2.1 では画像入力をサポートします。`images` セクションで静的画像を定義し、プロンプト内で `{name.img}` 記法で参照できます。

### 画像定義フォーマット

| フィールド    |  必須 | 型       | 説明                                    |
| ------------ | :--: | -------- | --------------------------------------- |
| `name`       |  ✓   | `string` | 画像の識別名（プロンプトで `{name.img}` として参照） |
| `path`       |      | `string` | ローカルファイルパス                      |
| `url`        |      | `string` | 画像URL（HTTPSを推奨）                   |
| `base64`     |      | `string` | Base64エンコードされた画像データ          |
| `media_type` |      | `string` | MIMEタイプ（既定: `image/png`）          |

**注**: `path`、`url`、`base64` のいずれか1つを指定します。

### 静的画像定義の例

```yaml
images:
  # ローカルファイル
  - name: logo
    path: ./assets/logo.png
  
  # URL参照
  - name: reference_image
    url: https://example.com/images/reference.png
  
  # Base64埋め込み
  - name: small_icon
    base64: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    media_type: "image/png"
```

### 入力データでの画像指定

JSOLNの入力データで画像を指定する場合、`_type: "image"` を持つオブジェクトを使用します：

```jsonl
{"UserInput": "画像を分析してください", "ProductImage": {"_type": "image", "path": "./images/product_001.png"}}
{"UserInput": "この画像は何ですか", "ProductImage": {"_type": "image", "url": "https://example.com/img.png"}}
{"UserInput": "分析してください", "ProductImage": {"_type": "image", "base64": "iVBORw0K...", "media_type": "image/png"}}
```

### プロンプト内での画像参照（`{name.img}` 記法）

プロンプト内で画像を埋め込むには `{name.img}` 記法を使用します：

| 記法                      | 説明                                   |
| ------------------------- | -------------------------------------- |
| `{name.img}`              | 画像を埋め込み                          |
| `{name.img:detail=low}`   | 低解像度モード（トークン節約）           |
| `{name.img:detail=high}`  | 高解像度モード（詳細分析）               |
| `{name.img:detail=auto}`  | 自動選択（既定）                        |

### 画像入力の使用例

```yaml
mabel:
  version: "2.1"

images:
  - name: category_guide
    path: ./assets/category_guide.png

models:
  - name: vision
    api_model: "gpt-4o"
    api_key: "${ENV.OPENAI_API_KEY}"
    capabilities: ["vision"]

blocks:
  - type: ai
    exec: 1
    model: vision
    prompts:
      - |
        商品画像を分析してください。
        
        商品画像:
        {ProductImage.img:detail=high}
        
        カテゴリ参照:
        {category_guide.img:detail=low}
    outputs:
      - name: Analysis
        select: full
```

**実装ポイント**

* 画像解決の優先順位: 1) 入力データ（コンテキスト）、2) `images` セクション
* OpenAI Vision API形式（`image_url` タイプ）でメッセージを構築
* サポートフォーマット: PNG, JPEG, GIF, WebP

---

## 5. 文字列テンプレート（`templates`）

任意の名前付きテンプレートを定義し、`{...}` で埋め込み可能です。`ai` の `prompts`、`end.final.value` など**任意の文字列**で利用できます。

```yaml
templates:
  - name: "card"
    text: |
      [TITLE] {Title}
      [BODY]
      {Body}
```

**実装ポイント**

* 置換は `utils.render_template()`。`{APP_NAME}` や `{Answer}` のように、**先に求められた出力／変数**が利用可能。

---

## 6. ブロック仕様（`blocks[]`）

### 6.1 AI ブロック（`type: ai`）

**役割**: モデルへメッセージを送り、応答テキストから出力項目を抽出します。

```yaml
- type: ai
  exec: 1
  id: "ask"
  name: "Ask the planner"
  model: planner                   # ← models[].name
  system_prompt: |
    You are a concise planner.
  prompts:
    - |                             # 任意個の user メッセージ
      Summarize the following text:
      {UserInput}
  params:                           # モデル既定値を上書き
    temperature: 0.2
    max_tokens: 300
  mode: text                        # text | json
  outputs:                          # 応答テキストから抽出
    - name: Answer
      select: full                  # full | tag | regex | jsonpath
    - name: Title
      select: regex
      regex: "(?s)^(.*?)\\n"        # 先頭行をタイトル化
    - name: FirstCode
      select: tag
      tag: "code"                   # <code>...</code> を抽出
      join_with: "\\n\\n"
    - name: Meta
      select: jsonpath
      path: "$.data.value"          # JSON 返却時に抽出
      type_hint: json               # string|number|boolean|json
  save_to:                          # v2: 出力をグローバル変数へ保存
    vars:
      memo: Answer
```

**補足**

* `mode: json` の場合、`response_format: {type: "json_object"}` が付与されます（実装が設定）。
* `outputs` の抽出器:

  * `full`: そのまま全文
  * `tag`: `<tag>...</tag>` 抽出（`utils.extract_by_tag`）
  * `regex`: 正規表現（`utils.extract_by_regex`、`join_with` 併用可）
  * `jsonpath`: `$.a.b` 形式。`jsonpath_ng` が無い場合は **簡易パス走査**にフォールバック
  * `type_hint`: `json` を指定すると JSON として復元を試みる

**実装ポイント**

* データクラス: `AIBlock(model, system_prompt, prompts, outputs, params, attachments, mode, save_to)`（`config.py`）。
* 抽出処理: `executors._apply_outputs()` が `OutputDef(select/tag/regex/path/join_with/type_hint)` で分岐。
* `save_to.vars` は `globals.vars` に代入（`executors.run_pipeline()`）。

---

### 6.2 Logic ブロック（`type: logic`）

v1 の簡易条件に加え、v2 では **MEX（MABEL Expression Language）** による**式評価・制御構造**を持ちます。

#### 6.2.1 v1 条件・反復（互換）

```yaml
# if / and / or / not
- type: logic
  exec: 10
  name: "short_check"
  op: if
  cond:
    and:
      - { ne: ["{Answer}", ""] }
      - { lt: [ { len: "{Answer}" }, 280 ] }  # 文字数 < 280
  then: "short"
  else: "long"
  outputs:
    - name: Flag
      from: boolean                           # boolean|value などを選ぶ

# for
- type: logic
  exec: 11
  name: "loop_lines"
  op: for
  list: "{Answer}"               # 反復対象（リスト or 区切り文字列）
  parse: lines                   # lines | csv | json | regex
  regex_pattern: "^(.+)$"        # parse: regex のときに使用
  var: item                      # ループ変数名（既定: item）
  drop_empty: true
  where: { "ne": ["{item}", ""] } # 行フィルタ
  map: "Line: {item}"            # 変換
  outputs:
    - name: Joined
      from: join
      source: mapped             # raw|filtered|mapped のどれを集計
      join_with: "\n"
```

#### 6.2.2 v2 の式言語 MEX（`mex.py`）

**式は JSON 風**で、`run_if`・`cond`・値計算などに使えます。

```yaml
{"add": [1, {"mul": [{"var": "x"}, 2]}]}
{"if": {"cond": {"gt":[{"var":"n"}, 0]}, "then": "pos", "else": "non-pos"}}
{"and": [ {"eq":[{"var":"a"}, 1]}, {"not":{"lt":[{"var":"b"}, 3]}} ]}
```

主な演算子（実装に存在）

* 論理: `and`, `or`, `not`
* 比較: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
* 算術: `add`, `sub`, `mul`, `div`, `mod`, `min`, `max`
* 文字列: `lower`, `upper`, `trim`, `len`
* 正規表現: `match`, `replace`, `split`
* 乱択: `randint`, `choose`
* 変数: `var`（ローカル/グローバル参照）, `get`（安全取得）
* 制御: `if`, `let`（束縛）
* コレクション: `map`, `filter`, `reduce`, `any`, `all`, `first`, `last`

> **実装ポイント**: `mex.py` の `MEXEvaluator.eval()` が上記の主要演算子を実装。深さは `max_depth=256` で制限。

#### 6.2.3 while（反復）＋ emit（収集）

```yaml
- type: logic
  exec: 20
  name: "count_up"
  op: while
  cond: {"lt": [{"var":"counter"}, 3] }    # MEX 条件
  step:                                    # ループ本体（順に実行）
    - op: set
      var: counter
      value: {"add": [{"var":"counter"}, 1]}   # counter += 1
    - op: emit
      value: {"var":"counter"}              # 収集値を push
  budget:                                   # ブロック局所上書き（任意）
    loops: { max_iters: 10, on_exceed: "truncate" }
  outputs:
    - name: Iterations
      from: list                            # 収集された配列
    - name: IterCount
      from: count                           # 要素数
```

**実装ポイント**

* ループ進行は `executors._execute_logic_step()` と `check_loop_budget()` により制御。`emit` は内部バッファへ追加し、`outputs.from: list/count` で取り出し。

#### 6.2.4 set / let（代入・束縛）

```yaml
- type: logic
  exec: 30
  op: set
  var: memo
  value: {"concat": ["[FLAG] ", "{Flag}"] }   # 文字列結合（concat は map/テンプレートでも可）

- type: logic
  exec: 31
  op: let
  bindings:
    x: 41
    y: {"add":[{"var":"x"}, 1]}               # x は同ブロック中に可視
  outputs:
    - name: Sum
      from: value
      value: {"add":[{"var":"x"}, {"var":"y"}]}   # 41 + 42 = 83
```

#### 6.2.5 reduce（畳み込み）

```yaml
- type: logic
  exec: 32
  op: reduce
  list: "{Iterations}"           # 例えば [1,2,3]
  var: acc
  init: 0
  step: {"add":[{"var":"acc"}, {"var":"item"}]}
  outputs:
    - name: Total
      from: accumulator
```

#### 6.2.6 call（ロジック関数呼び出し）

`functions.logic` で MEX ベースの関数を定義し、`op: call` で利用。

```yaml
functions:
  logic:
    - name: inc
      params: [x]
      body:
        let:
          var: y
          value: {"add": [{"var":"x"}, 1]}

blocks:
  - type: logic
    exec: 40
    op: call
    name: "use_inc"
    with: { x: 41 }      # 実引数
    returns: [Answer]    # 戻り値の束縛名（配列）
```

#### 6.2.7 Logic 出力定義（`outputs[].from`）

* `boolean` / `value` / `join` / `count` / `any` / `all` / `first` / `last` / `list` / `var` / `accumulator`

  * `join` は `join_with` が必要。
  * `list` は `emit` された配列。
  * `var` はローカル束縛をそのまま返す。

**実装ポイント**

* データクラス: `LogicBlock(op, ...)`（`config.py`）。`op` 値は実装コメントにある通り `if|and|or|not|for|while|recurse|set|let|reduce|call|emit` をサポート。
* while/emit/for の具体挙動は `executors.py` のロジック分岐で確認できます。

---

### 6.3 Python ブロック（`type: python`）

#### 6.3.1 v1 互換

```yaml
- type: python
  exec: 50
  name: "format_result"
  function: format_output         # ← モジュール内の関数名
  inputs: [Answer, Flag]          # ← 位置引数として渡る
  code_path: examples/helpers.py  # ← ファイルからロード
  outputs: [Formatted]            # ← 関数の戻り値に含むキー名
```

> 例の `examples/helpers.py`:
>
> ```python
> def format_output(answer: str, flag: bool):
>     short = (answer or "").split("\n", 1)[0]
>     return {"Formatted": f"[SHORT]\n{short}\n\n[LONG]\n{answer}\n\n[FLAG]\n{flag}"}
> ```

#### 6.3.2 v2 拡張（インライン関数と統合環境）

```yaml
- type: python
  exec: 51
  id: "format_v2"
  entrypoint: format_inline            # function と同義
  function_code: |                     # ← インラインで関数実装
    def format_inline(answer: str, flag: bool, ctx=None):
        # ctx: 実行コンテキスト（executors.ExecutionContext 由来のプロキシ）
        # 利用可能メソッド（構成により制限）: ctx.get(k), ctx.set(k, v), ctx.emit(v), ctx.log(level, msg)
        ctx.log("info", f"flag={flag}")
        memo = ctx.get("memo") or ""
        out = f"[SHORT]{answer.splitlines()[0] if answer else ''}\n[MEMO]{memo}"
        # グローバルへ書き戻し
        ctx.set("memo", out)
        return {"FormattedResult": out}
  inputs: [Answer, Flag]
  outputs: [FormattedResult]
  use_env: global                     # global | override
  override_env: null                  # use_env: override のときのみ
  timeout_ms: 30000                   # 実装はこの値で関数実行をタイムアウト（将来拡張）
  ctx_access: ["get", "set", "emit", "log"]   # 許可する API
```

**実装ポイント**

* データクラス: `PyBlock(function, entrypoint, inputs, code_path, function_code, outputs, use_env, override_env, timeout_ms, ctx_access)`（`config.py`）。
* 実行: `executors.run_pipeline()` が `function_code` を `exec()` でロード or `code_path` を importlib でロードし、`entrypoint/function` を解決。
* `ctx` は `executors` 内部のラッパ（`get/set/emit/log` を持つ）を渡します。

---

### 6.4 End ブロック（`type: end`）

**最終ペイロード**を組み立ててレコードを終了します。

```yaml
- type: end
  exec: 999
  reason: "completed"
  exit_code: "success"
  final:
    - name: answer
      value: "{FormattedResult}"
    - name: status
      value: "{Flag}"
    - name: iterations
      value: "{IterCount}"
  final_mode: "map"                 # map | list
  include_vars: ["counter", "memo"] # 返却へ同梱するグローバル変数
```

**実装ポイント**

* データクラス: `EndBlock(reason, exit_code, final, final_mode, include_vars)`。
* `final_mode` は既定 `map`。`include_vars` に列挙した `globals.vars` を応答へ同梱。

---

## 7. 予算（Budgets）

**目的**: ループ／再帰暴走、過大な AI 呼び出しを遮断して安全に停止。

```yaml
budgets:
  loops:     { max_iters: 1000, on_exceed: "truncate" }   # while など
  recursion: { max_depth: 64,   on_exceed: "error"     }  # recurse
  wall_time_ms: 20000
  ai: { max_calls: 8, max_tokens: 16000 }
```

* グローバル既定（上）に対し、**各ブロックの `budget` で上書き**可。
* 超過時の振る舞い: `on_exceed: "error"|"truncate"`。

**実装ポイント**

* `executors.ExecutionContext.check_loop_budget()` / `check_recursion_budget()` が評価し、超過時に `BudgetExceeded`。

---

## 8. 明示配線（`connections`）

**同名の入出力は暗黙で流れる**設計ですが、任意で明示配線も書けます。

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

**実装ポイント（重要）**

* 現バージョンの実装は `connections` を **設定として受け取るだけ**で、実行エンジンは **暗黙配線（同名束縛）**を用います。設計書（`mabel_v2.md`）に沿った将来拡張の予約項目です。

---

## 9. セキュリティ / サンドボックス

* 既定で `runtime.python.allow_network: false` を想定（外部通信を抑止）。
* Python ブロックの `ctx_access` によりコンテキスト API を **最小権限**で付与。
* 秘匿値は `${ENV.*}` の形で**外部注入**し、YAML に直書きしない。

**実装ポイント**

* 実際のネットワーク遮断は環境側のセットアップに依存しますが、設計として **遮断前提**。`ctx` API の露出は `PyBlock.ctx_access` で制御。

---

## 10. エラー処理 / 再試行 / ログ

* 各ブロックは `on_error: "fail"|"continue"|"retry"` を持てます。

  * `retry` には `retry: { max_attempts: N, backoff: { type: "exponential"|"fixed", base_ms: 500 } }` を指定。
* 予算超過は §7 の方針に従います。
* ログは `ctx.log(level, message)` で記録でき、実行トレースの保存に使えます。

**実装ポイント**

* LLM 呼び出しはタイムアウトや 429/5xx をリトライ対象（`llm_client.BatchOptimizer` と再試行ロジック）。
* `on_error: continue` はブロック単位でエラーを無視し、`error_block_{exec}` に文字列化した例外を残します。

---

# 付録：最小から実用までの YAML 全体例

以下は **v2 の主要機能**をすべて踏む、ほぼそのまま動く統合例です。

```yaml
mabel:
  version: "2.0"
  id: "com.example.nexus.demo"
  name: "Nexus Demo"
  description: "AI + Logic (MEX while/set/emit) + Python + End"

runtime:
  python:
    venv: ".venv"
    requirements_file: "requirements.txt"
    allow_network: false
    env:
      OPENAI_API_KEY: "${ENV.OPENAI_API_KEY}"

globals:
  const:
    APP_NAME: "SDG Nexus v2"
    MAX_TURNS: 3
  vars:
    counter: 0
    memo: ""

budgets:
  loops:     { max_iters: 10, on_exceed: "truncate" }
  recursion: { max_depth: 64, on_exceed: "error" }
  wall_time_ms: 20000
  ai: { max_calls: 8, max_tokens: 16000 }

models:
  - name: planner
    api_model: "phi4-mini"
    api_key: "${ENV.OPENAI_API_KEY}"
    base_url: "http://127.0.0.1:11500/v1"
    request_defaults:
      temperature: 0.2
      max_tokens: 400
  - name: writer
    api_model: "gemma-3-1b-it"
    api_key: "${ENV.OPENAI_API_KEY}"
    base_url: "http://127.0.0.1:11500/v1"
    request_defaults:
      temperature: 0.3
      max_tokens: 400

templates:
  - name: "report"
    text: |
      App: {APP_NAME}
      Title: {Title}
      ----
      {Answer}

blocks:

  # 1) AI: 下書き作成
  - type: ai
    exec: 1
    id: "ask"
    model: planner
    system_prompt: |
      You are a concise planner.
    prompts:
      - |
        Summarize the following text:
        {UserInput}
    params:
      temperature: 0.2
    mode: text
    outputs:
      - name: Answer
        select: full
      - name: Title
        select: regex
        regex: "(?s)^(.*?)\\n"

  # 2) Logic: 短文判定（v1 互換）
  - type: logic
    exec: 2
    name: "short_check"
    op: if
    cond:
      and:
        - { ne: ["{Answer}", ""] }
        - { lt: [ { len: "{Answer}" }, 280 ] }
    then: "short"
    else: "long"
    outputs:
      - name: Flag
        from: boolean

  # 3) Logic (v2): while/set/emit でカウンタを回す
  - type: logic
    exec: 3
    name: "count_up"
    op: while
    cond: {"lt": [{"var":"counter"}, {"var":"MAX_TURNS"}]}
    step:
      - op: set
        var: counter
        value: {"add": [{"var":"counter"}, 1]}
      - op: emit
        value: {"var":"counter"}
    budget:
      loops: { max_iters: 10, on_exceed: "truncate" }
    outputs:
      - name: Iterations
        from: list
      - name: IterCount
        from: count

  # 4) Python (v2): インラインで整形し、ctx でメモを更新
  - type: python
    exec: 4
    id: "format_v2"
    entrypoint: format_inline
    function_code: |
      def format_inline(answer: str, flag: bool, ctx=None):
          ctx.log("info", f"flag={flag}")
          memo = ctx.get("memo") or ""
          short = (answer or "").split("\n", 1)[0]
          out = f"[SHORT]\\n{short}\\n\\n[FLAG]\\n{flag}\\n\\n[MEMO]\\n{memo}"
          ctx.set("memo", out)
          return {"FormattedResult": out}
    inputs: [Answer, Flag]
    outputs: [FormattedResult]
    ctx_access: ["get", "set", "log"]

  # 5) End: 最終出力
  - type: end
    exec: 100
    final:
      - name: answer
        value: "{FormattedResult}"
      - name: status
        value: "{Flag}"
      - name: iterations
        value: "{IterCount}"
    final_mode: "map"
    include_vars: ["counter", "memo"]
```

---

## 実装確認の要点（抜粋）

* **データクラス**（`config.py`）

  * `AIBlock` … `model, system_prompt, prompts, outputs, params, attachments, mode, save_to`
  * `LogicBlock` … `op`（`if|and|or|not|for|while|recurse|set|let|reduce|call|emit`）ほか
  * `PyBlock` … `function/entrypoint, inputs, code_path, function_code, outputs, use_env, override_env, timeout_ms, ctx_access`
  * `EndBlock` … `reason, exit_code, final, final_mode, include_vars`
  * `OutputDef` … `name, select, tag, regex, path, join_with, type_hint` / `from, var, source`（logic 用）
  * `BudgetConfig` … `loops, recursion, wall_time_ms, ai`
* **AI 呼び出し**は Chat Completions 互換（`llm_client.py`）、**`mode: json`** で `response_format` が付与。
* **MEX 式エンジン**は `mex.py` の `MEXEvaluator`。`len/upper/trim/map` 等を実装、深さ上限 256。
* **実行エンジン**は `executors.run_pipeline()`：

  * `exec` の昇順でブロックを選別し、`run_if` を評価。
  * `ai` 出力は `_apply_outputs()` で抽出。`save_to.vars` は `globals.vars` に書き込み。
  * `logic: while` は `check_loop_budget()` を通過しつつ `emit` を収集、`outputs.from: list/count` に対応。
  * `python` は `function_code` を `exec()` または `code_path` を `importlib` でロード、`entrypoint/function` を起動。`ctx` 経由で `get/set/emit/log`。
  * エラー時は `on_error` に従う（`continue` は埋め込みログを残して継続）。
* **connections** は現状**解析のみ**（暗黙配線で同名束縛が流れる）。

---

### 参考：CLI 実行

```bash
# 例
python -m sdg.cli run \
  --yaml examples/sdg_demo_v2.yaml \
  --input examples/data/input.jsonl \
  --output out/output.jsonl \
  --max-batch 8 --min-batch 1 --target-latency-ms 3000 \
  --save-intermediate
```

---

## まとめ

* 本ドキュメントは、同梱実装のクラス・関数に合わせて **1〜10 節を完全網羅**しました。
* ここに示した YAML スニペットは、そのままの記法で **`config.py / executors.py` が解釈・実行**する形に整えています。
* v2 の中核（MEX、`while/set/emit`、インライン Python、`final/include_vars`、予算とエラー処理）は、いずれも実装内で確認済みのフィールド・制御経路です。

必要に応じて、さらに詳細なサンプル（特定モデルや JSON 抽出規則のバリエーション、`reduce/call/recurse` の複合例）も作成できます。
