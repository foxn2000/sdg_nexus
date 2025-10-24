# SDG‑Nexus (Scalable Data Generator — Nexus)

[English README](README.md)

SDG‑Nexus は、LLM（大規模言語モデル）を用いたデータ生成／加工を YAML 設計図で定義し、JSONL/CSV の入力データに対してスケーラブルに実行する CLI ツール兼 Python ライブラリです。OpenAI 互換の推論エンドポイント（OpenAI / vLLM / SGLang など、`/v1/chat/completions` API）に対応し、遅延とエラー率に基づいて同時並行実行数（バッチサイズ）を自動調整する適応型の並行制御を備えています。

注: 本実装は、MABEL Studio v1.1 と互換のある YAML 仕様サブセット（models / blocks / connections 等）をサポートします。下記の例と仕様メモを参照してください。

## 特徴

- YAML 設計図のロードと基本検証（特に `models` / `blocks` を中心）
- `ai` / `logic` / `python` / `end` ブロックの実行エンジン
- OpenAI 互換の Chat Completions（`/v1/chat/completions`）に対応
- 適応型並行実行: 遅延とエラー率に応じてバッチサイズを自動で増減
- JSONL/CSV をレコード単位で処理し、最終結果を JSONL に保存
- デバッグ用途で中間生成物を保存可能（`--save-intermediate`）

## インストール

要件: Python >= 3.10

方法 A: ローカル編集（editable）インストール
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

方法 B: 通常インストール（ローカル編集なし）
```bash
pip install -U pip
pip install -r requirements.txt
pip install .
```

CLI エントリーポイント `sdg` は `pyproject.toml` で提供されます。

## クイックスタート

1) 認証情報の設定

- OpenAI を使用する場合:
  - シェルで `OPENAI_API_KEY` をエクスポートしてください。
- OpenAI 互換サーバ（vLLM / SGLang 等）を使用する場合:
  - `models[].base_url` にサーバの URL を設定します。
  - `models[].api_key` に API キーを設定します。`${ENV.OPENAI_API_KEY}` と書いた場合、実行時に環境変数 `OPENAI_API_KEY` が使用されます（現実装は `${ENV.***}` の中身に関わらず `OPENAI_API_KEY` 固定で参照します）。

2) サンプルの実行
```bash
sdg run \
  --yaml examples/sdg_demo.yaml \
  --input examples/data/input.jsonl \
  --output out/output.jsonl \
  --max-batch 8 --min-batch 1
```

- 各 JSONL レコード（`UserInput` を含む）に対し、`ai -> ai -> logic -> python -> end` のパイプラインを実行し、`out/output.jsonl` に最終結果を書き出します。

3) CSV 入力の例
```bash
sdg run \
  --yaml examples/sdg_demo.yaml \
  --input examples/data/input.csv \
  --output out/out.jsonl
```

## CLI リファレンス

推奨（サブコマンド）:
```text
sdg run --yaml PATH --input PATH --output PATH [options]
```

オプション:
- `--yaml`（必須）: YAML 設計図のパス
- `--input`（必須）: 入力データセット（.jsonl または .csv）
- `--output`（必須）: 出力 JSONL ファイル
- `--max-batch`（int, 既定 8）: 最大同時実行数
- `--min-batch`（int, 既定 1）: 最小同時実行数
- `--target-latency-ms`（int, 既定 3000）: 目標平均レイテンシ（ミリ秒）
- `--save-intermediate`（フラグ）: 中間結果を保存

後方互換のレガシーモードも利用可能:
```bash
sdg --yaml ... --input ... --output ...
```

## 入出力フォーマット

- JSONL 入力: 1 行に 1 つの JSON オブジェクト。各キーは後続ブロックのテンプレートで参照可能になります。
  - 例: `{"UserInput": "What is the capital of Japan?"}`

- CSV 入力: 先頭行をヘッダとして読み込み、各行を文字列値の辞書に変換します。

- 出力: 既定では、各レコードについて `end` ブロックの `final` で指定したフィールドを 1 行の JSON として書き出します。

- 中間出力: `--save-intermediate` を付けると、各ブロックの値を `_{exec}_{name}`（例: `_2_ShortAnswer`）というキーでレコードごとに格納します。

## YAML 設計図（サポートされるサブセット）

トップレベル:
```yaml
mabel: { version: "1.0" }     # 任意
models: [ ... ]               # 必須
blocks:  [ ... ]              # 必須
connections: [ ... ]          # 解析のみ、将来拡張予約
```

### models

`ai` ブロックから `name` で参照されます。

フィールド:
- `name` (str): モデル識別子
- `api_model` (str): バックエンド側のモデル名（例: `gpt-4o-mini`, `phi4-mini` 等）
- `api_key` (str): API キー。`${ENV.OPENAI_API_KEY}` と書くと、環境変数 `OPENAI_API_KEY` を使用（注: 現行実装は `${ENV.***}` の変数名を解釈せず、`OPENAI_API_KEY` 固定）
- `base_url` (str, 任意): ベース URL。末尾が `/v1` でなければ自動的に付与（例: `http://127.0.0.1:8000` → `http://127.0.0.1:8000/v1`）
- `organization` (str, 任意): OpenAI の組織 ID（必要な場合）
- `headers` (map, 任意): ベンダ固有の追加ヘッダ（標準ヘッダは SDK が処理）
- `request_defaults` (map, 任意): 各 `ai` 呼び出しにマージされる既定パラメータ  
  - 代表例: `temperature`, `top_p`, `max_tokens`, `timeout_sec`, `retry`  
  - `retry` 構造:
    ```yaml
    retry:
      max_attempts: 3
      backoff:
        initial_ms: 250
        factor: 2.0
    ```

例:
```yaml
models:
  - name: writer
    api_model: gpt-4o-mini
    api_key: "${ENV.OPENAI_API_KEY}"
    base_url: https://api.openai.com
    headers:
      HTTP-Referer: "https://your-app.example"
    request_defaults:
      temperature: 0.2
      top_p: 0.95
      max_tokens: 400
      timeout_sec: 60
      retry:
        max_attempts: 3
        backoff: { initial_ms: 250, factor: 2.0 }
```

パラメータの優先順位: `request_defaults`（モデル側）と `params`（ブロック側）はマージされ、ブロック側 `params` が優先されます。

### blocks

`exec` の昇順で実行されます。共通:
- `exec` (int): 実行順序
- `run_if` (オブジェクトまたは JSON 文字列, 任意): 偽の場合、そのレコードではスキップ
- `on_error` (`fail` | `continue`, 既定 `fail`): ブロック失敗時に継続するか

サポートされるブロック種別:

#### ai

フィールド:
- `type: ai`
- `model` (str): `models` 内の `name`
- `system_prompt` (str, 任意): レコード文脈でテンプレート展開
- `prompts` (list[str]): テンプレート展開後に結合してユーザメッセージとして送信
- `outputs` (list[OutputDef], 任意): LLM 応答から値を抽出  
  - `name` (str): 出力変数名  
  - `select` (`full` | `tag` | `regex`): 抽出モード  
  - `tag` (str, `tag` 時必須): `<tag> ... </tag>`（大文字小文字無視・複数行対応）で抽出  
  - `regex` (str, `regex` 時必須): Python 正規表現。キャプチャ群があれば group(1)、なければマッチ全体  
  - `join_with` (str, 任意): 複数値を 1 つの文字列に結合する区切り
- `params` (map, 任意): 呼び出しごとの上書き（例: `temperature`, `max_tokens`, `timeout_sec`, `retry`）

`outputs` を省略した場合の既定:
```yaml
outputs:
  - name: full
    select: full
```

#### logic

フィールド:
- `type: logic`
- `name` (str, 任意)
- `op` (`if` | `and` | `or` | `not` | `for`)
- `cond` (object): `if` 用の条件（JSON オブジェクト）  
  - 対応: `equals`, `not_equals`, `contains`, `is_empty`, `gt`, `lt`, `gte`, `lte`  
  - 論理合成: `and`, `or`, `not`  
  - オペランドはテンプレート展開後に比較
- `then`, `else` (str, 任意): `if` のテキスト結果
- `operands` (list): `and`/`or`/`not` のオペランド
- 繰り返し（`op: for`）:
  - `list` (str): 文脈上のリスト（またはカンマ区切り文字列）名
  - `parse` (str, 任意: `"regex"` 指定で正規表現パース。省略時はカンマ区切り）
  - `regex_pattern` (str): `parse: regex` のときのパターン
  - `var` (str, 任意): ループ変数名（既定 `"item"`）
  - `drop_empty` (bool, 任意): 空要素を除去
  - `where` (object): 各要素に適用する条件（`cond` と同等の演算子）
  - `map` (str): 各要素に適用するテンプレート（例: `"* {item}"`）
- `outputs` (list of map): op ごとの出力定義  
  - `if` の場合:  
    - `name`: 出力キー  
    - `from`: `boolean` | `text` | `source`  
      - `boolean` → 真偽値  
      - `text` → `then`/`else` の文字列  
      - `source` → 文脈から指定名の値をコピー（`source: FieldName`）
  - `and`/`or`/`not` の場合:  
    - `name`: 出力キー。値は真偽結果
  - `for` の場合:  
    - `name`: 出力キー  
    - `join_with` (str, 任意): リストを結合して文字列化  
    - `limit` (int, 任意), `offset` (int, 任意)

#### python

フィールド:
- `type: python`
- `name` (str, 任意)
- `function` (str, 必須): `code_path` 内の関数名
- `inputs` (list[str]): 文脈から位置引数として渡すフィールド名
- `code_path` (str): 動的にロードする Python ファイルのパス
- `venv_path` (無視): 実行中の Python 環境を使用
- `outputs` (list[str]): 関数の戻り値を文脈へマップ  
  - 戻り値が dict の場合 → 指定したキーのみ抽出  
  - 非 dict の場合 → 値（単一または list/tuple）を位置対応でマップ

#### end

フィールド:
- `type: end`
- `final` (list of `{ name, value }`): テンプレートを評価し、最終 JSONL に書き出すフィールド
- `exit_code` (str, 任意)
- `reason` (str, 任意)

### テンプレートと抽出ヘルパ

- テンプレート: `{VarName}` や `{foo.bar}` のように参照可能。欠損キーは空文字になります。
- タグ抽出: `<tag> ... </tag>`（大文字小文字無視・複数行）
- 正規表現抽出: キャプチャ群があれば group(1)、なければマッチ全体

## 適応型並行実行（バッチ最適化）

各 `ai` ラウンドで複数レコードを処理する際、平均レイテンシとエラー数を計測し、同時実行数を調整します。
- エラーが発生、または平均レイテンシが `--target-latency-ms` を超過 → `--min-batch` まで段階的に削減
- 安定かつ速い → `--max-batch` まで段階的に増加

これにより、スループットを高めつつ、レート制限やサーバ遅延に頑健になります。

## バックエンド（OpenAI 互換）

- OpenAI Python SDK (>= 1.40) を使用し、`base_url` を差し替え可能
- `base_url` が `/v1` で終わらない場合、自動で `/v1` を付与
- ベンダ固有の追加ヘッダは `models[].headers` に設定可能（標準ヘッダは SDK が管理）

例:
- OpenAI: `base_url: https://api.openai.com`
- vLLM プロキシ: `base_url: http://127.0.0.1:8000`
- SGLang: `base_url: http://127.0.0.1:30000`

注: 本プロジェクトは OpenAI の Batches API ではなく、通常の HTTP チャット呼び出しの並行化でスループットを最適化します。

## Python からの利用

```python
from sdg.runner import run

run(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="out/output.jsonl",
    max_batch=8,
    min_batch=1,
    target_latency_ms=3000,
    save_intermediate=False,
)
```

## サンプル設計図

`examples/sdg_demo.yaml` は、以下を示す実行可能な例です:
- 2 つの `ai` ブロック（`planner` → `writer`）
- 短文回答の有無を確認する `logic` ブロック
- 回答を整形する `python` ブロック（`examples/helpers.py`）
- 最終フィールドを選択する `end` ブロック

## エラーハンドリング

- ブロックレベルのエラー:
  - 既定は `on_error: fail` で、パイプラインを停止して例外を送出します。
  - `on_error: continue` を設定すると処理を継続し、レコード文脈に `error_block_{exec}`（例: `error_block_2`）というキーでエラー文字列が入ります。
- 条件付き実行:
  - `run_if` が偽と評価されたレコードでは、そのブロックはスキップされます。

## トラブルシューティング

- 401 Unauthorized: `OPENAI_API_KEY` またはサーバ/API キー設定を確認
- レート制限／低速化: `--max-batch` を下げる、モデル側 `request_defaults.max_tokens` を下げる、`--target-latency-ms` を上げる
- 出力が空:
  - `regex`: パターンおよびキャプチャ群の指定を確認
  - `tag`: LLM 出力が指定タグで囲まれているか確認
- CSV の値は文字列: 必要に応じて `python` ブロックで数値変換等を行う

## ライセンス

MIT
