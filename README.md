# SDG-Nexus (Scalable Data Generator - Nexus)

**SDG** は、AIエージェントのYAML設計図を入力として、大規模言語モデル（LLM）によるデータ生成・加工をスケーラブルに実行するためのCLIツール兼ライブラリです。
OpenAI API互換の推論バックエンド（OpenAI / vLLM / SGLang 等、`base_url`で指定）を利用し、**バッチ数（同時並行数）の自動最適化**を行います。

> 本実装は、MABEL Studio v1.1 が生成・取り込む YAML 仕様と互換のあるサブセットをサポートしています（models / blocks / connections など）。
> 仕様の出典は、ユーザー添付ドキュメントを参照してください。

## 特徴
- YAML設計図のロードと検証（`models`/`blocks`セクションを中心にサポート）
- `ai` / `logic` / `python` / `end` ブロックの実行エンジン
- OpenAI互換のチャット補完エンドポイント（`/v1/chat/completions`）を利用
- **自動バッチ（同時並行）最適化**: 遅延とエラー率に応じて同時実行数を調整
- JSONL/CSV 入力データをレコード単位に処理し、最終出力を JSONL で保存

## インストール
```bash
conda create --name sdg python=3.10
conda activate sdg
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 使い方
### 1) 環境変数
最小限、以下のいずれかを設定してください。

- `OPENAI_API_KEY`（OpenAI使用時）
- 互換サーバ使用時は YAML `models[].api_key` に `${ENV.OPENAI_API_KEY}` を指定し、`models[].base_url` にエンドポイントのベースURLを記述します。

### 2) サンプルを実行
```bash
sdg run   --yaml examples/sdg_demo.yaml   --input examples/data/input.jsonl   --output out/output.jsonl   --max-batch 8 --min-batch 1
```

> 上記は `UserInput` フィールドを含む JSONL を入力に、1つの `ai` ブロック→`logic`→`python`→`end` を実行し、最終出力を JSONL で保存します。

### 3) CSV入力
```bash
sdg run --yaml examples/sdg_demo.yaml --input examples/data/input.csv --output out/out.jsonl
```

## YAMLの書き方（対応サブセット）
- `models`: `name`, `api_model`, `api_key`, `base_url`, `headers`, `request_defaults`（`temperature`/`top_p`/`max_tokens`/`timeout_sec`/`retry`）
- `blocks`:
  - `ai`: `model`, `system_prompt`, `prompts[]`, `outputs[]({name,select,tag,regex,join_with})`, `params`
  - `logic`: `op: if/and/or/not/for`, `cond`, `then/else`, `operands`, `list/parse/regex_pattern/var/drop_empty/where/map`, `outputs`
  - `python`: `name`, `function`, `inputs[]`, `code_path`, `venv_path`（無視されます。お好みの仮想環境で実行してください）, `outputs[]`
  - `end`: `final[]({name,value})`, `exit_code`, `reason`

本実装のYAML解釈は、添付仕様の重要部分に準拠しています（詳細はドキュメントを参照）。

## バッチ最適化（同時並行数）
- `--max-batch` と `--min-batch` の範囲で並列度を自動調整します。
- 1ラウンドの `ai` ブロックを **全レコードに対して** 実行する際、HTTP 429/5xx やタイムアウト・高遅延を検知すると次ラウンドで並列度を下げます。安定して高速な場合は徐々に上げます。

## 出力
- 既定では、各入力レコードごとに `end` ブロックの `final` で指定したフィールドを JSON 1行として `--output` に保存します。
- `--save-intermediate` を指定すると、中間生成物も含めます（デバッグ用途）。

## 注意事項
- 内部実装は OpenAI Python SDK (openai>=1.40) を使用します。OpenAI互換サーバーへ接続する場合は `models[].base_url` にベースURL（末尾に /v1 を含まなくても可。自動で `/v1` を付与）を指定してください。
- `python` ブロックの `venv_path` は無視し、**実行中のPython環境**で `code_path` を動的importします。
- OpenAIの [Batches API] ではなく、**同時並行HTTP呼び出し**でスループットを稼ぐ設計です。
- vLLM/SGLang など **OpenAI互換エンドポイント** を `models[].base_url` で指定してください。

## ライセンス
MIT
