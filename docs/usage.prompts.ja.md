# MABELフロー作成ガイド：AIコーディングエージェント向けプロンプト

## 目次

1. [概要](#概要)
2. [前提条件](#前提条件)
3. [基本プロンプトテンプレート](#基本プロンプトテンプレート)
4. [使用方法](#使用方法)
5. [プロンプトのカスタマイズ](#プロンプトのカスタマイズ)
6. [実装例](#実装例)
7. [動作確認方法](#動作確認方法)
8. [ベストプラクティス](#ベストプラクティス)
9. [トラブルシューティング](#トラブルシューティング)

---

## 概要

このドキュメントでは、Roo Cline、Cline、Claude Code、Codexなどの**AIコーディングエージェント**を使用して、MABELフロー（YAML形式のAIエージェント定義ファイル）を効率的に作成する方法を説明します。

**MABELとは？**

MABEL（Model And Blocks Expansion Language）は、AIエージェントの処理フローをYAML形式で宣言的に記述できる言語仕様です。SDG Nexusで使用され、複雑なAIワークフローを構造化して定義できます。

詳細は以下を参照してください：
- [MABEL v2 完全仕様](./mabel/mabel_v2.md)
- [SDG使用ガイド](./usage.ja.md)

---

## 前提条件

### 必要なツール

- **AIコーディングエージェント**（以下のいずれか）
  - Roo Cline
  - Cline
  - Claude Code
  - Cursor
  - その他のAIアシスタント

### 推奨モデル

高品質なMABELフローを生成するため、以下のような高性能モデルの使用を推奨します：

- **Claude Opus 4.5**（推奨）
- **GPT-5.1 Codex**
- **Gemini 3 pro**
- **Grok 4**
- その他の最新のフロンティアモデル

### 必要な知識

- YAMLの基本的な構文
- AIエージェントの基本概念
- （推奨）MABELの基本仕様

---

## 基本プロンプトテンプレート

以下のプロンプトをAIコーディングエージェントに送信して、MABELフローを生成してください。

```markdown
## プロンプトの目的

現在運用中のAIエージェントシステム（SDG Nexus）の実装を参考に、指定されたロジックをMABELフロー形式で明確かつ動作保証のあるYAMLファイルとして作成してください。

## 実施する作業内容

あなたはAIエージェント向けロジック設計を担当するロジックエンジニアです。以下の手順に従ってタスクを実施してください。

### ステップ1: 既存仕様の確認

以下のファイルを確認し、MABELの仕様や規約を把握してください：

1. **ソースコード**: `sdg/` ディレクトリ以下のPythonコード
2. **サンプルYAML**: `examples/` ディレクトリに格納された各YAMLファイル
3. **ドキュメント**: `docs/` 以下のドキュメント、特に以下を重点的に確認
   - `docs/mabel/mabel_v2.md` - MABEL v2 完全仕様
   - `docs/usage.ja.md` - 使用ガイド

### ステップ2: MABELフローの設計・実装

上記の調査を踏まえた上で、私が提示するロジックを具体的なMABELフローとして設計・実装してください。

**実装要件:**

- 必ず動作可能なYAML形式で記述すること
- MABEL v2.0の仕様に準拠すること
- 既存の `examples/` 内のYAMLファイルのフォーマットを参考にすること

### ステップ3: ファイル配置

完成したファイルを以下のように配置してください：

- **YAMLファイル**: `/flows/yaml/` ディレクトリ以下に格納
  - ファイル名は処理内容がわかるように命名（例: `data_analysis_flow.yaml`）
- **補助スクリプト**: Pythonなどの補助コードが必要な場合
  - `/flows/codes/` ディレクトリ以下にPythonファイルとして作成
  - YAMLから参照できるように適切に設定

## 出力における注意事項

### 品質要件

1. **動作保証**
   - 提供されている既存の `examples/` に収録されたYAMLファイルのフォーマットや仕様を遵守すること
   - 必ず動作可能な状態にすること（構文エラーがないこと）

2. **コード品質**
   - すべての実装内容は明確かつ一貫した構造を持つように整理すること
   - 適切なコメントを付与すること
   - 変数名は分かりやすい命名規則に従うこと

3. **エラーハンドリング**
   - 適切なエラー処理を含めること（`on_error`, `retry` の活用）
   - 予算制御（`budgets`）を適切に設定すること

4. **ドキュメンテーション**
   - YAMLファイル内に説明コメントを含めること
   - 複雑な処理には補足説明を追加すること

### 事前確認事項

不明点や不備がある場合は、実装前に以下を明示的に指摘し、改善案を提示してください：

- 仕様の不明確な点
- 必要な追加情報
- 代替実装案
- リスクや制約事項

## 実装して欲しいロジックの内容

以下に私が希望するロジックを示します。

＜ここにロジック本体を文章形式で記入してください＞

---

## 期待される成果物

1. **YAMLファイル** (`/flows/yaml/*.yaml`)
   - 動作可能なMABELフロー定義
   - 適切なコメント付き

2. **補助スクリプト** (`/flows/codes/*.py`) ※必要な場合のみ
   - Python関数の実装
   - 適切なドキュメント文字列付き

3. **README** (`/flows/README.md`) ※推奨
   - フローの概要説明
   - 使用方法
   - 入力データの形式
   - 出力データの形式

## 実装後の動作確認

実装が完了したら、以下を実行して動作確認を行ってください：

1. **構文チェック**
   ```bash
   # YAMLの構文チェック
   python -c "import yaml; yaml.safe_load(open('flows/yaml/your_flow.yaml'))"
   ```

2. **実行テスト**
   ```bash
   # 小規模なテストデータで実行
   sdg run --yaml flows/yaml/your_flow.yaml \
           --input test_data.jsonl \
           --output test_output.jsonl
   ```

3. **結果確認**
   - 出力ファイルが正しく生成されているか
   - エラーが発生していないか
   - 期待した結果が得られているか
```

---

## 使用方法

### 基本的な使い方

1. **プロンプトをコピー**
   - 上記の「基本プロンプトテンプレート」をコピーします

2. **ロジックを記述**
   - `＜ここにロジック本体を文章形式で記入してください＞` の部分を、実装したいロジックに置き換えます

3. **AIエージェントに送信**
   - コピーしたプロンプトをAIコーディングエージェントに送信します

4. **生成結果を確認**
   - 生成されたYAMLファイルを確認し、必要に応じて修正します

### 例：簡単なQ&Aフローの生成

```markdown
## 実装して欲しいロジックの内容

ユーザーからの質問を受け取り、AIモデルで回答を生成するシンプルなQ&Aフローを作成してください。

**要件:**
- 入力: `UserInput` フィールドに質問が入っている
- 処理: GPT-4o-miniを使用して簡潔な回答を生成
- 出力: `Answer` フィールドに回答を格納
- システムプロンプト: 「あなたは親切で簡潔な回答を提供するアシスタントです」
```

---

## プロンプトのカスタマイズ

### 高度な機能を含める場合

MABEL v2の高度な機能（再帰、ループ、条件分岐など）を使用する場合は、プロンプトに以下を追加してください：

```markdown
## 追加要件

以下のMABEL v2高度機能を活用してください：

- **再帰処理** (`op: recurse`): ［使用場面を記述］
- **Whileループ** (`op: while`): ［使用場面を記述］
- **ユーザー定義関数** (`functions.logic`): ［使用場面を記述］
- **MEX式言語**: 条件判定や値の計算に活用

参考: `docs/mabel/mabel_v2.md` の §6.2（Logicブロック）を参照してください。
```

### Python統合を含める場合

Pythonコードを含むフローを生成する場合：

```markdown
## Python統合の要件

以下のPython機能を実装してください：

- **インラインPython** (`function_code`): ［実装内容］
- **外部Pythonファイル** (`code_path`): ［使用するライブラリや処理内容］
- **統合仮想環境**: `runtime.python`で必要なパッケージを指定

必要なライブラリ:
- numpy
- pandas
- その他［必要に応じて追加］
```

---

## 実装例

### 例1: シンプルなデータ処理フロー

**ロジック記述:**

```
CSVデータを読み込み、各行の数値データを処理して統計情報を出力するフローを作成してください。

入力: 数値のCSV文字列
処理:
1. CSVをパースして数値リストに変換
2. 合計、平均、最大値、最小値を計算
3. Python関数で標準偏差を計算
出力: 統計情報をJSON形式で出力
```

**期待される生成物:**

```yaml
mabel:
  version: "2.0"
  name: "Data Statistics Flow"
  description: "CSVデータの統計処理"

runtime:
  python:
    interpreter: "python>=3.11,<3.13"
    venv: ".venv"

globals:
  vars:
    numbers: []
    stats: {}

blocks:
  - type: logic
    exec: 1
    name: "Parse CSV"
    op: for
    list: "{InputCSV}"
    parse: csv
    var: num
    map: {"to_number": "{num}"}
    outputs:
      - name: Numbers
        from: list

  - type: logic
    exec: 2
    name: "Calculate Sum"
    op: reduce
    list: "{Numbers}"
    value: 0
    var: item
    accumulator: sum
    body:
      - op: set
        var: sum
        value: {"add": [{"var": "sum"}, {"var": "item"}]}
    outputs:
      - name: Total
        from: accumulator

  # ... 追加の統計計算ブロック ...

  - type: end
    exec: 100
    final:
      - name: statistics
        value: "{stats}"
```

### 例2: AI駆動の反復処理

**ロジック記述:**

```
ドキュメントを要約し、要約が十分に簡潔になるまで繰り返し改善するフローを作成してください。

要件:
- 最初の要約を生成
- 文字数が200文字以下になるまで要約を繰り返す
- 最大5回まで試行
- 各反復の履歴を保存
```

---

## 動作確認方法

### 1. 構文検証

YAMLファイルの構文が正しいか確認：

```bash
# Pythonで構文チェック
python -c "import yaml; print(yaml.safe_load(open('flows/yaml/your_flow.yaml')))"

# または、SDGの設定読み込みでチェック
python -c "from sdg.config import load_config; load_config('flows/yaml/your_flow.yaml')"
```

### 2. テスト実行

小規模なテストデータで実行：

```bash
# テストデータを準備
echo '{"UserInput": "テストです"}' > test_input.jsonl

# 実行
sdg run --yaml flows/yaml/your_flow.yaml \
        --input test_input.jsonl \
        --output test_output.jsonl \
        --max-concurrent 1

# 結果確認
cat test_output.jsonl
```

### 3. デバッグモード

問題がある場合は、詳細ログを有効化：

```bash
# 環境変数でログレベルを設定
export PYTHONPATH=.
python -m sdg run --yaml flows/yaml/your_flow.yaml \
                  --input test_input.jsonl \
                  --output test_output.jsonl
```

---

## ベストプラクティス

### 1. 段階的な実装

- 最初はシンプルなフローから始める
- 動作確認後に機能を追加していく
- 各段階でテストを実行

### 2. 明確な命名

- ブロックには分かりやすい`name`を付ける
- 一意な`id`を設定して参照を明確にする
- 出力名は処理内容を反映した名前にする

### 3. エラーハンドリング

```yaml
blocks:
  - type: ai
    exec: 1
    model: gpt4
    prompts: ["..."]
    on_error: "retry"  # エラー時は再試行
    retry:
      max_attempts: 3
      backoff:
        type: "exponential"
        base_ms: 1000
    outputs:
      - name: Answer
        select: full
```

### 4. 予算制御

```yaml
budgets:
  loops:
    max_iters: 1000
    on_exceed: "error"
  recursion:
    max_depth: 128
    on_exceed: "error"
  wall_time_ms: 300000  # 5分
  ai:
    max_calls: 100
    max_tokens: 100000
```

### 5. ドキュメンテーション

- YAMLファイル内にコメントを追加
- READMEファイルで使用方法を説明
- サンプル入力データを提供

---

## トラブルシューティング

### よくある問題と解決方法

#### 問題1: YAMLの構文エラー

**エラーメッセージ:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**解決方法:**
- インデントを確認（スペースとタブの混在に注意）
- コロン（:）の後にスペースがあるか確認
- 引用符のエスケープを確認

#### 問題2: ブロック実行順序の問題

**症状:**
出力が参照される前に使用されている

**解決方法:**
```yaml
# execの値を確認し、依存関係に従って設定
blocks:
  - type: ai
    exec: 1  # 先に実行
    outputs:
      - name: Answer
        select: full
  
  - type: logic
    exec: 2  # Answerが使用可能になった後
    op: if
    cond: {"ne": ["{Answer}", ""]}
    # ...
```

#### 問題3: APIキーが見つからない

**エラーメッセージ:**
```
Error: Missing API key
```

**解決方法:**
```bash
# 環境変数を設定
export OPENAI_API_KEY="your-api-key-here"

# またはYAMLで直接指定（非推奨）
models:
  - name: gpt4
    api_key: "sk-..."  # 本番環境では環境変数を使用すること
```

#### 問題4: Python関数が見つからない

**エラーメッセージ:**
```
ModuleNotFoundError: No module named 'helpers'
```

**解決方法:**
```yaml
# code_pathを正しく設定
- type: python
  exec: 10
  code_path: ./flows/codes/helpers.py  # 相対パスを確認
  function: process_data
  # ...
```

#### 問題5: メモリ不足

**症状:**
大規模データ処理時にメモリエラー

**解決方法:**
```bash
# 並行数を減らす
sdg run --yaml flow.yaml \
        --input large_data.jsonl \
        --output result.jsonl \
        --max-concurrent 4  # デフォルトの8から減らす

# またはPhase 2最適化を有効化
sdg run --yaml flow.yaml \
        --input large_data.jsonl \
        --output result.jsonl \
        --enable-memory-optimization \
        --max-cache-size 500
```

---

## 関連リソース

- **[MABEL v2 完全仕様](./mabel/mabel_v2.md)** - 詳細な言語仕様
- **[SDG使用ガイド](./usage.ja.md)** - CLI・Python APIの使用方法
- **[サンプル集](../examples/)** - 動作するMABELフローの例
- **[Phase 2最適化ガイド](./others/phase2_optimization.md)** - 大規模データ処理の最適化

---

## まとめ

このガイドで紹介したプロンプトテンプレートを使用することで、AIコーディングエージェントを活用して効率的にMABELフローを作成できます。

**重要なポイント:**

1. 既存の仕様とサンプルを参照させる
2. 明確な要件を伝える
3. 段階的に実装する
4. 動作確認を必ず行う
5. ドキュメントを残す

高品質なMABELフローを作成し、AIエージェントシステムを最大限に活用してください。
