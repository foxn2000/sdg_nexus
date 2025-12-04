# SDG (Scalable Data Generator) 使用ガイド

このドキュメントでは、既存のYAMLファイル（MABEL形式）を使用してSDGパイプラインを実行する方法について説明します。

## 目次

1. [概要](#概要)
2. [CLIからの使用方法](#cliからの使用方法)
3. [Python APIからの使用方法](#python-apiからの使用方法)
4. [高度な最適化機能](#高度な最適化機能)
5. [パーサーについて](#パーサーについて)
6. [入出力データ形式](#入出力データ形式)

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

SDGにはストリーミングデータ処理のための2つの実行モードがあります:

#### 1. 固定並行数のストリーミングモード（デフォルト）

固定された並行レベルで各データ行を並列処理し、完了した行から順次出力ファイルへ書き込みます。

```bash
# ストリーミングモード（デフォルト）
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl

# 固定並行数を指定
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --max-concurrent 16

# 進捗表示を無効化
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --no-progress
```

**オプション:**
| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--max-concurrent` | 8 | 同時処理行数の上限（固定） |
| `--no-progress` | false | 進捗表示を無効化 |

**特徴:**
- 実行中は並行数が固定
- 途中結果が失われにくい（リアルタイム書き込み）
- メモリ効率が良い
- 出力順序は処理完了順（入力順序と異なる場合あり）

> **Note:** 元の順序が必要な場合は、出力の `_row_index` フィールドでソートしてください。

#### 2. 適応的並行性制御のストリーミングモード

観測されたレイテンシやオプションのバックエンドメトリクスに基づいて、並行数を動的に調整します。`--adaptive` フラグで有効化します。

```bash
# 適応的並行性制御を有効化
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --adaptive --min-batch 1 --max-batch 32 --target-latency-ms 2000

# vLLMバックエンドメトリクスを使用
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --adaptive --use-vllm-metrics --min-batch 1 --max-batch 64

# 高スループット向けリクエストバッチングを有効化
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --adaptive --use-vllm-metrics --enable-request-batching
```

**オプション:**
| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--adaptive` | false | 適応的並行性制御を有効化 |
| `--min-batch` | 1 | 最小並行処理数 |
| `--max-batch` | 64 | 最大並行処理数 |
| `--target-latency-ms` | 3000 | 目標P95レイテンシ（ミリ秒） |
| `--target-queue-depth` | 32 | 目標バックエンドキュー深度 |
| `--use-vllm-metrics` | false | vLLMのPrometheusメトリクスを使用 |
| `--use-sglang-metrics` | false | SGLangのPrometheusメトリクスを使用 |
| `--enable-request-batching` | false | リクエストバッチングを有効化 |
| `--max-batch-size` | 32 | バッチあたりの最大リクエスト数 |
| `--max-wait-ms` | 50 | バッチ形成の最大待機時間（ミリ秒） |

**特徴:**
- `--min-batch` と `--max-batch` の間で並行数を自動調整
- レイテンシが低い場合は並行数を増加
- エラー発生やレイテンシ急上昇時は並行数を減少
- バックエンドメトリクス（vLLM/SGLang）を監視してより良い最適化
- 最大スループットのためのオプションのリクエストバッチング

**動作原理:**
適応的コントローラーはAIMD（加法増加乗法減少）アルゴリズムを使用:
- **増加**: P95レイテンシが目標値の0.7倍未満の場合、並行数に+2を加算
- **減少**: レイテンシが目標値の1.5倍を超えるかエラーが発生した場合、50%減少
- **監視**: バックエンドメトリクスを500msごとにポーリング（有効化時）
- **調整**: 直近50サンプルに基づいて1秒ごとに評価

### 共通オプション

| オプション | 説明 |
|-----------|------|
| `--save-intermediate` | 中間結果を保存する |

### 最適化オプション

SDGは高速化とリソース効率化のための最適化オプションを提供しています：

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--use-shared-transport` | false | 共有HTTPトランスポートを使用（コネクションプール共有） |
| `--no-http2` | false | HTTP/2を無効化（デフォルトは有効） |

**最適化オプションの詳細:**

- **`--use-shared-transport`**: 複数のリクエスト間でHTTPコネクションプールを共有します。これにより、新しいコネクションの確立のオーバーヘッドが削減され、特に多数の短いリクエストを処理する場合にパフォーマンスが向上します。

- **`--no-http2`**: デフォルトではHTTP/2が有効になっています。このフラグを使用するとHTTP/1.1にフォールバックします。一部のバックエンドやプロキシとの互換性問題がある場合に使用します。

**使用例:**

```bash
# 共有トランスポートを使用して接続効率を向上
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --use-shared-transport

# HTTP/2を無効化してHTTP/1.1を使用
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --no-http2

# 適応的並行性制御と最適化オプションを組み合わせ
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --adaptive --use-vllm-metrics \
  --use-shared-transport
```

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

# 最適化オプションを使用したストリーミング実行
run_streaming(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_concurrent=8,
    save_intermediate=False,
    show_progress=True,
    use_shared_transport=True,  # 共有HTTPトランスポートを使用
    http2=True,                  # HTTP/2を有効化（デフォルト）
)
```

### 適応的並行性制御での実行

```python
from sdg.runner import run_streaming_adaptive

# 適応的並行性制御で実行
run_streaming_adaptive(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_concurrent=64,
    min_concurrent=1,
    target_latency_ms=2000,
    target_queue_depth=32,
    metrics_type="vllm",  # "none", "vllm", "sglang"
    save_intermediate=False,
    show_progress=True,
    use_shared_transport=True,
    http2=True,
)
```

### リクエストバッチングでの実行

```python
from sdg.runner import run_streaming_adaptive_batched

# リクエストバッチングを有効化した実行
run_streaming_adaptive_batched(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_concurrent=64,
    min_concurrent=1,
    target_latency_ms=2000,
    target_queue_depth=32,
    metrics_type="vllm",
    max_batch_size=32,
    max_wait_ms=50,
    save_intermediate=False,
    show_progress=True,
    use_shared_transport=True,
    http2=True,
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

## 高度な最適化機能

SDG Nexusは、高スループットのLLM推論ワークロードのための高度な最適化機能を提供します。これらの機能は、vLLMやSGLangバックエンドを使用する際に特に有効です。

### 適応型並行制御

[`AdaptiveController`](../sdg/adaptive/controller.py:28)は、AIMD（加法増加乗法減少）アルゴリズムを使用して、観測されたレイテンシとバックエンドメトリクスに基づいて並行レベルを自動調整します。

**主な機能:**
- レイテンシが低い場合は並行度を自動的に増加
- エラー発生やレイテンシの急上昇時には並行度を素早く減少
- バックエンドのキュー深度とキャッシュ使用率を監視（vLLM/SGLang）
- 目標レイテンシ境界を維持

**基本的な使用方法:**

```python
from sdg.adaptive.controller import AdaptiveController

controller = AdaptiveController(
    min_concurrency=1,
    max_concurrency=64,
    target_latency_ms=2000.0,  # 目標P95レイテンシ
    target_queue_depth=32,
)

# 現在の並行度制限を取得
limit = controller.current_concurrency

# 完了したリクエストを記録
controller.record_latency(latency_ms=150.0, is_error=False)

# 統計情報を取得
stats = controller.get_stats()
print(f"現在の並行度: {stats['current_concurrency']}")
print(f"P95レイテンシ: {stats['p95_latency_ms']}ms")
```

**詳細設定:**

```python
controller = AdaptiveController(
    min_concurrency=1,
    max_concurrency=128,
    target_latency_ms=2000.0,
    target_queue_depth=32,
    # AIMDパラメータ
    increase_step=2,              # サイクルごとの加法増加量
    decrease_factor=0.5,          # 乗法減少係数（50%）
    # 感度
    latency_tolerance=1.5,        # レイテンシが目標の1.5倍を超えたら減少
    error_rate_threshold=0.05,    # エラー率が5%を超えたら減少
    # タイミング
    adjustment_interval_ms=1000,  # 1秒ごとに調整
    window_size=50,               # 直近50サンプルを追跡
)
```

### メトリクス収集

[`MetricsCollector`](../sdg/adaptive/metrics.py:75)は、バックエンドのメトリクスエンドポイントをポーリングして、キュー深度、キャッシュ使用率、スループットに関するリアルタイム情報を収集します。

**サポートされているバックエンド:**
- **vLLM**: `/metrics`エンドポイントのPrometheusメトリクス
- **SGLang**: `/metrics`エンドポイントのPrometheusメトリクス

**基本的な使用方法:**

```python
from sdg.adaptive.metrics import MetricsCollector, MetricsType

collector = MetricsCollector(
    base_url="http://localhost:8000",
    metrics_type=MetricsType.VLLM,
    poll_interval_ms=500,
)

await collector.start()

# 最新のメトリクスを取得
metrics = collector.get_latest()
if metrics and metrics.is_valid:
    print(f"キュー深度: {metrics.queue_depth}")
    print(f"キャッシュ使用率: {metrics.cache_usage_percent}%")
    print(f"実行中のリクエスト: {metrics.num_requests_running}")

await collector.stop()
```

**利用可能なメトリクス:**

| メトリクス | 説明 | vLLM | SGLang |
|--------|-------------|------|--------|
| `num_requests_waiting` | キュー内のリクエスト数 | ✓ | ✓ |
| `num_requests_running` | 現在処理中のリクエスト数 | ✓ | ✓ |
| `queue_depth` | 総キュー深度（待機中+実行中） | ✓ | ✓ |
| `cache_usage_percent` | KVキャッシュ使用率 | ✓ | ✓ |
| `prompt_tokens_total` | 総入力トークン数 | ✓ | ✓ |
| `generation_tokens_total` | 総出力トークン数 | ✓ | ✓ |

### リクエストバッチング

[`RequestBatcher`](../sdg/adaptive/batcher.py:38)は、連続的バッチングの利点を最大化するため、複数のリクエストをまとめて送信します。

**主な機能:**
- キュー状態に基づく動的バッチサイジング
- レイテンシ境界を保証する最大待機時間
- トークン認識型バッチング（オプション）
- 優先度キューサポート

**基本的な使用方法:**

```python
from sdg.adaptive.batcher import RequestBatcher

async def batch_processor(payloads):
    # バッチ処理ロジック
    results = await client.batch_chat(payloads)
    return results

batcher = RequestBatcher(
    batch_processor=batch_processor,
    max_batch_size=64,
    max_wait_ms=50,
)

async with batcher:
    # リクエストを送信
    result = await batcher.submit({
        "messages": [{"role": "user", "content": "こんにちは"}]
    })
```

**詳細設定:**

```python
batcher = RequestBatcher(
    batch_processor=batch_processor,
    max_batch_size=64,
    max_wait_ms=50,
    max_tokens_per_batch=8192,  # 総トークン数を制限
    token_estimator=custom_estimator,  # カスタムトークンカウンター
    enabled=True,
)

# 統計情報を取得
stats = batcher.get_stats()
print(f"平均バッチサイズ: {stats['avg_batch_size']}")
print(f"保留中のリクエスト: {stats['pending_count']}")
```

### 統合最適化

[`AdaptiveConcurrencyManager`](../sdg/adaptive/controller.py:293)は、すべての最適化機能を組み合わせて、ターンキーの高性能推論を実現します。

**完全な例:**

```python
from sdg.adaptive.controller import AdaptiveConcurrencyManager
from sdg.adaptive.metrics import MetricsType

async def main():
    manager = AdaptiveConcurrencyManager(
        base_url="http://localhost:8000",
        metrics_type=MetricsType.VLLM,
        min_concurrency=1,
        max_concurrency=64,
        target_latency_ms=2000.0,
        target_queue_depth=32,
        enabled=True,
    )
    
    async with manager:
        # 自動並行制御でリクエストを実行
        async with manager.acquire():
            start = time.time()
            result = await execute_request()
            latency = (time.time() - start) * 1000
            manager.record_latency(latency)
        
        # 統計情報を監視
        stats = manager.get_stats()
        print(f"現在の並行度: {stats['current_concurrency']}")
        print(f"バックエンドキュー: {stats.get('backend_queue_depth', 'N/A')}")

asyncio.run(main())
```

**CLI統合:**

ストリーミングモードで `--adaptive` フラグを使用して適応的並行性制御を有効化:

```bash
# 基本的な適応的並行性制御
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --min-batch 1 \
  --max-batch 32 \
  --target-latency-ms 2000

# vLLMバックエンドメトリクスを使用
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --min-batch 1 \
  --max-batch 64

# リクエストバッチングを有効化
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --enable-request-batching \
  --max-batch-size 32
```

### マルチバックエンドサポート

複数のバックエンドインスタンスによる負荷分散デプロイメントの場合:

```python
from sdg.adaptive.metrics import MultiBackendMetricsCollector, MetricsType

collector = MultiBackendMetricsCollector(
    backends={
        "http://backend1:8000": MetricsType.VLLM,
        "http://backend2:8000": MetricsType.VLLM,
        "http://backend3:8000": MetricsType.VLLM,
    },
    poll_interval_ms=500,
)

await collector.start()

# 全バックエンドの集約メトリクスを取得
metrics = collector.get_aggregated_metrics()
print(f"総キュー深度: {collector.get_total_queue_depth()}")

await collector.stop()
```

### ベストプラクティス

1. **保守的に開始**: 低い並行度制限から開始し、コントローラーに自動増加させる
2. **メトリクスの監視**: [`get_stats()`](../sdg/adaptive/controller.py:422)を使用してパフォーマンスを追跡
3. **ワークロードに合わせて調整**: 要件に基づいて`target_latency_ms`を調整
4. **バックエンドメトリクス**: より良い最適化のためvLLM/SGLangのメトリクス収集を有効化
5. **バッチング**: 予測可能なリクエストパターンを持つワークロードにはリクエストバッチングを使用

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
# examples/sdg_demo.yaml を固定並行数で使用
sdg run \
  --yaml examples/sdg_demo.yaml \
  --input examples/data/input.jsonl \
  --output output/qa_result.jsonl \
  --max-concurrent 16
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

### 例3: 固定並行数で大量データの処理

```bash
# 大量データを固定並行数でストリーミング処理
sdg run \
  --yaml pipeline.yaml \
  --input large_dataset.jsonl \
  --output output/large_result.jsonl \
  --max-concurrent 16
```

### 例4: vLLMバックエンドでの適応的並行性制御

```bash
# レイテンシに基づいて並行数を動的に調整
sdg run \
  --yaml examples/question_generator_agent_v2.yaml \
  --input examples/data/question_generator_input.jsonl \
  --output output/generated_questions_v2.jsonl \
  --adaptive \
  --min-batch 1 \
  --max-batch 32 \
  --target-latency-ms 2000
```

### 例5: vLLMメトリクスを使用した最適化

```bash
# vLLMのPrometheusメトリクスを使用して最適な並行制御
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --min-batch 1 \
  --max-batch 64 \
  --target-latency-ms 2000
```

### 例6: リクエストバッチングで最大スループット

```bash
# 最大スループットのためリクエストバッチングを有効化
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --enable-request-batching \
  --max-batch 64 \
  --max-batch-size 32
```

### 例7: 最適化オプションを使用した高速実行

```bash
# 共有HTTPトランスポートを使用して接続効率を最大化
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --max-concurrent 16 \
  --use-shared-transport

# 適応的並行性制御と最適化オプションを組み合わせ
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --min-batch 1 \
  --max-batch 64 \
  --use-shared-transport

# HTTP/2を無効化して互換性を確保
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --max-concurrent 8 \
  --no-http2
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