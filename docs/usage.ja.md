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

# Hugging Face Datasetsを使用する例
sdg run --yaml examples/sdg_demo.yaml --dataset squad --split validation --output output/result.jsonl
```

### ログ出力オプション

SDGは、`rich`ライブラリを使用した美しく読みやすいログ出力を提供します。

```bash
# 詳細ログを有効化（デバッグ出力）
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --verbose

# または短縮形
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl -v

# 日本語UIを使用（デフォルトは英語）
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --ui-locale ja

# レガシーログ形式を使用（richフォーマット無効化）
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --legacy-logs

# 進捗表示を無効化
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --no-progress
```

**ログオプション:**

| オプション | 説明 |
|-----------|------|
| `--verbose`, `-v` | 詳細ログを有効化（デバッグメッセージを表示） |
| `--ui-locale {en,ja}` | ログ出力のUI言語（デフォルト: en） |
| `--legacy-logs` | レガシーログ形式を使用（richフォーマット無効化） |
| `--no-progress` | 進捗表示を無効化 |

**ログ出力の特徴:**

- **デフォルトモード**: `rich`ライブラリによる美しいフォーマット、プログレスバー、カラー出力（英語UI）
- **日本語UI** (`--ui-locale ja`): ログメッセージを日本語で表示
- **Verboseモード** (`--verbose`): デバッグメッセージを含む詳細な出力
- **レガシーモード** (`--legacy-logs`): シンプルなテキスト出力（richが利用できない環境向け）
- **Quietモード** (`--no-progress`): 進捗表示を無効化（ログファイルへのリダイレクト時に便利）

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
| `--dataset` | - | Hugging Faceデータセット名 |
| `--subset` | - | データセットサブセット名 |
| `--split` | train | データセットスプリット |
| `--mapping` | - | キーマッピング（`orig:new`形式、複数指定可） |

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
適応的コントローラーはTCP輻輳制御（Vegas/Reno/BBR）にインスパイアされたアルゴリズムを使用:
- **Slow Start**: レイテンシが目標値の0.7倍未満の場合、並行数を2倍に増加（指数増加）
- **Congestion Avoidance**: ssthresh到達後は+2ずつ線形増加
- **段階的減少**: 輻輳の深刻度に応じて15%〜50%減少（軽度の輻輳は無視）
- **EMA平滑化**: ノイズを除去してトレンドを検出
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

[`AdaptiveController`](../sdg/adaptive/controller.py:286)は、TCP輻輳制御アルゴリズム（Vegas、Reno、BBR）にインスパイアされた高度な制御ロジックを実装し、観測されたレイテンシとバックエンドメトリクスに基づいて並行レベルを自動調整します。

**主な機能:**
- **制御フェーズ**: Slow Start（指数増加）とCongestion Avoidance（線形増加）の2フェーズ制御
- **EMAベースの平滑化**: 指数移動平均によるノイズ除去とトレンド検出
- **Vegas-style輻輳検出**: RTTベースのプロアクティブな輻輳検出
- **段階的減少ロジック**: エラー時は即座に、レイテンシ悪化時は緩やかに減少
- バックエンドのキュー深度とキャッシュ使用率を監視（vLLM/SGLang）
- 目標レイテンシ境界を維持

**アルゴリズムの詳細:**

1. **Slow Start フェーズ**:
   - 並行数がssthresh（スロースタート閾値）未満、かつレイテンシが目標の70%以下の場合
   - 並行数を指数関数的に増加（2倍）
   - 最適な並行数に素早く収束

2. **Congestion Avoidance フェーズ**:
   - ssthresh到達後は線形増加（Additive Increase）
   - 慎重に上限を探る

3. **段階的減少ロジック**:
   - エラー発生時: 即座に乗算減少（Multiplicative Decrease）
   - 深刻な輻輳: 乗算減少（50%）
   - 中程度の輻輳: 緩やかな減少（15%）
   - 軽度の輻輳: 3回連続で検出された場合のみ線形減少

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

# 現在の制御フェーズを取得
phase = controller.phase  # ControlPhase.SLOW_START or ControlPhase.CONGESTION_AVOIDANCE

# 完了したリクエストを記録
controller.record_latency(latency_ms=150.0, is_error=False)

# 統計情報を取得
stats = controller.get_stats()
print(f"現在の並行度: {stats['current_concurrency']}")
print(f"P95レイテンシ: {stats['p95_latency_ms']}ms")
print(f"制御フェーズ: {stats['phase']}")
print(f"EMA レイテンシ: {stats['ema_latency_ms']}ms")
print(f"輻輳シグナル: {stats['vegas_congestion_signal']}")
```

**詳細設定:**

```python
controller = AdaptiveController(
    min_concurrency=1,
    max_concurrency=128,
    target_latency_ms=2000.0,
    target_queue_depth=32,
    # 基本AIMDパラメータ
    increase_step=2,              # CAフェーズでのサイクルごとの加法増加量
    decrease_factor=0.5,          # エラー時の乗法減少係数（50%）
    # 感度
    latency_tolerance=1.5,        # レイテンシが目標の1.5倍を超えたら減少
    error_rate_threshold=0.05,    # エラー率が5%を超えたら減少
    # タイミング
    adjustment_interval_ms=1000,  # 1秒ごとに調整
    window_size=50,               # 直近50サンプルを追跡
    # 高度なパラメータ
    ema_alpha=0.3,                # EMA平滑化係数（0-1、高いほど反応的）
    slow_start_threshold=32,      # 初期ssthresh
    vegas_alpha=2.0,              # Vegas下限閾値
    vegas_beta=4.0,               # Vegas上限閾値
    mild_decrease_factor=0.85,    # 軽度輻輳時の減少係数（15%減少）
    trend_sensitivity=0.1,        # トレンド検出感度
)
```

**EMA統計と輻輳統計の取得:**

```python
# EMA統計情報（ノイズ除去されたレイテンシ）
ema_stats = controller.get_ema_stats()
print(f"平滑化レイテンシ: {ema_stats['latency']['value']}ms")
print(f"レイテンシトレンド: {ema_stats['latency']['trend']}")  # 正=増加中、負=減少中
print(f"レイテンシ分散: {ema_stats['latency']['variance']}")

# 輻輳検出統計（Vegas-style）
congestion_stats = controller.get_congestion_stats()
print(f"ベースレイテンシ: {congestion_stats['base_latency_ms']}ms")
print(f"輻輳レベル: {congestion_stats['congestion_level']}")  # none/mild/moderate/severe
print(f"輻輳シグナル: {congestion_stats['congestion_signal']}")
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

## Phase 2 最適化

SDG Nexus Phase 2では、スケーラビリティとメモリ効率を向上させる高度な最適化機能を導入しました。これらの機能はすべて**オプトイン方式**で、デフォルトでは無効になっており、後方互換性を保証します。

### 階層的タスクスケジューリング

大規模データセットを効率的に処理するため、データをチャンク単位に分割し、段階的にタスクを生成・実行します。

**特徴:**
- **チャンク単位のデータ分割**: データセットを適切なサイズのチャンクに分割
- **最大保留タスク数の制限**: メモリ使用量を制御
- **処理開始遅延の最小化**: 全タスクを一度に生成しないため、処理開始が高速
- **条件変数による効率的な制御**: Pythonの`Condition`を用いた協調制御

**CLIでの使用:**

```bash
# 階層的スケジューリングを有効化
sdg run \
  --yaml pipeline.yaml \
  --input large_data.jsonl \
  --output result.jsonl \
  --enable-scheduling \
  --max-pending-tasks 100 \
  --chunk-size 50
```

**Python APIでの使用:**

```python
import asyncio
from sdg.config import load_config
from sdg.executors import run_pipeline_streaming

async def main():
    cfg = load_config("config.yaml")
    dataset = [{"id": i, "text": f"sample_{i}"} for i in range(10000)]
    
    # 階層的スケジューリングを有効化
    async for result in run_pipeline_streaming(
        cfg,
        dataset,
        max_concurrent=16,
        enable_scheduling=True,          # スケジューリングを有効化
        max_pending_tasks=100,            # 最大保留タスク数
        chunk_size=50,                    # チャンクサイズ
    ):
        if result.error:
            print(f"Error in row {result.row_index}: {result.error}")
        else:
            print(f"Completed row {result.row_index}")

asyncio.run(main())
```

**設定パラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `enable_scheduling` | `False` | 階層的スケジューリングを有効化 |
| `max_pending_tasks` | `1000` | 最大保留タスク数（メモリ制御用） |
| `chunk_size` | `100` | データセット分割サイズ |

### メモリ最適化

#### ストリーミングコンテキストマネージャ

LRUキャッシュを使用してコンテキストを管理し、処理が完了したコンテキストを自動的に解放します。

**特徴:**
- **LRUキャッシュ**: 最近使用されていないコンテキストを自動削除
- **自動メモリ解放**: 処理完了時にコンテキストをクリア
- **参照カウント**: 安全なメモリ解放
- **オプションのメモリ監視**: psutilを使用したメモリ使用状況監視

**CLIでの使用:**

```bash
# メモリ最適化を有効化
sdg run \
  --yaml pipeline.yaml \
  --input large_data.jsonl \
  --output result.jsonl \
  --enable-memory-optimization \
  --max-cache-size 500 \
  --enable-memory-monitoring
```

**Python APIでの使用:**

```python
async def main():
    cfg = load_config("config.yaml")
    dataset = [{"id": i} for i in range(50000)]
    
    # メモリ最適化を有効化
    async for result in run_pipeline_streaming(
        cfg,
        dataset,
        max_concurrent=32,
        enable_memory_optimization=True,  # メモリ最適化を有効化
        max_cache_size=500,               # キャッシュサイズ
        enable_memory_monitoring=True,    # メモリ監視を有効化
    ):
        # 処理...
        pass
```

**設定パラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `enable_memory_optimization` | `False` | メモリ最適化を有効化 |
| `max_cache_size` | `500` | コンテキストキャッシュの最大サイズ |
| `enable_memory_monitoring` | `False` | メモリ使用状況監視を有効化 |

#### バッチ処理用段階的メモリ解放

従来の`run_pipeline`（バッチモード）でも段階的なメモリ解放を実現します。

**Python APIでの使用:**

```python
async def main():
    cfg = load_config("config.yaml")
    dataset = [{"id": i} for i in range(10000)]
    
    # バッチモードでメモリ最適化を有効化
    results = await run_pipeline(
        cfg,
        dataset,
        enable_memory_optimization=True,  # メモリ最適化を有効化
        gc_interval=100,                  # GC実行間隔
        memory_threshold_mb=1024,         # メモリ警告閾値（MB）
    )
```

**設定パラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `enable_memory_optimization` | `False` | メモリ最適化を有効化 |
| `enable_memory_monitoring` | `False` | メモリ使用状況監視を有効化 |
| `gc_interval` | `100` | ガベージコレクション実行間隔（処理行数） |
| `memory_threshold_mb` | `1024` | メモリ使用量警告閾値（MB） |

### 適応的パイプラインとの統合

階層的スケジューリングとメモリ最適化は、適応的並行性制御と組み合わせて使用できます。

**CLIでの使用:**

```bash
# すべての最適化を有効化
sdg run \
  --yaml pipeline.yaml \
  --input huge_data.jsonl \
  --output result.jsonl \
  --adaptive \
  --min-batch 4 \
  --max-batch 64 \
  --target-latency-ms 2000 \
  --use-vllm-metrics \
  --enable-scheduling \
  --max-pending-tasks 200 \
  --chunk-size 100 \
  --enable-memory-optimization \
  --max-cache-size 1000
```

**Python APIでの使用:**

```python
from sdg.executors import run_pipeline_streaming_adaptive

async def main():
    cfg = load_config("config.yaml")
    dataset = [{"id": i} for i in range(100000)]
    
    # すべての最適化を有効化
    async for result in run_pipeline_streaming_adaptive(
        cfg,
        dataset,
        # 適応的制御
        max_concurrent=64,
        min_concurrent=4,
        target_latency_ms=2000,
        metrics_type="vllm",
        # Phase 2: スケジューリング
        enable_scheduling=True,
        max_pending_tasks=200,
        chunk_size=100,
        # Phase 2: メモリ最適化
        enable_memory_optimization=True,
        max_cache_size=1000,
        enable_memory_monitoring=True,
    ):
        # 処理...
        pass
```

### パフォーマンスガイドライン

#### 小規模データセット（< 1,000行）

```python
# スケジューリング: 無効（オーバーヘッドが大きい）
enable_scheduling=False

# メモリ最適化: 無効（必要ない）
enable_memory_optimization=False
```

#### 中規模データセット（1,000 - 10,000行）

```python
# スケジューリング: 有効
enable_scheduling=True
max_pending_tasks=100
chunk_size=50

# メモリ最適化: 有効
enable_memory_optimization=True
max_cache_size=500
```

#### 大規模データセット（> 10,000行）

```python
# スケジューリング: 有効
enable_scheduling=True
max_pending_tasks=500
chunk_size=200

# メモリ最適化: 有効
enable_memory_optimization=True
max_cache_size=1000
enable_memory_monitoring=True  # メモリ監視を有効化
gc_interval=100
```

### チューニングのヒント

1. **max_pending_tasks**:
   - 小さすぎる: スループットが低下
   - 大きすぎる: メモリ使用量が増加
   - 推奨: `max_concurrent` の 5-10倍

2. **chunk_size**:
   - 小さすぎる: スケジューリングオーバーヘッドが増加
   - 大きすぎる: メモリ使用量が増加
   - 推奨: `max_pending_tasks` の 20-50%

3. **max_cache_size**:
   - LRUキャッシュのサイズ
   - 推奨: データセットサイズの 5-10%

### トラブルシューティング

**メモリ使用量が高い:**
1. `enable_memory_optimization=True` を設定
2. `max_cache_size` を減らす
3. `gc_interval` を小さくする（より頻繁にGCを実行）
4. `enable_memory_monitoring=True` でメモリ使用状況を監視

**処理が遅い:**
1. `max_concurrent` を増やす
2. `max_pending_tasks` を増やす
3. `chunk_size` を大きくする
4. `enable_scheduling=False` を試す（小規模データセットの場合）

**メモリ不足エラー:**
1. `max_pending_tasks` を減らす
2. `chunk_size` を小さくする
3. `max_cache_size` を減らす
4. `enable_memory_optimization=True` を設定

詳細な実装とAPIリファレンスについては、[Phase 2最適化ガイド](phase2_optimization.md)を参照してください。

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

#### Hugging Face Datasets

Hugging Face Hub上のデータセットを直接読み込むことができます。

**基本的な使用方法:**

```bash
# squadデータセットのvalidationスプリットを使用
sdg run --yaml pipeline.yaml --dataset squad --split validation --output result.jsonl

# サブセットを指定する場合
sdg run --yaml pipeline.yaml --dataset glue --subset mrpc --split train --output result.jsonl
```

**キーマッピング機能:**

データセットのキー名とパイプラインで期待される入力キー名が異なる場合、`--mapping` オプションを使用してキーをマッピングできます。

```bash
# 例: データセットの "context" を "text" として、"question" を "query" として扱う
sdg run --yaml pipeline.yaml \
  --dataset squad \
  --mapping context:text \
  --mapping question:query \
  --output result.jsonl

# 複数のキーマッピングを指定
sdg run --yaml pipeline.yaml \
  --dataset my_dataset \
  --mapping original_field:UserInput \
  --mapping label:Category \
  --output result.jsonl
```

**オプション:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--dataset` | - | Hugging Face データセット名（必須） |
| `--subset` | - | データセットのサブセット名（オプション） |
| `--split` | train | データセットのスプリット（train/validation/test等） |
| `--mapping` | - | キーマッピング（`orig:new`形式、複数指定可） |

**注意事項:**

- `--dataset` を指定した場合、`--input` は指定できません
- キーマッピングは、データセットの各行に対して適用されます
- マッピングされたキーは、パイプラインのYAMLファイル内で使用できます

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