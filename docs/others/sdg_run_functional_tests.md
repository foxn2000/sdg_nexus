# SDG RUN 機能テスト仕様書

## 概要

このドキュメントは、`sdg run`コマンドの全機能が正しく動作することを確認するための機能テスト仕様書です。`examples/3d_scaling_agent.yaml`を使用して各オプションをテストします。

## テスト環境

- **テスト用YAMLファイル**: `examples/3d_scaling_agent.yaml`
- **テスト用入力データ**: `examples/data/3d_scaling_input.jsonl`
- **出力ディレクトリ**: `output/tests/`（テスト実行前に作成）

## 前提条件

```bash
# 出力ディレクトリを作成
mkdir -p output/tests

# LLMサーバーが起動していることを確認
# 例: http://localhost:8000/v1 でqwen3モデルが利用可能
```

---

## テストケース一覧

### 1. 基本機能テスト

#### 1.1 基本実行（最小限のオプション）

**目的**: 必須パラメータのみでのコマンド実行を確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_1.1_basic.jsonl
```

**期待結果**:
- ✅ コマンドが正常に完了する
- ✅ `output/tests/test_1.1_basic.jsonl`が作成される
- ✅ 出力ファイルに有効なJSONL形式のデータが含まれる
- ✅ デフォルト並行数8で処理される

---

#### 1.2 ヘルプ表示

**目的**: ヘルプオプションが正しく表示されることを確認

```bash
# 英語ヘルプ
sdg run --help

# 日本語ヘルプ
sdg run --help.ja

# メインヘルプ
sdg --help

# メイン日本語ヘルプ
sdg --help.ja
```

**期待結果**:
- ✅ すべてのオプションが表示される
- ✅ 日本語ヘルプが正しく表示される
- ✅ 使用例が含まれている

---

#### 1.3 中間出力保存

**目的**: `--save-intermediate`オプションの動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_1.3_intermediate.jsonl \
  --save-intermediate
```

**期待結果**:
- ✅ 最終出力ファイルが作成される
- ✅ 中間出力ファイルが作成される（ブロックごとの出力）
- ✅ 中間ファイル名に対応するブロックIDが含まれる

---

### 2. ストリーミングモード機能テスト

#### 2.1 並行数制御

**目的**: `--max-concurrent`オプションの動作確認

```bash
# 並行数を16に設定
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_2.1_concurrent_16.jsonl \
  --max-concurrent 16
```

**期待結果**:
- ✅ 16並行で処理される
- ✅ 処理速度が並行数8のテストより向上する

---

#### 2.2 プログレス表示無効化

**目的**: `--no-progress`オプションの動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_2.2_no_progress.jsonl \
  --no-progress
```

**期待結果**:
- ✅ プログレスバーが表示されない
- ✅ 処理は正常に完了する
- ✅ ログ出力のみが表示される

---

### 3. 適応的並行性制御テスト

#### 3.1 基本的な適応的制御

**目的**: `--adaptive`オプションの基本動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_3.1_adaptive.jsonl \
  --adaptive
```

**期待結果**:
- ✅ 適応的並行性制御が有効化される
- ✅ レイテンシに応じて並行数が動的に調整される
- ✅ デフォルト範囲（1〜64）で並行数が変動する

---

#### 3.2 適応的制御の範囲指定

**目的**: `--min-batch`と`--max-batch`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_3.2_adaptive_range.jsonl \
  --adaptive \
  --min-batch 2 \
  --max-batch 32
```

**期待結果**:
- ✅ 並行数が2〜32の範囲で調整される
- ✅ 範囲外の並行数にならない

---

#### 3.3 レイテンシ目標設定

**目的**: `--target-latency-ms`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_3.3_target_latency.jsonl \
  --adaptive \
  --target-latency-ms 2000
```

**期待結果**:
- ✅ P95レイテンシが2000ms前後に維持される
- ✅ レイテンシが目標を超えると並行数が減少する

---

#### 3.4 キュー深度目標設定

**目的**: `--target-queue-depth`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_3.4_queue_depth.jsonl \
  --adaptive \
  --target-queue-depth 16
```

**期待結果**:
- ✅ バックエンドキュー深度が16前後に維持される
- ✅ キュー深度に応じて並行数が調整される

---

### 4. バックエンドメトリクス統合テスト

#### 4.1 vLLMメトリクス使用

**目的**: `--use-vllm-metrics`の動作確認

**前提条件**: vLLMサーバーがPrometheusメトリクスを公開している

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_4.1_vllm_metrics.jsonl \
  --adaptive \
  --use-vllm-metrics
```

**期待結果**:
- ✅ vLLMのメトリクスを取得して並行数を最適化する
- ✅ バックエンドの負荷状況に応じて調整される

---

#### 4.2 SGLangメトリクス使用

**目的**: `--use-sglang-metrics`の動作確認

**前提条件**: SGLangサーバーがPrometheusメトリクスを公開している

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_4.2_sglang_metrics.jsonl \
  --adaptive \
  --use-sglang-metrics
```

**期待結果**:
- ✅ SGLangのメトリクスを取得して並行数を最適化する
- ✅ バックエンドの負荷状況に応じて調整される

---

### 5. リクエストバッチングテスト

#### 5.1 基本的なバッチング

**目的**: `--enable-request-batching`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_5.1_batching.jsonl \
  --adaptive \
  --enable-request-batching
```

**期待結果**:
- ✅ 複数のリクエストがバッチ化される
- ✅ スループットが向上する
- ✅ デフォルトバッチサイズ32で動作する

---

#### 5.2 バッチサイズ設定

**目的**: `--max-batch-size`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_5.2_batch_size.jsonl \
  --adaptive \
  --enable-request-batching \
  --max-batch-size 16
```

**期待結果**:
- ✅ 1バッチあたり最大16リクエストが含まれる
- ✅ 指定サイズを超えるバッチが作成されない

---

#### 5.3 バッチ待機時間設定

**目的**: `--max-wait-ms`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_5.3_batch_wait.jsonl \
  --adaptive \
  --enable-request-batching \
  --max-wait-ms 100
```

**期待結果**:
- ✅ バッチ形成の最大待機時間が100msに制限される
- ✅ タイムアウト後にバッチが送信される

---

### 6. Phase 2 最適化機能テスト

#### 6.1 階層的タスクスケジューリング

**目的**: `--enable-scheduling`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_6.1_scheduling.jsonl \
  --enable-scheduling
```

**期待結果**:
- ✅ 階層的タスクスケジューリングが有効化される
- ✅ 大規模データセットの処理が効率化される
- ✅ メモリ使用量が適切に管理される

---

#### 6.2 保留タスク数制限

**目的**: `--max-pending-tasks`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_6.2_pending_tasks.jsonl \
  --enable-scheduling \
  --max-pending-tasks 500
```

**期待結果**:
- ✅ 保留タスク数が最大500に制限される
- ✅ メモリ使用量が抑制される

---

#### 6.3 データセット分割サイズ

**目的**: `--chunk-size`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_6.3_chunk_size.jsonl \
  --enable-scheduling \
  --chunk-size 50
```

**期待結果**:
- ✅ データセットが50行ごとに分割される
- ✅ 処理が効率的に行われる

---

### 7. メモリ最適化機能テスト

#### 7.1 基本的なメモリ最適化

**目的**: `--enable-memory-optimization`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_7.1_memory_opt.jsonl \
  --enable-memory-optimization
```

**期待結果**:
- ✅ LRUキャッシュによるコンテキスト管理が有効化される
- ✅ メモリ使用量が最適化される

---

#### 7.2 キャッシュサイズ設定

**目的**: `--max-cache-size`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_7.2_cache_size.jsonl \
  --enable-memory-optimization \
  --max-cache-size 100
```

**期待結果**:
- ✅ キャッシュサイズが最大100に制限される
- ✅ 古いエントリが適切に削除される

---

#### 7.3 メモリ監視

**目的**: `--enable-memory-monitoring`の動作確認

**前提条件**: `psutil`パッケージがインストールされている

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_7.3_memory_mon.jsonl \
  --enable-memory-monitoring
```

**期待結果**:
- ✅ メモリ使用状況が監視される
- ✅ メモリ使用量がログに出力される

---

#### 7.4 ガベージコレクション間隔

**目的**: `--gc-interval`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_7.4_gc_interval.jsonl \
  --gc-interval 50
```

**期待結果**:
- ✅ 50行処理ごとにGCが実行される
- ✅ メモリが定期的に解放される

---

#### 7.5 メモリ警告閾値

**目的**: `--memory-threshold-mb`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_7.5_memory_threshold.jsonl \
  --enable-memory-monitoring \
  --memory-threshold-mb 512
```

**期待結果**:
- ✅ メモリ使用量が512MBを超えると警告が出力される
- ✅ 適切なメモリ管理が行われる

---

### 8. LLMリトライ機能テスト

#### 8.1 空返答リトライ無効化

**目的**: `--no-retry-on-empty`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_8.1_no_retry.jsonl \
  --no-retry-on-empty
```

**期待結果**:
- ✅ 空の返答が返された場合にリトライしない
- ✅ 処理が継続される

---

### 9. 出力クリーニング機能テスト

#### 9.1 出力クリーニング無効化

**目的**: `--disable-output-cleaning`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_9.1_no_cleaning.jsonl \
  --disable-output-cleaning
```

**期待結果**:
- ✅ 出力JSONLのクリーニングが無効化される
- ✅ 生のLLM出力が保持される

---

### 10. ネットワーク最適化機能テスト

#### 10.1 共有HTTPトランスポート

**目的**: `--use-shared-transport`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_10.1_shared_transport.jsonl \
  --use-shared-transport
```

**期待結果**:
- ✅ 共有HTTPトランスポートが使用される
- ✅ コネクションプールが共有される
- ✅ ネットワーク効率が向上する

---

#### 10.2 HTTP/2無効化

**目的**: `--no-http2`の動作確認

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_10.2_no_http2.jsonl \
  --no-http2
```

**期待結果**:
- ✅ HTTP/2が無効化される
- ✅ HTTP/1.1で通信が行われる

---

### 11. 統合テスト（複数オプション組み合わせ）

#### 11.1 高性能設定

**目的**: 複数の最適化オプションを組み合わせたテスト

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_11.1_high_performance.jsonl \
  --adaptive \
  --min-batch 4 \
  --max-batch 32 \
  --target-latency-ms 2000 \
  --enable-request-batching \
  --max-batch-size 16 \
  --use-shared-transport \
  --enable-memory-optimization \
  --max-cache-size 300
```

**期待結果**:
- ✅ すべてのオプションが正しく動作する
- ✅ 高いスループットが達成される
- ✅ メモリ使用量が最適化される

---

#### 11.2 大規模データセット設定

**目的**: 大規模データセット処理に最適化された設定

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_11.2_large_dataset.jsonl \
  --enable-scheduling \
  --max-pending-tasks 500 \
  --chunk-size 50 \
  --enable-memory-optimization \
  --max-cache-size 200 \
  --enable-memory-monitoring \
  --gc-interval 50 \
  --memory-threshold-mb 1024
```

**期待結果**:
- ✅ 大規模データセットが効率的に処理される
- ✅ メモリ使用量が適切に管理される
- ✅ 処理が安定して完了する

---

#### 11.3 フル機能設定

**目的**: 利用可能なすべての主要オプションを有効化

```bash
sdg run \
  --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_11.3_full_features.jsonl \
  --save-intermediate \
  --adaptive \
  --min-batch 2 \
  --max-batch 32 \
  --target-latency-ms 2500 \
  --target-queue-depth 24 \
  --enable-request-batching \
  --max-batch-size 16 \
  --max-wait-ms 50 \
  --enable-scheduling \
  --max-pending-tasks 500 \
  --chunk-size 50 \
  --enable-memory-optimization \
  --max-cache-size 200 \
  --enable-memory-monitoring \
  --gc-interval 50 \
  --memory-threshold-mb 1024 \
  --use-shared-transport
```

**期待結果**:
- ✅ すべての機能が正常に動作する
- ✅ オプション間の競合がない
- ✅ 処理が正常に完了する

---

### 12. レガシーモードテスト

#### 12.1 レガシー構文での実行

**目的**: `sdg --yaml ...`形式（レガシーモード）の動作確認

```bash
sdg --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_12.1_legacy.jsonl \
  --max-concurrent 8
```

**期待結果**:
- ✅ レガシーモードで正常に動作する
- ✅ 新しい`sdg run`形式と同じ結果が得られる

---

## テスト実行方法

### 個別テスト実行

各テストケースのコマンドを個別に実行し、期待結果を確認します。

### 一括テスト実行スクリプト

以下のスクリプトで全テストを実行できます：

```bash
#!/bin/bash

# 出力ディレクトリ作成
mkdir -p output/tests

echo "=== SDG RUN 機能テスト開始 ==="

# テスト1.1: 基本実行
echo "Test 1.1: 基本実行"
sdg run --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_1.1_basic.jsonl

# テスト1.3: 中間出力保存
echo "Test 1.3: 中間出力保存"
sdg run --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_1.3_intermediate.jsonl \
  --save-intermediate

# テスト2.1: 並行数制御
echo "Test 2.1: 並行数16"
sdg run --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_2.1_concurrent_16.jsonl \
  --max-concurrent 16

# テスト3.1: 適応的並行性制御
echo "Test 3.1: 適応的制御"
sdg run --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_3.1_adaptive.jsonl \
  --adaptive

# テスト11.3: フル機能
echo "Test 11.3: フル機能設定"
sdg run --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/test_11.3_full_features.jsonl \
  --save-intermediate \
  --adaptive \
  --min-batch 2 \
  --max-batch 32 \
  --target-latency-ms 2500 \
  --enable-request-batching \
  --enable-memory-optimization \
  --use-shared-transport

echo "=== テスト完了 ==="
```

## テスト結果検証

### 出力ファイル検証

各テストの出力ファイルを以下の観点で検証：

```bash
# ファイルの存在確認
ls -lh output/tests/

# JSONL形式の検証
cat output/tests/test_1.1_basic.jsonl | jq .

# 行数確認（入力と同じか確認）
wc -l examples/data/3d_scaling_input.jsonl
wc -l output/tests/test_1.1_basic.jsonl
```

### パフォーマンス比較

```bash
# 実行時間の計測と比較
time sdg run --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/perf_test_baseline.jsonl \
  --max-concurrent 8

time sdg run --yaml examples/3d_scaling_agent.yaml \
  --input examples/data/3d_scaling_input.jsonl \
  --output output/tests/perf_test_optimized.jsonl \
  --adaptive --enable-request-batching
```

## トラブルシューティング

### よくある問題

1. **LLMサーバーに接続できない**
   - `base_url`が正しいか確認
   - サーバーが起動しているか確認
   - ポートが開放されているか確認

2. **メモリ不足エラー**
   - `--enable-memory-optimization`を使用
   - `--max-concurrent`を減らす
   - `--chunk-size`を小さくする

3. **処理が遅い**
   - `--adaptive`を有効化
   - `--enable-request-batching`を使用
   - `--use-shared-transport`を使用

## まとめ

このテスト仕様書に従って全テストを実行することで、SDG RUNパーサーの全機能が正常に動作することを確認できます。各テストケースは独立して実行可能で、特定の機能のみをテストすることも可能です。

---

**作成日**: 2025-12-07
**バージョン**: 1.0
**対象**: SDG Nexus v2.0+