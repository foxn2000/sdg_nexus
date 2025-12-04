# SDG Nexus Phase 2 最適化ガイド

## 概要

Phase 2最適化では、SDG Nexusのスケーラビリティとメモリ効率を大幅に向上させる機能を導入しました。これらの機能はすべて**オプトイン方式**で、デフォルトでは無効になっており、既存の動作に影響を与えません。

## 主要機能

### 1. 階層的タスクスケジューリング

大規模データセットを効率的に処理するため、データをチャンク単位に分割し、段階的にタスクを生成・実行します。

#### 特徴

- **チャンク単位のデータ分割**: データセットを適切なサイズのチャンクに分割
- **最大保留タスク数の制限**: メモリ使用量を制御
- **処理開始遅延の最小化**: 全タスクを一度に生成しないため、処理開始が高速
- **条件変数による効率的な制御**: Pythonの`Condition`を用いた協調制御

#### 使用例

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

#### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `enable_scheduling` | `False` | 階層的スケジューリングを有効化 |
| `max_pending_tasks` | `1000` | 最大保留タスク数（メモリ制御用） |
| `chunk_size` | `100` | データセット分割サイズ |

### 2. メモリ効率化

#### 2.1 ストリーミングコンテキストマネージャ

LRUキャッシュを使用してコンテキストを管理し、処理が完了したコンテキストを自動的に解放します。

#### 特徴

- **LRUキャッシュ**: 最近使用されていないコンテキストを自動削除
- **自動メモリ解放**: 処理完了時にコンテキストをクリア
- **参照カウント**: 安全なメモリ解放
- **オプションのメモリ監視**: psutilを使用したメモリ使用状況監視

#### 使用例

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

#### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `enable_memory_optimization` | `False` | メモリ最適化を有効化 |
| `max_cache_size` | `500` | コンテキストキャッシュの最大サイズ |
| `enable_memory_monitoring` | `False` | メモリ使用状況監視を有効化 |

#### 2.2 バッチ処理用段階的メモリ解放

従来の`run_pipeline`（バッチモード）でも段階的なメモリ解放を実現します。

#### 使用例

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

#### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `enable_memory_optimization` | `False` | メモリ最適化を有効化 |
| `enable_memory_monitoring` | `False` | メモリ使用状況監視を有効化 |
| `gc_interval` | `100` | ガベージコレクション実行間隔（処理行数） |
| `memory_threshold_mb` | `1024` | メモリ使用量警告閾値（MB） |

### 3. 適応的パイプラインとの統合

階層的スケジューリングとメモリ最適化は、適応的並行性制御と組み合わせて使用できます。

#### 使用例

```python
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

## パフォーマンスガイドライン

### 推奨設定

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

## API リファレンス

### SchedulerConfig

階層的タスクスケジューラの設定クラス。

```python
from sdg.executors import SchedulerConfig

config = SchedulerConfig(
    max_pending_tasks=1000,  # 最大保留タスク数
    chunk_size=100,          # チャンクサイズ
    enable_scheduling=False, # スケジューリング有効化
)
```

### MemoryConfig

メモリ効率化の設定クラス。

```python
from sdg.executors import MemoryConfig

config = MemoryConfig(
    max_cache_size=500,              # キャッシュサイズ
    enable_memory_optimization=False, # メモリ最適化有効化
    enable_monitoring=False,          # メモリ監視有効化
    gc_interval=100,                  # GC実行間隔
    memory_threshold_mb=1024,         # メモリ警告閾値
)
```

### HierarchicalTaskScheduler

階層的タスクスケジューラクラス。

```python
from sdg.executors import HierarchicalTaskScheduler, SchedulerConfig

config = SchedulerConfig(enable_scheduling=True, max_pending_tasks=100)
scheduler = HierarchicalTaskScheduler(config=config)

# データセットをスケジューリング
async for item in scheduler.schedule(dataset):
    # タスクを処理
    await process_task(item)
    # 完了を通知
    await scheduler.mark_task_completed()

# 統計情報を取得
stats = scheduler.get_stats()
print(f"Progress: {stats['progress_percent']}%")
```

### StreamingContextManager

ストリーミング対応コンテキストマネージャ。

```python
from sdg.executors import StreamingContextManager, MemoryConfig

config = MemoryConfig(enable_memory_optimization=True, max_cache_size=500)
ctx_manager = StreamingContextManager(config=config)

# コンテキストを取得または作成
ctx = await ctx_manager.get_or_create(row_index, initial_data)

# 処理...

# 完了を通知してメモリ解放
await ctx_manager.mark_completed(row_index)

# 統計情報を取得
stats = ctx_manager.get_stats()
print(f"Cache hit rate: {stats['cache_stats']['hit_rate']}")
```

### LRUCache

LRU（Least Recently Used）キャッシュクラス。

```python
from sdg.executors import LRUCache

cache = LRUCache[str](max_size=100)

# アイテムを追加
evicted = cache.put("key1", "value1")

# アイテムを取得
value = cache.get("key1")

# 統計情報を取得
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
```

## トラブルシューティング

### メモリ使用量が高い

1. `enable_memory_optimization=True` を設定
2. `max_cache_size` を減らす
3. `gc_interval` を小さくする（より頻繁にGCを実行）
4. `enable_memory_monitoring=True` でメモリ使用状況を監視

### 処理が遅い

1. `max_concurrent` を増やす
2. `max_pending_tasks` を増やす
3. `chunk_size` を大きくする
4. `enable_scheduling=False` を試す（小規模データセットの場合）

### メモリ不足エラー

1. `max_pending_tasks` を減らす
2. `chunk_size` を小さくする
3. `max_cache_size` を減らす
4. `enable_memory_optimization=True` を設定

## ベストプラクティス

1. **段階的な有効化**: まず小規模データセットでテストし、徐々に機能を有効化
2. **メモリ監視**: `enable_memory_monitoring=True` で監視し、適切な設定を見つける
3. **統計情報の活用**: `get_stats()` メソッドで最適化の効果を確認
4. **psutilのインストール**: メモリ監視機能を使用する場合は `pip install psutil`
5. **適応的制御との組み合わせ**: 大規模データセットでは適応的制御と組み合わせて使用

## 後方互換性

Phase 2の全機能はデフォルトで無効になっており、既存のコードは変更なしで動作します。

```python
# 既存コード（Phase 2機能なし）
async for result in run_pipeline_streaming(cfg, dataset):
    # 処理...
    pass

# Phase 2機能を有効化
async for result in run_pipeline_streaming(
    cfg, 
    dataset,
    enable_scheduling=True,
    enable_memory_optimization=True,
):
    # 処理...
    pass
```

## まとめ

Phase 2最適化により、SDG Nexusは以下を実現しました：

- ✅ 大規模データセットの効率的な処理
- ✅ メモリ使用量の大幅な削減
- ✅ 処理開始遅延の最小化
- ✅ 完全な後方互換性
- ✅ オプトイン方式による柔軟な有効化

詳細な実装とテストケースは、`sdg/executors/scheduling.py` および `tests/test_scheduling.py` を参照してください。