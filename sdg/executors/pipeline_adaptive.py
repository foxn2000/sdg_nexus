from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, Iterable, List, Optional, Set

from ..config import SDGConfig, PyBlock
from .core import ExecutionContext, StreamingResult
from .python import _load_python_function
from .ai import _build_clients
from .scheduling import (
    HierarchicalTaskScheduler,
    StreamingContextManager,
    SchedulerConfig,
    MemoryConfig,
)
from .pipeline_core import process_single_row
from .pipeline_streaming import run_pipeline_streaming

try:
    from ..adaptive import (
        AdaptiveController,
        MetricsCollector,
        MetricsType,
    )

    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


async def run_pipeline_streaming_adaptive(
    cfg: SDGConfig,
    dataset: Iterable[Dict[str, Any]],
    *,
    max_concurrent: int = 64,
    min_concurrent: int = 1,
    target_latency_ms: int = 3000,
    target_queue_depth: int = 32,
    metrics_type: str = "none",
    save_intermediate: bool = False,
    # Phase 2: スケジューリングオプション（デフォルト無効）
    max_pending_tasks: int = 1000,
    chunk_size: int = 100,
    enable_scheduling: bool = False,
    # Phase 2: メモリ最適化オプション（デフォルト無効）
    max_cache_size: int = 500,
    enable_memory_optimization: bool = False,
    enable_memory_monitoring: bool = False,
    # 処理再開オプション
    processed_indices: Optional[Set[int]] = None,
):
    """
    適応的並行性制御付きストリーミング版パイプライン

    レイテンシとオプションのバックエンドメトリクス（vLLM/SGLang）に基づいて、
    並行処理数を動的に調整しながら処理を行う。

    Args:
        cfg: SDG設定
        dataset: 入力データセット（イテラブル）
        max_concurrent: 同時処理行数の上限 (デフォルト: 64)
        min_concurrent: 同時処理行数の下限 (デフォルト: 1)
        target_latency_ms: 目標P95レイテンシ (ミリ秒、デフォルト: 3000)
        target_queue_depth: 目標バックエンドキュー深度 (デフォルト: 32)
        metrics_type: メトリクスタイプ ("none", "vllm", "sglang")
        save_intermediate: 中間結果を保存するか
        max_pending_tasks: 最大保留タスク数（スケジューリング有効時）
        chunk_size: データセット分割サイズ（スケジューリング有効時）
        enable_scheduling: 階層的タスクスケジューリングを有効化
        max_cache_size: コンテキストキャッシュの最大サイズ
        enable_memory_optimization: メモリ最適化を有効化
        enable_memory_monitoring: メモリ使用状況監視を有効化
        processed_indices: 処理済み行インデックスのセット（再開時に使用）
            このセットに含まれるインデックスの行はスキップされる

    Yields:
        StreamingResult: 各行の処理結果
    """
    # 処理済みインデックスのセットを初期化
    skip_indices = processed_indices or set()

    if not ADAPTIVE_AVAILABLE:
        # Fall back to standard streaming if adaptive module not available
        async for result in run_pipeline_streaming(
            cfg,
            dataset,
            max_concurrent=max_concurrent,
            save_intermediate=save_intermediate,
            max_pending_tasks=max_pending_tasks,
            chunk_size=chunk_size,
            enable_scheduling=enable_scheduling,
            max_cache_size=max_cache_size,
            enable_memory_optimization=enable_memory_optimization,
            enable_memory_monitoring=enable_memory_monitoring,
            processed_indices=processed_indices,
        ):
            yield result
        return

    # モデルクライアント構築
    clients = _build_clients(cfg)

    # Python関数をプリロード
    python_functions: Dict[str, Any] = {}
    for block in cfg.blocks:
        if isinstance(block, PyBlock):
            fn_key = f"{block.exec}_{block.function or block.entrypoint}"
            python_functions[fn_key] = _load_python_function(block)

    # AdaptiveController設定 - 最大並行数から開始し、エラー時に下げる
    controller = AdaptiveController(
        min_concurrency=min_concurrent,
        max_concurrency=max_concurrent,
        target_latency_ms=float(target_latency_ms),
        target_queue_depth=target_queue_depth,
        initial_concurrency=max_concurrent,  # 最大並行数から開始
    )

    # MetricsCollector設定（メトリクスタイプに応じて）
    metrics_collector: Optional[MetricsCollector] = None
    if metrics_type != "none":
        # モデル設定から最初のbase_urlを取得
        base_url = None
        for model in cfg.models:
            if model.base_url:
                base_url = model.base_url
                break

        if base_url:
            if metrics_type == "vllm":
                metrics_collector = MetricsCollector(
                    base_url=base_url,
                    metrics_type=MetricsType.VLLM,
                )
            elif metrics_type == "sglang":
                metrics_collector = MetricsCollector(
                    base_url=base_url,
                    metrics_type=MetricsType.SGLANG,
                )

    # 結果キュー
    result_queue: asyncio.Queue[StreamingResult] = asyncio.Queue()

    # 処理完了カウンター（totalは事前に不明な場合がある）
    completed = 0
    total_started = 0

    # メトリクス更新タスク
    metrics_update_task: Optional[asyncio.Task] = None

    # Phase 2: 階層的タスクスケジューラの初期化
    scheduler_config = SchedulerConfig(
        max_pending_tasks=max_pending_tasks,
        chunk_size=chunk_size,
        enable_scheduling=enable_scheduling,
    )
    scheduler = HierarchicalTaskScheduler(config=scheduler_config)

    # Phase 2: ストリーミングコンテキストマネージャの初期化
    memory_config = MemoryConfig(
        max_cache_size=max_cache_size,
        enable_memory_optimization=enable_memory_optimization,
        enable_monitoring=enable_memory_monitoring,
    )
    ctx_manager = StreamingContextManager(config=memory_config)

    async def update_metrics_loop():
        """バックエンドメトリクスを定期的にコントローラーに反映"""
        if metrics_collector is None:
            return

        await metrics_collector.start()
        try:
            while True:
                await asyncio.sleep(0.5)
                metrics = metrics_collector.get_latest()
                if metrics and metrics.is_valid:
                    controller.update_with_metrics(metrics)
        except asyncio.CancelledError:
            pass
        finally:
            await metrics_collector.stop()

    async def process_row(row_index: int, row_data: Dict[str, Any]):
        """1行を処理してキューに結果を入れる（適応的並行性制御付き）"""
        # 動的セマフォで制御
        async with controller.semaphore:
            start_time = time.time()

            # 行ごとに独立したExecutionContextを作成
            row_exec_ctx = ExecutionContext(cfg)

            try:
                result_data = await process_single_row(
                    row_index=row_index,
                    initial_context=row_data,
                    cfg=cfg,
                    clients=clients,
                    exec_ctx=row_exec_ctx,
                    save_intermediate=save_intermediate,
                    python_functions=python_functions,
                )

                # レイテンシを記録
                latency_ms = (time.time() - start_time) * 1000
                controller.record_latency(latency_ms, is_error=False)

                await result_queue.put(
                    StreamingResult(
                        row_index=row_index,
                        data=result_data,
                        error=None,
                    )
                )
            except Exception as e:
                # エラーもレイテンシとして記録
                latency_ms = (time.time() - start_time) * 1000
                controller.record_latency(latency_ms, is_error=True)

                await result_queue.put(
                    StreamingResult(
                        row_index=row_index,
                        data={},
                        error=e,
                    )
                )
            finally:
                # Phase 2: スケジューラに完了を通知
                if scheduler.is_enabled:
                    await scheduler.mark_task_completed()
                # Phase 2: コンテキストマネージャに完了を通知してメモリ解放
                if ctx_manager.is_enabled:
                    await ctx_manager.mark_completed(row_index)

    async def progressive_task_launcher(data_iter):
        """
        Progressive Task Launcher - セマフォ容量に応じて段階的にタスクを起動

        全タスクを一度に起動するのではなく、現在のセマフォ容量に応じて
        新しいタスクを追加する。エラー発生時はセマフォ容量が下がるため、
        新規タスクの起動も自動的に抑制される。

        処理済みインデックス（skip_indices）はスキップされ、元のインデックスが維持される。

        Args:
            data_iter: データのイテレータ
        """
        nonlocal total_started
        tasks: List[asyncio.Task] = []
        active_count = 0
        data_exhausted = False
        current_index = 0  # 元のインデックスを追跡

        # イテレータをイテレート可能にする
        data_iterator = iter(data_iter)

        while not data_exhausted or active_count > 0:
            # 現在のセマフォ容量を取得
            current_capacity = controller.current_concurrency
            available_slots = controller.get_available_slots()

            # 新しいタスクを起動できるスロット数を計算
            pending_tasks = active_count
            slots_to_fill = max(
                0,
                min(
                    current_capacity - pending_tasks,  # 現在の並行数制限
                    available_slots,  # 利用可能なスロット
                ),
            )

            # 新しいタスクを起動
            launched = 0
            for _ in range(slots_to_fill):
                try:
                    row_data = next(data_iterator)
                    row_index = current_index
                    current_index += 1

                    # 処理済みインデックスをスキップ
                    if row_index in skip_indices:
                        continue

                    task = asyncio.create_task(process_row(row_index, row_data))
                    tasks.append(task)
                    total_started += 1
                    active_count += 1
                    launched += 1
                except StopIteration:
                    data_exhausted = True
                    break

            # 少し待機してから次のチェック
            if active_count > 0:
                # 結果を待つ（タイムアウト付き）
                try:
                    result = await asyncio.wait_for(
                        result_queue.get(), timeout=0.1  # 100ms待機
                    )
                    active_count -= 1
                    yield result
                except asyncio.TimeoutError:
                    # タイムアウト時は次のループへ
                    pass
            else:
                # アクティブなタスクがない場合は短い待機
                await asyncio.sleep(0.01)

        # 残りの結果を取得
        while not result_queue.empty():
            result = await result_queue.get()
            yield result

        # 全タスクの完了を待機
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    try:
        # メトリクス収集を開始（有効な場合）
        if metrics_collector is not None:
            metrics_update_task = asyncio.create_task(update_metrics_loop())

        if enable_scheduling:
            # Phase 2: 階層的スケジューリングでタスクを段階的に起動
            tasks = []
            async for item in scheduler.schedule(dataset):
                # 処理済みインデックスをスキップ
                if item.index in skip_indices:
                    continue
                task = asyncio.create_task(process_row(item.index, item.data))
                tasks.append(task)
                total_started += 1

            # 完了した結果を順次yield
            while completed < total_started:
                result = await result_queue.get()
                completed += 1
                yield result

            # すべてのタスク完了を待機
            await asyncio.gather(*tasks)
        else:
            # Progressive Task Launcher を使用してタスクを段階的に起動
            async for result in progressive_task_launcher(dataset):
                completed += 1
                yield result

    finally:
        # Phase 2: リソース解放
        if ctx_manager.is_enabled:
            await ctx_manager.release_all()

        # メトリクス収集を停止
        if metrics_update_task is not None:
            metrics_update_task.cancel()
            try:
                await metrics_update_task
            except asyncio.CancelledError:
                pass
