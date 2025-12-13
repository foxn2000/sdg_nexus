from __future__ import annotations
import asyncio
from typing import Any, Dict, List

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


async def run_pipeline_streaming(
    cfg: SDGConfig,
    dataset: List[Dict[str, Any]],
    *,
    max_concurrent: int = 8,
    save_intermediate: bool = False,
    # Phase 2: スケジューリングオプション（デフォルト無効）
    max_pending_tasks: int = 1000,
    chunk_size: int = 100,
    enable_scheduling: bool = False,
    # Phase 2: メモリ最適化オプション（デフォルト無効）
    max_cache_size: int = 500,
    enable_memory_optimization: bool = False,
    enable_memory_monitoring: bool = False,
):
    """
    ストリーミング版パイプライン - 完了した行から順次yield

    Args:
        cfg: SDG設定
        dataset: 入力データセット
        max_concurrent: 同時処理行数の上限
        save_intermediate: 中間結果を保存するか
        max_pending_tasks: 最大保留タスク数（スケジューリング有効時）
        chunk_size: データセット分割サイズ（スケジューリング有効時）
        enable_scheduling: 階層的タスクスケジューリングを有効化
        max_cache_size: コンテキストキャッシュの最大サイズ
        enable_memory_optimization: メモリ最適化を有効化
        enable_memory_monitoring: メモリ使用状況監視を有効化

    Yields:
        StreamingResult: 各行の処理結果
    """
    # モデルクライアント構築
    clients = _build_clients(cfg)

    # Python関数をプリロード
    python_functions: Dict[str, Any] = {}
    for block in cfg.blocks:
        if isinstance(block, PyBlock):
            fn_key = f"{block.exec}_{block.function or block.entrypoint}"
            python_functions[fn_key] = _load_python_function(block)

    # 同時実行数制御用セマフォ
    semaphore = asyncio.Semaphore(max_concurrent)

    # 結果キュー
    result_queue: asyncio.Queue[StreamingResult] = asyncio.Queue()

    # 処理完了カウンター
    completed = 0
    total = len(dataset)

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

    async def process_row(row_index: int, row_data: Dict[str, Any]):
        """1行を処理してキューに結果を入れる"""
        async with semaphore:
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
                await result_queue.put(
                    StreamingResult(
                        row_index=row_index,
                        data=result_data,
                        error=None,
                    )
                )
            except Exception as e:
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

    try:
        if enable_scheduling:
            # Phase 2: 階層的スケジューリングでタスクを段階的に起動
            tasks = []
            async for item in scheduler.schedule(dataset):
                task = asyncio.create_task(process_row(item.index, item.data))
                tasks.append(task)

            # 完了した結果を順次yield
            while completed < total:
                result = await result_queue.get()
                completed += 1
                yield result

            # すべてのタスク完了を待機
            await asyncio.gather(*tasks)
        else:
            # 従来の動作: 全行のタスクを一度に起動
            tasks = [
                asyncio.create_task(process_row(i, row))
                for i, row in enumerate(dataset)
            ]

            # 完了した結果を順次yield
            while completed < total:
                result = await result_queue.get()
                completed += 1
                yield result

            # すべてのタスク完了を待機（エラー発生時の例外を伝播）
            await asyncio.gather(*tasks)

    finally:
        # Phase 2: リソース解放
        if ctx_manager.is_enabled:
            await ctx_manager.release_all()
