from __future__ import annotations
import argparse
import sys
from .runner import (
    run,
    run_streaming,
    run_streaming_adaptive,
    run_streaming_adaptive_batched,
)


# 日本語ヘルプメッセージ
HELP_JA = """使い方: sdg [--help.ja] {run} ...

SDG (Scalable Data Generator) CLI

オプション:
  -h, --help      このヘルプメッセージを表示して終了
  --help.ja       このヘルプメッセージを日本語で表示して終了

サブコマンド:
  {run}
    run           YAMLブループリントを入力データセットに対して実行

'sdg run --help.ja' でサブコマンドの詳細なヘルプを表示できます。

レガシーモード（後方互換性のため）:
  sdg --yaml <file> --input <file> --output <file> [オプション]
"""

RUN_HELP_JA = """使い方: sdg run --yaml YAML --input INPUT --output OUTPUT [オプション]

YAMLブループリントを入力データセットに対して実行

必須引数:
  --yaml YAML              YAMLブループリントパス
  --input INPUT            入力データセット (.jsonl または .csv)
  --output OUTPUT          出力JSONLファイル

オプション引数:
  -h, --help               このヘルプメッセージを表示して終了
  --help.ja                このヘルプメッセージを日本語で表示して終了
  --save-intermediate      中間出力を保存

Hugging Face データセットオプション:
  --dataset DATASET       Hugging Face データセット名
  --subset SUBSET         データセットのサブセット名
  --split SPLIT           データセットの分割 (デフォルト: train)
  --mapping MAPPING       'orig:new' 形式のキーマッピング (複数回使用可)

ストリーミングモードオプション（デフォルトモード）:
  --max-concurrent MAX_CONCURRENT
                          並行処理する最大行数 (デフォルト: 8)
  --no-progress           プログレス表示を無効化

適応的並行性制御オプション:
  --adaptive               適応的並行性制御を有効化（レイテンシに応じて動的に調整）
                          ※ --adaptive-concurrency でも可
  --min-batch MIN_BATCH   最小並行処理数（適応的制御時、デフォルト: 1）
  --max-batch MAX_BATCH   最大並行処理数（適応的制御時、デフォルト: 64）
  --target-latency-ms TARGET_LATENCY_MS
                          目標P95レイテンシ（ミリ秒、デフォルト: 3000）
  --target-queue-depth TARGET_QUEUE_DEPTH
                          目標バックエンドキュー深度（デフォルト: 32）

バックエンドメトリクスオプション（適応的制御時）:
  --use-vllm-metrics      vLLMのPrometheusメトリクスを使用して並行性を最適化
  --use-sglang-metrics    SGLangのPrometheusメトリクスを使用して並行性を最適化

リクエストバッチングオプション（適応的制御時）:
  --enable-request-batching
                          リクエストバッチングを有効化（複数リクエストを集約して送信）
  --max-batch-size MAX_BATCH_SIZE
                          バッチあたりの最大リクエスト数（デフォルト: 32）
  --max-wait-ms MAX_WAIT_MS
                          バッチ形成の最大待機時間（ミリ秒、デフォルト: 50）

Phase 2 最適化オプション:
  --enable-scheduling     階層的タスクスケジューリングを有効化（大規模データセット用）
  --max-pending-tasks MAX_PENDING_TASKS
                          最大保留タスク数（スケジューリング有効時、デフォルト: 1000）
  --chunk-size CHUNK_SIZE データセット分割サイズ（スケジューリング有効時、デフォルト: 100）
  --enable-memory-optimization
                          メモリ最適化を有効化（LRUキャッシュによるコンテキスト管理）
  --max-cache-size MAX_CACHE_SIZE
                          コンテキストキャッシュの最大サイズ（デフォルト: 500）
  --enable-memory-monitoring
                          メモリ使用状況監視を有効化（psutilが必要）
  --gc-interval GC_INTERVAL
                          ガベージコレクション実行間隔（処理行数、デフォルト: 100）
  --memory-threshold-mb MEMORY_THRESHOLD_MB
                          メモリ使用量警告閾値（MB、デフォルト: 1024）

LLMリトライオプション:
  --no-retry-on-empty     空返答時のリトライを無効化（デフォルトは有効）

JSONL出力クリーニングオプション:
  --disable-output-cleaning
                          出力JSONLのクリーニングを無効化（デフォルトは有効）

最適化オプション:
  --use-shared-transport  共有HTTPトランスポートを使用（コネクションプール共有）
  --no-http2              HTTP/2を無効化（デフォルトは有効）

例:
  # ストリーミングモード（デフォルト・固定並行数）
  sdg run --yaml config.yaml --input data.jsonl --output result.jsonl --max-concurrent 16

  # 適応的並行性制御を有効化（並行数が動的に調整される）
  sdg run --yaml config.yaml --input data.jsonl --output result.jsonl \\
    --adaptive --min-batch 1 --max-batch 32 --target-latency-ms 2000

  # vLLMメトリクスを使用した適応的並行性制御
  sdg run --yaml config.yaml --input data.jsonl --output result.jsonl \\
    --adaptive --use-vllm-metrics --min-batch 1 --max-batch 64

  # リクエストバッチングを有効化（高スループット向け）
  sdg run --yaml config.yaml --input data.jsonl --output result.jsonl \\
    --adaptive --use-vllm-metrics --enable-request-batching

  # 中間出力を保存
  sdg run --yaml config.yaml --input data.jsonl --output result.jsonl --save-intermediate
"""

# Legacy mode help message in Japanese
LEGACY_HELP_JA = """使い方: sdg --yaml YAML --input INPUT --output OUTPUT [オプション]

SDG (Scalable Data Generator) CLI [レガシーモード: sdg --yaml ...]

オプション:
  -h, --help            このヘルプメッセージを表示して終了
  --help.ja             このヘルプメッセージを日本語で表示して終了
  --yaml YAML           YAMLブループリントパス
  --input INPUT         入力データセット (.jsonl または .csv)
  --output OUTPUT       出力JSONLファイル
  --save-intermediate   中間出力を保存
  --dataset DATASET     Hugging Face データセット名
  --subset SUBSET       データセットのサブセット名
  --split SPLIT         データセットの分割 (デフォルト: train)
  --mapping MAPPING     'orig:new' 形式のキーマッピング (複数回使用可)
  --max-concurrent MAX_CONCURRENT
                        並行処理する最大行数 (デフォルト: 8)
  --no-progress         プログレス表示を無効化
  --adaptive            適応的並行性制御を有効化
  --min-batch MIN_BATCH
                        最小並行処理数（適応的制御時、デフォルト: 1）
  --max-batch MAX_BATCH
                        最大並行処理数（適応的制御時、デフォルト: 64）
  --target-latency-ms TARGET_LATENCY_MS
                        目標P95レイテンシ（ミリ秒、デフォルト: 3000）
  --target-queue-depth TARGET_QUEUE_DEPTH
                        目標バックエンドキュー深度（デフォルト: 32）
  --use-vllm-metrics    vLLMのメトリクスを使用
  --use-sglang-metrics  SGLangのメトリクスを使用
  --enable-request-batching
                        リクエストバッチングを有効化
  --max-batch-size MAX_BATCH_SIZE
                        バッチあたりの最大リクエスト数（デフォルト: 32）
  --max-wait-ms MAX_WAIT_MS
                        バッチ形成の最大待機時間（ミリ秒、デフォルト: 50）
  --enable-scheduling   階層的タスクスケジューリングを有効化
  --max-pending-tasks MAX_PENDING_TASKS
                        最大保留タスク数（デフォルト: 1000）
  --chunk-size CHUNK_SIZE
                        データセット分割サイズ（デフォルト: 100）
  --enable-memory-optimization
                        メモリ最適化を有効化
  --max-cache-size MAX_CACHE_SIZE
                        コンテキストキャッシュの最大サイズ（デフォルト: 500）
  --enable-memory-monitoring
                        メモリ使用状況監視を有効化
  --gc-interval GC_INTERVAL
                        ガベージコレクション実行間隔（デフォルト: 100）
  --memory-threshold-mb MEMORY_THRESHOLD_MB
                        メモリ使用量警告閾値（MB、デフォルト: 1024）
  --no-retry-on-empty   空返答時のリトライを無効化（デフォルトは有効）
  --disable-output-cleaning
                        出力JSONLのクリーニングを無効化（デフォルトは有効）
  --use-shared-transport
                        共有HTTPトランスポートを使用（コネクションプール共有）
  --no-http2            HTTP/2を無効化（デフォルトは有効）
"""


def build_run_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--yaml", required=True, help="YAML blueprint path")
    p.add_argument("--input", help="Input dataset (.jsonl or .csv)")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument(
        "--save-intermediate", action="store_true", help="Save intermediate outputs"
    )

    # Hugging Face Dataset options
    p.add_argument("--dataset", help="Hugging Face dataset name")
    p.add_argument("--subset", help="Dataset subset name")
    p.add_argument(
        "--split", default="train", help="Dataset split (default: train)"
    )
    p.add_argument(
        "--mapping",
        action="append",
        help="Key mapping in format 'orig:new' (can be used multiple times)",
    )
    p.add_argument(
        "--help.ja", action="store_true", help="Show this help message in Japanese"
    )

    # Streaming mode options (default mode)
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Max concurrent rows to process (default: 8)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display",
    )

    # Adaptive concurrency options
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive concurrency control (adjusts dynamically based on latency)",
    )
    p.add_argument(
        "--adaptive-concurrency",
        action="store_true",
        dest="adaptive",
        help=argparse.SUPPRESS,  # Hidden alias for --adaptive
    )
    p.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Min concurrency (adaptive mode, default: 1)",
    )
    p.add_argument(
        "--max-batch",
        type=int,
        default=64,
        help="Max concurrency (adaptive mode, default: 64)",
    )
    p.add_argument(
        "--target-latency-ms",
        type=int,
        default=3000,
        help="Target P95 latency in ms (default: 3000)",
    )
    p.add_argument(
        "--target-queue-depth",
        type=int,
        default=32,
        help="Target backend queue depth (default: 32)",
    )

    # Backend metrics options (for adaptive mode)
    p.add_argument(
        "--use-vllm-metrics",
        action="store_true",
        help="Use vLLM Prometheus metrics for adaptive optimization",
    )
    p.add_argument(
        "--use-sglang-metrics",
        action="store_true",
        help="Use SGLang Prometheus metrics for adaptive optimization",
    )

    # Request batching options (for adaptive mode)
    p.add_argument(
        "--enable-request-batching",
        action="store_true",
        help="Enable request batching (groups multiple requests before sending)",
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Max requests per batch (default: 32)",
    )
    p.add_argument(
        "--max-wait-ms",
        type=int,
        default=50,
        help="Max wait time for batch formation in ms (default: 50)",
    )

    # Phase 2: Hierarchical scheduling options
    p.add_argument(
        "--enable-scheduling",
        action="store_true",
        help="Enable hierarchical task scheduling (for large datasets)",
    )
    p.add_argument(
        "--max-pending-tasks",
        type=int,
        default=1000,
        help="Max pending tasks (scheduling mode, default: 1000)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Dataset chunk size (scheduling mode, default: 100)",
    )

    # Phase 2: Memory optimization options
    p.add_argument(
        "--enable-memory-optimization",
        action="store_true",
        help="Enable memory optimization (LRU cache for context management)",
    )
    p.add_argument(
        "--max-cache-size",
        type=int,
        default=500,
        help="Max context cache size (default: 500)",
    )
    p.add_argument(
        "--enable-memory-monitoring",
        action="store_true",
        help="Enable memory usage monitoring (requires psutil)",
    )
    p.add_argument(
        "--gc-interval",
        type=int,
        default=100,
        help="Garbage collection interval in rows (default: 100)",
    )
    p.add_argument(
        "--memory-threshold-mb",
        type=int,
        default=1024,
        help="Memory usage warning threshold in MB (default: 1024)",
    )

    # LLM retry options
    p.add_argument(
        "--no-retry-on-empty",
        action="store_true",
        help="Disable retry on empty response (enabled by default)",
    )

    # JSONL cleaning options
    p.add_argument(
        "--disable-output-cleaning",
        action="store_true",
        help="Disable output JSONL cleaning (enabled by default)",
    )

    # Optimization options
    p.add_argument(
        "--use-shared-transport",
        action="store_true",
        help="Use shared HTTP transport (connection pooling)",
    )
    p.add_argument(
        "--no-http2",
        action="store_true",
        help="Disable HTTP/2 (enabled by default)",
    )

    # Legacy options (hidden, for backward compatibility)
    p.add_argument(
        "--batch-mode",
        action="store_true",
        help=argparse.SUPPRESS,  # Hidden, use --adaptive instead
    )
    p.add_argument(
        "--streaming", action="store_true", help=argparse.SUPPRESS
    )  # Now default, kept for compatibility
    p.add_argument(
        "--max-concurrent-rows", type=int, default=None, help=argparse.SUPPRESS
    )  # Alias for --max-concurrent
    p.add_argument(
        "--min-concurrent", type=int, default=None, help=argparse.SUPPRESS
    )  # Alias for --min-batch
    return p


def _execute_run(args):
    """Execute the run command based on args"""
    # Validation
    if not args.input and not args.dataset:
        print("Error: Either --input or --dataset must be provided.", file=sys.stderr)
        sys.exit(1)
    if args.input and args.dataset:
        print("Error: Cannot specify both --input and --dataset.", file=sys.stderr)
        sys.exit(1)

    # Parse mapping
    mapping = {}
    if args.mapping:
        for m in args.mapping:
            if ":" not in m:
                print(
                    f"Error: Invalid mapping format '{m}'. Expected 'orig:new'.",
                    file=sys.stderr,
                )
                sys.exit(1)
            k, v = m.split(":", 1)
            mapping[k] = v

    # Resolve legacy option aliases
    max_concurrent = (
        args.max_concurrent_rows
        if args.max_concurrent_rows is not None
        else args.max_concurrent
    )
    min_concurrent = (
        args.min_concurrent if args.min_concurrent is not None else args.min_batch
    )

    # Legacy batch mode (hidden, for backward compatibility)
    if args.batch_mode:
        run(
            args.yaml,
            args.input,
            args.output,
            max_batch=args.max_batch,
            min_batch=args.min_batch,
            target_latency_ms=args.target_latency_ms,
            save_intermediate=args.save_intermediate,
            dataset_name=args.dataset,
            subset=args.subset,
            split=args.split,
            mapping=mapping,
        )
    elif args.adaptive:
        # Adaptive streaming mode: dynamic concurrency control
        # Determine metrics type
        metrics_type = "none"
        if args.use_vllm_metrics:
            metrics_type = "vllm"
        elif args.use_sglang_metrics:
            metrics_type = "sglang"

        if args.enable_request_batching:
            # Use batched adaptive mode
            run_streaming_adaptive_batched(
                args.yaml,
                args.input,
                args.output,
                max_concurrent=args.max_batch,
                min_concurrent=min_concurrent,
                target_latency_ms=args.target_latency_ms,
                target_queue_depth=args.target_queue_depth,
                metrics_type=metrics_type,
                max_batch_size=args.max_batch_size,
                max_wait_ms=args.max_wait_ms,
                save_intermediate=args.save_intermediate,
                show_progress=not args.no_progress,
                use_shared_transport=args.use_shared_transport,
                http2=not args.no_http2,
                # LLM retry options
                retry_on_empty=not args.no_retry_on_empty,
                # JSONL cleaning options
                clean_output=not args.disable_output_cleaning,
                # Phase 2: Scheduling options
                enable_scheduling=args.enable_scheduling,
                max_pending_tasks=args.max_pending_tasks,
                chunk_size=args.chunk_size,
                # Phase 2: Memory optimization options
                enable_memory_optimization=args.enable_memory_optimization,
                max_cache_size=args.max_cache_size,
                enable_memory_monitoring=args.enable_memory_monitoring,
                # HF Dataset options
                dataset_name=args.dataset,
                subset=args.subset,
                split=args.split,
                mapping=mapping,
            )
        else:
            run_streaming_adaptive(
                args.yaml,
                args.input,
                args.output,
                max_concurrent=args.max_batch,
                min_concurrent=min_concurrent,
                target_latency_ms=args.target_latency_ms,
                target_queue_depth=args.target_queue_depth,
                metrics_type=metrics_type,
                save_intermediate=args.save_intermediate,
                show_progress=not args.no_progress,
                use_shared_transport=args.use_shared_transport,
                http2=not args.no_http2,
                # LLM retry options
                retry_on_empty=not args.no_retry_on_empty,
                # JSONL cleaning options
                clean_output=not args.disable_output_cleaning,
                # Phase 2: Scheduling options
                enable_scheduling=args.enable_scheduling,
                max_pending_tasks=args.max_pending_tasks,
                chunk_size=args.chunk_size,
                # Phase 2: Memory optimization options
                enable_memory_optimization=args.enable_memory_optimization,
                max_cache_size=args.max_cache_size,
                enable_memory_monitoring=args.enable_memory_monitoring,
                # HF Dataset options
                dataset_name=args.dataset,
                subset=args.subset,
                split=args.split,
                mapping=mapping,
            )
    else:
        # Streaming mode: row-by-row processing with fixed concurrency (default)
        run_streaming(
            args.yaml,
            args.input,
            args.output,
            max_concurrent=max_concurrent,
            save_intermediate=args.save_intermediate,
            show_progress=not args.no_progress,
            use_shared_transport=args.use_shared_transport,
            http2=not args.no_http2,
            # LLM retry options
            retry_on_empty=not args.no_retry_on_empty,
            # JSONL cleaning options
            clean_output=not args.disable_output_cleaning,
            # Phase 2: Scheduling options
            enable_scheduling=args.enable_scheduling,
            max_pending_tasks=args.max_pending_tasks,
            chunk_size=args.chunk_size,
            # Phase 2: Memory optimization options
            enable_memory_optimization=args.enable_memory_optimization,
            max_cache_size=args.max_cache_size,
            enable_memory_monitoring=args.enable_memory_monitoring,
            gc_interval=args.gc_interval,
            memory_threshold_mb=args.memory_threshold_mb,
            # HF Dataset options
            dataset_name=args.dataset,
            subset=args.subset,
            split=args.split,
            mapping=mapping,
        )


def main():
    argv = sys.argv[1:]

    # Check for --help.ja option (before argparse processes it)
    if "--help.ja" in argv:
        # Backward compatibility: detect legacy mode
        legacy_mode = (
            len(argv) > 0 and not argv[0] in {"run"} and argv[0].startswith("--")
        )

        # Determine if this is for 'run' subcommand, legacy mode, or main help
        if len(argv) >= 2 and argv[0] == "run":
            # sdg run --help.ja
            print(RUN_HELP_JA)
            sys.exit(0)
        elif legacy_mode:
            # sdg --yaml ... --help.ja (legacy mode)
            print(LEGACY_HELP_JA)
            sys.exit(0)
        else:
            # sdg --help.ja (main help)
            print(HELP_JA)
            sys.exit(0)

    # Backward compatibility: support legacy usage `sdg --yaml ...`
    legacy_mode = len(argv) > 0 and not argv[0] in {"run"} and argv[0].startswith("--")

    if legacy_mode:
        p = argparse.ArgumentParser(
            description="SDG (Scalable Data Generator) CLI [legacy mode: sdg --yaml ...]"
        )
        build_run_parser(p)
        args = p.parse_args(argv)
        _execute_run(args)
        return

    # Subcommand style: `sdg run --yaml ...`
    p = argparse.ArgumentParser(description="SDG (Scalable Data Generator) CLI")
    sub = p.add_subparsers(dest="command")
    # Python 3.10 supports required for subparsers
    try:
        sub.required = True  # type: ignore[attr-defined]
    except Exception:
        pass

    run_p = sub.add_parser("run", help="Run a YAML blueprint over an input dataset")
    build_run_parser(run_p)

    args = p.parse_args(argv)

    if args.command == "run":
        _execute_run(args)
