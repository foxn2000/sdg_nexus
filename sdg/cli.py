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
"""


def build_run_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--yaml", required=True, help="YAML blueprint path")
    p.add_argument("--input", required=True, help="Input dataset (.jsonl or .csv)")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument(
        "--save-intermediate", action="store_true", help="Save intermediate outputs"
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
