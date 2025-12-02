from __future__ import annotations
import argparse
import sys
from .runner import run, run_streaming


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

RUN_HELP_JA = """使い方: sdg run --yaml YAML --input INPUT --output OUTPUT [--save-intermediate] [--max-concurrent MAX_CONCURRENT] [--no-progress] [--batch-mode] [--max-batch MAX_BATCH] [--min-batch MIN_BATCH] [--target-latency-ms TARGET_LATENCY_MS]

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
                          並行処理する最大行数 (ストリーミングモード、デフォルト: 8)
  --no-progress           プログレス表示を無効化 (ストリーミングモード)

バッチモードオプション（ブロック単位の処理）:
  --batch-mode            バッチモードを使用（ストリーミングの代わりにブロック単位で処理）
  --max-batch MAX_BATCH   ブロックあたりの最大並行リクエスト数 (バッチモードのみ、デフォルト: 8)
  --min-batch MIN_BATCH   ブロックあたりの最小並行リクエスト数 (バッチモードのみ、デフォルト: 1)
  --target-latency-ms TARGET_LATENCY_MS
                          リクエストあたりの目標平均レイテンシ (バッチモードのみ、デフォルト: 3000)

例:
  # ストリーミングモード（デフォルト）
  sdg run --yaml config.yaml --input data.jsonl --output result.jsonl

  # バッチモード
  sdg run --yaml config.yaml --input data.jsonl --output result.jsonl --batch-mode

  # 中間出力を保存
  sdg run --yaml config.yaml --input data.jsonl --output result.jsonl --save-intermediate
"""

# Legacy mode help message in Japanese
LEGACY_HELP_JA = """使い方: sdg --yaml YAML --input INPUT --output OUTPUT [--save-intermediate] [--max-concurrent MAX_CONCURRENT] [--no-progress] [--batch-mode] [--max-batch MAX_BATCH] [--min-batch MIN_BATCH] [--target-latency-ms TARGET_LATENCY_MS]

SDG (Scalable Data Generator) CLI [レガシーモード: sdg --yaml ...]

オプション:
  -h, --help            このヘルプメッセージを表示して終了
  --help.ja             このヘルプメッセージを日本語で表示して終了
  --yaml YAML           YAMLブループリントパス
  --input INPUT         入力データセット (.jsonl または .csv)
  --output OUTPUT       出力JSONLファイル
  --save-intermediate   中間出力を保存
  --max-concurrent MAX_CONCURRENT
                        並行処理する最大行数 (ストリーミングモード、デフォルト: 8)
  --no-progress         プログレス表示を無効化 (ストリーミングモード)
  --batch-mode          バッチモードを使用（ストリーミングの代わりにブロック単位で処理）
  --max-batch MAX_BATCH
                        ブロックあたりの最大並行リクエスト数 (バッチモードのみ、デフォルト: 8)
  --min-batch MIN_BATCH
                        ブロックあたりの最小並行リクエスト数 (バッチモードのみ、デフォルト: 1)
  --target-latency-ms TARGET_LATENCY_MS
                        リクエストあたりの目標平均レイテンシ (バッチモードのみ、デフォルト: 3000)
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
        help="Max concurrent rows to process (streaming mode, default)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display (streaming mode)",
    )

    # Batch mode options (block-by-block processing)
    p.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use batch mode (block-by-block processing instead of streaming)",
    )
    p.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Max concurrent requests per block (batch mode only)",
    )
    p.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Min concurrent requests per block (batch mode only)",
    )
    p.add_argument(
        "--target-latency-ms",
        type=int,
        default=3000,
        help="Target average latency per request (batch mode only)",
    )

    # Legacy option (hidden, for backward compatibility)
    p.add_argument(
        "--streaming", action="store_true", help=argparse.SUPPRESS
    )  # Now default, kept for compatibility
    p.add_argument(
        "--max-concurrent-rows", type=int, default=None, help=argparse.SUPPRESS
    )  # Alias for --max-concurrent
    return p


def _execute_run(args):
    """Execute the run command based on args"""
    # Batch mode is now opt-in; streaming is default
    if args.batch_mode:
        # Batch mode: block-by-block processing
        run(
            args.yaml,
            args.input,
            args.output,
            max_batch=args.max_batch,
            min_batch=args.min_batch,
            target_latency_ms=args.target_latency_ms,
            save_intermediate=args.save_intermediate,
        )
    else:
        # Streaming mode: row-by-row processing (default)
        # Support legacy --max-concurrent-rows option
        max_concurrent = (
            args.max_concurrent_rows
            if args.max_concurrent_rows is not None
            else args.max_concurrent
        )
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
