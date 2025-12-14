from __future__ import annotations
import asyncio
import random
import time
from typing import Any, Dict, Optional, Callable

from ..config import (
    load_config,
    PyBlock,
    AIBlock,
    LogicBlock,
    EndBlock,
    OutputDef,
)
from ..executors.ai import _build_clients, _execute_ai_block_single
from ..executors.core import ExecutionContext, _eval_cond, _execute_end_block_single
from ..executors.logic import _apply_logic_block
from ..executors.python import _load_python_function, _execute_python_block_single
from ..logger import init_logger, SDGLogger
from ..utils import render_template
from ..io import (
    read_jsonl,
    read_csv,
    read_hf_dataset,
    apply_mapping,
)


async def _test_run_pipeline(
    cfg,
    data: Dict[str, Any],
    logger: SDGLogger,
    clients: Dict[str, Any],
    exec_ctx: ExecutionContext,
    python_functions: Dict[str, Callable],
) -> Dict[str, Any]:
    """
    テスト実行用のパイプライン処理。

    各ブロックの実行時に詳細なログを出力し、
    AIの出力を視覚的に強調表示する。

    Args:
        cfg: パイプライン設定
        data: 入力データ（1件）
        logger: SDGLoggerインスタンス
        clients: LLMクライアント辞書
        exec_ctx: 実行コンテキスト
        python_functions: プリロードされたPython関数

    Returns:
        実行結果の辞書
    """
    result: Dict[str, Any] = {}
    ctx = dict(data)
    total_blocks = len(cfg.blocks)

    for block_index, block in enumerate(cfg.blocks):
        # グローバル変数をコンテキストに追加（run_if評価用）
        extended_ctx = {
            **exec_ctx.globals_const,
            **exec_ctx.globals_vars,
            **ctx,
        }

        # run_if評価
        run_ok = True
        if block.run_if:
            run_ok = _eval_cond(extended_ctx, block.run_if, exec_ctx)

        # ブロック名を取得
        block_name = block.name or block.id or f"block_{block.exec}"
        block_type = block.type or "unknown"

        if not run_ok:
            logger.block_skipped(block_name, reason="run_if=False")
            continue

        # ブロック開始ログ
        extra_info = {}
        if isinstance(block, AIBlock):
            extra_info["model"] = block.model

        logger.block_start(
            block_name=block_name,
            block_type=block_type,
            block_index=block_index,
            total_blocks=total_blocks,
            extra_info=extra_info if extra_info else None,
        )

        block_start_time = time.time()

        try:
            if isinstance(block, AIBlock):
                # AIブロック: プロンプトと出力を詳細表示

                # プロンプトを構築して表示
                ai_extended_ctx = {
                    **exec_ctx.globals_const,
                    **exec_ctx.globals_vars,
                    **ctx,
                }
                raw_user_content = "\n\n".join(
                    [render_template(p, ai_extended_ctx) for p in (block.prompts or [])]
                )
                logger.ai_prompt(raw_user_content, model=block.model)

                # AIブロック実行
                out_map = await _execute_ai_block_single(
                    block, ctx, cfg, clients, exec_ctx
                )

                # AI出力を強調表示
                for out_name, out_value in out_map.items():
                    # 最初の出力を主要出力として扱う
                    is_primary = out_name == list(out_map.keys())[0]
                    logger.ai_output(
                        output=str(out_value),
                        output_name=out_name,
                        is_primary=is_primary,
                    )

                result.update({f"_{block.exec}_{k}": v for k, v in out_map.items()})
                ctx.update(out_map)

            elif isinstance(block, LogicBlock):
                # Logicブロック
                out_map = _apply_logic_block(block, ctx, exec_ctx)

                if out_map:
                    logger.step(
                        f"Logic output: {list(out_map.keys())}", step_type="data"
                    )

                result.update({f"_{block.exec}_{k}": v for k, v in out_map.items()})
                ctx.update(out_map)

            elif isinstance(block, PyBlock):
                # Pythonブロック
                fn_key = f"{block.exec}_{block.function or block.entrypoint}"
                if fn_key not in python_functions:
                    python_functions[fn_key] = _load_python_function(block)
                fn = python_functions[fn_key]

                out_map = _execute_python_block_single(block, ctx, cfg, exec_ctx, fn)

                if out_map:
                    logger.step(
                        f"Python output: {list(out_map.keys())}", step_type="data"
                    )

                result.update({f"_{block.exec}_{k}": v for k, v in out_map.items()})
                ctx.update(out_map)

            elif isinstance(block, EndBlock):
                # Endブロック
                out_map = _execute_end_block_single(block, ctx, exec_ctx)
                result.update(out_map)

                logger.step(
                    f"Final output keys: {list(out_map.keys())}", step_type="success"
                )

            else:
                raise ValueError(f"Unknown block class: {type(block)}")

            # ブロック完了ログ
            elapsed_ms = int((time.time() - block_start_time) * 1000)
            logger.block_end(block_name, elapsed_ms=elapsed_ms, success=True)

        except Exception as e:
            elapsed_ms = int((time.time() - block_start_time) * 1000)
            logger.block_end(block_name, elapsed_ms=elapsed_ms, success=False)
            logger.error(f"Block error: {e}")

            if block.on_error != "continue":
                raise
            # continue on error
            ctx[f"error_block_{block.exec}"] = str(e)

    return result


async def _test_run_async(
    cfg,
    data: Dict[str, Any],
    logger: SDGLogger,
):
    """
    テスト実行用の非同期関数。

    1件のデータに対してパイプラインを実行し、詳細なログを出力する。

    Args:
        cfg: パイプライン設定
        data: 入力データ（1件）
        logger: SDGLoggerインスタンス

    Returns:
        実行結果の辞書
    """
    logger.separator("light")
    logger.info("Starting pipeline execution...")

    # 入力データを表示
    logger.input_data(data)

    # パイプライン実行
    start_time = time.time()

    # モデルクライアント構築
    clients = _build_clients(cfg)

    # Python関数をプリロード
    python_functions = {}
    for block in cfg.blocks:
        if isinstance(block, PyBlock):
            fn_key = f"{block.exec}_{block.function or block.entrypoint}"
            python_functions[fn_key] = _load_python_function(block)

    # 実行コンテキストを作成
    exec_ctx = ExecutionContext(cfg)

    logger.separator("heavy")

    try:
        result = await _test_run_pipeline(
            cfg=cfg,
            data=data,
            logger=logger,
            clients=clients,
            exec_ctx=exec_ctx,
            python_functions=python_functions,
        )

        elapsed_time = time.time() - start_time

        logger.separator("heavy")
        logger.success(f"Pipeline execution completed in {elapsed_time:.2f}s")

        return {
            "_row_index": 0,
            "_elapsed_time_ms": int(elapsed_time * 1000),
            **result,
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.separator("heavy")
        logger.error(f"Pipeline execution error: {e}")
        return {
            "_row_index": 0,
            "_error": str(e),
            "_elapsed_time_ms": int(elapsed_time * 1000),
        }


def test_run(
    yaml_path: str,
    input_path: Optional[str] = None,
    # HF Dataset options
    dataset_name: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "train",
    mapping: Optional[Dict[str, str]] = None,
    # UI options
    verbose: bool = True,
    locale: str = "en",
    show_meta: bool = False,
    # Data selection options
    random_input: bool = False,
) -> Dict[str, Any]:
    """
    テスト実行: YAMLブループリントを1件のデータに対して実行し、動作確認を行う。

    開発者がエージェントの挙動を素早く検証できるようにするためのコマンド。
    詳細なログを出力し、各ステップの実行状況を可視化する。

    Args:
        yaml_path: YAMLブループリントのパス
        input_path: 入力データセット (.jsonl or .csv)
        dataset_name: Hugging Faceデータセット名
        subset: データセットサブセット
        split: データセットスプリット
        mapping: キーマッピング辞書
        verbose: 詳細ログを有効化（デフォルト: True）
        locale: UIロケール ('en' or 'ja')
        show_meta: メタ情報を表示するか（デフォルト: False）
        random_input: ランダムにデータを選択するか（デフォルト: False）

    Returns:
        実行結果の辞書

    Example:
        >>> result = test_run("config.yaml", "data.jsonl")
        >>> print(result)
    """

    # ロガーを初期化（test-runは常に詳細ログを有効化）
    logger = init_logger(
        verbose=verbose,
        quiet=False,
        use_rich=True,
        locale=locale,
    )

    # ヘッダーを表示
    if locale == "ja":
        subtitle = "AIエージェントの動作確認モード（1件のみ実行）"
    else:
        subtitle = "AI Agent verification mode (single item execution)"
    logger.header("SDG Test Run", subtitle)

    # 設定を読み込み
    cfg = load_config(yaml_path)

    # 最適化オプションを設定（テスト実行用）
    cfg.optimization = {
        "use_shared_transport": False,
        "http2": True,
        "retry_on_empty": True,
    }

    # 設定情報を表示
    if locale == "ja":
        config_info = {
            "YAMLファイル": yaml_path,
            "MABELバージョン": cfg.get_version(),
            "モデル数": len(cfg.models),
            "ブロック数": len(cfg.blocks),
        }
        logger.table("設定情報", config_info)
    else:
        config_info = {
            "YAML File": yaml_path,
            "MABEL Version": cfg.get_version(),
            "Model Count": len(cfg.models),
            "Block Count": len(cfg.blocks),
        }
        logger.table("Configuration", config_info)

    # モデル情報を表示
    if cfg.models:
        if locale == "ja":
            model_info = {}
            for i, m in enumerate(cfg.models):
                model_info[f"モデル {i+1}"] = f"{m.name} ({m.api_model})"
            logger.table("使用モデル", model_info)
        else:
            model_info = {}
            for i, m in enumerate(cfg.models):
                model_info[f"Model {i+1}"] = f"{m.name} ({m.api_model})"
            logger.table("Models", model_info)

    # ブロック情報を表示
    if cfg.blocks:
        if locale == "ja":
            block_info = {}
            for b in cfg.blocks:
                block_name = b.name or b.id or f"Block {b.exec}"
                block_info[block_name] = f"type={b.type}, exec={b.exec}"
            logger.table("ブロック構成", block_info)
        else:
            block_info = {}
            for b in cfg.blocks:
                block_name = b.name or b.id or f"Block {b.exec}"
                block_info[block_name] = f"type={b.type}, exec={b.exec}"
            logger.table("Block Structure", block_info)

    # データを読み込み
    if random_input:
        # ランダム選択の場合: 最大100件読み込んでその中から1件をランダムに選択
        max_read = 100
        if input_path:
            if input_path.endswith(".jsonl"):
                ds = list(read_jsonl(input_path, max_inputs=max_read))
            elif input_path.endswith(".csv"):
                ds = list(read_csv(input_path, max_inputs=max_read))
            else:
                raise ValueError("Unsupported input format. Use .jsonl or .csv")
        elif dataset_name:
            ds = list(read_hf_dataset(dataset_name, subset, split, max_inputs=max_read))
        else:
            raise ValueError("Either input_path or dataset_name must be provided")

        if not ds:
            raise ValueError("No data found in input")

        # マッピングを適用
        if mapping:
            ds = list(apply_mapping(ds, mapping))

        # ランダムに1件選択
        selected_index = random.randint(0, len(ds) - 1)
        data = ds[selected_index]

        # ランダム選択された情報をログに表示
        if locale == "ja":
            logger.info(f"ランダム選択: {len(ds)}件中の{selected_index + 1}件目を使用")
        else:
            logger.info(
                f"Random selection: using item {selected_index + 1} of {len(ds)}"
            )
    else:
        # 通常の場合: 1件目を使用
        if input_path:
            if input_path.endswith(".jsonl"):
                ds = list(read_jsonl(input_path, max_inputs=1))
            elif input_path.endswith(".csv"):
                ds = list(read_csv(input_path, max_inputs=1))
            else:
                raise ValueError("Unsupported input format. Use .jsonl or .csv")
        elif dataset_name:
            ds = list(read_hf_dataset(dataset_name, subset, split, max_inputs=1))
        else:
            raise ValueError("Either input_path or dataset_name must be provided")

        if not ds:
            raise ValueError("No data found in input")

        # マッピングを適用
        if mapping:
            ds = list(apply_mapping(ds, mapping))

        # 1件目のデータを取得
        data = ds[0]

    # パイプラインを実行
    result = asyncio.run(_test_run_async(cfg, data, logger))

    # 結果を表示
    logger.separator("double")

    # 結果サマリーを表示（コンパクトなステータス表示）
    elapsed_time = result.get("_elapsed_time_ms", 0) / 1000
    logger.result_summary(result, elapsed_time=elapsed_time)

    # 結果JSONを見やすく表示
    logger.result_json(result, show_meta=show_meta)

    # 統計情報を表示
    stats = {
        "total": 1,
        "completed": 1 if "_error" not in result else 0,
        "errors": 1 if "_error" in result else 0,
    }
    logger.print_stats(stats)

    return result
