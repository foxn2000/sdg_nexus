from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, Optional

from ..config import load_config, PyBlock
from ..executors import process_single_row
from ..executors.ai import _build_clients
from ..executors.core import ExecutionContext
from ..executors.python import _load_python_function
from ..logger import init_logger
from ..io import (
    read_jsonl,
    read_csv,
    read_hf_dataset,
    apply_mapping,
)


async def _test_run_async(
    cfg,
    data: Dict[str, Any],
    logger,
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
    logger.info("Starting test run with single data item...")
    
    # 入力データを表示
    logger.table("Input Data", data)
    
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
    
    try:
        result = await process_single_row(
            row_index=0,
            initial_context=data,
            cfg=cfg,
            clients=clients,
            exec_ctx=exec_ctx,
            save_intermediate=True,  # 中間結果を保存
            python_functions=python_functions,
        )
        
        elapsed_time = time.time() - start_time
        
        # process_single_row returns a Dict, not StreamingResult
        logger.success(f"Pipeline execution completed in {elapsed_time:.2f}s")
        return {
            "_row_index": 0,
            "_elapsed_time_ms": int(elapsed_time * 1000),
            **result,
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
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
    
    # データを読み込み（1件のみ）
    if input_path:
        if input_path.endswith(".jsonl"):
            ds = read_jsonl(input_path, max_inputs=1)
        elif input_path.endswith(".csv"):
            ds = read_csv(input_path, max_inputs=1)
        else:
            raise ValueError("Unsupported input format. Use .jsonl or .csv")
    elif dataset_name:
        ds = read_hf_dataset(dataset_name, subset, split, max_inputs=1)
    else:
        raise ValueError("Either input_path or dataset_name must be provided")
    
    if not ds:
        raise ValueError("No data found in input")
    
    # マッピングを適用
    if mapping:
        ds = apply_mapping(ds, mapping)
    
    # 1件目のデータを取得
    data = ds[0]
    
    # パイプラインを実行
    result = asyncio.run(_test_run_async(cfg, data, logger))
    
    # 結果を表示
    if locale == "ja":
        logger.info("実行結果:")
    else:
        logger.info("Execution Result:")
    
    # 結果をテーブル形式で表示
    result_display = {}
    for key, value in result.items():
        if key.startswith("_"):
            continue
        # 長い値は省略
        str_value = str(value)
        if len(str_value) > 100:
            str_value = str_value[:100] + "..."
        result_display[key] = str_value
    
    if result_display:
        if locale == "ja":
            logger.table("出力データ", result_display)
        else:
            logger.table("Output Data", result_display)
    
    # メタ情報を表示
    meta_info = {}
    if "_elapsed_time_ms" in result:
        if locale == "ja":
            meta_info["実行時間"] = f"{result['_elapsed_time_ms']}ms"
        else:
            meta_info["Elapsed Time"] = f"{result['_elapsed_time_ms']}ms"
    if "_error" in result:
        if locale == "ja":
            meta_info["エラー"] = result["_error"]
        else:
            meta_info["Error"] = result["_error"]
    
    if meta_info:
        if locale == "ja":
            logger.table("メタ情報", meta_info)
        else:
            logger.table("Meta Information", meta_info)
    
    # 統計情報を表示
    stats = {
        "total": 1,
        "completed": 1 if "_error" not in result else 0,
        "errors": 1 if "_error" in result else 0,
    }
    logger.print_stats(stats)
    
    return result
