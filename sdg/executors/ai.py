from __future__ import annotations
import os, sys
from typing import Any, Dict, List, Optional

from ..config import AIBlock, SDGConfig, OutputDef
from ..llm_client import LLMClient
from ..utils import (
    render_template,
    has_image_placeholders,
    extract_image_placeholders,
    resolve_image_to_data_uri,
    is_image_data,
    load_image_as_base64,
)
from .core import ExecutionContext, _apply_outputs


def _build_clients(cfg: SDGConfig) -> Dict[str, LLMClient]:
    """モデルクライアントを構築"""
    clients: Dict[str, LLMClient] = {}

    # 最適化オプションの取得
    use_shared_transport = cfg.optimization.get("use_shared_transport", False)
    http2 = cfg.optimization.get("http2", True)

    for m in cfg.models:
        # 環境変数注入
        api_key = m.api_key
        if api_key.startswith("${ENV."):
            env_name = api_key[6:-1]  # ${ENV.NAME} -> NAME
            api_key = os.environ.get(env_name, "")

        base_url = m.base_url or "https://api.openai.com"
        timeout = (m.request_defaults or {}).get("timeout_sec")
        clients[m.name] = LLMClient(
            base_url=base_url,
            api_key=api_key,
            organization=m.organization,
            headers=m.headers,
            timeout_sec=timeout,
            use_shared_transport=use_shared_transport,
            http2=http2,
        )
    return clients


def _build_multimodal_content(
    text: str,
    ctx: Dict[str, Any],
    cfg: SDGConfig,
    base_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    テキストと画像を含むマルチモーダルコンテンツを構築

    Args:
        text: プロンプトテキスト（{name.img}プレースホルダーを含む可能性あり）
        ctx: コンテキスト辞書
        cfg: SDG設定
        base_path: 画像パス解決用のベースディレクトリ

    Returns:
        OpenAI Vision API形式のcontentリスト
    """
    # 画像プレースホルダーがなければシンプルなテキストを返す
    if not has_image_placeholders(text):
        return [{"type": "text", "text": text}]

    content_parts: List[Dict[str, Any]] = []
    last_end = 0

    for img_name, options, start, end in extract_image_placeholders(text):
        # 画像の前のテキスト部分を追加
        if start > last_end:
            text_part = text[last_end:start]
            if text_part.strip():
                content_parts.append({"type": "text", "text": text_part})

        # 画像を解決
        img_url = None
        detail = options.get("detail", "auto")

        # 1. 入力データから検索
        img_data = ctx.get(img_name)
        if img_data and is_image_data(img_data):
            try:
                img_url = resolve_image_to_data_uri(img_data, base_path)
            except Exception as e:
                # 画像解決に失敗した場合はスキップ
                print(
                    f"Warning: Failed to resolve image '{img_name}' from context: {e}",
                    file=sys.stderr,
                )

        # 2. imagesセクションから検索
        if img_url is None:
            img_def = cfg.image_by_name(img_name)
            if img_def:
                try:
                    if img_def.base64:
                        img_url = f"data:{img_def.media_type};base64,{img_def.base64}"
                    elif img_def.url:
                        img_url = img_def.url
                    elif img_def.path:
                        path = img_def.path
                        if base_path and not os.path.isabs(path):
                            path = os.path.join(base_path, path)
                        b64, media_type = load_image_as_base64(path)
                        img_url = f"data:{media_type};base64,{b64}"
                except Exception as e:
                    print(
                        f"Warning: Failed to resolve image '{img_name}' from images section: {e}",
                        file=sys.stderr,
                    )

        # 画像が見つかった場合のみ追加
        if img_url:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                        "detail": detail,
                    },
                }
            )

        last_end = end

    # 残りのテキストを追加
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining.strip():
            content_parts.append({"type": "text", "text": remaining})

    # contentが空の場合はエラー回避のためテキストのみ返す
    if not content_parts:
        return [{"type": "text", "text": text}]

    return content_parts


def _has_images_in_prompts(
    prompts: List[str], ctx: Dict[str, Any], cfg: SDGConfig
) -> bool:
    """プロンプトに画像が含まれているかチェック"""
    for p in prompts:
        rendered = render_template(p, ctx)
        if has_image_placeholders(rendered):
            return True
    return False


async def _execute_ai_block_single(
    block: AIBlock,
    ctx: Dict[str, Any],
    cfg: SDGConfig,
    clients: Dict[str, LLMClient],
    exec_ctx: ExecutionContext,
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """単一行のAIブロック実行"""
    # メッセージ構築
    msgs = []
    if block.system_prompt:
        msgs.append(
            {
                "role": "system",
                "content": render_template(block.system_prompt, ctx),
            }
        )

    # プロンプト内に画像があるかチェック
    raw_user_content = "\n\n".join(
        [render_template(p, ctx) for p in (block.prompts or [])]
    )

    # 画像プレースホルダーがある場合はマルチモーダルコンテンツを構築
    if has_image_placeholders(raw_user_content):
        multimodal_content = _build_multimodal_content(
            raw_user_content, ctx, cfg, base_path
        )
        msgs.append({"role": "user", "content": multimodal_content})
    else:
        msgs.append({"role": "user", "content": raw_user_content})

    client = clients[block.model]
    model_def = cfg.model_by_name(block.model)
    req_params = dict((model_def.request_defaults or {}))
    req_params.update(block.params or {})

    # v2: JSONモード
    if block.mode == "json":
        req_params["response_format"] = {"type": "json_object"}

    # 単一チャット呼び出し
    retry_cfg = dict(req_params.get("retry") or {})
    # 最適化オプションからretry_on_empty設定を取得
    if hasattr(cfg, "optimization") and cfg.optimization:
        retry_on_empty = cfg.optimization.get("retry_on_empty", True)
        retry_cfg["retry_on_empty"] = retry_on_empty
    payload = {"model": model_def.api_model, "messages": msgs, **req_params}
    text, err, _ = await client._one_chat(payload, retry_cfg)

    if err:
        raise err

    text = text or ""
    out_map = _apply_outputs(
        text,
        block.outputs or [OutputDef(name="full", select="full")],
    )

    # v2: save_to
    if block.save_to and "vars" in block.save_to:
        for var_name, out_name in block.save_to["vars"].items():
            if out_name in out_map:
                exec_ctx.set_global(var_name, out_map[out_name])

    return out_map
