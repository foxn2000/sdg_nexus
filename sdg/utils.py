from __future__ import annotations
import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_\.\-]+)\}")
# 画像プレースホルダー: {name.img} or {name.img:detail=high,resize=512x512}
IMAGE_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_\-]+)\.img(?::([^}]+))?\}")


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_json_obj(v: Any) -> Dict[str, Any]:
    """Accept dict or JSON string and return dict."""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        return json.loads(v)
    return {}


def render_template(s: str, ctx: Dict[str, Any]) -> str:
    """Replace {VarName} with ctx[VarName] string values.
    Supports dotted keys like {foo.bar}. Missing keys -> empty string.
    """

    def repl(m: re.Match):
        key = m.group(1)
        cur = ctx
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return ""
        return str(cur)

    return PLACEHOLDER_RE.sub(repl, s)


def extract_by_tag(text: str, tag: str) -> list[str]:
    # Extract <tag>...</tag> ignoring case and allowing newlines
    pattern = re.compile(
        rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    results = pattern.findall(text)

    # 抽出結果から他タグの残骸を除去（例: </think> が先頭に残る場合）
    cleaned = []
    for result in results:
        # 先頭の閉じタグ残骸を削除
        result = re.sub(r"^[\s]*</\w+>[\s]*", "", result, flags=re.IGNORECASE)
        # 末尾の閉じタグ残骸を削除（念のため）
        result = re.sub(r"[\s]*</\w+>[\s]*$", "", result, flags=re.IGNORECASE)
        cleaned.append(result.strip())

    return cleaned


def extract_by_regex(text: str, pattern: str) -> list[str]:
    rx = re.compile(pattern, re.DOTALL)
    return [m.group(1) if m.groups() else m.group(0) for m in rx.finditer(text)]


# ============================================================
# 画像処理ユーティリティ (v2.1)
# ============================================================


def get_media_type_from_path(path: str) -> str:
    """ファイルパスからMIMEタイプを推定"""
    suffix = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/png")


def load_image_as_base64(path: str) -> Tuple[str, str]:
    """
    画像ファイルをbase64エンコードして返す

    Args:
        path: 画像ファイルのパス

    Returns:
        (base64エンコード文字列, MIMEタイプ) のタプル
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    media_type = get_media_type_from_path(path)

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return encoded, media_type


def parse_image_options(options_str: str) -> Dict[str, str]:
    """
    画像オプション文字列をパース

    例: "detail=high,resize=512x512" -> {"detail": "high", "resize": "512x512"}
    """
    if not options_str:
        return {}

    options = {}
    for opt in options_str.split(","):
        opt = opt.strip()
        if "=" in opt:
            key, value = opt.split("=", 1)
            options[key.strip()] = value.strip()

    return options


def extract_image_placeholders(text: str) -> List[Tuple[str, Dict[str, str], int, int]]:
    """
    テキストから画像プレースホルダーを抽出

    Args:
        text: 検索対象のテキスト

    Returns:
        List of (画像名, オプション辞書, 開始位置, 終了位置)
    """
    results = []
    for match in IMAGE_PLACEHOLDER_RE.finditer(text):
        name = match.group(1)
        options_str = match.group(2) or ""
        options = parse_image_options(options_str)
        results.append((name, options, match.start(), match.end()))

    return results


def has_image_placeholders(text: str) -> bool:
    """テキストに画像プレースホルダーが含まれているかチェック"""
    return bool(IMAGE_PLACEHOLDER_RE.search(text))


def resolve_image_to_data_uri(
    img_data: Dict[str, Any],
    base_path: Optional[str] = None,
) -> str:
    """
    画像データをdata URI形式に変換

    Args:
        img_data: 画像データ辞書 {"_type": "image", "path"|"url"|"base64": ...}
        base_path: パス解決用のベースディレクトリ

    Returns:
        data URI文字列 または URL
    """
    if "base64" in img_data:
        media_type = img_data.get("media_type", "image/png")
        return f"data:{media_type};base64,{img_data['base64']}"

    elif "url" in img_data:
        # URLはそのまま返す（OpenAI APIが直接取得可能）
        return img_data["url"]

    elif "path" in img_data:
        path = img_data["path"]
        # 相対パスの場合、base_pathを基準に解決
        if base_path and not Path(path).is_absolute():
            path = str(Path(base_path) / path)

        b64, media_type = load_image_as_base64(path)
        return f"data:{media_type};base64,{b64}"

    raise ValueError("Image data must contain 'path', 'url', or 'base64'")


def is_image_data(value: Any) -> bool:
    """値が画像データ形式かどうかをチェック"""
    if not isinstance(value, dict):
        return False
    return value.get("_type") == "image"
