from __future__ import annotations
import json, re, time
from typing import Any, Dict

PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_\.\-]+)\}")


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
