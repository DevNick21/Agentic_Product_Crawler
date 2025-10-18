"""Utilities for homepage markdown classification with Claude."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, MutableMapping, Optional

CLAUDE_DEFAULT_MODEL = "claude-3-5-haiku-latest"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_VERSION = "2023-06-01"
DEFAULT_MARKDOWN_INSTRUCTION = (
    "You are an expert ecommerce analyst. Review the provided markdown snapshot of a website's homepage "
    "and identify all high-level navigation destinations you can find. Classify each URL as either a PRODUCT_CATEGORY page "
    "(a landing page that lists or describes a group of products) or an INFO page (store information, policies, "
    "services, financing, contact, blog). Exclude individual product detail URLs, marketing campaigns, or external domains. "
    "Include every clearly distinct navigation destination you detect, even if there are many. "
    "Respond strictly in JSON with the shape {\"product_categories\": [ {\"url\": str, \"title\": str, \"confidence\": float, \"reason\": str } ], "
    "\"info_pages\": [ {\"url\": str, \"title\": str, \"confidence\": float, \"reason\": str } ]}. Confidence must be between 0 and 1. "
    "Produce JSON only—no code fences, explanations, or trailing text."
)


class ClaudeClassificationError(RuntimeError):
    """Raised when Claude classification fails."""


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    # remove opening fence
    stripped = stripped[3:]
    stripped = stripped.lstrip()

    # drop language identifiers like json/jsonc/javascript
    for prefix in ("jsonc", "json5", "json", "javascript"):
        if stripped.lower().startswith(prefix):
            stripped = stripped[len(prefix):]
            break

    stripped = stripped.lstrip("\r\n ")

    # remove trailing fence (handle both same-line and newline fences)
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    else:
        fence_index = stripped.rfind("\n```")
        if fence_index != -1:
            stripped = stripped[:fence_index]

    return stripped.strip()


def _truncate_markdown(markdown: str, max_chars: int) -> str:
    text = markdown.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    suffix = "\n\n[truncated for brevity]"
    return text[: max_chars - len(suffix)] + suffix


def classify_markdown_with_claude(
    markdown: str,
    *,
    site_url: str,
    instruction: Optional[str] = None,
    model: str = CLAUDE_DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = 2500,
    request_timeout: int = 60,
    max_markdown_chars: int = 15000,
) -> Dict[str, Any]:
    """Ask Claude to identify product category and info pages from homepage markdown."""

    if not markdown.strip():
        return {
            "product_categories": [],
            "info_pages": [],
            "raw_response": "",
            "elapsed_seconds": 0.0,
            "model": model,
        }

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ClaudeClassificationError(
            "Anthropic API key not provided. Set ANTHROPIC_API_KEY or pass api_key."
        )

    system_instruction = instruction or DEFAULT_MARKDOWN_INSTRUCTION
    truncated_markdown = _truncate_markdown(markdown, max_markdown_chars)
    message_text = (
        "Website URL: "
        f"{site_url}\n\n"
        "Homepage markdown snapshot:\n"
        f"{truncated_markdown}\n\n"
        "Return JSON only."
    )

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_instruction,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message_text,
                    }
                ],
            }
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    anthropic_version = os.getenv("ANTHROPIC_API_VERSION", CLAUDE_VERSION)
    request = urllib.request.Request(
        CLAUDE_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
        },
        method="POST",
    )

    start = time.time()
    try:
        # type: ignore[arg-type]
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network failure path
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ClaudeClassificationError(
            f"Claude request failed: {exc.code} {exc.reason}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network failure path
        raise ClaudeClassificationError(
            f"Claude request error: {exc}") from exc

    elapsed = time.time() - start

    try:
        parsed = json.loads(response_body)
        content = parsed["content"][0]["text"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:  # pragma: no cover - response shape unexpected
        raise ClaudeClassificationError(
            f"Unexpected Claude response format: {response_body}"
        ) from exc

    content = content.strip()
    try:
        summary = json.loads(content)
    except json.JSONDecodeError:
        cleaned = _strip_code_fence(content)
        try:
            summary = json.loads(cleaned)
        except json.JSONDecodeError as exc:  # pragma: no cover - final parse failure
            raise ClaudeClassificationError(
                f"Claude returned non-JSON output: {content}"
            ) from exc

    product_categories = summary.get("product_categories", []) if isinstance(
        summary, MutableMapping) else []
    info_pages = summary.get("info_pages", []) if isinstance(
        summary, MutableMapping) else []

    return {
        "product_categories": product_categories,
        "info_pages": info_pages,
        "raw_response": content,
        "elapsed_seconds": elapsed,
        "model": model,
    }


__all__ = [
    "classify_markdown_with_claude",
    "ClaudeClassificationError",
    "CLAUDE_DEFAULT_MODEL",
]
