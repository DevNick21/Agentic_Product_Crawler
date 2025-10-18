"""OpenAI helper for applying schemas to product pages."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import requests

from .config import SchemaValidatorConfig


OPENAI_RESPONSES_PATH = "/responses"


class SchemaApplierError(RuntimeError):
    """Raised when schema application with OpenAI fails."""


@dataclass
class _Prompt:
    text: str
    truncated: bool = False


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    stripped = stripped[3:].lstrip()
    for prefix in ("jsonc", "json5", "json", "javascript"):
        if stripped.lower().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    stripped = stripped.lstrip("\r\n ")

    if stripped.endswith("```"):
        stripped = stripped[:-3]
    else:
        fence_index = stripped.rfind("\n```")
        if fence_index != -1:
            stripped = stripped[:fence_index]
    return stripped.strip()


def _extract_json_object(content: str) -> Optional[str]:
    if not content:
        return None

    in_string = False
    escape = False
    depth = 0
    start_index: Optional[int] = None

    for index, char in enumerate(content):
        if char == '"' and not escape:
            in_string = not in_string
        if char == "\\" and in_string:
            escape = not escape
            continue
        else:
            escape = False

        if in_string:
            continue

        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = content[start_index: index + 1]
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    start_index = None
                    continue
                return candidate

    return None


def _trim_text(value: str, limit: int) -> tuple[str, bool]:
    if limit and len(value) > limit:
        return value[:limit], True
    return value, False


def _build_payload(
    *,
    schema: Mapping[str, Any],
    page_url: str,
    structured_data: Optional[Mapping[str, Any]],
    text_content: Optional[str],
    config: SchemaValidatorConfig,
) -> _Prompt:
    payload: Dict[str, Any] = {
        "page_url": page_url,
        "schema": schema,
    }

    truncated = False

    if structured_data:
        try:
            payload["structured_data"] = json.loads(
                json.dumps(structured_data, ensure_ascii=False, default=str)
            )
        except (TypeError, ValueError):
            payload["structured_data"] = {}

    if text_content and text_content.strip():
        trimmed, cut = _trim_text(
            text_content.strip(), config.max_text_content_chars)
        payload["text_content"] = trimmed
        truncated = truncated or cut

    payload_text = json.dumps(
        payload, ensure_ascii=False, indent=2, default=str)
    if config.max_prompt_chars and len(payload_text) > config.max_prompt_chars:
        payload_text = (
            f"{payload_text[:config.max_prompt_chars]}\n\n[Payload truncated to {config.max_prompt_chars} characters to respect context limits.]"
        )
        truncated = True

    return _Prompt(text=payload_text, truncated=truncated)


def _extract_response_text(response: Mapping[str, Any]) -> str:
    if "output_text" in response and isinstance(response["output_text"], str):
        return response["output_text"].strip()

    output = response.get("output")
    if isinstance(output, Sequence):
        for item in output:
            if isinstance(item, Mapping):
                content = item.get("content")
                if isinstance(content, Sequence):
                    for block in content:
                        if isinstance(block, Mapping):
                            text = block.get("text") or block.get("value")
                            if isinstance(text, str):
                                return text.strip()

    content = response.get("content")
    if isinstance(content, Sequence):
        for block in content:
            if isinstance(block, Mapping):
                text = block.get("text") or block.get("value")
                if isinstance(text, str):
                    return text.strip()

    data = response.get("data")
    if isinstance(data, Sequence):
        for item in data:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("value")
                if isinstance(text, str):
                    return text.strip()

    raise SchemaApplierError("OpenAI response did not contain textual output")


def _make_openai_request(
    *,
    prompt: _Prompt,
    config: SchemaValidatorConfig,
    api_key: str,
) -> tuple[str, Dict[str, Any], float]:
    base_url = config.openai_base_url.rstrip(
        "/") or "https://api.openai.com/v1"
    url = f"{base_url}{OPENAI_RESPONSES_PATH}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if config.openai_organization:
        headers["OpenAI-Organization"] = config.openai_organization

    payload: Dict[str, Any] = {
        "model": config.openai_model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt.text}
                ],
            }
        ],
        "max_output_tokens": config.openai_max_output_tokens,
    }
    if config.openai_system_instruction:
        payload["instructions"] = config.openai_system_instruction

    start = time.time()
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=config.openai_request_timeout,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network failure
        detail = exc.response.text if exc.response is not None else str(exc)
        raise SchemaApplierError(
            f"OpenAI schema applier request failed (model={config.openai_model}): {detail}"
        ) from exc
    except requests.RequestException as exc:  # pragma: no cover - network failure
        raise SchemaApplierError(
            f"OpenAI schema applier request error: {exc}") from exc

    elapsed = time.time() - start

    try:
        parsed = response.json()
    except ValueError as exc:
        raise SchemaApplierError(
            f"OpenAI returned non-JSON response: {response.text}") from exc

    return _extract_response_text(parsed), parsed, elapsed


class OpenAISchemaApplier:
    """GPT-powered instance for applying schemas to extract product data."""

    def __init__(self, config: SchemaValidatorConfig, schema: Mapping[str, Any]):
        self.config = config
        self.schema = schema

    def validate_product(
        self,
        *,
        page_url: str,
        structured_data: Optional[Mapping[str, Any]] = None,
        text_content: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SchemaApplierError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
            )

        if not structured_data and not text_content:
            return {
                "status": "warn",
                "product": {},
                "missing_fields": [],
                "warnings": ["No content provided for validation"],
                "notes": "Product page had no extractable content",
                "raw_response": "",
                "model": self.config.openai_model,
                "elapsed_seconds": 0.0,
            }

        prompt = _build_payload(
            schema=self.schema,
            page_url=page_url,
            structured_data=structured_data,
            text_content=text_content,
            config=self.config,
        )

        text, raw_response, elapsed = _make_openai_request(
            prompt=prompt,
            config=self.config,
            api_key=api_key,
        )

        candidates = [text]
        cleaned = _strip_code_fence(text)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
        extracted = _extract_json_object(cleaned)
        if extracted and extracted not in candidates:
            candidates.append(extracted)

        result: MutableMapping[str, Any]
        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, MutableMapping):
                result = parsed
                break
        else:
            raise SchemaApplierError(
                f"OpenAI returned non-JSON validation output: {text}"
            ) from last_error

        result.setdefault("status", "warn")
        result.setdefault("product", {})
        result.setdefault("missing_fields", [])
        result.setdefault("warnings", [])
        result.setdefault("notes", "")

        usage = raw_response.get("usage") if isinstance(
            raw_response, Mapping) else None
        prompt_tokens = None
        completion_tokens = None
        if isinstance(usage, Mapping):
            prompt_tokens = usage.get("input_tokens")
            completion_tokens = usage.get("output_tokens")

        # Override status to "fail" if product data is empty or has no title
        product_data = result.get("product", {})
        status = result.get("status")
        if not product_data or not product_data.get("title"):
            status = "fail"
            if not result.get("warnings"):
                result["warnings"] = []
            if isinstance(result["warnings"], list):
                result["warnings"].append("Product extraction returned no usable data")

        response_payload: Dict[str, Any] = {
            "status": status,
            "product": product_data,
            "missing_fields": result.get("missing_fields"),
            "warnings": result.get("warnings"),
            "notes": result.get("notes"),
            "raw_response": text,
            "model": self.config.openai_model,
            "elapsed_seconds": elapsed,
        }
        if prompt.truncated:
            response_payload["prompt_truncated"] = True
        if isinstance(prompt_tokens, int):
            response_payload["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            response_payload["completion_tokens"] = completion_tokens

        return response_payload


def validate_with_schema(
    schema: Mapping[str, Any],
    *,
    page_url: str,
    structured_data: Optional[Mapping[str, Any]] = None,
    text_content: Optional[str] = None,
    config: Optional[SchemaValidatorConfig] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    config = config or SchemaValidatorConfig()
    applier = OpenAISchemaApplier(config, schema)
    return applier.validate_product(
        page_url=page_url,
        structured_data=structured_data,
        text_content=text_content,
        api_key=api_key,
    )
