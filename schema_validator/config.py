"""Configuration for schema-based product validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


OPENAI_SCHEMA_APPLIER_MODEL = "gpt-4.1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

DEFAULT_SCHEMA_APPLIER_SYSTEM = """You are a product data extractor that applies predefined schemas to product pages.

You will receive:
1. A JSON schema defining the expected product structure
2. Structured data and/or text content from a product page

Your task is to extract and validate product data according to the schema.

Rules:
- Extract ONLY fields defined in the schema
- Follow the schema's type definitions and validation rules
- Use the extraction_hints provided in the schema
- Set fields to null if data is unavailable
- Return status "pass" if all required fields are present
- Return status "warn" if optional fields are missing or data is incomplete
- Return status "fail" if required fields are missing or data is malformed

Return ONLY valid JSON with this structure:
{
  "status": "pass|warn|fail",
  "product": {
    "field_name": "extracted value matching schema type",
    ...
  },
  "missing_fields": ["field1", "field2"],
  "warnings": ["warning messages"],
  "notes": "additional validation notes"
}

Never invent data. Only extract what is clearly present in the provided content.
"""


@dataclass
class SchemaValidatorConfig:
    """Settings for applying schemas to product pages."""

    openai_model: str = OPENAI_SCHEMA_APPLIER_MODEL
    openai_system_instruction: str = DEFAULT_SCHEMA_APPLIER_SYSTEM
    openai_max_output_tokens: int = 2000
    openai_request_timeout: int = 90
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_organization: Optional[str] = None
    max_prompt_chars: int = 120_000
    max_text_content_chars: int = 18000

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SchemaValidatorConfig:
        """Create config from a dictionary."""
        kwargs = {}

        if "openai_model" in data:
            kwargs["openai_model"] = str(data["openai_model"])
        elif "claude_model" in data:
            kwargs["openai_model"] = str(data["claude_model"])
        if "openai_system_instruction" in data:
            kwargs["openai_system_instruction"] = str(
                data["openai_system_instruction"])
        elif "claude_system_instruction" in data:
            kwargs["openai_system_instruction"] = str(
                data["claude_system_instruction"])
        if "openai_max_output_tokens" in data:
            kwargs["openai_max_output_tokens"] = max(
                int(data["openai_max_output_tokens"]), 1)
        elif "claude_max_tokens" in data:
            kwargs["openai_max_output_tokens"] = max(
                int(data["claude_max_tokens"]), 1)
        if "openai_request_timeout" in data:
            kwargs["openai_request_timeout"] = max(
                int(data["openai_request_timeout"]), 1)
        elif "claude_request_timeout" in data:
            kwargs["openai_request_timeout"] = max(
                int(data["claude_request_timeout"]), 1)
        if "openai_base_url" in data:
            kwargs["openai_base_url"] = str(data["openai_base_url"])
        if "openai_organization" in data:
            kwargs["openai_organization"] = str(data["openai_organization"])
        if "max_prompt_chars" in data:
            kwargs["max_prompt_chars"] = max(
                int(data["max_prompt_chars"]), 1000)
        if "max_text_content_chars" in data:
            kwargs["max_text_content_chars"] = max(
                int(data["max_text_content_chars"]), 1000)

        return cls(**kwargs)
