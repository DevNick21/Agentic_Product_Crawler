"""AI-powered product data extractor using Firecrawl extract endpoint."""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

FIRECRAWL_STATUS_ENDPOINT = "https://api.firecrawl.dev/v2/extract/{job_id}"
MAX_POLL_ATTEMPTS = 60  # Increased from 30 to handle slow extractions
POLL_INTERVAL_SECONDS = 2


def _poll_extract_job(
    job_id: str,
    api_key: str,
    max_wait_seconds: int = 60,
) -> Dict[str, Any]:
    """Poll Firecrawl extract job until completion."""
    status_url = FIRECRAWL_STATUS_ENDPOINT.format(job_id=job_id)
    max_attempts = min(MAX_POLL_ATTEMPTS, max_wait_seconds //
                       POLL_INTERVAL_SECONDS)

    for attempt in range(max_attempts):
        time.sleep(POLL_INTERVAL_SECONDS)

        request = urllib.request.Request(
            status_url,
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)

                status = parsed.get("status")

                # Check if completed
                if status == "completed" or parsed.get("success"):
                    extracted_data = parsed.get("data", [])
                    if extracted_data:
                        product_data = extracted_data[0] if isinstance(
                            extracted_data, list) else extracted_data
                        return {
                            "data": product_data,
                            "status_code": response.getcode(),
                            "poll_attempts": attempt + 1,
                        }

                # Check if failed
                if status in ("failed", "error"):
                    return {
                        "error": f"Extract job failed: {parsed.get('error', 'Unknown error')}",
                        "response": parsed,
                        "poll_attempts": attempt + 1,
                    }

                # Still processing, continue polling
                continue

        except urllib.error.HTTPError as exc:
            # Retry on 502 Bad Gateway and 503 Service Unavailable
            if exc.code in (502, 503):
                # Don't fail immediately - continue polling
                continue
            else:
                return {
                    "error": f"Polling failed: HTTP Error {exc.code}: {exc.reason}",
                    "poll_attempts": attempt + 1,
                }
        except Exception as exc:
            # For other exceptions, also continue polling (network issues, etc.)
            # Only fail if we run out of attempts
            continue

    return {
        "error": f"Extract job timed out after {max_attempts} attempts",
        "job_id": job_id,
        "poll_attempts": max_attempts,
    }


def extract_product_with_schema(
    url: str,
    schema: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Extract structured product data using Firecrawl v2 /extract endpoint.

    This uses AI to extract data matching your schema, much more accurate than regex.

    Args:
        url: Product page URL
        schema: JSON schema defining product structure
        api_key: Firecrawl API key (defaults to FIRECRAWL_API_KEY env var)
        timeout: Request timeout in seconds

    Returns:
        Dict with 'data' (extracted product) or 'error'
    """
    if not api_key:
        api_key = os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        return {"error": "FIRECRAWL_API_KEY is required"}

    endpoint = "https://api.firecrawl.dev/v2/extract"

    # Build extraction request
    payload = {
        "urls": [url],
        "schema": schema,
        "prompt": "Extract all product information from this page, including specifications, images, pricing, and technical details. Be thorough and extract everything visible.",
    }

    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            status_code = response.getcode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return {
            "error": f"Firecrawl extract failed: {exc.code} {exc.reason}: {detail}",
            "elapsed_seconds": time.perf_counter() - start,
        }
    except (TimeoutError, urllib.error.URLError) as exc:
        return {
            "error": f"Firecrawl extract error: {exc}",
            "elapsed_seconds": time.perf_counter() - start,
        }

    elapsed = time.perf_counter() - start

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {
            "error": "Firecrawl returned invalid JSON",
            "response_body": body[:1000],
            "elapsed_seconds": elapsed,
        }

    # v2 extract response: {"success": true, "data": [...]}
    if not parsed.get("success"):
        return {
            "error": parsed.get("error", "Extract failed"),
            "response": parsed,
            "elapsed_seconds": elapsed,
        }

    # Check if this is an async job response (has 'id' but no 'data')
    job_id = parsed.get("id")
    if job_id and "data" not in parsed:
        # Poll for results
        poll_result = _poll_extract_job(job_id, api_key, timeout)
        poll_result["elapsed_seconds"] = time.perf_counter() - start
        return poll_result

    extracted_data = parsed.get("data", [])

    if not extracted_data:
        return {
            "error": "No data extracted",
            "response": parsed,
            "elapsed_seconds": elapsed,
        }

    # Return first result (single URL)
    product_data = extracted_data[0] if isinstance(
        extracted_data, list) else extracted_data

    return {
        "data": product_data,
        "elapsed_seconds": elapsed,
        "status_code": status_code,
    }


def build_generic_product_schema() -> Dict[str, Any]:
    """
    Build a flexible product schema that works for any e-commerce site.

    Returns generic schema that captures common product fields.
    """
    return {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Product name or title"
            },
            "description": {
                "type": "string",
                "description": "Full product description"
            },
            "price": {
                "type": "number",
                "description": "Current price (numeric value only)"
            },
            "original_price": {
                "type": "number",
                "description": "Original price before discount, if shown"
            },
            "currency": {
                "type": "string",
                "description": "Currency code (EUR, USD, GBP, etc.)"
            },
            "availability": {
                "type": "string",
                "description": "Stock status (in stock, out of stock, etc.)"
            },
            "brand": {
                "type": "string",
                "description": "Brand or manufacturer name"
            },
            "sku": {
                "type": "string",
                "description": "Product SKU or model number"
            },
            "category": {
                "type": "string",
                "description": "Product category or type"
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of product image URLs"
            },
            "specifications": {
                "type": "array",
                "description": "All technical specifications and product attributes",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Specification name or label"},
                        "value": {"type": "string", "description": "Specification value"}
                    },
                    "required": ["name", "value"]
                }
            },
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of key product features or bullet points"
            },
            "dimensions": {
                "type": "object",
                "properties": {
                    "length": {"type": "string"},
                    "width": {"type": "string"},
                    "height": {"type": "string"},
                    "weight": {"type": "string"}
                },
                "description": "Product dimensions and weight"
            },
            "warranty": {
                "type": "string",
                "description": "Warranty information"
            },
            "rating": {
                "type": "number",
                "description": "Average customer rating (0-5)"
            },
            "review_count": {
                "type": "integer",
                "description": "Number of customer reviews"
            }
        },
        "required": ["title"]
    }
