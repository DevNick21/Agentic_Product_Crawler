"""LangChain tools for the product extraction pipeline.

This module wraps existing pipeline functionality as LangChain tools,
enabling agentic orchestration with LangChain and LangGraph.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Mapping, Optional, Sequence

from langchain_core.tools import tool

# Import existing pipeline modules
from firecrawl_utils import normalise_firecrawl_formats
from retry_utils import retry_with_backoff, is_retryable_api_error


def _domain_matches(url: str, allowed_domain: Optional[str]) -> bool:
    if not allowed_domain:
        return True
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.netloc or "").lower()
    if not hostname:
        return False
    allowed = allowed_domain.lower()
    return hostname == allowed or hostname.endswith(f".{allowed}")


def _prune_structured_context(
    value: Any,
    *,
    depth: int = 4,
    max_items: int = 12,
    max_string: int = 800,
) -> Any:
    if depth <= 0:
        return None
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        pruned_list = []
        for item in value[:max_items]:
            pruned = _prune_structured_context(
                item,
                depth=depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
            if pruned is not None:
                pruned_list.append(pruned)
        return pruned_list
    if isinstance(value, Mapping):
        pruned_dict: Dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            if count >= max_items:
                break
            pruned = _prune_structured_context(
                item,
                depth=depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
            if pruned is not None:
                pruned_dict[str(key)] = pruned
            count += 1
        return pruned_dict
    return value


def _fetch_product_page(
    url: str,
    *,
    timeout: int,
    max_bytes: int,
    logger: logging.Logger,
    allowed_domain: Optional[str],
    firecrawl_endpoint: str,
    firecrawl_formats: Sequence[str],
    firecrawl_wait_seconds: float,
) -> Dict[str, Any]:
    if not _domain_matches(url, allowed_domain):
        return {
            "html": "",
            "metadata": {
                "status_code": None,
                "elapsed_seconds": 0.0,
                "final_url": url,
                "content_type": None,
                "content_length": 0,
                "truncated": False,
            },
            "error": f"URL outside allowed domain: {url}",
        }

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return {
            "html": "",
            "markdown": "",
            "structured_data": None,
            "metadata": {
                "status_code": None,
                "elapsed_seconds": 0.0,
                "final_url": url,
                "content_type": None,
                "content_length": 0,
                "truncated": False,
                "fetch_mode": "firecrawl",
            },
            "error": "FIRECRAWL_API_KEY is required for product detail scraping",
        }

    normalised_formats = normalise_firecrawl_formats(
        firecrawl_formats,
        default=("markdown", "html", "rawHtml", "json"),
    )

    request_body = json.dumps(
        {
            "url": url,
            "formats": list(normalised_formats),
            "proxy": "stealth",
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        firecrawl_endpoint,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    start = time.perf_counter()
    try:
        # type: ignore[arg-type]
        with urllib.request.urlopen(request, timeout=max(timeout, 1)) as response:
            body = response.read().decode("utf-8", errors="ignore")
            http_status = response.getcode()
            http_content_type = response.headers.get("Content-Type")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        elapsed_seconds = time.perf_counter() - start
        return {
            "html": "",
            "metadata": {
                "status_code": exc.code,
                "elapsed_seconds": elapsed_seconds,
                "final_url": url,
                "content_type": exc.headers.get("Content-Type") if exc.headers else None,
                "content_length": 0,
                "truncated": False,
            },
            "error": f"Firecrawl scrape failed: {exc.code} {exc.reason}: {detail[:500]}",
        }
    except (TimeoutError, urllib.error.URLError) as exc:
        elapsed_seconds = time.perf_counter() - start
        return {
            "html": "",
            "metadata": {
                "status_code": None,
                "elapsed_seconds": elapsed_seconds,
                "final_url": url,
                "content_type": None,
                "content_length": 0,
                "truncated": False,
                "fetch_mode": "firecrawl",
            },
            "error": f"Firecrawl scrape error: {exc}",
        }

    elapsed_seconds = time.perf_counter() - start

    if firecrawl_wait_seconds:
        time.sleep(max(firecrawl_wait_seconds, 0.0))

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {
            "html": "",
            "metadata": {
                "status_code": http_status,
                "elapsed_seconds": time.perf_counter() - start,
                "final_url": url,
                "content_type": http_content_type,
                "content_length": 0,
                "truncated": False,
                "fetch_mode": "firecrawl",
            },
            "error": "Firecrawl returned invalid JSON",
        }

    if isinstance(parsed, Mapping) and parsed.get("error"):
        return {
            "html": "",
            "metadata": {
                "status_code": http_status,
                "elapsed_seconds": time.perf_counter() - start,
                "final_url": url,
                "content_type": http_content_type,
                "content_length": 0,
                "truncated": False,
                "fetch_mode": "firecrawl",
            },
            "error": str(parsed.get("error")),
        }

    html = ""
    markdown = ""
    structured_context: Optional[Dict[str, Any]] = None
    metadata_block: Dict[str, Any] = {}
    final_url = url
    content_type = http_content_type
    status_code = http_status

    if isinstance(parsed, Mapping):
        data = parsed.get("data") if parsed.get("success") else parsed
        if not isinstance(data, Mapping):
            data = parsed

        html = str(
            data.get("html")
            or data.get("rawHtml")
            or data.get("raw_html")
            or data.get("content")
            or ""
        )
        markdown = str(data.get("markdown") or "")

        combined: Dict[str, Any] = {}
        for key in ("json", "extract", "attributes", "changeTracking", "change_tracking"):
            value = data.get(key)
            if isinstance(value, Mapping):
                combined[key] = value
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                combined[key] = list(value)
        if combined:
            structured_context = _prune_structured_context(combined)

        metadata_candidate = data.get("metadata")
        if isinstance(metadata_candidate, Mapping):
            metadata_block = dict(metadata_candidate)
            status_code = (
                metadata_block.get("statusCode")
                or metadata_block.get("status_code")
                or metadata_block.get("status")
                or status_code
            )
            content_type = (
                metadata_block.get("contentType")
                or metadata_block.get("content_type")
                or metadata_block.get("mimeType")
                or content_type
            )
            final_url = (
                metadata_block.get("resolvedUrl")
                or metadata_block.get("finalUrl")
                or metadata_block.get("url")
                or final_url
            )

        if not html and markdown:
            html = markdown
    else:
        html = str(parsed)

    if not html.strip() and not markdown.strip():
        return {
            "html": "",
            "metadata": {
                "status_code": status_code,
                "elapsed_seconds": elapsed_seconds,
                "final_url": final_url,
                "content_type": content_type,
                "content_length": 0,
                "truncated": False,
                "fetch_mode": "firecrawl",
            },
            "error": "Firecrawl returned empty content",
        }

    if not _domain_matches(final_url, allowed_domain):
        return {
            "html": "",
            "metadata": {
                "status_code": status_code,
                "elapsed_seconds": elapsed_seconds,
                "final_url": final_url,
                "content_type": content_type,
                "content_length": 0,
                "truncated": False,
                "fetch_mode": "firecrawl",
            },
            "error": f"Redirected to disallowed domain: {final_url}",
        }

    truncated = False
    max_bytes = max(max_bytes, 1000)
    if len(html) > max_bytes:
        html = html[:max_bytes]
        truncated = True

    metadata: Dict[str, Any] = {
        "status_code": status_code,
        "elapsed_seconds": elapsed_seconds,
        "final_url": final_url,
        "content_type": content_type,
        "content_length": len(html),
        "truncated": truncated,
        "fetch_mode": "firecrawl",
    }
    if metadata_block:
        metadata["firecrawl"] = metadata_block

    return {
        "html": html,
        "markdown": markdown,
        "structured_data": structured_context,
        "metadata": metadata,
        "error": None,
    }


logger = logging.getLogger(__name__)


@tool
def scrape_homepage(url: str, formats: str = "markdown,html", wait_seconds: float = 0.0) -> Dict[str, Any]:
    """Scrape a homepage using Firecrawl to extract content and navigation links.

    Args:
        url: The homepage URL to scrape
        formats: Comma-separated list of formats to extract (markdown, html, rawHtml)
        wait_seconds: Seconds to wait for JavaScript to load (default: 0.0)

    Returns:
        Dictionary containing:
        - html: HTML content
        - markdown: Markdown content
        - metadata: Page metadata (status, title, etc.)
        - error: Error message if scraping failed
    """
    try:
        format_list = tuple(f.strip() for f in formats.split(","))
        result = _fetch_product_page(
            url=url,
            timeout=60,
            max_bytes=1000000,
            logger=logger,
            allowed_domain=None,
            firecrawl_endpoint="https://api.firecrawl.dev/v2/scrape",
            firecrawl_formats=format_list,
            firecrawl_wait_seconds=wait_seconds,
        )

        has_error = "error" in result and result.get("error") is not None
        logger.info(
            f"Scrape result: html={len(result.get('html', ''))}, markdown={len(result.get('markdown', ''))}, error={result.get('error')}")

        return {
            "success": not has_error,
            "html": result.get("html", ""),
            "markdown": result.get("markdown", ""),
            "metadata": result.get("metadata", {}),
            "error": result.get("error"),
        }
    except Exception as e:
        logger.error(f"Homepage scrape failed: {e}")
        return {"success": False, "error": str(e)}


@tool
def classify_navigation_links(markdown: str, site_url: str, html: str = "") -> Dict[str, Any]:
    """Classify extracted links into product categories vs accessories.

    Uses AI to analyze navigation structure and identify product categories.

    Args:
        markdown: Markdown content from homepage
        site_url: The URL of the site being analyzed
        html: Optional HTML content for additional context

    Returns:
        Dictionary containing:
        - product_categories: List of product category links
        - accessories: List of accessory/info links
        - other: Other links found
    """
    from homepage_scraper.claude_classifier import classify_markdown_with_claude

    def _classify():
        return classify_markdown_with_claude(
            markdown=markdown,
            site_url=site_url,
            instruction="Identify and extract all product categories and accessories from this e-commerce homepage.",
        )

    try:
        # Retry classification with backoff for API errors
        result = retry_with_backoff(
            _classify,
            max_retries=3,
            initial_delay=2.0,
            backoff_factor=2.0,
            retryable_exceptions=(ConnectionError, TimeoutError, Exception),
            logger_name="langchain_tools",
        )

        return {
            "success": True,
            "product_categories": result.get("product_categories", []),
            "accessories": result.get("info_pages", []),
            "other": [],
        }
    except Exception as e:
        error_msg = str(e)
        if is_retryable_api_error(error_msg):
            logger.error(
                f"Classification failed after retries (API overloaded): {e}")
        else:
            logger.error(f"Classification failed: {e}")
        return {"success": False, "error": error_msg}


@tool
def crawl_category_page(url: str, max_products: int = 50) -> Dict[str, Any]:
    """Crawl a category page to extract product links.

    Tries static crawler first, falls back to Firecrawl crawl if needed.

    Args:
        url: Category page URL
        max_products: Maximum number of products to extract

    Returns:
        Dictionary containing:
        - products: List of product dictionaries with title and url
        - total_found: Total number of products found
        - metadata: Page metadata
        - method: "static" or "firecrawl_crawl"
    """
    from scraper import StaticProductCrawler, CrawlConfig, ProductSelectors

    try:
        # Try static crawler first (fast, no API credits)
        selectors = ProductSelectors(
            card="li.item.product.product-item",
            title=".product-item-link",
            link=".product-item-link",
        )

        config = CrawlConfig(
            start_urls=[url],
            max_pages=1,
            selectors=selectors,
        )

        crawler = StaticProductCrawler(config)
        products_list = crawler.run()

        # If static crawler found products, use them
        if products_list and len(products_list) > 0:
            products = products_list[:max_products]
            logger.info(f"Static crawler found {len(products)} products")
            return {
                "success": True,
                "products": products,
                "total_found": len(products),
                "metadata": {},
                "method": "static",
            }

        # Static crawler found 0 products - fallback to Firecrawl crawl
        logger.warning(
            f"Static crawler found 0 products, trying Firecrawl crawl fallback...")

        result = firecrawl_crawl_category.invoke({
            "url": url,
            "limit": 10  # Limit crawl to 10 pages
        })

        if result.get("success"):
            product_urls = result.get("product_urls", [])
            # Convert URLs to product dictionaries
            products = [{"title": u.split("/")[-1], "url": u}
                        for u in product_urls[:max_products]]

            logger.info(f"Firecrawl crawl found {len(products)} products")
            return {
                "success": True,
                "products": products,
                "total_found": len(products),
                "metadata": {"credits_used": result.get("credits_used", 0)},
                "method": "firecrawl_crawl",
            }
        else:
            logger.error(
                f"Firecrawl crawl fallback also failed: {result.get('error')}")
            return {"success": False, "error": f"Both static and Firecrawl crawl failed"}

    except Exception as e:
        logger.error(f"Category crawl failed: {e}")
        return {"success": False, "error": str(e)}


@tool
def firecrawl_crawl_category(url: str, limit: int = 50) -> Dict[str, Any]:
    """Crawl a category page using Firecrawl to extract all product URLs.

    Uses Firecrawl's crawl endpoint which can handle JavaScript-rendered content
    and automatically follows links to discover all products in a category.

    Args:
        url: Category page URL to crawl
        limit: Maximum number of pages to crawl (default: 50)

    Returns:
        Dictionary containing:
        - success: Boolean indicating if crawl succeeded
        - product_urls: List of product URLs found
        - total_found: Total number of products found
        - credits_used: Firecrawl credits consumed
        - error: Error message if crawl failed
    """
    try:
        from firecrawl import Firecrawl

        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return {"success": False, "error": "FIRECRAWL_API_KEY not set"}

        client = Firecrawl(api_key=api_key)

        # Start crawl with parameters optimized for product extraction
        logger.info(f"Starting Firecrawl crawl for {url}")

        crawl_job = client.start_crawl(
            url,
            limit=limit,
            ignore_query_parameters=True,  # Treat ?page=1 and ?page=2 as same URL
        )

        logger.info(f"Crawl started with job ID: {crawl_job.id}")

        # Wait for crawl to complete (poll status)
        max_wait = 300  # 5 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status = client.get_crawl_status(crawl_job.id)

            if status.status == "completed":
                logger.info(
                    f"Crawl completed: {status.completed}/{status.total} pages")
                break
            elif status.status == "failed":
                return {
                    "success": False,
                    "error": f"Crawl failed"
                }

            # Wait before next poll
            time.sleep(5)
        else:
            # Timeout
            client.cancel_crawl(crawl_job.id)
            return {
                "success": False,
                "error": f"Crawl timed out after {max_wait}s"
            }

        # Extract product URLs from crawl results
        product_urls = []

        # Normalize the base category URL for comparison
        base_url = url.rstrip('/')

        if status.data:
            for doc in status.data:
                # Document is a Pydantic model with metadata attribute
                doc_url = doc.metadata.url if hasattr(
                    doc, "metadata") and hasattr(doc.metadata, "url") else None

                if not doc_url:
                    continue

                # Skip the category page itself
                if doc_url.rstrip('/') == base_url:
                    continue

                # Accept URLs that start with the category URL (child pages)
                # Example: captures product pages like /product-name-123
                if doc_url.startswith(base_url):
                    if doc_url not in product_urls:
                        product_urls.append(doc_url)

        credits_used = status.credits_used if hasattr(
            status, "credits_used") else 0

        logger.info(
            f"Found {len(product_urls)} product URLs, used {credits_used} credits")

        return {
            "success": True,
            "product_urls": product_urls,
            "total_found": len(product_urls),
            "credits_used": credits_used,
            "pages_crawled": status.completed if hasattr(status, "completed") else 0,
        }

    except Exception as e:
        logger.error(f"Firecrawl crawl failed: {e}")
        return {"success": False, "error": str(e)}


# Export all tools as a list for easy registration
ALL_TOOLS = [
    scrape_homepage,
    classify_navigation_links,
    crawl_category_page,
    firecrawl_crawl_category,
]
