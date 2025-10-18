"""LangChain-based product extraction pipeline - with Per-Category Schemas.

This implements a hybrid approach:
1. Homepage Scrape (Firecrawl) → Get all links
2. Classify (Claude) → Identify product categories
3. Crawl Categories (Static + Firecrawl fallback) → Get product URLs
4. Generate Schemas (GPT per category) → Create category-specific schemas
5. Extract Products (Firecrawl /extract with category schemas) → Get structured data
6. Finalize Output

Key differences from main pipeline:
- Generates per-category schemas using GPT (not generic schema)
- Uses category-specific schemas for extraction (better accuracy)

"""

import json
import logging
import warnings
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Suppress pydantic warnings
warnings.filterwarnings(
    "ignore", message=".*Field name.*shadows an attribute.*")

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


# Define the pipeline state
class PipelineState(TypedDict):
    """State for the 6-stage pipeline."""
    # Input
    start_url: str
    request: str

    # Stage 1: Homepage Scrape
    homepage_markdown: str
    homepage_metadata: Dict[str, Any]
    stage1_credits: int

    # Stage 2: Classification
    product_categories: List[Dict[str, Any]]
    accessories: List[Dict[str, Any]]
    stage2_cost: Dict[str, Any]

    # Stage 3: Category Crawling
    category_products: Dict[str, List[str]]  # category_url -> [product_urls]
    all_product_urls: List[str]
    stage3_metadata: Dict[str, Any]

    # Stage 4: Schema Generation
    category_schemas: Dict[str, Dict[str, Any]]  # category_url -> schema
    stage4_cost: Dict[str, Any]

    # Stage 5: Product Extraction
    product_details: List[Dict[str, Any]]
    extraction_stats: Dict[str, int]
    stage5_credits: int

    # Stage 6: Output
    final_output: Dict[str, Any]
    errors: List[str]
    current_stage: str
    total_elapsed_seconds: float


def create_pipeline() -> StateGraph:
    """Create the pipeline with per-category schemas."""

    def stage1_scrape_homepage(state: PipelineState) -> PipelineState:
        """Stage 1: Scrape homepage with Firecrawl."""
        import time
        logger.info("=" * 60)
        logger.info("STAGE 1: Homepage Scrape (Firecrawl)")
        logger.info("=" * 60)

        start_time = time.time()

        from langchain_tools import scrape_homepage

        result = scrape_homepage.invoke({
            "url": state["start_url"],
            "formats": "markdown,html"
        })

        if result.get("success"):
            state["homepage_markdown"] = result.get("markdown", "")
            state["homepage_metadata"] = result.get("metadata", {})
            state["stage1_credits"] = result["metadata"].get(
                "firecrawl", {}).get("creditsUsed", 0)
            state["current_stage"] = "classify"

            elapsed = time.time() - start_time
            logger.info(
                f"[OK] Homepage scraped: {len(state['homepage_markdown'])} chars")
            logger.info(f"  Credits used: {state['stage1_credits']}")
            logger.info(f"  Time: {elapsed:.1f}s")
        else:
            state["errors"].append(
                f"Homepage scrape failed: {result.get('error')}")
            logger.error(
                f"[FAIL] Homepage scrape failed: {result.get('error')}")

        return state

    def stage2_classify_links(state: PipelineState) -> PipelineState:
        """Stage 2: Classify navigation links with Claude."""
        import time
        logger.info("=" * 60)
        logger.info("STAGE 2: Classify Links (Claude)")
        logger.info("=" * 60)

        start_time = time.time()

        from homepage_scraper.claude_classifier import classify_markdown_with_claude

        try:
            result = classify_markdown_with_claude(
                markdown=state["homepage_markdown"],
                site_url=state["start_url"]
            )

            state["product_categories"] = result.get("product_categories", [])
            state["accessories"] = result.get("info_pages", [])
            state["current_stage"] = "crawl"
            state["stage2_cost"] = {"model": "claude-3-5-haiku",
                                    "estimated_tokens": len(state["homepage_markdown"]) // 4}

            elapsed = time.time() - start_time
            logger.info(f"[OK] Classified links:")
            logger.info(
                f"  Product categories: {len(state['product_categories'])}")
            logger.info(f"  Accessories: {len(state['accessories'])}")
            logger.info(f"  Time: {elapsed:.1f}s")
        except Exception as e:
            error_msg = f"Classification failed: {e}"
            state["errors"].append(error_msg)
            logger.error(error_msg)

        return state

    def stage3_crawl_categories(state: PipelineState) -> PipelineState:
        """Stage 3: Crawl category pages with static crawler + Firecrawl fallback."""
        import time
        logger.info("=" * 60)
        logger.info("STAGE 3: Crawl Categories (Static + Fallback)")
        logger.info("=" * 60)

        start_time = time.time()

        from langchain_tools import crawl_category_page

        category_products = {}
        all_products = []
        total_static = 0
        total_firecrawl = 0
        total_credits = 0

        for i, category in enumerate(state["product_categories"], 1):
            category_url = category.get("url")
            category_title = category.get("title", category_url)

            if not category_url:
                continue

            logger.info(
                f"[{i}/{len(state['product_categories'])}] Crawling: {category_title}")

            result = crawl_category_page.invoke(
                {"url": category_url, "max_products": 100})

            if result.get("success"):
                products = result.get("products", [])
                product_urls = [p["url"] for p in products if "url" in p]

                category_products[category_url] = product_urls
                all_products.extend(product_urls)

                method = result.get("method", "unknown")
                if method == "static":
                    total_static += len(product_urls)
                elif method == "firecrawl_crawl":
                    total_firecrawl += len(product_urls)
                    total_credits += result.get("metadata",
                                                {}).get("credits_used", 0)

                logger.info(
                    f"  [OK] Found {len(product_urls)} products (method: {method})")
            else:
                logger.warning(f"  [FAIL] Failed: {result.get('error')}")

        state["category_products"] = category_products
        state["all_product_urls"] = list(set(all_products))  # Deduplicate
        state["stage3_metadata"] = {
            "static_products": total_static,
            "firecrawl_products": total_firecrawl,
            "credits_used": total_credits
        }
        state["current_stage"] = "schemas"

        elapsed = time.time() - start_time
        logger.info(
            f"[OK] Total products found: {len(state['all_product_urls'])}")
        logger.info(f"  Static: {total_static}, Firecrawl: {total_firecrawl}")
        logger.info(f"  Credits: {total_credits}")
        logger.info(f"  Time: {elapsed:.1f}s")

        return state

    def stage4_extract_and_infer_schemas(state: PipelineState) -> PipelineState:
        """Stage 4: Extract first product from each category, then infer schema from data."""
        import time
        import json
        logger.info("=" * 60)
        logger.info("STAGE 4: Extract Samples & Infer Schemas")
        logger.info("=" * 60)

        start_time = time.time()
        stage_timeout = 600  # 10 minutes max for entire stage

        from product_extractor import extract_product_with_schema, build_generic_product_schema
        import re

        category_schemas = {}
        category_samples = {}  # Store extracted samples
        total_samples = 0
        total_tokens = 0
        extraction_stats = {"attempted": 0,
                            "succeeded": 0, "failed": 0, "timeouts": 0}
        # Use generic schema for initial extraction
        generic_schema = build_generic_product_schema()

        logger.info(
            f"Stage timeout: {stage_timeout}s | Categories to process: {len(state['category_products'])}")

        # Step 1: Extract first product from each category using Firecrawl /extract
        for i, (category_url, product_urls) in enumerate(state["category_products"].items(), 1):
            # Check stage timeout
            elapsed = time.time() - start_time
            if elapsed > stage_timeout:
                logger.error(
                    f"[TIME]  Stage timeout reached ({stage_timeout}s) - stopping extraction")
                break

            if not product_urls:
                continue

            # Find category title
            category_title = "Unknown"
            for cat in state["product_categories"]:
                if cat.get("url") == category_url:
                    category_title = cat.get("title", category_url)
                    break

            logger.info(
                f"[{i}/{len(state['category_products'])}] Extracting sample from: {category_title}")
            logger.info(
                f"  URL: {product_urls[0][:80]}... | Elapsed: {elapsed:.1f}s/{stage_timeout}s")
            extraction_stats["attempted"] += 1

            # Get first product URL to extract
            first_url = product_urls[0] if product_urls else None
            if not first_url or not first_url.startswith("http"):
                logger.warning(f"  [FAIL] Invalid URL")
                extraction_stats["failed"] += 1
                continue

            # Extract using Firecrawl with generic schema (with timeout)
            extraction_start = time.time()
            try:
                result = extract_product_with_schema(
                    url=first_url,
                    schema=generic_schema,
                    timeout=90  # 90s timeout per extraction
                )

                extraction_time = time.time() - extraction_start

                if result.get("data"):
                    extracted_data = result["data"]
                    category_samples[category_url] = {
                        "url": first_url,
                        "data": extracted_data,
                        "category_title": category_title
                    }
                    total_samples += 1
                    extraction_stats["succeeded"] += 1
                    logger.info(
                        f"  [OK] Extracted {len(extracted_data)} fields in {extraction_time:.1f}s")
                else:
                    error = result.get("error", "Unknown error")
                    if "timed out" in str(error).lower() or "timeout" in str(error).lower():
                        extraction_stats["timeouts"] += 1
                        logger.warning(
                            f"  [TIME]  Extraction timeout after {extraction_time:.1f}s")
                    else:
                        extraction_stats["failed"] += 1
                        logger.warning(f"  [FAIL] Failed: {str(error)[:100]}")

            except Exception as e:
                extraction_stats["failed"] += 1
                logger.warning(f"  [FAIL] Exception: {str(e)[:100]}")

            time.sleep(5)  # Rate limit pause

        # Step 2: Infer schemas from extracted samples
        logger.info("")
        logger.info("Inferring schemas from extracted samples...")

        for category_url, sample in category_samples.items():
            category_title = sample["category_title"]
            extracted_data = sample["data"]

            logger.info(f"  Generating schema for {category_title}...")

            try:
                # Ask GPT to infer schema from the extracted data
                import os
                import requests

                api_key = os.getenv("OPENAI_API_KEY")
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [{
                            "role": "user",
                            "content": f"""Based on this extracted product data, generate a reusable schema.

Product type: {category_title}
Extracted data:
{json.dumps(extracted_data, indent=2)[:10000]}

Create a schema in this format:
{{
  "schema_version": "1.0",
  "product_type": "{category_title}",
  "fields": {{
    "field_name": {{"type": "string|number|array|object", "required": true|false, "description": "..."}}
  }}
}}

Return ONLY the schema JSON."""
                        }],
                        "max_tokens": 2000
                    },
                    timeout=60
                )

                if response.ok:
                    schema_text = response.json(
                    )["choices"][0]["message"]["content"]
                    json_match = re.search(r'\{[\s\S]*\}', schema_text)
                    if json_match:
                        schema = json.loads(json_match.group())
                        category_schemas[category_url] = schema
                        logger.info(
                            f"  [OK] Schema created with {len(schema.get('fields', {}))} fields")
                    else:
                        logger.warning(f"  [FAIL] Could not parse schema JSON")
                else:
                    logger.warning(
                        f"  [FAIL] GPT request failed: {response.status_code}")

            except Exception as e:
                logger.warning(f"  [FAIL] Schema inference failed: {e}")

        state["category_schemas"] = category_schemas
        state["stage4_cost"] = {"model": "gpt-4",
                                "estimated_tokens": total_tokens}
        state["stage4_stats"] = extraction_stats
        state["current_stage"] = "extract"

        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 4 COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"[OK] Extraction: {extraction_stats['succeeded']}/{extraction_stats['attempted']} succeeded")
        logger.info(
            f"  Failures: {extraction_stats['failed']} | Timeouts: {extraction_stats['timeouts']}")
        logger.info(
            f"[OK] Schemas: {len(category_schemas)} generated from {total_samples} samples")
        logger.info(
            f"[TIME]  Time: {elapsed:.1f}s / {stage_timeout}s ({elapsed/stage_timeout*100:.1f}%)")

        return state

    def stage5_extract_products(state: PipelineState) -> PipelineState:
        """Stage 5: Extract products using Firecrawl /scrape + Claude schema validation."""
        import time
        logger.info("=" * 60)
        logger.info("STAGE 5: Extract Products (Scrape + Schema Validation)")
        logger.info("=" * 60)

        start_time = time.time()
        stage_timeout = 1800  # 30 minutes max for entire stage

        from langchain_tools import scrape_homepage
        from schema_validator.openai_schema_applier import validate_with_schema

        product_details = []
        stats = {"attempted": 0, "succeeded": 0, "failed": 0,
                 "rate_limited": 0, "validation_errors": 0}
        total_credits = 0
        request_count = 0

        # TODO: Remove this limit in production
        total_products = sum(len(urls[:2])
                             for urls in state["category_products"].values())
        logger.info(
            f"Stage timeout: {stage_timeout}s | Products to extract: {total_products}")

        # Build generic fallback schema once
        from product_extractor import build_generic_product_schema
        generic_fallback_schema = build_generic_product_schema()

        # Group products by category for schema application
        for category_url, product_urls in state["category_products"].items():
            schema = state["category_schemas"].get(category_url)

            # Use fallback generic schema if category schema is missing
            if not schema:
                logger.warning(f"  No schema found for {category_url}, using generic fallback schema")
                schema = generic_fallback_schema
            else:
                logger.info(
                    f"  Using schema for {category_url}: {schema.get('product_type', 'Unknown')} with {len(schema.get('fields', {}))} fields")

            # Limit to 2 per category for testing
            # TODO: Remove this limit in production
            for i, url in enumerate(product_urls[:2], 1):
                # Check stage timeout
                elapsed = time.time() - start_time
                if elapsed > stage_timeout:
                    logger.error(
                        f"[TIME]  Stage timeout reached ({stage_timeout}s) - stopping extraction")
                    break

                logger.info(
                    f"  [{stats['attempted'] + 1}/{total_products}] Processing: {url[:80]}...")
                logger.info(
                    f"  [TIME]  Elapsed: {elapsed:.1f}s / {stage_timeout}s ({elapsed/stage_timeout*100:.1f}%)")
                stats["attempted"] += 1
                request_count += 1

                # Rate limit prevention: pause every 40 requests
                if request_count % 40 == 0:
                    wait_time = 65
                    logger.info(
                        f"  [PAUSE]  Pausing {wait_time}s to avoid rate limit ({request_count} requests made)")
                    time.sleep(wait_time)

                # Step 1: Scrape the page
                max_retries = 3
                markdown = None

                for attempt in range(max_retries):
                    scrape_result = scrape_homepage.invoke({"url": url})

                    error_msg = scrape_result.get("error", "")
                    if "429" in str(error_msg) or "Rate limit" in str(error_msg):
                        stats["rate_limited"] += 1
                        if attempt < max_retries - 1:
                            wait_time = 35
                            logger.warning(
                                f"    [WAIT] Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.warning(
                                f"    [FAIL] Rate limit exceeded, skipping")
                            break

                    if scrape_result.get("markdown"):
                        markdown = scrape_result["markdown"]
                        total_credits += 1
                        break
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(5)
                            continue
                        else:
                            logger.warning(
                                f"    [FAIL] Scrape failed: {error_msg}")
                            break

                if not markdown:
                    stats["failed"] += 1
                    continue

                # Step 2: Apply schema validation with Claude
                try:
                    result = validate_with_schema(
                        schema=schema,
                        page_url=url,
                        text_content=markdown
                    )

                    # validate_with_schema returns {'status': ..., 'product': {...}, ...}
                    # Extract the actual product data
                    product_data = result.get("product", {}) if result else {}

                    if product_data and product_data.get("title"):
                        product_details.append({
                            "url": url,
                            "category_url": category_url,
                            "product": product_data,
                            "method": "scrape_and_validate",
                        })
                        stats["succeeded"] += 1
                        logger.info(
                            f"    [OK] Extracted: {product_data.get('title', 'Unknown')}")
                    else:
                        stats["failed"] += 1
                        if product_data:
                            logger.warning(
                                f"    [FAIL] No title - product_data keys: {list(product_data.keys())}")
                            logger.warning(
                                f"    [FAIL] Status: {result.get('status')}, missing_fields: {result.get('missing_fields')}")
                        else:
                            logger.warning(
                                f"    [FAIL] Empty product data - status: {result.get('status')}")

                except Exception as e:
                    stats["validation_errors"] += 1
                    stats["failed"] += 1
                    error_str = str(e)[:200]
                    logger.warning(
                        f"    [FAIL] Validation exception: {error_str}")

                # Small delay between requests
                time.sleep(0.5)

        state["product_details"] = product_details
        state["extraction_stats"] = stats
        state["stage5_credits"] = total_credits
        state["current_stage"] = "finalize"

        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 5 COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"[OK] Extracted: {stats['succeeded']}/{stats['attempted']} succeeded ({stats['succeeded']/max(stats['attempted'],1)*100:.1f}%)")
        logger.info(
            f"  Rate limited: {stats['rate_limited']} | Validation errors: {stats['validation_errors']}")
        logger.info(f"  Firecrawl credits: ~{total_credits}")
        logger.info(
            f"[TIME]  Time: {elapsed:.1f}s / {stage_timeout}s ({elapsed/stage_timeout*100:.1f}%)")

        return state

    def stage6_finalize_output(state: PipelineState) -> PipelineState:
        """Stage 6: Build final output."""
        logger.info("=" * 60)
        logger.info("STAGE 6: Finalize Output")
        logger.info("=" * 60)

        state["final_output"] = {
            "request": {
                "start_url": state["start_url"],
                "query": state["request"]
            },
            "pipeline": {
                "version": "langchain-v3-per-category-schemas",
                "stages_completed": 6
            },
            "stage1_homepage_scrape": {
                "markdown_length": len(state.get("homepage_markdown", "")),
                "firecrawl_credits": state.get("stage1_credits", 0)
            },
            "stage2_classification": {
                "product_categories": len(state.get("product_categories", [])),
                "accessories": len(state.get("accessories", [])),
                "cost": state.get("stage2_cost", {})
            },
            "stage3_category_crawling": {
                "total_products_found": len(state.get("all_product_urls", [])),
                "products_by_category": {
                    cat.get("title", cat.get("url", "Unknown")): len(state["category_products"].get(cat.get("url"), []))
                    for cat in state.get("product_categories", [])
                },
                "metadata": state.get("stage3_metadata", {})
            },
            "stage4_schema_generation": {
                "schemas_generated": len(state.get("category_schemas", {})),
                "cost": state.get("stage4_cost", {})
            },
            "stage5_product_extraction": {
                "products_extracted": state.get("extraction_stats", {}).get("succeeded", 0),
                "firecrawl_credits": state.get("stage5_credits", 0),
                "stats": state.get("extraction_stats", {})
            },
            "products": state.get("product_details", []),
            "usage_summary": {
                "total_firecrawl_credits": state.get("stage1_credits", 0) + state.get("stage3_metadata", {}).get("credits_used", 0) + state.get("stage5_credits", 0),
                "claude_calls": 1,
                "gpt_calls": len(state.get("category_schemas", {}))
            },
            "errors": state.get("errors", [])
        }

        state["current_stage"] = "complete"

        logger.info("[OK] Output finalized")
        logger.info(
            f"  Products extracted: {state.get('extraction_stats', {}).get('succeeded', 0)}")
        logger.info(
            f"  Total credits: {state['final_output']['usage_summary']['total_firecrawl_credits']}")

        return state

    # Build the graph
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("scrape_homepage", stage1_scrape_homepage)
    workflow.add_node("classify", stage2_classify_links)
    workflow.add_node("crawl", stage3_crawl_categories)
    workflow.add_node("schemas", stage4_extract_and_infer_schemas)
    workflow.add_node("extract", stage5_extract_products)
    workflow.add_node("finalize", stage6_finalize_output)

    # Define edges
    workflow.set_entry_point("scrape_homepage")
    workflow.add_edge("scrape_homepage", "classify")
    workflow.add_edge("classify", "crawl")
    workflow.add_edge("crawl", "schemas")
    workflow.add_edge("schemas", "extract")
    workflow.add_edge("extract", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def run_pipeline(start_url: str, request: str, output_file: str = None):
    """Run the V3 pipeline."""
    import time

    logger.info("=" * 60)
    logger.info("LANGCHAIN PRODUCT EXTRACTION PIPELINE V3")
    logger.info("Per-Category Schemas with Static Crawler")
    logger.info("=" * 60)

    start_time = time.time()

    pipeline = create_pipeline()

    initial_state = PipelineState(
        start_url=start_url,
        request=request,
        homepage_markdown="",
        homepage_metadata={},
        stage1_credits=0,
        product_categories=[],
        accessories=[],
        stage2_cost={},
        category_products={},
        all_product_urls=[],
        stage3_metadata={},
        category_schemas={},
        stage4_cost={},
        product_details=[],
        extraction_stats={},
        stage5_credits=0,
        final_output={},
        errors=[],
        current_stage="scrape",
        total_elapsed_seconds=0.0,
    )

    try:
        final_state = pipeline.invoke(initial_state)

        final_state["total_elapsed_seconds"] = time.time() - start_time
        final_state["final_output"]["total_elapsed_seconds"] = final_state["total_elapsed_seconds"]

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_state["final_output"], f,
                          indent=2, ensure_ascii=False)
            logger.info(f"\n[OK] Results saved to: {output_file}")

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"Products extracted: {final_state.get('extraction_stats', {}).get('succeeded', 0)}")
        logger.info(
            f"Firecrawl credits used: {final_state['final_output']['usage_summary']['total_firecrawl_credits']}")
        logger.info(f"Time: {final_state['total_elapsed_seconds']:.1f}s")
        logger.info(f"Errors: {len(final_state.get('errors', []))}")
        logger.info("=" * 60)

        return final_state

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the V3 product extraction pipeline")
    parser.add_argument("--start-url", required=True,
                        help="Homepage URL to scrape")
    parser.add_argument("--request", required=True,
                        help="Extraction request description")
    parser.add_argument(
        "--output", default="outputs/v3_results.json", help="Output JSON file path")

    args = parser.parse_args()

    run_pipeline(
        start_url=args.start_url,
        request=args.request,
        output_file=args.output,
    )
