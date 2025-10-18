# Agentic Product Crawler

An intelligent, autonomous product extraction system that crawls e-commerce websites and extracts structured product data using AI agents. Built with **LangChain** and **LangGraph** for reliable, scalable web scraping.

## Overview

This agentic crawler autonomously navigates e-commerce sites, identifies product categories, and extracts comprehensive product information using a combination of static crawling and AI-powered extraction. The system learns product schemas from actual data and applies them consistently across categories.

## Key Features

- **Extract-First Architecture** - Samples products to infer schemas, ensuring accurate extraction
- **Hybrid Crawling** - Fast static HTML parsing with AI fallback for complex sites
- **Category-Specific Schemas** - Automatically generates tailored schemas per product category
- **Modular Design** - Independent tools that work together seamlessly

## Quick Start

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
FIRECRAWL_API_KEY=fc-your_api_key_here
ANTHROPIC_API_KEY=sk-ant-your_api_key_here
OPENAI_API_KEY=sk-your_api_key_here
```

### Run the Crawler

```bash
python langchain_pipeline.py \
  --start-url "https://www.example.com" \
  --request "Extract all products" \
  --output "outputs/products.json"
```

## Architecture

The crawler operates through 6 autonomous stages:

```text
┌──────────────────────────────────────────────────────────┐
│  STAGE 1: Homepage Scrape                                │
│  Extract homepage content with Firecrawl                 │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│  STAGE 2: Link Classification                            │
│  AI identifies product categories vs info pages          │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│  STAGE 3: Category Crawling                              │
│  Static crawler extracts product URLs (fast)             │
│  Firecrawl fallback for complex pages                    │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│  STAGE 4: Schema Inference                               │
│  Sample first product → Infer category schema with GPT   │
│  Firecrawl /extract + GPT-4o                             │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│  STAGE 5: Product Extraction                             │
│  Scrape markdown → Validate against learned schema       │
│  Firecrawl /scrape + GPT-4.1                             │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│  STAGE 6: Output Finalization                            │
│  Compile results with statistics and metadata            │
└──────────────────────────────────────────────────────────┘
```

## Detailed Workflow

The LangGraph state machine in `langchain_pipeline.py` wires six sequential nodes that pass a shared `PipelineState` dictionary. Each stage both performs work and augments the state for the next stage.

1. **Stage 1 – Homepage Scrape (`stage1_scrape_homepage`)**  
  Invokes `langchain_tools.scrape_homepage.invoke(...)`, which wraps the Firecrawl `/v2/scrape` endpoint to collect markdown, HTML, and metadata. The stage records `homepage_markdown`, `homepage_metadata`, and Firecrawl credit usage before handing control to classification.

2. **Stage 2 – Link Classification (`stage2_classify_links`)**  
  Uses `homepage_scraper.claude_classifier.classify_markdown_with_claude` (Claude 3.5 Haiku) to segment navigation links into `product_categories` and `accessories`. Estimated token cost is logged to the state via `stage2_cost`.

3. **Stage 3 – Category Crawling (`stage3_crawl_categories`)**  
  For each category URL, `langchain_tools.crawl_category_page.invoke(...)` first applies the fast static crawler and falls back to Firecrawl crawl if needed. The stage deduplicates URLs, tracks how many came from static vs Firecrawl sources, and stores per-category collections in `category_products` along with crawl metadata.

4. **Stage 4 – Sample & Infer Schemas (`stage4_extract_and_infer_schemas`)**  
  Pulls the first product URL per category and extracts it with `product_extractor.extract_product_with_schema` using the generic schema from `build_generic_product_schema`. The resulting samples seed an OpenAI `gpt-4o` prompt (via direct HTTPS request) that synthesizes category-specific schemas, which are stored in `category_schemas`. Extraction success, failures, and timeouts are summarized in `stage4_stats`.

5. **Stage 5 – Product Extraction & Validation (`stage5_extract_products`)**  
  Processes up to 10 products per category. Each URL is scraped through the same `scrape_homepage` tool (Firecrawl `/scrape`) and then validated with `schema_validator.openai_schema_applier.validate_with_schema`, which calls the OpenAI Responses API (default model `gpt-4.1`). The stage enforces retries, rate-limit pauses, and aggregates detailed counters in `extraction_stats`.

6. **Stage 6 – Finalize Output (`stage6_finalize_output`)**  
  Compiles all prior artifacts into `final_output`, including the navigation map, product payloads, aggregated usage metrics, and stage-level statistics. The completed state is returned to the caller or written to disk when `--output` is supplied.

### Why Extract-First?

Traditional crawlers require pre-defined schemas. This crawler:

1. **Extracts a sample product** from each category using a generic schema
2. **Analyzes the extracted data** to understand what fields actually exist
3. **Generates a category-specific schema** based on real data
4. **Applies the schema** to extract remaining products with high accuracy

This approach eliminates schema guesswork and adapts to each site's unique structure.

## Project Structure

```text
.
├── langchain_pipeline.py       # Main agentic crawler (use this)
├── langchain_tools.py          # LangChain tool definitions
├── product_extractor.py        # Firecrawl extraction utilities
├── firecrawl_utils.py          # Firecrawl helper functions
│
├── homepage_scraper/           # Stage 2 modules
│   └── claude_classifier.py    # AI link classification
│
├── scraper/                    # Stage 3 modules
│   ├── static_crawler.py       # Fast HTML parser
│   └── config.py
│
├── schema_validator/           # Stage 5 modules
│   ├── openai_schema_applier.py # GPT-4.1 validation
│   └── config.py
│
├── .env                        # API keys (create this)
├── requirements.txt            # Python dependencies
└── outputs/                    # Extraction results
```

## Usage Examples

### Basic Extraction

```bash
python langchain_pipeline.py \
  --start-url "https://example.com" \
  --request "Extract all products with specifications" \
  --output "outputs/example_products.json"
```

### Programmatic Usage

```python
from langchain_pipeline import run_pipeline

result = run_pipeline(
    start_url="https://www.example.com",
    request="Extract all products",
    output_file="outputs/results.json"
)

final_output = result["final_output"]

print(f"Categories: {len(final_output['navigation']['product_categories'])}")
print(f"Products: {len(final_output['products'])}")
stats = final_output['statistics']['extraction_stats']
print(f"Success rate: {stats['succeeded']} / {stats['attempted']}")
```

### Custom Category Limit

```python
# Modify line 422 and 445  in langchain_pipeline.py:
for i, url in enumerate(product_urls[:10], 1):

# Increase the per-category limit, for example:
for i, url in enumerate(product_urls[:50], 1):
```

## Output Format

```json
{
  "request": {
    "start_url": "https://www.example.com",
    "query": "Extract all products"
  },
  "pipeline": {
    "version": "langchain-v3-per-category-schemas",
    "stages_completed": 6
  },
  "stage1_homepage_scrape": {
    "markdown_length": 15299,
    "firecrawl_credits": 5
  },
  "stage2_classification": {
    "product_categories": 6,
    "accessories": 8
  },
  "stage3_category_crawling": {
    "total_products_found": 200,
    "static_crawler_used": 200,
    "firecrawl_fallback_used": 0
  },
  "stage4_schema_generation": {
    "schemas_generated": 6,
    "extraction_stats": {
      "attempted": 6,
      "succeeded": 6,
      "failed": 0,
      "timeouts": 0
    }
  },
  "stage5_product_extraction": {
    "products_extracted": 60,
    "extraction_stats": {
      "attempted": 60,
      "succeeded": 58,
      "failed": 2,
      "rate_limited": 0,
      "validation_errors": 0
    }
  },
  "products": [
    {
      "url": "https://www.example.com/product-1",
      "category_url": "https://www.example.com/category",
      "product": {
        "title": "Product Name",
        "price": 2995,
        "currency": "EUR",
        "description": "Full product description...",
        "specifications": [
          {"name": "Weight", "value": "120 kg"},
          {"name": "Dimensions", "value": "136 x 62 x 100 cm"}
        ],
        "features": ["Feature 1", "Feature 2"],
        "images": ["image1.jpg", "image2.jpg"],
        "availability": "in stock",
        "warranty": "2 years"
      },
      "method": "scrape_and_validate"
    }
  ]
}
```

## Performance

| Metric | Value |
|--------|-------|
| **Stage 1** | ~6s (homepage scrape) |
| **Stage 2** | ~10s (AI classification) |
| **Stage 3** | ~15s for 200 products (static crawler) |
| **Stage 4** | ~60s per category (sample + schema) |
| **Stage 5** | ~3-5s per product (scrape + validate) |
| **Accuracy** | 95%+ with category-specific schemas |
| **Success Rate** | 90%+ (with retry logic) |

## Timeout Configuration

Built-in timeout guards prevent infinite waits:

| Stage | Timeout | Per-Item Timeout |
|-------|---------|------------------|
| Stage 4 | 600s (10 min) | 60s per extraction |
| Stage 5 | 1800s (30 min) | 60s per product |

Timeouts are logged as `[TIMEOUT]` with elapsed time percentages.

## Rate Limiting

Automatic rate limit handling:

- **Stage 4**: 5s pause between categories
- **Stage 5**: 65s pause every 40 requests
- **Retry logic**: 3 attempts with 35s waits for 429 errors
- **Backoff**: Exponential backoff for 502/503 errors

## Logging

The crawler provides comprehensive logging:

```text
[OK] - Successful operations
[FAIL] - Failed operations
[TIME] - Timeout events
[PAUSE] - Rate limit pauses
[WAIT] - Retry waits
```

Progress tracking shows elapsed time vs stage timeout:

```text
[TIME] Elapsed: 45.2s / 600s (7.5%)
```

## Pipeline Configuration

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `FIRECRAWL_API_KEY` | Yes | Web scraping (Stages 1, 3, 4, 5) |
| `ANTHROPIC_API_KEY` | Yes | Link classification (Stage 2) |
| `OPENAI_API_KEY` | Yes | Schema inference & validation (Stages 4, 5) |

### Pipeline Parameters

- `--start-url` - Starting URL for crawling (required)
- `--request` - Description of extraction task (required)
- `--output` - Output JSON file path (default: `outputs/results_test.json`)

## API Credits Usage

Estimated credits per typical e-commerce site (6 categories, 60 products):

| Service | Operation | Credits |
|---------|-----------|---------|
| **Firecrawl** | Homepage scrape | 1 |
| **Firecrawl** | Sample extraction (6) | 30 |
| **Firecrawl** | Product scraping (60) | 60 |
| **Anthropic** | Classification | ~1,000 tokens |
| **OpenAI** | Schema inference (6) | ~12,000 tokens |
| **OpenAI** | Product validation (60) | ~120,000 tokens |

**Total**: ~91 Firecrawl credits + ~133k tokens across APIs

## Advanced Features

### Modular Tools

Each stage uses independent LangChain tools:

```python
from langchain_tools import (
  scrape_homepage,
  classify_navigation_links,
  crawl_category_page,
  firecrawl_crawl_category,
)

# Use individual tools
homepage = scrape_homepage.invoke({"url": "https://example.com"})
categories = classify_navigation_links.invoke({
    "markdown": homepage["markdown"],
    "site_url": "https://example.com"
})
```

### Static Crawler Configuration

The static crawler uses CSS selectors defined in `scraper/config.py`:

```python
STATIC_SELECTORS = {
    "default": {
        "card": "div.product-card",
        "link": "a.product-link",
        "title": "h3.product-title"
    }
}
```

Add site-specific selectors for better accuracy.

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

MIT License

## Support

For issues and questions, please open an issue on the repository.

---

**Built with LangChain & LangGraph** - Autonomous agents for intelligent web scraping
