"""Scraping utilities for extracting product navigation data."""

from .config import CrawlConfig, ProductSelectors
from .static_crawler import StaticProductCrawler

__all__ = [
    "CrawlConfig",
    "ProductSelectors",
    "StaticProductCrawler",
]
