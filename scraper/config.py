"""Configuration primitives for product crawling."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


@dataclass
class ProductSelectors:
    """CSS selectors used to extract product titles and URLs."""

    card: str = "li.item.product.product-item"
    title: str = ".product-item-link"
    link: str = ".product-item-link"
    title_attribute_fallbacks: Sequence[str] = (
        "data-name", "aria-label", "title")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ProductSelectors":
        if not data:
            return cls()
        valid_keys = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items()
                    if key in valid_keys}
        return cls(**filtered)


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36"
)


@dataclass
class CrawlConfig:
    """High-level crawler configuration for pagination and extraction."""

    start_urls: List[str]
    max_pages: int = 1
    wait_for_network_idle: bool = True
    settle_timeout_ms: int = 1000
    headless: bool = True
    user_agent: Optional[str] = DEFAULT_USER_AGENT
    user_agent_pool: Optional[Sequence[str]] = None
    locale: str = "en-US"
    locale_pool: Optional[Sequence[str]] = None
    selectors: ProductSelectors = field(default_factory=ProductSelectors)
    selector_overrides: Dict[str, ProductSelectors] = field(
        default_factory=dict)
    fallback_link_selectors: Sequence[str] = (
        "header a[href]",
        "nav a[href]",
        "main a[href]",
        ".menu a[href]",
        "[role='navigation'] a[href]",
    )
    product_link_keywords: Sequence[str] = (
        "product",
        "shop",
        "catalog",
        "item",
        "category",
        "collection",
        "detail"
    )
    viewport_width: int = 1280
    viewport_height: int = 720
    viewport_pool: Optional[Sequence[Tuple[int, int]]] = None
    throttle_seconds: float = 1.5
    throttle_jitter_range: Tuple[float, float] = (0.5, 1.5)
    capture_html: bool = False

    def normalized_start_urls(self) -> List[str]:
        """Return de-duplicated start URLs while preserving order."""

        seen = set()
        ordered: List[str] = []
        for url in self.start_urls:
            if url not in seen:
                seen.add(url)
                ordered.append(url)
        return ordered

    def pick_user_agent(self) -> str:
        """Return a runtime user-agent with optional rotation."""

        if self.user_agent_pool:
            return random.choice(tuple(self.user_agent_pool))
        if self.user_agent:
            return self.user_agent
        return DEFAULT_USER_AGENT

    def pick_locale(self) -> str:
        """Return the locale to use for this crawl."""

        if self.locale_pool:
            return random.choice(tuple(self.locale_pool))
        return self.locale

    def pick_viewport(self) -> Tuple[int, int]:
        """Return a viewport size, optionally randomized from presets."""

        if self.viewport_pool:
            return random.choice(tuple(self.viewport_pool))
        return self.viewport_width, self.viewport_height

    def resolve_throttle_delay(self) -> float:
        """Calculate the next throttle delay using configured jitter."""

        base = max(self.throttle_seconds, 0)
        low, high = self.throttle_jitter_range
        if high < low:
            low, high = high, low
        if base == 0 or high == 0:
            return base
        factor = random.uniform(low, high) if high != low else low
        return max(base * factor, 0)

    def selectors_for_url(self, url: str) -> ProductSelectors:
        domain = urlparse(url).netloc.lower()
        for override_domain, override_selectors in self.selector_overrides.items():
            normalized = override_domain.lower()
            if domain == normalized or domain.endswith(f".{normalized}"):
                return override_selectors
        return self.selectors

    def register_selector_override(self, domain: str, selectors: ProductSelectors) -> None:
        if domain:
            self.selector_overrides[domain.lower()] = selectors
