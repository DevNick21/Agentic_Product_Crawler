"""Lightweight HTML crawler for extracting product links without a browser."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .config import CrawlConfig, ProductSelectors

logger = logging.getLogger(__name__)


@dataclass
class StaticFetchResult:
    url: str
    status_code: Optional[int]
    elapsed_seconds: float
    html: Optional[str]
    error: Optional[str] = None


class StaticProductCrawler:
    """Fetches pages via HTTP requests and extracts product-like links."""

    def __init__(
        self,
        config: CrawlConfig,
        *,
        request_timeout: int = 15,
        max_retries: int = 2,
    ) -> None:
        self.config = config
        self.request_timeout = max(request_timeout, 1)
        self.max_retries = max(max_retries, 0)
        self.page_snapshots: List[Dict[str, str]] = []
        self.last_mode: Optional[str] = None
        self._runtime_user_agent = self.config.pick_user_agent()

    def run(self) -> List[Dict[str, str]]:
        products: List[Dict[str, str]] = []
        for url in self._iter_target_urls():
            fetch = self._fetch(url)
            if not fetch.html:
                logger.warning("Static fetch yielded no HTML for %s", url)
                continue
            batch = self._extract_from_html(fetch.html, url)
            products.extend(batch)
            if self.config.capture_html:
                self.page_snapshots.append({"url": url, "html": fetch.html})
        logger.info(
            "Static crawler collected %d product candidates", len(products))
        self.last_mode = "static"
        return products

    def _iter_target_urls(self) -> Iterable[str]:
        for raw_url in self.config.normalized_start_urls():
            if "{page}" in raw_url:
                for page_number in range(1, self.config.max_pages + 1):
                    yield raw_url.replace("{page}", str(page_number))
            else:
                yield raw_url

    def _fetch(self, url: str) -> StaticFetchResult:
        session = requests.Session()
        session.headers.update({"User-Agent": self._runtime_user_agent})
        attempt = 0
        start_time = time.time()
        response = None
        last_error: Optional[str] = None

        while attempt <= self.max_retries:
            try:
                response = session.get(url, timeout=self.request_timeout)
                if response.status_code and response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                else:
                    break
            except requests.RequestException as exc:
                last_error = str(exc)
            attempt += 1
            if attempt <= self.max_retries:
                backoff = min(2 ** attempt, 5)
                logger.debug(
                    "Retrying %s after failure (%s); sleeping %.1fs",
                    url,
                    last_error,
                    backoff,
                )
                time.sleep(backoff)

        elapsed = time.time() - start_time
        html = response.text if response and response.ok else None
        status = response.status_code if response else None
        return StaticFetchResult(
            url=url,
            status_code=status,
            elapsed_seconds=elapsed,
            html=html,
            error=last_error,
        )

    def _extract_from_html(self, html: str, source_url: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        selectors = self.config.selectors_for_url(source_url)

        extracted = self._extract_with_selectors(soup, source_url, selectors)
        if not extracted:
            logger.debug(
                "Static selectors yielded no products for %s; using fallback anchors",
                source_url,
            )
            extracted = self._fallback_discover(soup, source_url)
        return extracted

    def _extract_with_selectors(
        self,
        soup: BeautifulSoup,
        source_url: str,
        selectors: ProductSelectors,
    ) -> List[Dict[str, str]]:
        card_selector = selectors.card
        if not card_selector:
            return []
        cards = soup.select(card_selector)
        logger.debug(
            "Static parse found %d elements for selector '%s'",
            len(cards),
            card_selector,
        )
        results: List[Dict[str, str]] = []
        for card in cards:
            product = self._extract_product(card, source_url, selectors)
            if product:
                results.append(product)
        return results

    def _extract_product(
        self,
        element,
        source_url: str,
        selectors: ProductSelectors,
    ) -> Optional[Dict[str, str]]:
        link_el = element.select_one(
            selectors.link) if selectors.link else None
        title_el = element.select_one(
            selectors.title) if selectors.title else None
        if not title_el and link_el:
            title_el = link_el

        title = None
        if title_el and getattr(title_el, "get_text", None):
            raw_title = title_el.get_text(strip=True)
            if raw_title:
                title = " ".join(raw_title.split())
        if (not title) and link_el:
            for attr in getattr(selectors, "title_attribute_fallbacks", tuple()):
                if not attr:
                    continue
                candidate = link_el.get(attr)
                if candidate:
                    title = " ".join(candidate.strip().split())
                    break
        if not title and link_el and getattr(link_el, "get_text", None):
            fallback_text = link_el.get_text(strip=True)
            if fallback_text:
                title = " ".join(fallback_text.split())

        url = None
        if link_el:
            href = link_el.get("href")
            if href:
                url = urljoin(source_url, href)

        if not title or not url:
            return None

        return {
            "title": title,
            "url": url,
            "source_url": source_url,
        }

    def _fallback_discover(self, soup: BeautifulSoup, source_url: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        seen: set[str] = set()

        def add_record(record: Optional[Dict[str, str]]) -> None:
            if not record:
                return
            url = record.get("url")
            if not url or url in seen:
                return
            seen.add(url)
            results.append(record)

        selectors = self.config.fallback_link_selectors or ()
        for selector in selectors:
            anchors = soup.select(selector)
            logger.debug(
                "Static fallback selector '%s' yielded %d anchors",
                selector,
                len(anchors),
            )
            for anchor in anchors:
                record = self._record_from_anchor(anchor, source_url)
                if self._looks_like_product(record):
                    add_record(record)

        if not results:
            anchors = soup.select("a[href]")
            logger.debug(
                "Static global anchor scan inspecting %d anchors",
                len(anchors),
            )
            for anchor in anchors:
                record = self._record_from_anchor(anchor, source_url)
                if self._looks_like_product(record):
                    add_record(record)

        return results

    def _record_from_anchor(self, anchor, source_url: str) -> Optional[Dict[str, str]]:
        if anchor is None:
            return None
        text = anchor.get_text(strip=True) if getattr(
            anchor, "get_text", None) else ""
        href = anchor.get("href") if hasattr(anchor, "get") else None
        if not href or not text:
            return None
        normalized_title = " ".join(text.split())
        absolute_url = urljoin(source_url, href)
        return {
            "title": normalized_title,
            "url": absolute_url,
            "source_url": source_url,
        }

    def _looks_like_product(self, record: Optional[Dict[str, str]]) -> bool:
        if not record:
            return False
        title = (record.get("title") or "").lower()
        url = (record.get("url") or "").lower()
        if not title or not url:
            return False
        keywords: Sequence[str] = self.config.product_link_keywords or ()
        if not keywords:
            return True
        return any(keyword in title or keyword in url for keyword in keywords)
