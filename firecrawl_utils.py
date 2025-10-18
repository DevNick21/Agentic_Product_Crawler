"""Utility helpers for working with the Firecrawl API."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

_CANONICAL_FORMATS = {
    "markdown": "markdown",
    "md": "markdown",
    "html": "html",
    "rawhtml": "rawHtml",
    "raw_html": "rawHtml",
    "raw-html": "rawHtml",
    "raw": "rawHtml",
    "extract": "extract",
    "json": "json",
    "data": "data",
    "links": "links",
    "metadata": "metadata",
    "summary": "summary",
    "summaries": "summary",
    "structured": "structured",
    "screenshot": "screenshot",
    "screenshotbase64": "screenshotBase64",
    "screenshot_base64": "screenshotBase64",
}


def _iter_formats(formats: Union[str, Sequence[Union[str, Iterable[str]]]]) -> Iterable[str]:
    if isinstance(formats, str):
        for part in formats.split(","):
            candidate = part.strip()
            if candidate:
                yield candidate
        return

    for item in formats:
        if isinstance(item, str):
            candidate = item.strip()
            if candidate:
                yield candidate
        elif isinstance(item, Iterable):
            for candidate in _iter_formats(tuple(item)):
                if candidate:
                    yield candidate
        else:
            text = str(item).strip()
            if text:
                yield text


def normalise_firecrawl_formats(
    formats: Union[str, Sequence[Union[str, Iterable[str]]], None],
    *,
    default: Sequence[str] = ("markdown",),
) -> Tuple[str, ...]:
    """Canonicalise the list of formats passed to Firecrawl.

    The Firecrawl API accepts a handful of string identifiers. Callers across
    the codebase may provide these identifiers in slightly different shapes or
    casings (comma separated string, lists, tuples, nested iterables, etc.).

    This helper collapses the input into a unique, order-preserving tuple of
    canonical identifiers. Unknown identifiers are preserved verbatim so that
    newly introduced Firecrawl response formats continue to work without
    requiring an immediate library update.
    """

    if not formats:
        candidates = list(default)
    else:
        candidates = list(_iter_formats(formats))
        if not candidates:
            candidates = list(default)

    normalised: list[str] = []
    seen: set[str] = set()

    for raw_value in candidates:
        key = raw_value.replace("-", "").replace("_", "").lower()
        canonical = _CANONICAL_FORMATS.get(key, raw_value)
        if canonical not in seen:
            normalised.append(canonical)
            seen.add(canonical)

    return tuple(normalised)
