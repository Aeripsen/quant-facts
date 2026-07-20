"""Headlines for a ticker from free RSS feeds, reported as facts.

Each item is (headline, source, date, link), verbatim. The tool adds no
sentiment score, no rating, no read on whether news is "good": deciding
what a headline means for money is exactly the judgment this tool refuses
to automate.

Feed content is third-party and treated as hostile input (parsed with
defusedxml, tags stripped, control characters removed, length-capped).
In rendered reports every headline line carries compliance.QUOTE_PREFIX,
marking it as quoted third-party text, not the tool's voice.

Sources: Yahoo Finance per-ticker RSS first, Google News RSS as fallback.
Both are free and keyless.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

import requests
from defusedxml import ElementTree as SafeET

from compliance import QUOTE_PREFIX

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_YAHOO = ("https://feeds.finance.yahoo.com/rss/2.0/headline"
          "?s={symbol}&region=US&lang=en-US")
_GOOGLE = ("https://news.google.com/rss/search"
           "?q={symbol}+stock&hl=en-CA&gl=CA&ceid=CA:en")

_TAG_RE = re.compile(r"<[^>]+>")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")
_MAX_TITLE = 300


@dataclass
class Headline:
    title: str
    source: str
    date: str
    link: str


def clean_text(raw: str) -> str:
    """Sanitize third-party text: unescape, strip tags and control chars,
    collapse whitespace, cap length."""
    text = html.unescape(raw or "")
    text = _TAG_RE.sub(" ", text)
    text = _CTRL_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:_MAX_TITLE]


def parse_rss(xml_bytes: bytes, default_source: str) -> list[Headline]:
    """Parse an RSS 2.0 feed defensively. Items missing a title are
    dropped; missing source falls back to the feed name."""
    root = SafeET.fromstring(xml_bytes)
    out: list[Headline] = []
    for item in root.iter("item"):
        title = clean_text(item.findtext("title") or "")
        if not title:
            continue
        source_tag = clean_text(item.findtext("source") or "")
        source = source_tag or default_source
        # Google News appends " - Source" to the title; strip exactly that
        # suffix and nothing else, so a natural " - " in a headline stays.
        if source_tag and title.endswith(f" - {source_tag}"):
            title = title[: -(len(source_tag) + 3)].rstrip()
        out.append(Headline(
            title=title,
            source=source,
            date=clean_text(item.findtext("pubDate") or ""),
            link=clean_text(item.findtext("link") or ""),
        ))
    return out


def fetch_news(symbol: str, limit: int = 8, timeout: int = 20) -> list[Headline]:
    """Yahoo per-ticker feed first, Google News fallback. Returns [] only
    when both feeds are empty; network errors from the last source raise."""
    feeds = (
        (_YAHOO.format(symbol=symbol), "Yahoo Finance"),
        (_GOOGLE.format(symbol=symbol), "Google News"),
    )
    last_err: Exception | None = None
    for url, name in feeds:
        try:
            resp = requests.get(url, headers=_UA, timeout=timeout)
            resp.raise_for_status()
            items = parse_rss(resp.content, name)
            if items:
                return items[:limit]
        except Exception as exc:  # try the next feed, keep the failure visible
            last_err = exc
    if last_err is not None:
        raise RuntimeError(f"news feeds failed for {symbol!r}: {last_err}")
    return []


def render_news(symbol: str, items: list[Headline]) -> str:
    """Factual block: what was published, where, when. No scoring."""
    lines = [
        f"NEWS for {symbol} (headlines as published; inclusion is not "
        "endorsement; the tool adds no rating or sentiment)",
    ]
    if not items:
        lines.append("  no headlines returned by the feeds right now")
    for h in items:
        lines.append(f"{QUOTE_PREFIX}{h.title}")
        meta = "  ".join(p for p in (h.source, h.date) if p)
        lines.append(f"    [{meta}]")
    return "\n".join(lines)
