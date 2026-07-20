"""News parsing on canned RSS. Feeds are hostile input; titles get
sanitized, and rendered headlines are marked as quoted text."""
from compliance import QUOTE_PREFIX
from news import clean_text, parse_rss, render_news

YAHOO_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item><title>Fake Corp reports quarterly results - a record</title>
<link>https://example.com/1</link>
<pubDate>Tue, 14 Jul 2026 12:00:00 +0000</pubDate></item>
<item><title>  Fake Corp &amp; partner announce&#13; new&#x09;product </title>
<link>https://example.com/2</link>
<pubDate>Tue, 14 Jul 2026 09:00:00 +0000</pubDate></item>
<item><title></title><link>https://example.com/skip</link></item>
</channel></rss>"""

GOOGLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item><title>Fake Corp shares move after earnings - Example Wire</title>
<link>https://news.example/3</link>
<pubDate>Mon, 13 Jul 2026 20:00:00 +0000</pubDate>
<source url="https://examplewire.com">Example Wire</source></item>
</channel></rss>"""

HOSTILE_RSS = (b"<?xml version='1.0'?><rss version='2.0'><channel>"
               b"<item><title>&lt;script&gt;alert(1)&lt;/script&gt;"
               b"Breaking news</title><link>x</link></item>"
               b"</channel></rss>")


def test_yahoo_shape_keeps_natural_dashes_in_titles():
    items = parse_rss(YAHOO_RSS, "Yahoo Finance")
    assert len(items) == 2  # empty-title item dropped
    assert items[0].title == "Fake Corp reports quarterly results - a record"
    assert items[0].source == "Yahoo Finance"
    assert items[0].date.startswith("Tue, 14 Jul 2026")


def test_yahoo_title_entities_and_whitespace_cleaned():
    items = parse_rss(YAHOO_RSS, "Yahoo Finance")
    assert items[1].title == "Fake Corp & partner announce new product"


def test_google_shape_strips_exact_source_suffix():
    items = parse_rss(GOOGLE_RSS, "Google News")
    assert items[0].title == "Fake Corp shares move after earnings"
    assert items[0].source == "Example Wire"


def test_hostile_title_sanitized():
    items = parse_rss(HOSTILE_RSS, "X")
    assert "<script>" not in items[0].title
    assert "Breaking news" in items[0].title
    # raw control characters cannot survive XML parsing, but clean_text
    # also strips them from any path that bypasses the parser
    assert clean_text("Breaking\x07 news\x00") == "Breaking news"


def test_clean_text_caps_length():
    assert len(clean_text("a" * 5000)) == 300


def test_render_marks_headlines_as_quotes_and_adds_no_rating():
    items = parse_rss(YAHOO_RSS, "Yahoo Finance")
    text = render_news("FAKE", items)
    headline_lines = [
        line for line in text.splitlines() if line.startswith(QUOTE_PREFIX)]
    assert len(headline_lines) == 2
    low = text.lower()
    assert "not endorsement" in low
    assert "no rating" in low
    for scored in ("bullish", "bearish", "positive", "negative", "score"):
        assert scored not in low
