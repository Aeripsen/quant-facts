"""The money boundary, tested end to end: reports carry the advisor
pointer and zero advice phrasing; sources and docs are scanned too; the
what-should-I-do path refuses and points at a licensed advisor."""
from datetime import date, timedelta
from pathlib import Path

import pytest

import backtest as bt
import compliance
from analyze import build_ticker_report, main
from data import Bars
from news import Headline

ROOT = Path(__file__).resolve().parents[1]


def make_bars(n=400, symbol="FAKE"):
    closes = [100.0 * (1.0015 ** i) * (1.0 + 0.01 * ((i // 9) % 2))
              for i in range(n)]
    d0 = date(2024, 1, 1)
    return Bars(
        symbol=symbol, currency="USD",
        dates=[d0 + timedelta(days=i) for i in range(n)],
        open=list(closes), high=[c * 1.01 for c in closes],
        low=[c * 0.99 for c in closes], close=list(closes),
        adjclose=list(closes), volume=[1000.0] * n,
    )


def full_report():
    bars = make_bars()
    news = [Headline("Fake Corp does a thing", "Example Wire",
                     "Tue, 14 Jul 2026", "https://example.com")]
    result = bt.run(bars, fast=20, slow=50)
    return build_ticker_report(bars, False, news, result)


def test_full_report_holds_the_boundary():
    text = full_report()
    low = text.lower()
    # the disclaimers are present and loud
    assert "not a prediction" in low
    assert "not advice" in low
    assert "historical simulation" in low
    # and no advisory phrasing anywhere in tool voice
    compliance.check_output(text)
    for phrase in compliance.BANNED:
        for line in text.splitlines():
            if line.startswith(compliance.QUOTE_PREFIX):
                continue
            assert phrase not in line.lower()


def test_quoted_headline_is_exempt_but_tool_voice_is_not():
    spicy = "Analysts say you should " + "buy Fake Corp"
    # quoted: a fact about what a third party published; passes
    compliance.check_output(compliance.QUOTE_PREFIX + spicy)
    # the same words in the tool's own voice: hard failure
    with pytest.raises(compliance.ComplianceError):
        compliance.check_output(spicy)


def test_guard_blocks_bad_output_instead_of_printing_it():
    bad = "The data suggests it is time to " + "buy this ticker."
    with pytest.raises(compliance.ComplianceError):
        compliance.guard(bad)


def test_report_carries_no_recommendation_vocabulary():
    text = full_report().lower()
    for word in ("recommend", "recommendation", "advises", "you should",
                 "rating:", "target price"):
        assert word not in text


def test_sources_and_docs_carry_no_advice_phrasing():
    files = sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("tests/*.py"))
    # every doc at the root, not a hand-kept list: a new CHANGELOG or
    # CITATION is scanned the day it is added
    files += sorted(ROOT.glob("*.md")) + sorted(ROOT.glob("*.cff"))
    assert ROOT / "README.md" in files and ROOT / "RESEARCH.md" in files
    for path in files:
        low = path.read_text(encoding="utf-8").lower()
        for phrase in compliance.BANNED:
            assert phrase not in low, f"{phrase!r} appears in {path.name}"


def test_what_should_i_do_is_refused(capsys):
    rc = main(["should", "I", "keep", "AAPL"])
    out = capsys.readouterr().out.lower()
    assert rc == 2
    assert "licensed-advisor question" in out
    assert "does not answer it" in out


def test_question_detection():
    assert compliance.looks_like_question("should")
    assert compliance.looks_like_question("AAPL?")
    assert compliance.looks_like_question("buy")
    assert not compliance.looks_like_question("SPY")
    assert not compliance.looks_like_question("VFV.TO")
    assert not compliance.looks_like_question("^GSPC")


def test_advisor_line_ends_every_report_path():
    assert "licensed-advisor" in compliance.ADVISOR_LINE
    # the CLI prints ADVISOR_LINE last; build_ticker_report output itself
    # must already be clean without relying on it
    compliance.check_output(full_report())
