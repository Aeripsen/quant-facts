"""The money boundary, enforced in code, not just in docs.

This tool reports facts and standard indicators. It never says what to do
with money. Every rendered report passes through guard() before it reaches
stdout; if advisory phrasing ever appears in the tool's own voice, the run
fails loudly with ComplianceError instead of printing it.

Two voices are distinguished:
- Tool voice: everything the tool itself says. Scanned, zero tolerance.
- Quoted voice: third-party headlines reproduced verbatim as facts about
  what was published. These lines carry the QUOTE_PREFIX marker and are
  exempt from the scan, because a newspaper headline is a fact we report,
  not a view we hold. The label around them stays in tool voice and is
  scanned.

The banned list is assembled from word tuples so the literal phrases never
appear in this source file; that lets the test suite scan every source and
doc file for the joined phrases with no exemptions.
"""
from __future__ import annotations

QUOTE_PREFIX = "> "

# Joined at import time; the literal advisory phrase never appears in source.
_PHRASE_PARTS: tuple[tuple[str, ...], ...] = (
    ("you", "should", "buy"),
    ("you", "should", "sell"),
    ("you", "should", "hold"),
    ("we", "recommend"),
    ("i", "recommend"),
    ("recommend", "buying"),
    ("recommend", "selling"),
    ("recommend", "holding"),
    ("consider", "buying"),
    ("consider", "selling"),
    ("buy", "now"),
    ("sell", "now"),
    ("strong", "buy"),
    ("strong", "sell"),
    ("time", "to", "buy"),
    ("time", "to", "sell"),
    ("invest", "in"),
    ("buy", "the", "dip"),
    ("buy", "more"),
    ("sell", "everything"),
    ("good", "entry", "point"),
    ("take", "profits"),
    ("this", "is", "a", "buy"),
    ("this", "is", "a", "sell"),
    ("worth", "buying"),
    ("worth", "selling"),
)
BANNED: tuple[str, ...] = tuple(" ".join(parts) for parts in _PHRASE_PARTS)

ADVISOR_LINE = (
    "Facts only. What to DO with these numbers (whether to trade, what, "
    "when, how much) is a licensed-advisor question; this tool does not "
    "answer it."
)

BACKTEST_DISCLAIMER = """\
================ READ THIS BEFORE THE NUMBERS ================
HISTORICAL SIMULATION. NOT A PREDICTION. NOT ADVICE.
- This replays one rule against one past path of one ticker.
  The past path happened once; it is not a sample you can buy.
- Testing a still-listed ticker is itself survivorship-biased:
  the tickers that died are not in the file.
- Every extra parameter combination tried raises the odds the
  result is an overfit artifact (Bailey, Borwein, Lopez de
  Prado, Zhu 2014). The trials count is printed below, along
  with the multiple-testing haircut, the deflated Sharpe and
  the probability of backtest overfitting for that search.
- Costs and slippage below are estimates; real fills differ.
See RESEARCH.md for the cited methodology.
==============================================================="""

_QUESTION_WORDS = frozenset(
    {"should", "shall", "what", "when", "recommend", "advice", "advise",
     "invest", "worth", "good", "bad"}
)
_ACTION_WORDS = frozenset({"buy", "sell", "hold", "short", "long"})


class ComplianceError(RuntimeError):
    """Raised when tool-voice output would contain advisory phrasing."""


def looks_like_question(token: str) -> bool:
    """True when a CLI argument reads as an advice question, not a ticker.

    Tickers are short and conventionally uppercase (SPY, VFV.TO, ^GSPC).
    A lowercase English word like the ones below, or anything with a
    question mark, is someone asking what to do.
    """
    if "?" in token:
        return True
    low = token.lower()
    if low in _QUESTION_WORDS:
        return True
    # 'buy'/'sell' lowercase is a verb; uppercase could be a real ticker.
    return low in _ACTION_WORDS and not token.isupper()


def refuse_advice(query: str) -> str:
    """The one answer to every what-should-I-do phrasing."""
    return (
        f'"{query.strip()}" is a question about what to do with money. '
        "That is a licensed-advisor question and this tool does not answer "
        "it, by design. It can show you prices, indicators, headlines, and "
        "historical simulations; the decision stays yours (or your "
        "advisor's)."
    )


def check_output(text: str) -> None:
    """Scan tool-voice lines for advisory phrasing. Raise on any hit.

    Lines starting with QUOTE_PREFIX are verbatim third-party headlines
    and are skipped; everything else is the tool's own voice.
    """
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith(QUOTE_PREFIX):
            continue
        low = line.lower()
        for phrase in BANNED:
            if phrase in low:
                raise ComplianceError(
                    f"advisory phrasing {phrase!r} in tool-voice output "
                    f"(line {lineno}): {line.strip()!r}"
                )


def guard(text: str) -> str:
    """Gate every report through the boundary check before printing."""
    check_output(text)
    return text
