"""Daily OHLCV ingestion from a free, no-key public source.

Source: the Yahoo Finance public chart API (query1.finance.yahoo.com).
This is the same endpoint the yfinance library wraps; calling it directly
keeps the module dependency-light (requests only). Stooq's CSV endpoint
was the first choice but sits behind a JavaScript anti-bot check as of
2026-07-15, so it is documented here as unavailable rather than silently
half-working.

Everything fetched is cached as plain CSV next to this module (gitignored)
so repeat runs on the same day are offline and reproducible, and so tests
never need the network.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import requests

PROVIDER = "Yahoo Finance chart API (public, no key)"
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
VALID_RANGES = ("3mo", "6mo", "1y", "2y", "5y", "10y", "max")
DEFAULT_CACHE = Path(__file__).resolve().parent / "cache"


@dataclass
class Bars:
    """One ticker's daily history. Parallel lists, oldest first."""

    symbol: str
    currency: str
    dates: list[date] = field(default_factory=list)
    open: list[float] = field(default_factory=list)
    high: list[float] = field(default_factory=list)
    low: list[float] = field(default_factory=list)
    close: list[float] = field(default_factory=list)
    adjclose: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.dates)

    def return_series(self) -> list[float]:
        """Close preferred for display; adjusted close for returns, because
        it folds dividends and splits back in. Falls back to close when the
        provider sent no adjclose."""
        return self.adjclose if self.adjclose else self.close


def fetch_daily(symbol: str, range_: str = "5y", timeout: int = 30) -> Bars:
    """Pull daily bars for one symbol. Fails loudly, never silently."""
    if range_ not in VALID_RANGES:
        raise ValueError(f"range {range_!r} not in {VALID_RANGES}")
    resp = requests.get(
        _CHART_URL.format(symbol=symbol),
        params={"range": range_, "interval": "1d", "includeAdjustedClose": "true"},
        headers=_UA,
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(
            f"{PROVIDER}: HTTP {resp.status_code} for {symbol!r}; "
            f"body starts: {resp.text[:120]!r}"
        )
    payload = resp.json()
    return parse_chart_payload(symbol, payload)


def parse_chart_payload(symbol: str, payload: dict) -> Bars:
    """Turn the chart JSON into Bars. Split out so tests run offline."""
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"{PROVIDER}: error for {symbol!r}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"{PROVIDER}: empty result for {symbol!r}")
    res = results[0]
    meta = res.get("meta") or {}
    quote = (res.get("indicators", {}).get("quote") or [{}])[0]
    adj_list = res.get("indicators", {}).get("adjclose") or []
    adj = adj_list[0].get("adjclose", []) if adj_list else []
    stamps = res.get("timestamp") or []
    if not stamps:
        raise RuntimeError(f"{PROVIDER}: no timestamps for {symbol!r}")

    bars = Bars(symbol=meta.get("symbol", symbol), currency=meta.get("currency", "?"))
    closes = quote.get("close") or []
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    vols = quote.get("volume") or []
    for i, ts in enumerate(stamps):
        c = closes[i] if i < len(closes) else None
        if c is None:
            continue  # Yahoo pads holidays/half-days with null rows
        bars.dates.append(datetime.fromtimestamp(ts, tz=timezone.utc).date())
        bars.close.append(float(c))
        bars.open.append(float(opens[i]) if i < len(opens) and opens[i] is not None else float(c))
        bars.high.append(float(highs[i]) if i < len(highs) and highs[i] is not None else float(c))
        bars.low.append(float(lows[i]) if i < len(lows) and lows[i] is not None else float(c))
        a = adj[i] if i < len(adj) else None
        bars.adjclose.append(float(a) if a is not None else float(c))
        v = vols[i] if i < len(vols) else None
        bars.volume.append(float(v) if v is not None else 0.0)
    if len(bars) < 2:
        raise RuntimeError(f"{PROVIDER}: under 2 usable bars for {symbol!r}")
    return bars


def save_csv(bars: Bars, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# symbol={bars.symbol} currency={bars.currency} "
                 f"fetched={date.today().isoformat()} source={PROVIDER}\n")
        w = csv.writer(fh)
        w.writerow(["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"])
        for i in range(len(bars)):
            w.writerow([
                bars.dates[i].isoformat(),
                f"{bars.open[i]:.6f}", f"{bars.high[i]:.6f}",
                f"{bars.low[i]:.6f}", f"{bars.close[i]:.6f}",
                f"{bars.adjclose[i]:.6f}", f"{bars.volume[i]:.0f}",
            ])
    return path


def load_csv(path: Path) -> Bars:
    path = Path(path)
    symbol, currency = path.stem, "?"
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline()
        if first.startswith("#"):
            for part in first[1:].split():
                if part.startswith("symbol="):
                    symbol = part.split("=", 1)[1]
                elif part.startswith("currency="):
                    currency = part.split("=", 1)[1]
        else:
            fh.seek(0)
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"no data rows in {path}")
    bars = Bars(symbol=symbol, currency=currency)
    for r in rows:
        bars.dates.append(date.fromisoformat(r["Date"]))
        bars.open.append(float(r["Open"]))
        bars.high.append(float(r["High"]))
        bars.low.append(float(r["Low"]))
        bars.close.append(float(r["Close"]))
        bars.adjclose.append(float(r.get("AdjClose") or r["Close"]))
        bars.volume.append(float(r.get("Volume") or 0))
    return bars


def _cache_path(symbol: str, range_: str, cache_dir: Path) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in symbol)
    return Path(cache_dir) / f"{safe}_{range_}.csv"


def _fetched_today(path: Path) -> bool:
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return False
    return f"fetched={date.today().isoformat()}" in first


def get_bars(symbol: str, range_: str = "5y",
             cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> tuple[Bars, bool]:
    """Cached fetch. Returns (bars, from_cache)."""
    cpath = _cache_path(symbol, range_, cache_dir)
    if not refresh and cpath.exists() and _fetched_today(cpath):
        return load_csv(cpath), True
    bars = fetch_daily(symbol, range_)
    save_csv(bars, cpath)
    return bars, False
