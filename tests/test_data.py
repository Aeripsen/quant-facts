"""Data layer tests: payload parsing and CSV round-trip, fully offline."""
from datetime import date, datetime, timezone

import pytest

import data as d


def _ts(y, m, day):
    return int(datetime(y, m, day, 13, 30, tzinfo=timezone.utc).timestamp())


def payload(closes, adj=None, with_null=False):
    stamps = [_ts(2025, 1, i + 2) for i in range(len(closes))]
    if with_null:
        closes = list(closes) + [None]
        stamps.append(_ts(2025, 1, len(closes) + 2))
    quote = {
        "close": closes,
        "open": [c if c is None else c - 1 for c in closes],
        "high": [c if c is None else c + 1 for c in closes],
        "low": [c if c is None else c - 2 for c in closes],
        "volume": [1000 for _ in closes],
    }
    out = {"chart": {"result": [{
        "meta": {"symbol": "FAKE", "currency": "USD"},
        "timestamp": stamps,
        "indicators": {"quote": [quote]},
    }], "error": None}}
    if adj is not None:
        out["chart"]["result"][0]["indicators"]["adjclose"] = [
            {"adjclose": adj}]
    return out


def test_parse_payload_basic():
    bars = d.parse_chart_payload("FAKE", payload([10.0, 11.0, 12.0]))
    assert bars.symbol == "FAKE" and bars.currency == "USD"
    assert len(bars) == 3
    assert bars.dates[0] == date(2025, 1, 2)
    assert bars.close == [10.0, 11.0, 12.0]
    assert bars.high[1] == 12.0 and bars.low[1] == 9.0


def test_parse_payload_drops_null_rows():
    bars = d.parse_chart_payload("FAKE", payload([10.0, 11.0], with_null=True))
    assert len(bars) == 2


def test_parse_payload_adjclose_used_for_returns():
    bars = d.parse_chart_payload(
        "FAKE", payload([10.0, 11.0, 12.0], adj=[9.0, 10.0, 11.0]))
    assert bars.return_series() == [9.0, 10.0, 11.0]
    assert bars.close == [10.0, 11.0, 12.0]  # display stays unadjusted


def test_parse_payload_error_raises():
    bad = {"chart": {"result": None, "error": {"code": "Not Found"}}}
    with pytest.raises(RuntimeError, match="Not Found"):
        d.parse_chart_payload("NOPE", bad)


def test_csv_round_trip(tmp_path):
    bars = d.parse_chart_payload(
        "FAKE", payload([10.0, 11.0, 12.0], adj=[9.0, 10.0, 11.0]))
    p = d.save_csv(bars, tmp_path / "FAKE_5y.csv")
    back = d.load_csv(p)
    assert back.symbol == "FAKE" and back.currency == "USD"
    assert back.dates == bars.dates
    assert back.close == pytest.approx(bars.close)
    assert back.adjclose == pytest.approx(bars.adjclose)


def test_get_bars_uses_todays_cache(tmp_path, monkeypatch):
    bars = d.parse_chart_payload("FAKE", payload([10.0, 11.0]))
    d.save_csv(bars, tmp_path / "FAKE_5y.csv")

    def boom(*a, **k):
        raise AssertionError("network hit despite fresh cache")

    monkeypatch.setattr(d, "fetch_daily", boom)
    got, from_cache = d.get_bars("FAKE", "5y", cache_dir=tmp_path)
    assert from_cache is True
    assert got.close == pytest.approx([10.0, 11.0])


def test_bad_range_rejected():
    with pytest.raises(ValueError, match="range"):
        d.fetch_daily("SPY", range_="7weeks")
