"""
Data layer for the PBL project:
"Aggregated daily sentiment analysis for predicting short-term crypto returns".

Sources
-------
- LunarCrush API v4  : per-coin daily social metrics (sentiment, posts, contributors,
                       interactions) plus OHLC price. Needs LUNARCRUSH_API_KEY.
- CoinMarketCap API  : daily historical quotes and the CMC Fear & Greed index.
                       Needs CMC_API_KEY. Free Basic plan covers roughly 1 year of daily data.
- Free, no key       : Binance public klines (prices) and alternative.me Fear & Greed.
- Sample             : synthetic data, ONLY for testing that the pipeline runs.

Every fetch is cached to data/<name>.csv so you don't burn API credits re-running cells.
All timestamps are converted to UTC calendar dates.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TIMEOUT = 30


# ----------------------------------------------------------------------------- helpers
def _cache(name: str, fetch_fn, refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch_fn()
    df.to_csv(path)
    return df


def _get(url, headers=None, params=None, retries=3):
    for attempt in range(retries):
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if r.status_code == 429:                    # rate limited: back off and retry
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} from {url}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"Rate limited too many times on {url}")


def _to_date_index(df: pd.DataFrame, col: str, unit: str | None = "s") -> pd.DataFrame:
    ts = pd.to_datetime(df[col], unit=unit, utc=True) if unit else pd.to_datetime(df[col], utc=True)
    df = df.drop(columns=[col])
    df.index = ts.dt.tz_convert(None).dt.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _key(env_name: str) -> str:
    k = os.getenv(env_name)
    if not k:
        raise RuntimeError(
            f"{env_name} is not set. Copy .env.example to .env and paste your key there."
        )
    return k


# ----------------------------------------------------------------------------- LunarCrush
LC_BASE = "https://lunarcrush.com/api4/public"


def fetch_lunarcrush(coin: str = "BTC", start: str = "2020-01-01", end: str | None = None,
                     refresh: bool = False) -> pd.DataFrame:
    """Daily time series for one coin from LunarCrush v4.

    Columns kept (when the plan returns them): open, high, low, close, volume_24h,
    sentiment (% of posts that are positive, 0-100), galaxy_score, alt_rank,
    social_dominance, interactions, posts_active, posts_created,
    contributors_active, contributors_created, spam.
    """
    def fetch():
        key = _key("LUNARCRUSH_API_KEY")
        start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
        end_ts = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")).timestamp())
        js = _get(f"{LC_BASE}/coins/{coin.lower()}/time-series/v2",
                  headers={"Authorization": f"Bearer {key}"},
                  params={"bucket": "day", "start": start_ts, "end": end_ts})
        rows = js.get("data", [])
        if not rows:
            raise RuntimeError(f"LunarCrush returned no rows for {coin}. Response: {str(js)[:300]}")
        df = pd.DataFrame(rows)
        return _to_date_index(df, "time")

    df = _cache(f"lunarcrush_{coin.lower()}", fetch, refresh)
    keep = ["open", "high", "low", "close", "volume_24h", "sentiment", "galaxy_score",
            "alt_rank", "social_dominance", "interactions", "posts_active", "posts_created",
            "contributors_active", "contributors_created", "spam"]
    return df[[c for c in keep if c in df.columns]].apply(pd.to_numeric, errors="coerce")


# ----------------------------------------------------------------------------- CoinMarketCap
CMC_BASE = "https://pro-api.coinmarketcap.com"


def fetch_cmc_prices(symbol: str = "BTC", days: int = 365, refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV from CMC. Basic plan allows about a year of daily history."""
    def fetch():
        key = _key("CMC_API_KEY")
        end = pd.Timestamp.now(tz="UTC").normalize()
        start = end - pd.Timedelta(days=days)
        js = _get(f"{CMC_BASE}/v2/cryptocurrency/ohlcv/historical",
                  headers={"X-CMC_PRO_API_KEY": key},
                  params={"symbol": symbol, "time_start": start.isoformat(),
                          "time_end": end.isoformat(), "interval": "daily", "convert": "USD"})
        data = js["data"]
        data = data[symbol][0] if isinstance(data.get(symbol), list) else data.get(symbol, data)
        rows = [{"time": q["time_open"], **{k: q["quote"]["USD"][k]
                                            for k in ("open", "high", "low", "close", "volume")}}
                for q in data["quotes"]]
        return _to_date_index(pd.DataFrame(rows), "time", unit=None)

    return _cache(f"cmc_prices_{symbol.lower()}", fetch, refresh)


def fetch_cmc_fear_greed(refresh: bool = False) -> pd.DataFrame:
    """CMC's own Fear & Greed index history (0 = extreme fear, 100 = extreme greed)."""
    def fetch():
        key = _key("CMC_API_KEY")
        rows, start = [], 1
        while True:
            js = _get(f"{CMC_BASE}/v3/fear-and-greed/historical",
                      headers={"X-CMC_PRO_API_KEY": key}, params={"start": start, "limit": 500})
            batch = js.get("data", [])
            rows += batch
            if len(batch) < 500:
                break
            start += 500
        df = pd.DataFrame(rows)[["timestamp", "value"]].rename(columns={"value": "fng"})
        df["timestamp"] = pd.to_numeric(df["timestamp"])
        return _to_date_index(df, "timestamp")

    return _cache("cmc_fear_greed", fetch, refresh)


# ----------------------------------------------------------------------------- free sources
def fetch_binance_prices(symbol: str = "BTC", start: str = "2020-01-01",
                         refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV from Binance's public market-data API (no key needed)."""
    def fetch():
        pair = f"{symbol.upper()}USDT"
        out, start_ms = [], int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        hosts = ["https://data-api.binance.vision", "https://api.binance.com"]
        host = None
        while True:
            for h in ([host] if host else hosts):
                try:
                    js = _get(f"{h}/api/v3/klines",
                              params={"symbol": pair, "interval": "1d",
                                      "startTime": start_ms, "limit": 1000})
                    host = h
                    break
                except Exception as e:
                    err = e
            else:
                raise RuntimeError(f"Could not reach Binance: {err}")
            if not js:
                break
            out += js
            start_ms = js[-1][0] + 86_400_000
            if len(js) < 1000:
                break
        df = pd.DataFrame(out).iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df = _to_date_index(df, "time", unit="ms").astype(float)
        return df[df.index < pd.Timestamp.utcnow().tz_localize(None).normalize()]  # drop partial day

    return _cache(f"binance_{symbol.lower()}", fetch, refresh)


def fetch_alternative_fear_greed(refresh: bool = False) -> pd.DataFrame:
    """alternative.me Crypto Fear & Greed index, daily since Feb 2018 (no key needed)."""
    def fetch():
        js = _get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"})
        df = pd.DataFrame(js["data"])[["timestamp", "value"]].rename(columns={"value": "fng"})
        df["timestamp"] = pd.to_numeric(df["timestamp"])
        df["fng"] = pd.to_numeric(df["fng"])
        return _to_date_index(df, "timestamp")

    return _cache("alt_fear_greed", fetch, refresh)


# ----------------------------------------------------------------------------- sample data
def make_sample_data(coin: str = "BTC", n_days: int = 1400, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC data shaped like the LunarCrush output. For testing the code only.

    Built so that sentiment partly reacts to past returns and weakly leads the next
    day's return, with a stronger lead in high-volatility periods. Results from this
    data say nothing about real markets.
    """
    rng = np.random.default_rng(seed + (0 if coin == "BTC" else 7))
    idx = pd.date_range(end=pd.Timestamp("2025-10-31"), periods=n_days, freq="D", name="date")
    sigma = np.empty(n_days); r = np.empty(n_days); s = np.empty(n_days)
    sigma[0], r[0], s[0] = 0.03, 0.0, 0.0
    w, a, b = 0.03 ** 2 * 0.05, 0.10, 0.85
    for t in range(1, n_days):
        sigma[t] = np.sqrt(w + a * r[t - 1] ** 2 + b * sigma[t - 1] ** 2)
        s[t] = 0.6 * s[t - 1] + 8 * r[t - 1] + rng.normal(0, 0.5)          # mood follows price
        beta = 0.012 if sigma[t] > 0.035 else 0.003                         # stronger when volatile
        r[t] = beta * s[t - 1] + sigma[t] * rng.standard_t(5) / np.sqrt(5 / 3)
    close = 30000 * np.exp(np.cumsum(r)) if coin == "BTC" else 2000 * np.exp(np.cumsum(r))
    rng_hl = np.abs(rng.normal(0, 1, n_days)) * sigma * close
    posts = np.exp(9 + 15 * sigma + rng.normal(0, 0.2, n_days))
    df = pd.DataFrame({
        "open": np.r_[close[0], close[:-1]], "close": close,
        "high": close + rng_hl, "low": np.maximum(close - rng_hl, close * 0.5),
        "volume_24h": posts * 1e5,
        "sentiment": np.clip(70 + 6 * s + rng.normal(0, 2, n_days), 0, 100),
        "posts_active": posts, "contributors_active": posts / 3,
        "interactions": posts * 40 * np.exp(rng.normal(0, 0.3, n_days)),
        "fng": np.clip(50 + 10 * pd.Series(s).rolling(7, min_periods=1).mean().values
                       + rng.normal(0, 5, n_days), 0, 100),
    }, index=idx)
    return df


# ----------------------------------------------------------------------------- one entry point
def load_dataset(coin: str, mode: str = "free", start: str = "2020-01-01",
                 refresh: bool = False) -> pd.DataFrame:
    """Return one daily dataframe per coin with price columns plus whatever sentiment
    columns the chosen mode provides.

    mode = "lunarcrush" : LunarCrush social + price, joined with Fear & Greed
           "cmc"        : CMC prices + CMC Fear & Greed (about 1 year on the free plan)
           "free"       : Binance prices + alternative.me Fear & Greed
           "sample"     : synthetic, for testing
    """
    if mode == "sample":
        return make_sample_data(coin)

    if mode == "lunarcrush":
        df = fetch_lunarcrush(coin, start=start, refresh=refresh)
        try:
            fng = fetch_cmc_fear_greed(refresh) if os.getenv("CMC_API_KEY") \
                else fetch_alternative_fear_greed(refresh)
            df = df.join(fng, how="left")
        except Exception as e:
            print(f"[warn] Fear & Greed not added: {e}")
    elif mode == "cmc":
        df = fetch_cmc_prices(coin, refresh=refresh).join(fetch_cmc_fear_greed(refresh), how="left")
    elif mode == "free":
        df = fetch_binance_prices(coin, start=start, refresh=refresh) \
            .join(fetch_alternative_fear_greed(refresh), how="left")
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    df = df[df.index >= pd.Timestamp(start)]
    df = df.asfreq("D")                                  # crypto trades every day
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
    return df
