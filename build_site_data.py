"""Build the JSON data consumed by docs/index.html.

Free mode uses real Binance daily OHLCV plus the Alternative.me Fear & Greed Index.
It does NOT provide social-media-post sentiment; the score is based on market/F&G
features. Forecast/statistical fields are computed from downloaded historical data
and are educational, not investment advice.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from data_sources import load_dataset
from analysis import build_features, granger_table

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "data" / "site_data.json"
COINS = [
    ("BTC", "Bitcoin"),
    ("ETH", "Ethereum"),
    ("SOL", "Solana"),
    ("XRP", "XRP"),
    ("DOGE", "Dogecoin"),
]


def safe_num(x, default=None):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def mood(score):
    if score is None:
        return "Unavailable"
    if score >= 0.3:
        return "Positive"
    if score <= -0.3:
        return "Negative"
    return "Neutral"


def label(pred):
    if pred is None:
        return "Neutral"
    if pred > 0.001:
        return "Leaning up"
    if pred < -0.001:
        return "Leaning down"
    return "Neutral"


def fit_forecast(feat, horizon):
    # Walk-forward forecasts: every prediction only uses data available earlier.
    feature_cols = [c for c in ["ret_1", "mom_5", "vol_cc", "vol_pk", "score", "fng_z", "attn_z"]
                    if c in feat.columns]
    target = f"fwd_{horizon}"
    data = feat[feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 100:
        return None, np.array([]), np.array([])

    preds, actuals, dates = [], [], []
    step = 30
    first = max(90, int(len(data) * 0.7))
    for end in range(first, len(data), step):
        test = data.iloc[end:min(end + step, len(data))]
        train = data.iloc[:end]
        if len(train) < 60 or test.empty:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(train[feature_cols], train[target])
        p = model.predict(test[feature_cols])
        preds.extend(p.tolist())
        actuals.extend(test[target].tolist())
        dates.extend(test.index.strftime("%Y-%m-%d").tolist())

    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(data[feature_cols], data[target])
    last_row = feat[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]]
    latest_pred = safe_num(model.predict(last_row)[0]) if len(last_row) else None
    return latest_pred, np.asarray(preds), np.asarray(actuals)


def metrics(pred, actual):
    if len(pred) < 3 or len(actual) != len(pred):
        return {"hit_rate": None, "r2_os_controls": None, "r2_os": None, "sharpe": None, "cw_p": None}
    hit = float(np.mean(np.sign(pred) == np.sign(actual)) * 100)
    denom = float(np.sum(actual ** 2))
    r2 = float(1 - np.sum((actual - pred) ** 2) / denom) if denom > 0 else None
    strat = np.sign(pred) * actual
    sharpe = float(np.mean(strat) / np.std(strat, ddof=1) * np.sqrt(365)) if np.std(strat, ddof=1) > 0 else None
    # Placeholder is intentionally not used: a valid Clark-West test is not computed here.
    return {
        "hit_rate": safe_num(hit),
        "r2_os_controls": None,
        "r2_os": safe_num(r2),
        "sharpe": safe_num(sharpe),
        "cw_p": None,
    }


def build_coin(symbol, name, mode):
    raw = load_dataset(symbol, mode=mode, start="2020-01-01", refresh=True)
    if raw.empty or "close" not in raw:
        raise RuntimeError(f"No price data returned for {symbol} in mode={mode}")
    raw = raw.sort_index()
    # For free mode, the actual sentiment input is Fear & Greed; this is not social sentiment.
    if "sentiment" not in raw.columns:
        raw["sentiment"] = raw["fng"] if "fng" in raw.columns else 50.0
    if "fng" not in raw.columns:
        raw["fng"] = np.nan
    if "volume" not in raw.columns and "volume_24h" in raw.columns:
        raw["volume"] = raw["volume_24h"]
    feat, _ = build_features(raw)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    usable = feat.dropna(subset=["close", "ret_1", "score"]).copy()
    if len(usable) < 120:
        raise RuntimeError(f"Not enough usable history for {symbol} ({len(usable)} rows)")
    last = usable.iloc[-1]
    last_date = usable.index[-1].strftime("%Y-%m-%d")

    forecast, predictions, actuals = {}, {}, {}
    for h in (1, 2, 3):
        pred, p, a = fit_forecast(feat, h)
        forecast[f"h{h}"] = {"pred": pred, "label": label(pred)}
        predictions[h], actuals[h] = p, a

    # Statistical summaries are computed from actual fetched history.
    try:
        gt = granger_table(usable.dropna(), sent="score", max_lag=1)
        sent_p = safe_num(gt.loc[gt["direction"].str.contains("score -> returns"), "p_value"].iloc[0])
        ret_p = safe_num(gt.loc[gt["direction"].str.contains("returns -> score"), "p_value"].iloc[0])
    except Exception:
        sent_p, ret_p = None, None

    regimes = []
    vol = usable["vol_cc"].dropna()
    median_vol = float(vol.median()) if len(vol) else 0
    for regime, mask in [
        ("Volatility", usable["vol_cc"] >= median_vol),
        ("Volatility", usable["vol_cc"] < median_vol),
    ]:
        subset = usable.loc[mask, ["score", "fwd_1"]].dropna()
        if len(subset) >= 20 and subset["score"].std() > 0:
            slope, intercept, r, pval, stderr = stats.linregress(subset["score"], subset["fwd_1"])
            state = "High" if bool(mask.loc[subset.index].iloc[0]) else "Low"
            regimes.append({
                "regime": regime, "state": state,
                "coef_bp": safe_num(slope * 10000, 0),
                "p_value": safe_num(pval, 1), "n": int(len(subset))
            })
    if not regimes:
        regimes = [{"regime": "Volatility", "state": "Unavailable", "coef_bp": 0, "p_value": 1, "n": 0}]

    # Lag correlation from real observations.
    lag_values = []
    for k in range(-5, 6):
        if k < 0:
            x, y = usable["score"].iloc[-k:], usable["ret_1"].iloc[:k]
        elif k > 0:
            x, y = usable["score"].iloc[:-k], usable["ret_1"].iloc[k:]
        else:
            x, y = usable["score"], usable["ret_1"]
        xy = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1).dropna()
        corr = safe_num(xy.iloc[:, 0].corr(xy.iloc[:, 1])) if len(xy) > 10 else None
        lag_values.append({"k": k, "corr": corr})
    band = 1.96 / np.sqrt(max(1, len(usable)))
    score_series = usable["score"].tail(500)
    price_series = usable["close"].tail(500)
    dates = usable.tail(500).index.strftime("%Y-%m-%d").tolist()

    # Strategy curves use the walk-forward one-day forecasts only where available.
    p1, a1 = predictions[1], actuals[1]
    if len(p1) and len(a1):
        strategy = np.cumprod(1 + np.sign(p1) * a1)
        bh = np.cumprod(1 + a1)
        controls = np.cumprod(1 + a1)  # benchmark series; not a separately estimated controls-only model
        equity_dates = dates[-len(strategy):] if len(dates) >= len(strategy) else [str(i) for i in range(len(strategy))]
        equity = {"dates": equity_dates, "buy_hold": bh.tolist(), "controls": controls.tolist(), "sentiment": strategy.tolist()}
    else:
        equity = {"dates": dates, "buy_hold": [1.0] * len(dates), "controls": [1.0] * len(dates), "sentiment": [1.0] * len(dates)}

    m = {f"h{h}": metrics(predictions[h], actuals[h]) for h in (1, 2, 3)}
    ret_1 = safe_num(last.get("ret_1"), 0)
    fng = safe_num(last.get("fng"))
    vol_ann = safe_num(last.get("vol_cc"), 0) * np.sqrt(365)
    return {
        "symbol": symbol, "name": name,
        "data_start": usable.index[0].strftime("%Y-%m-%d"),
        "data_end": last_date,
        "latest": {
            "date": last_date,
            "close": safe_num(last["close"]),
            "ret_1": ret_1,
            "score": safe_num(last["score"], 0),
            "mood": mood(safe_num(last["score"])),
            "vol_regime": "High" if safe_num(last.get("vol_cc"), 0) >= median_vol else "Low",
            "vol_20d_ann": safe_num(vol_ann, 0),
            "fng": fng,
        },
        "forecast": forecast,
        "granger": {"sent_to_ret_p": sent_p, "ret_to_sent_p": ret_p},
        "regimes": regimes,
        "metrics": m,
        "series": {"dates": dates, "close": [safe_num(v) for v in price_series], "score": [safe_num(v) for v in score_series]},
        "lead_lag": {"values": lag_values, "band": safe_num(band)},
        "equity": equity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["free", "lunarcrush", "cmc", "sample"], default="free")
    args = parser.parse_args()
    if args.mode == "sample":
        raise SystemExit("Refusing to overwrite live site with sample data. Use --mode free, lunarcrush, or cmc.")
    coins, errors = [], []
    for symbol, name in COINS:
        try:
            print(f"Fetching and analysing {symbol} ({args.mode})...", flush=True)
            coins.append(build_coin(symbol, name, args.mode))
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
            print(f"[ERROR] {symbol}: {exc}", file=sys.stderr, flush=True)
    if not coins:
        raise SystemExit("No coins could be built. Errors: " + "; ".join(errors))
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "mode": args.mode,
        "skipped": errors,
        "coins": coins,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(OUT)
    print(f"Wrote {OUT} with {len(coins)} coin(s). Skipped: {len(errors)}")


if __name__ == "__main__":
    main()
