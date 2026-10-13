# Aggregated Daily Sentiment → Short-Term Crypto Returns

Working model for the PBL-1 project (DSE 2170), Manipal University Jaipur.
Tests whether daily aggregated sentiment predicts BTC and ETH returns 1 to 3 days ahead,
beyond momentum and volatility, and whether the effect changes across market regimes.

## Files

| File | What it does |
|---|---|
| `sentiment_returns.ipynb` | Main notebook. Run top to bottom. |
| `data_sources.py` | API clients for LunarCrush, CoinMarketCap, Binance, alternative.me, plus a synthetic sample generator. Caches every download in `data/`. |
| `analysis.py` | Features, HAC Granger tests, predictive regressions, walk-forward models (Ridge, Lasso, gradient boosting), Clark-West test, GARCH-X, Markov switching and regime splits. |
| `plots.py` | Report-ready charts. |
| `results/` | CSV tables saved by the last cell. |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then paste your keys into .env
jupyter notebook sentiment_returns.ipynb
```

In the first code cell, set `DATA_MODE`:

- `"free"`: no keys. Binance prices + alternative.me Fear & Greed, back to 2020. **Start here.**
- `"lunarcrush"`: per-coin social sentiment, post counts and active contributors. Needs `LUNARCRUSH_API_KEY`.
- `"cmc"`: CMC prices + CMC Fear & Greed. Needs `CMC_API_KEY`. Free plan gives roughly 1 year of daily data.
- `"sample"`: synthetic data for testing. The notebook ships with sample-mode output so you can see what every section produces. Those numbers are not real results.

## Getting keys

- **CoinMarketCap:** sign up for the free Basic plan at coinmarketcap.com/api, copy the key from the dashboard.
- **LunarCrush:** create an account at lunarcrush.com/developers. The free account is described as very limited test access; time-series history likely needs the Individual plan. Try your key first: if the LunarCrush cell errors with a 401/403 or returns no rows, your plan doesn't include that endpoint.

## Method notes

- No look-ahead: sentiment z-scores use only the previous 90 days; models train only on rows whose future return is already known; the volatility regime uses an expanding median. The Markov regime split uses smoothed probabilities, so it is descriptive only.
- Overlapping multi-day returns are handled with Newey-West errors (lag ≥ h) and non-overlapping strategy evaluation.
- The Fear & Greed index is partly built from price volatility and momentum. That's why the tests control for both; without those controls it would look more predictive than it is.
- Strategy Sharpe ratios ignore trading costs and slippage.
