"""
Features, statistical tests, forecasting models and regime analysis.

Design rule throughout: nothing at date t may use information from after t.
Rolling z-scores only look backward, walk-forward training stops h days before
each test block, and the volatility regime split uses an expanding median.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

HORIZONS = (1, 2, 3)
CONTROLS = ["ret_1", "mom_5", "vol_cc", "vol_pk"]


# ============================================================================ features
def rolling_z(x: pd.Series, window: int = 90, min_periods: int = 30) -> pd.Series:
    """Backward-looking z-score: compares today with the previous `window` days."""
    m = x.rolling(window, min_periods=min_periods).mean()
    s = x.rolling(window, min_periods=min_periods).std()
    return (x - m) / s.replace(0, np.nan)


def build_features(df: pd.DataFrame, z_window: int = 90) -> tuple[pd.DataFrame, list[str]]:
    """Add return targets, market controls and sentiment features.

    Returns the frame and the list of sentiment feature names that were built
    (depends on which columns the data source provided).
    """
    out = df.copy()
    c = out["close"]

    # --- returns and targets
    out["ret_1"] = np.log(c / c.shift(1))                       # today's log return
    for h in HORIZONS:
        out[f"fwd_{h}"] = np.log(c.shift(-h) / c)                # return over next h days

    # --- market controls
    out["mom_5"] = out["ret_1"].rolling(5).sum()                 # 5-day momentum
    out["vol_cc"] = out["ret_1"].rolling(20).std()               # close-to-close vol
    hl = np.log(out["high"] / out["low"]) ** 2
    out["vol_pk"] = np.sqrt(hl.rolling(20).mean() / (4 * np.log(2)))  # Parkinson range vol

    # --- sentiment features
    sent_cols = []
    if "sentiment" in out:
        polarity = (out["sentiment"] - 50) / 50                  # -1 (all negative) .. +1 (all positive)
        out["sent_z"] = rolling_z(out["sentiment"], z_window)
        sent_cols.append("sent_z")
        if "contributors_active" in out:                         # weight mood by number of users
            out["sent_users_z"] = rolling_z(polarity * np.log1p(out["contributors_active"]), z_window)
            sent_cols.append("sent_users_z")
        if "posts_active" in out:                                # weight mood by volume of posts
            out["sent_posts_z"] = rolling_z(polarity * np.log1p(out["posts_active"]), z_window)
            sent_cols.append("sent_posts_z")
        out["d_sent_z"] = out["sent_z"].diff()                   # change in mood
        sent_cols.append("d_sent_z")
    if "fng" in out:
        out["fng_z"] = rolling_z(out["fng"], z_window)
        out["d_fng_z"] = out["fng_z"].diff()
        sent_cols += ["fng_z", "d_fng_z"]
    if "posts_active" in out:                                    # attention, not mood
        out["attn_z"] = rolling_z(np.log1p(out["posts_active"]), z_window)
        sent_cols.append("attn_z")
    elif "volume" in out or "volume_24h" in out:                 # trading volume as attention proxy
        v = out["volume"] if "volume" in out else out["volume_24h"]
        out["attn_z"] = rolling_z(np.log1p(v), z_window)

    # --- one combined daily score (report section 4.3): average of the level z-scores
    level = [col for col in ("sent_z", "sent_users_z", "sent_posts_z", "fng_z") if col in out]
    if not level:
        raise ValueError("No sentiment columns found. Check the data mode.")
    out["score"] = rolling_z(out[level].mean(axis=1), z_window)
    sent_cols = ["score"] + sent_cols
    return out, sent_cols


# ============================================================================ Granger (HAC)
def granger_hac(df: pd.DataFrame, cause: str, effect: str, lags: int,
                controls: list[str] | None = None) -> dict:
    """Does `cause` help predict `effect` beyond effect's own lags and the controls?

    effect_t = a + sum b_i effect_{t-i} + sum g_i cause_{t-i} + d' controls_{t-1} + e_t
    H0: all g_i = 0, Wald test with Newey-West (HAC) standard errors.
    """
    controls = controls or []
    X = {}
    for i in range(1, lags + 1):
        X[f"{effect}_L{i}"] = df[effect].shift(i)
        X[f"{cause}_L{i}"] = df[cause].shift(i)
    for ctl in controls:
        if ctl not in (cause, effect):
            X[f"{ctl}_L1"] = df[ctl].shift(1)
    X = pd.DataFrame(X)
    data = pd.concat([df[effect].rename("y"), X], axis=1).dropna()
    res = sm.OLS(data["y"], sm.add_constant(data.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags + 2})
    names = [f"{cause}_L{i}" for i in range(1, lags + 1)]
    R = np.zeros((lags, len(res.params)))
    for j, n in enumerate(names):
        R[j, list(res.params.index).index(n)] = 1
    w = res.wald_test(R, scalar=True)
    return {"cause": cause, "effect": effect, "lags": lags,
            "F": float(w.statistic), "p_value": float(w.pvalue),
            "sum_coef": float(res.params[names].sum()), "n": int(res.nobs)}


def granger_table(df: pd.DataFrame, sent: str = "score", max_lag: int = 3) -> pd.DataFrame:
    """Both directions: sentiment -> returns and returns -> sentiment."""
    rows = []
    for L in range(1, max_lag + 1):
        rows.append(granger_hac(df, sent, "ret_1", L, ["vol_cc", "mom_5"]))
        rows.append(granger_hac(df, "ret_1", sent, L, ["vol_cc"]))
    t = pd.DataFrame(rows)
    t["direction"] = np.where(t["cause"] == sent, f"{sent} -> returns", f"returns -> {sent}")
    return t[["direction", "lags", "F", "p_value", "sum_coef", "n"]]


# ============================================================================ predictive regressions
def predictive_regression(df: pd.DataFrame, feature: str, h: int,
                          controls: list[str] = CONTROLS) -> dict:
    """fwd_h ~ feature + controls, HAC errors (overlapping targets need maxlags >= h)."""
    cols = [feature] + [c for c in controls if c != feature]
    data = df[[f"fwd_{h}"] + cols].dropna()
    X = sm.add_constant(data[cols])
    res = sm.OLS(data[f"fwd_{h}"], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 5)})
    return {"feature": feature, "h": h, "coef_bp": res.params[feature] * 1e4,
            "t_stat": res.tvalues[feature], "p_value": res.pvalues[feature],
            "r2": res.rsquared, "n": int(res.nobs)}


def regression_table(df, features, horizons=HORIZONS):
    return pd.DataFrame([predictive_regression(df, f, h) for f in features for h in horizons])


# ============================================================================ walk-forward forecasting
def _models(seed=0):
    tscv = TimeSeriesSplit(n_splits=3)
    return {
        "Ridge (controls only)": ("controls",
            make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))),
        "Ridge (+ sentiment)": ("all",
            make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))),
        "Lasso (+ sentiment)": ("all",
            make_pipeline(StandardScaler(), LassoCV(cv=tscv, alphas=30, max_iter=5000, random_state=seed))),
        "Gradient boosting (+ sentiment)": ("all",
            HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=150,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=seed)),
    }


def walk_forward(df: pd.DataFrame, sent_cols: list[str], h: int = 1,
                 min_train: int = 365, step: int = 30) -> pd.DataFrame:
    """Expanding-window out-of-sample forecasts of fwd_h.

    At each refit date t the model trains only on rows whose target is already known
    (rows up to t-1-h), then predicts the next `step` days. Returns one row per test day
    with the realised return and each model's prediction.
    """
    feats_all = CONTROLS + sent_cols
    data = df[[f"fwd_{h}"] + feats_all].dropna()
    y, idx = data[f"fwd_{h}"], data.index
    preds = {name: pd.Series(np.nan, index=idx) for name in _models()}
    preds["Historical mean"] = pd.Series(np.nan, index=idx)

    for start in range(min_train, len(data), step):
        train = data.iloc[: max(start - h, 1)]
        test = data.iloc[start: start + step]
        preds["Historical mean"].loc[test.index] = train[f"fwd_{h}"].mean()
        for name, (fs, model) in _models().items():
            cols = CONTROLS if fs == "controls" else feats_all
            model.fit(train[cols], train[f"fwd_{h}"])
            preds[name].loc[test.index] = model.predict(test[cols])

    out = pd.DataFrame(preds)
    out.insert(0, "actual", y)
    return out.dropna()


def clark_west(actual, pred_small, pred_big, h=1):
    """Clark-West test for nested models: does the bigger model (with sentiment)
    forecast better than the smaller one? One-sided p-value."""
    e1, e2 = actual - pred_small, actual - pred_big
    f = e1 ** 2 - (e2 ** 2 - (pred_small - pred_big) ** 2)
    res = sm.OLS(f.values, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 5)})
    t = float(res.tvalues[0])
    return t, 1 - stats.norm.cdf(t)


def score_forecasts(fc: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Out-of-sample R^2 vs historical mean, direction hit rate, and a simple
    long/short strategy (sign of the forecast, gross of costs, non-overlapping for h>1)."""
    y = fc["actual"]
    base_sse = ((y - fc["Historical mean"]) ** 2).sum()
    rows = []
    for m in fc.columns.drop("actual"):
        p = fc[m]
        r2_os = 1 - ((y - p) ** 2).sum() / base_sse
        hit = (np.sign(p) == np.sign(y)).mean()
        pos = np.sign(p).iloc[::h]
        pnl = (pos * y.iloc[::h])
        sharpe = pnl.mean() / pnl.std() * np.sqrt(365 / h) if pnl.std() > 0 else np.nan
        row = {"model": m, "R2_OS_%": 100 * r2_os, "hit_rate_%": 100 * hit,
               "strategy_sharpe_gross": sharpe}
        if m not in ("Historical mean", "Ridge (controls only)"):
            t, p_cw = clark_west(y, fc["Ridge (controls only)"], p, h)
            row["CW_p_vs_controls"] = p_cw
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# ============================================================================ GARCH-X
def garch_x(df: pd.DataFrame, feature: str = "score") -> pd.DataFrame:
    """AR(1)-GARCH(1,1) with Student-t errors and yesterday's sentiment in the mean.

    Note: the `arch` package supports exogenous regressors in the mean equation only,
    so sentiment enters the mean, not the variance.
    """
    from arch import arch_model
    data = pd.concat([100 * df["ret_1"].rename("r"),
                      df[feature].shift(1).rename(f"{feature}_L1"),
                      df["ret_1"].shift(1).rename("ret_L1")], axis=1).dropna()
    am = arch_model(data["r"], x=data[[f"{feature}_L1", "ret_L1"]], mean="LS",
                    vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    return pd.DataFrame({"coef": res.params, "t_stat": res.tvalues, "p_value": res.pvalues})


# ============================================================================ regimes
def markov_regimes(df: pd.DataFrame) -> pd.Series:
    """Two-state Markov switching model on daily returns with switching variance.
    Returns the smoothed probability of the high-volatility state.
    (Smoothed probabilities use the full sample, so this split is descriptive.)"""
    r = (100 * df["ret_1"]).dropna()
    mod = sm.tsa.MarkovRegression(r, k_regimes=2, trend="c", switching_variance=True)
    res = mod.fit(disp=False, search_reps=10)
    sig = [res.params[f"sigma2[{k}]"] for k in range(2)]
    hi = int(np.argmax(sig))
    return res.smoothed_marginal_probabilities[hi].rename("p_high_vol")


def regime_table(df: pd.DataFrame, feature: str = "score", h: int = 1) -> pd.DataFrame:
    """Predictive regression of fwd_h on the feature, separately inside each regime."""
    d = df.copy()
    vol_med = d["vol_cc"].expanding(min_periods=60).median()       # no look-ahead
    d["Volatility"] = np.where(d["vol_cc"] > vol_med, "High", "Low")
    if "p_high_vol" in d:
        d["Markov state"] = np.where(d["p_high_vol"] > 0.5, "Turbulent", "Calm")
    if "attn_z" in d:
        d["Attention"] = np.where(d["attn_z"] > 0, "High", "Low")
    d["Trend"] = np.where(d["close"] > d["close"].rolling(50).mean(), "Up", "Down")

    rows = []
    for col in [c for c in ("Volatility", "Markov state", "Attention", "Trend") if c in d]:
        for state, sub in d.groupby(col):
            if sub[[feature, f"fwd_{h}"]].dropna().shape[0] < 60:
                continue
            r = predictive_regression(sub, feature, h)
            rows.append({"regime": col, "state": state, **{k: r[k] for k in
                         ("coef_bp", "t_stat", "p_value", "n")}})
    return pd.DataFrame(rows)


def rolling_coef(df: pd.DataFrame, feature: str = "score", h: int = 1,
                 window: int = 180) -> pd.Series:
    """Coefficient of the feature in a rolling-window regression (stability check)."""
    data = df[[f"fwd_{h}", feature] + CONTROLS].dropna()
    vals = {}
    for end in range(window, len(data), 5):
        w = data.iloc[end - window: end]
        b = sm.OLS(w[f"fwd_{h}"], sm.add_constant(w[[feature] + CONTROLS])).fit().params[feature]
        vals[data.index[end - 1]] = b * 1e4
    return pd.Series(vals, name="coef_bp")
