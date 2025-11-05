# main.py
import pandas as pd
import numpy as np
import json
import argparse
import datetime
from typing import List, Optional, Dict, Any, Literal, Tuple

from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

# ------------------------------------------------------------
# Registry
# ------------------------------------------------------------
REGISTRY: Dict[str, Any] = {}

def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco

def run(command: str, /, **kwargs):
    if command not in REGISTRY:
        raise ValueError(f"Unknown command: {command}. Available: {sorted(REGISTRY)}")
    return REGISTRY[command](**kwargs)

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------
def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

def write_any(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False); return
    if path.lower().endswith((".parquet", ".pq")):
        df.to_parquet(path, index=False); return
    raise ValueError(f"Unsupported file type: {path}")

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _coerce_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _detect_numeric(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def _fill_by_ticker(df: pd.DataFrame, numeric_cols: List[str], sort_cols: List[str]):
    present = [c for c in sort_cols if c in df.columns]
    if not present:
        present = ["Ticker"]
    df = df.sort_values(present)
    for c in numeric_cols:
        df[c] = df.groupby("Ticker")[c].ffill().bfill()
    return df

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def _per_date_transform(
    df: pd.DataFrame,
    features: List[str],
    sector_neutral: bool,
    normalize: Literal["zscore", "rank"]
) -> pd.DataFrame:
    # Demean by sector (per-date), then normalize per-date
    out = []
    for dt, g in df.groupby("price_date", sort=True):
        g = g.copy()
        # winsorize
        for f in features:
            g[f] = _winsorize(g[f])

        # optional sector de-mean per feature
        if sector_neutral and "Sector" in g.columns and g["Sector"].nunique() > 1:
            for f in features:
                g[f] = g[f] - g.groupby("Sector")[f].transform("mean")

        # normalize
        if normalize == "zscore":
            for f in features:
                std = g[f].std(ddof=0)
                g[f] = (g[f] - g[f].mean()) / (std if std and std > 0 else 1.0)
        elif normalize == "rank":
            for f in features:
                # percentile rank in [0,1]
                g[f] = g[f].rank(pct=True)
        else:
            raise ValueError("normalize must be 'zscore' or 'rank'")

        out.append(g)
    return pd.concat(out, axis=0)

def _adaptive_decile_spread(g: pd.DataFrame, min_names: int) -> float:
    if len(g) < min_names or g["pred"].nunique() < 3:
        return np.nan
    ranked = g["pred"].rank(method="first")
    for q in (10, 5, 3):
        if q <= len(g):
            try:
                dec = pd.qcut(ranked, q, labels=False, duplicates="drop")
                if dec.nunique() >= 2:
                    top = g.loc[dec == dec.max(), "ret_fwd"].mean()
                    bot = g.loc[dec == dec.min(), "ret_fwd"].mean()
                    return float(top - bot)
            except ValueError:
                pass
    return np.nan

# ------------------------------------------------------------
# Commands
# ------------------------------------------------------------
@register("clean_data")
def clean_data(
    input: str,
    output: str,
    zero_to_nan_cols: Optional[List[str]] = None,
    sector_col: str = "Sector",
    sort_cols: Optional[List[str]] = None,
    fill_sector: bool = True,
    fill_global: bool = True,
) -> Dict[str, Any]:
    df = read_any(input)
    df = _coerce_datetime(df, ["Price Dates", "Filing Date", "Price Date", "FilingDate"])
    numeric_cols = _detect_numeric(df)

    # 0 -> NaN for selected columns (ratios etc.)
    if zero_to_nan_cols:
        for c in zero_to_nan_cols:
            if c in df.columns:
                df.loc[df[c] == 0, c] = np.nan

    # temporal ffill/bfill within ticker
    sort_cols = sort_cols or ["Ticker", "Price Dates"]
    df = _fill_by_ticker(df, numeric_cols, sort_cols)

    # sector-wise fill
    if fill_sector and sector_col in df.columns:
        for c in numeric_cols:
            if df[c].isna().any():
                df[c] = df.groupby(sector_col)[c].transform(lambda x: x.fillna(x.median(skipna=True)))

    # global fallback
    if fill_global:
        for c in numeric_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median(skipna=True))

    nan_left = df[numeric_cols].isna().sum().sort_values(ascending=False)
    write_any(df, output)
    return {
        "input": input,
        "output": output,
        "rows": int(len(df)),
        "numeric_cols": len(numeric_cols),
        "remaining_nans": {k: int(v) for k, v in nan_left[nan_left > 0].items()},
    }

@register("train_model")
def train_model(
    input: str = "return_prediction_model\data\us_stock_valuation_clean_sorted_zero_handled.csv",
    output: str = "data/model_results.json",
    min_names_per_date: int = 20,                 # evaluate only dates with >= this many names
    horizon_min_days: int = 60,                   # keep labels with horizon in [min, max]
    horizon_max_days: int = 180,
    model_type: Literal["ridge", "gbdt", "randomforest"] = "gbdt",
    sector_neutral: bool = False,
    normalize: Literal["zscore", "rank"] = "rank",
    log_file: Optional[str] = "results/model_runs.txt",
    csv_log: Optional[str] = "results/model_runs.csv"
) -> Dict[str, Any]:
    """
    Train a pooled cross-sectional model to predict next-observation returns.

    - Labels: log return to the next observed price for the same ticker (variable horizon).
    - Features: fundamentals/ratios as-of the current date (guarded by Filing Date).
    - Split: earliest 70% price_dates train, latest 30% test.
    - Evaluation: per-date Rank IC (only when enough names), adaptive decile spread, pooled Spearman IC.
    """
    # ---------- Load & parse ----------
    df = read_any(input).copy()
    if "Price Dates" not in df.columns or "Price" not in df.columns:
        raise ValueError("Expected columns 'Price Dates' and 'Price' in the dataset.")
    df["price_date"] = pd.to_datetime(df["Price Dates"], errors="coerce")
    if "Filing Date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["Filing Date"], errors="coerce")
    df = df.sort_values(["Ticker", "price_date"])
    df = df[df["price_date"].notna() & df["Price"].notna()].copy()

    # ---------- Labels: next observed snapshot (vectorized) ----------
    df["next_date"]  = df.groupby("Ticker")["price_date"].shift(-1)
    df["next_price"] = df.groupby("Ticker")["Price"].shift(-1)
    df["days_ahead"] = (df["next_date"] - df["price_date"]).dt.days
    df["ret_fwd"]    = np.log(df["next_price"] / df["Price"])
    df = df.dropna(subset=["ret_fwd", "days_ahead"])
    df = df[(df["days_ahead"] >= horizon_min_days) & (df["days_ahead"] <= horizon_max_days)].copy()

    # ---------- As-of guard ----------
    if "filing_date" in df.columns:
        df = df[(df["filing_date"].isna()) | (df["filing_date"] <= df["price_date"])].copy()

    # ---------- Feature set ----------
    base_feats = [
        "P/E Ratio", "P/S Ratio",
        "Return On Equity", "Return On Invested Capital", "Return On Assets", "Return On Capital Employed",
        "Gross Profit Margin", "Operating Profit Margin",
        "Beta", "Current Ratio", "Quick Ratio"
    ]
    feats = [c for c in base_feats if c in df.columns]

    if "P/E Ratio" in df.columns:
        df["inv_pe"] = 1.0 / df["P/E Ratio"]
        df["inv_pe"].replace([np.inf, -np.inf], np.nan, inplace=True)
    if "P/S Ratio" in df.columns:
        df["inv_ps"] = 1.0 / df["P/S Ratio"]
        df["inv_ps"].replace([np.inf, -np.inf], np.nan, inplace=True)

    for extra in ["inv_pe", "inv_ps"]:
        if extra in df.columns:
            feats.append(extra)
    feats = list(dict.fromkeys(feats))  # preserve order, dedupe

    if not feats:
        raise ValueError("No available feature columns found.")
    df = df.dropna(subset=feats)

    # ---------- Per-date transform ----------
    df = _per_date_transform(df, feats, sector_neutral=sector_neutral, normalize=normalize)

    # impute any remaining gaps
    imp = SimpleImputer(strategy="median")
    df[feats] = imp.fit_transform(df[feats])

    # ---------- Time split (70/30 by unique price dates) ----------
    unique_dates = np.sort(df["price_date"].unique())
    cut = int(len(unique_dates) * 0.7) if len(unique_dates) > 2 else 1
    train_dates = set(unique_dates[:cut])
    test_dates  = set(unique_dates[cut:])

    tr = df[df["price_date"].isin(train_dates)].copy()
    te = df[df["price_date"].isin(test_dates)].copy()

    X_tr, y_tr = tr[feats].values, tr["ret_fwd"].values
    X_te, y_te = te[feats].values, te["ret_fwd"].values

    # ---------- Model selection ----------
    model_name = model_type.lower()
    coef_source: Literal["coef_", "feature_importances_", "none"] = "none"

    if model_name == "ridge":
        model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        coef_source = "coef_"
    elif model_name == "randomforest":
        model = RandomForestRegressor(n_estimators=200, max_depth=6, n_jobs=-1, random_state=42)
        coef_source = "feature_importances_"
    elif model_name == "gbdt":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError("lightgbm is not installed. `pip install lightgbm` to use model_type='gbdt'.") from e
        model = LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=300, random_state=42)
        coef_source = "feature_importances_"
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    model.fit(X_tr, y_tr)
    te["pred"] = model.predict(X_te)

    # ---------- Evaluation ----------
    ics = []
    for dt, g in te.groupby("price_date", sort=True):
        if len(g) < min_names_per_date: 
            continue
        if g["pred"].nunique() < 3 or g["ret_fwd"].nunique() < 3:
            continue
        ics.append(g["pred"].corr(g["ret_fwd"], method="spearman"))
    ics = pd.Series(ics, dtype=float)
    mean_ic = float(ics.mean()) if len(ics) else np.nan
    tstat_ic = float(mean_ic / (ics.std(ddof=1) / np.sqrt(len(ics)))) if len(ics) > 2 else np.nan

    spreads = []
    for dt, g in te.groupby("price_date", sort=True):
        spreads.append(_adaptive_decile_spread(g, min_names_per_date))
    spreads = pd.Series([s for s in spreads if pd.notna(s)], dtype=float)
    avg_spread = float(spreads.mean()) if len(spreads) else np.nan

    pooled_ic = float(spearmanr(te["pred"], te["ret_fwd"]).statistic) if len(te) > 3 else np.nan

    if coef_source == "coef_" and hasattr(model, "coef_"):
        coef_values = list(map(float, model.coef_))
    elif coef_source == "feature_importances_" and hasattr(model, "feature_importances_"):
        coef_values = list(map(float, model.feature_importances_))
    else:
        coef_values = [np.nan] * len(feats)
    coef_map = {f: float(c) for f, c in zip(feats, coef_values)}

    summary: Dict[str, Any] = {
        "mean_rank_ic": None if pd.isna(mean_ic) else mean_ic,
        "rank_ic_tstat": None if pd.isna(tstat_ic) else tstat_ic,
        "avg_decile_spread": None if pd.isna(avg_spread) else avg_spread,
        "pooled_spearman_ic": None if pd.isna(pooled_ic) else pooled_ic,
        "n_test_periods_used": int(len(ics)),
        "min_names_per_date": int(min_names_per_date),
        "horizon_days_band": [int(horizon_min_days), int(horizon_max_days)],
        "sector_neutral": bool(sector_neutral),
        "normalize": normalize,
        "model_type": model_name,
        "features": feats,
        "coefficients": coef_map,
    }

    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---------- Human-readable TXT log (optional) ----------
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("-" * 60 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Sector neutral: {sector_neutral}\n")
            f.write(f"Normalize: {normalize}\n")
            f.write(f"Horizon: {horizon_min_days}-{horizon_max_days}\n")
            f.write(f"Min names/date: {min_names_per_date}\n")
            f.write(f"Mean Rank IC: {summary['mean_rank_ic']}\n")
            f.write(f"Rank IC t-stat: {summary['rank_ic_tstat']}\n")
            f.write(f"Avg Decile Spread: {summary['avg_decile_spread']}\n")
            f.write(f"Pooled Spearman IC: {summary['pooled_spearman_ic']}\n")
            f.write("Top features (by |value|):\n")
            for k, v in sorted(coef_map.items(), key=lambda kv: -abs(kv[1]))[:10]:
                f.write(f"  {k}: {v:+.4f}\n")
            f.write("-" * 60 + "\n\n")

    # ---------- Structured CSV log (optional) ----------
    if csv_log:
        import os, csv as pycsv
        row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model_type": model_name,
            "normalize": normalize,
            "sector_neutral": sector_neutral,
            "horizon_min": horizon_min_days,
            "horizon_max": horizon_max_days,
            "mean_rank_ic": mean_ic,
            "rank_ic_tstat": tstat_ic,
            "avg_decile_spread": avg_spread,
            "pooled_spearman_ic": pooled_ic,
            "n_test_periods_used": len(ics),
            "features_used": "|".join(feats),
            "top_features": "; ".join([f"{k}:{coef_map[k]:+.4f}" for k in sorted(coef_map, key=lambda kk: -abs(coef_map[kk]))[:5]])
        }
        file_exists = os.path.exists(csv_log)
        with open(csv_log, "a", newline="", encoding="utf-8") as f:
            writer = pycsv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    return summary

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def _parse_kv(args_list):
    out: Dict[str, Any] = {}
    for item in args_list or []:
        if "=" not in item:
            raise ValueError(f"Invalid param '{item}'. Expected key=value.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        # try JSON for lists/bools/numbers; fallback to string
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser("NBIM micro-CLI")
    p.add_argument("--command", required=True)
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--params", nargs="*")
    a = p.parse_args()
    extra = _parse_kv(a.params)
    if a.input:  extra.setdefault("input", a.input)
    if a.output: extra.setdefault("output", a.output)
    res = run(a.command, **extra)
    if res is not None:
        print(json.dumps(res, indent=2, default=str))
