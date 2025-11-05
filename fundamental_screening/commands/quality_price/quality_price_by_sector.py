import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

# ------------------------------
# Column aliases & sector config
# ------------------------------

RATIO_ALIASES: Dict[str, str] = {
    "PE": "P/E Ratio",
    "P/E": "P/E Ratio",
    "P/S": "P/S Ratio",
    "PS": "P/S Ratio",
}

# Choose which valuation metric to use for each sector
SECTOR_VALUATION_METRIC: Dict[str, str] = {
    "Technology": "P/S Ratio",
    "Communication Services": "P/S Ratio",
    "Healthcare": "P/E Ratio",
    "Financial Services": "P/E Ratio",
    "Industrials": "P/E Ratio",
    "Consumer Cyclical": "P/E Ratio",
    "Consumer Defensive": "P/E Ratio",
    "Energy": "P/E Ratio",
    "Utilities": "P/E Ratio",
    "Real Estate": "P/E Ratio",   # if you later have P/FFO, switch it here
    "Basic Materials": "P/E Ratio",
}

DEFAULT_VALUATION_METRIC = "P/E Ratio"


def _resolve_ratio_col(cols, ratio: str) -> str:
    """Best-effort resolver for a ratio column; supports short aliases."""
    if ratio in cols:
        return ratio
    if ratio in RATIO_ALIASES and RATIO_ALIASES[ratio] in cols:
        return RATIO_ALIASES[ratio]
    rlow = ratio.lower()
    cands = [c for c in cols if rlow in c.lower()]
    if len(cands) == 1:
        return cands[0]
    raise ValueError(f"Could not resolve ratio '{ratio}'. Available: {', '.join(cols)}")


# -----------------------------------
# Core helpers (pure, unit-testable)
# -----------------------------------

def _auto_or_specific_year(df: pd.DataFrame, year_col: str, year: Optional[int], auto_max_year: bool) -> int:
    if auto_max_year or year is None:
        y = pd.to_numeric(df[year_col], errors="coerce")
        y = y.dropna().astype(int)
        if y.empty:
            raise ValueError("No valid years found to auto-select.")
        return int(y.max())
    return int(year)


def _attach_sector_metric(df: pd.DataFrame, sector_col: str, price_is_relative: bool) -> Tuple[pd.DataFrame, str]:
    """
    Attach a 'price_metric' column selecting P/E or P/S per row based on sector.
    Returns df (copy) and the name 'price_metric'.
    """
    df = df.copy()
    # Resolve both price columns if present
    cols = set(df.columns)
    pe_col = _resolve_ratio_col(cols, "P/E Ratio") if any("P/E Ratio" == c or "P/E" in c for c in cols) else None
    ps_col = _resolve_ratio_col(cols, "P/S Ratio") if any("P/S Ratio" == c or "P/S" in c for c in cols) else None
    if pe_col is None and ps_col is None:
        raise ValueError("Neither P/E nor P/S columns were found in the ratios file.")

    # For each row, choose per-sector metric; fallback to whichever exists
    chosen = []
    for _, r in df.iterrows():
        sector = str(r[sector_col]).strip()
        want = SECTOR_VALUATION_METRIC.get(sector, DEFAULT_VALUATION_METRIC)
        if want.startswith("P/E") and pe_col is not None:
            chosen.append(r[pe_col])
        elif want.startswith("P/S") and ps_col is not None:
            chosen.append(r[ps_col])
        else:
            # graceful fallback
            fallback = r[pe_col] if pe_col is not None else r[ps_col]
            chosen.append(fallback)

    df["price_metric"] = pd.to_numeric(chosen, errors="coerce")

    # If NOT already relative, normalize to sector median (company ÷ sector median)
    if not price_is_relative:
        med = df.groupby(sector_col)["price_metric"].transform("median")
        df["price_metric"] = np.where((med == 0) | (~np.isfinite(med)), np.nan, df["price_metric"] / med)

    return df, "price_metric"


def _sector_relative(df: pd.DataFrame, sector_col: str, col: str, out_col: str) -> pd.DataFrame:
    """Make a column relative to the sector median: company ÷ sector_median(col)."""
    df = df.copy()
    med = df.groupby(sector_col)[col].transform("median")
    df[out_col] = np.where((med == 0) | (~np.isfinite(med)), np.nan, df[col] / med)
    return df


def _z_by_sector(df: pd.DataFrame, sector_col: str, col: str, out_col: str) -> pd.DataFrame:
    """Within-sector z-score of a column that’s already centered around ~1 (relative)."""
    df = df.copy()
    grp = df.groupby(sector_col)[col]
    std = grp.transform("std")
    # center on 1.0 (the sector median for relative metrics)
    df[out_col] = np.where((std == 0) | (~np.isfinite(std)), 0.0, (df[col] - 1.0) / std)
    return df


def _fit_lnprice_on_quality(df: pd.DataFrame, rel_price_col: str, rel_quality_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit ln(rel_price) ~ a + b * rel_quality using simple OLS (np.polyfit with deg=1).
    Returns (pred, resid).
    """
    x = pd.to_numeric(df[rel_quality_col], errors="coerce")
    y_raw = pd.to_numeric(df[rel_price_col], errors="coerce")
    # guard: positivity for log
    y = y_raw.copy()
    y[~np.isfinite(y) | (y <= 0)] = np.nan
    x_mask = x.notna()
    y_mask = y.notna()
    mask = x_mask & y_mask
    pred = np.full(len(df), np.nan)
    resid = np.full(len(df), np.nan)
    if mask.sum() < 8:
        # too few points; return NaNs (we can still use spread-based QARP)
        return pred, resid
    coeff = np.polyfit(x[mask], np.log(y[mask]), 1)  # ln(y) = a + b * x
    a, b = coeff[1], coeff[0]
    pred_ln = a + b * x
    pred = np.exp(pred_ln)
    resid = np.log(y) - pred_ln
    return pred, resid

def _plot_sector_subplot(df, sector_col, company_col, x_col, y_col, out_dir, log_x=True,
                        percentile_clip_x=None, label_top_n=6):
    """Create one scatter plot per sector showing relative price vs relative quality."""
    for sector, g in df.groupby(sector_col):
        if g.empty:
            continue
        x = g[x_col].values.copy()
        y = g[y_col].values.copy()

        if percentile_clip_x is not None and np.isfinite(x).any():
            cap = np.nanpercentile(x, float(percentile_clip_x))
            x = np.clip(x, 1e-9, cap)

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, s=22, alpha=0.7, edgecolor="none", color="#1f77b4")
        plt.axvline(1.0, color="gray", linestyle="--", linewidth=1)
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)

        if log_x:
            plt.xscale("log")
            try:
                plt.xticks([0.5, 1.0, 1.5, 2.0], ["0.5×", "1×", "1.5×", "2×"])
            except Exception:
                pass

        plt.xlabel("Relative Price (sector-normalized P/E or P/S)")
        plt.ylabel("Relative Quality (sector-normalized total_health)")
        plt.title(f"{sector} — QARP Map")

        # Label top-N within sector by highest quality-to-price ratio
        g_sorted = g.sort_values(y_col, ascending=False)
        for _, r in g_sorted.head(label_top_n).iterrows():
            xi, yi = r[x_col], r[y_col]
            if not np.isfinite(xi) or not np.isfinite(yi):
                continue
            xi_plot = min(xi, np.nanpercentile(g[x_col].values, float(percentile_clip_x))) if percentile_clip_x else xi
            plt.annotate(str(r[company_col])[:24], (xi_plot, yi), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, alpha=0.9)

        plt.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        sector_slug = sector.replace(" ", "_").replace("/", "-")
        path = os.path.join(out_dir, f"qarp_scatter_{sector_slug}.png")
        plt.savefig(path, dpi=300)
        plt.close()

# ---------------------------
# Main pipeline entrypoint
# ---------------------------

def run(
    input_quality: str = "out/quality_scores/quality_scores_2023.csv",
    input_ratios: str = "data\sector_benchmark\sector_performance_ratios_neg_handled.csv",
    out_ranked: str = "qarp_ranked_2023.csv",
    out_candidates: str = "qarp_candidates_2023.csv",
    out_plot: str = "qarp_scatter_2023.png",
    out_report: str = "qarp_report_2023.txt",
    sector_col: str = "Sector",
    company_col: str = "Company Name",
    year_col: str = "Calendar Year",
    year: Optional[int] = 2023,
    auto_max_year: bool = False,
    price_is_relative: bool = True,     # your ratios file already uses company ÷ sector median
    rel_quality_min: float = 1.05,      # baseline screen: above-sector quality
    rel_price_max: float = 1.00,        # baseline screen: at/below-sector price
    qarp_score_min: Optional[float] = None,  # optional extra filter on score (e.g., 0.5)
    min_obs_per_sector_for_z: int = 6,
    use_residual_model: bool = False,
    label_top_n: int = 10,
    log_x: bool = True,
    percentile_clip_x: Optional[float] = 99.5,
    by_sector: bool = True
) -> None:
    """
    Identify 'Quality at a Reasonable Price' (QARP) opportunities by combining sector-relative
    quality with sector-relative price (P/E or P/S chosen per sector), rank, plot, and report.
    """

    # ---------- I/O setup ----------
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    p_ranked = os.path.join(out_dir, out_ranked)
    p_cands  = os.path.join(out_dir, out_candidates)
    p_plot   = os.path.join(out_dir, out_plot)
    p_report = os.path.join(out_dir, out_report)

    # ---------- Load inputs ----------
    q = pd.read_csv(input_quality)
    r = pd.read_csv(input_ratios)

    # Year selection on ratios (quality CSV is already a single year)
    if year is None and not auto_max_year:
        auto_max_year = True
    target_year = _auto_or_specific_year(r, year_col, year, auto_max_year)
    r[year_col] = pd.to_numeric(r[year_col], errors="coerce")
    r_year = r[r[year_col] == target_year].copy()

    # Merge quality and ratios on Company + Sector
    base_cols = [company_col, sector_col]
    needed_q = base_cols + ["total_health"]
    for c in needed_q:
        if c not in q.columns:
            raise ValueError(f"Column '{c}' must exist in {input_quality}")
    for c in base_cols + [year_col]:
        if c not in r_year.columns:
            raise ValueError(f"Column '{c}' must exist in {input_ratios}")

    df = r_year.merge(q[needed_q], on=base_cols, how="inner")

    if df.empty:
        raise ValueError("No intersection between quality CSV and ratios CSV for the selected year.")

    # ---------- Attach sector price metric (P/E or P/S) ----------
    df, price_col = _attach_sector_metric(df, sector_col=sector_col, price_is_relative=price_is_relative)

    # Ensure numeric & finite for key columns
    df["total_health"] = pd.to_numeric(df["total_health"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # ---------- Sector-relative transforms ----------
    # Make quality relative to sector median:
    df = _sector_relative(df, sector_col, "total_health", "rel_quality")
    # Price metric is already relative if price_is_relative=True; if not, we normalized above.
    df["rel_price"] = df[price_col]

    # Drop rows missing either axis
    df = df[np.isfinite(df["rel_quality"]) & np.isfinite(df["rel_price"])].copy()

    # ---------- Optional: z-scores by sector (guard small sectors) ----------
    # Mark sectors large enough
    sector_counts = df[sector_col].value_counts()
    big_sectors = set(sector_counts[sector_counts >= min_obs_per_sector_for_z].index)

    df["z_quality"] = np.nan
    df["z_price"] = np.nan
    # Compute only for "big" sectors; else leave NaN (we'll fall back on rel filters)
    for s in big_sectors:
        mask = df[sector_col] == s
        df.loc[mask, "z_quality"] = _z_by_sector(df.loc[mask], sector_col, "rel_quality", "_tmp")._tmp.values
        df.loc[mask, "z_price"]   = _z_by_sector(df.loc[mask], sector_col, "rel_price", "_tmp")._tmp.values

    # ---------- Optional: residual model (ln price vs quality) ----------
    df["price_hat"] = np.nan
    df["price_resid"] = np.nan
    df["price_resid_std"] = np.nan
    if use_residual_model:
        pred, resid = _fit_lnprice_on_quality(df, rel_price_col="rel_price", rel_quality_col="rel_quality")
        df["price_hat"] = pred
        df["price_resid"] = resid
        # Standardize residuals (ignore NaNs)
        resid_std = np.nanstd(resid, ddof=1) if np.isfinite(resid).sum() > 1 else np.nan
        if np.isfinite(resid_std) and resid_std > 0:
            df["price_resid_std"] = df["price_resid"] / resid_std

    # ---------- QARP scores ----------
    # Spread-based (works even if no residual model or small sectors):
    df["QARP_spread"] = (df["z_quality"].fillna(0.0)) - (df["z_price"].fillna(0.0))

    # Residual-based (reward high quality, penalize positive price residuals)
    if use_residual_model and np.isfinite(df["price_resid_std"]).any():
        df["QARP_resid"] = (df["z_quality"].fillna(0.0)) - (df["price_resid_std"].fillna(0.0))
        primary_score_col = "QARP_resid"
    else:
        primary_score_col = "QARP_spread"

    # ---------- Baseline candidate filters ----------
    cand = df[
        (df["rel_quality"] >= float(rel_quality_min)) &
        (df["rel_price"]   <= float(rel_price_max))
    ].copy()
    if qarp_score_min is not None:
        cand = cand[cand[primary_score_col] >= float(qarp_score_min)]

    # ---------- Ranking ----------
    df = df.sort_values(primary_score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    df["rank_qarf"] = np.arange(1, len(df) + 1)

    cand = cand.sort_values(primary_score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    cand["rank_qarf"] = np.arange(1, len(cand) + 1)

    # ---------- Save tables ----------
    cols_ranked = [
        company_col, sector_col, year_col,
        "total_health", "rel_quality", "z_quality",
        price_col, "rel_price", "z_price",
        "price_hat", "price_resid", "price_resid_std",
        "QARP_spread", "QARP_resid" if "QARP_resid" in df.columns else primary_score_col,
        primary_score_col, "rank_qarf"
    ]
    # Keep only existing columns
    cols_ranked = [c for c in cols_ranked if c in df.columns]
    df[cols_ranked].to_csv(p_ranked, index=False)

    cols_cand = [c for c in cols_ranked if c in cand.columns]
    cand[cols_cand].to_csv(p_cands, index=False)

    x_all = df["rel_price"].values.copy()
    y_all = df["rel_quality"].values.copy()
    if percentile_clip_x is not None and np.isfinite(x_all).any():
        cap = np.nanpercentile(x_all, float(percentile_clip_x))
        x_all = np.clip(x_all, 1e-9, cap)  # keep >0 for potential log scale

    # Build a stable color map by sector
    sectors = sorted(df[sector_col].dropna().unique().tolist())
    # tab20 has 20 distinct colors; if more sectors, fallback to hsv spreading
    if len(sectors) <= 20:
        cmap = plt.get_cmap("tab20")
        color_map = {s: cmap(i % 20) for i, s in enumerate(sectors)}
    else:
        cmap = plt.get_cmap("hsv")
        color_map = {s: cmap(i / max(len(sectors), 1)) for i, s in enumerate(sectors)}

    plt.figure(figsize=(11, 7))

    # Plot each sector with its own color
    for s in sectors:
        g = df[df[sector_col] == s]
        x = g["rel_price"].values.copy()
        y = g["rel_quality"].values.copy()
        if percentile_clip_x is not None and np.isfinite(x).any():
            local_cap = np.nanpercentile(df["rel_price"].values, float(percentile_clip_x))
            x = np.clip(x, 1e-9, local_cap)
        plt.scatter(
            x, y,
            s=22, alpha=0.7, edgecolor="none",
            label=str(s), color=color_map[s]
        )

    # Quadrant reference lines
    plt.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    if log_x:
        plt.xscale("log")
        try:
            plt.xticks([0.5, 1.0, 1.5, 2.0], ["0.5×", "1×", "1.5×", "2×"])
        except Exception:
            pass

    plt.xlabel("Relative Price (sector-normalized P/E or P/S)")
    plt.ylabel("Relative Quality (sector-normalized total_health)")
    plt.title(f"QARP Map — {target_year}  (Top-left: High Quality, Low Price)")

    # Label top-N overall by the chosen primary score
    if label_top_n and label_top_n > 0:
        top = df.sort_values(primary_score_col, ascending=False).head(label_top_n)
        for _, rrow in top.iterrows():
            xi = rrow["rel_price"]; yi = rrow["rel_quality"]
            if not np.isfinite(xi) or not np.isfinite(yi) or xi <= 0:
                continue
            xi_plot = min(xi, np.nanpercentile(df["rel_price"].values, float(percentile_clip_x))) if percentile_clip_x else xi
            plt.annotate(
                str(rrow[company_col])[:28],
                (xi_plot, yi),
                textcoords="offset points", xytext=(4, 4),
                fontsize=8, alpha=0.9
            )

    # Place the legend below the plot to avoid clutter
    leg = plt.legend(
        title="Sector", fontsize=8, title_fontsize=9,
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(sectors), 4), frameon=False
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(p_plot, dpi=300)
    plt.close()

    # ---------- Report ----------
    with open(p_report, "w", encoding="utf-8") as f:
        f.write("# QARP Report\n")
        f.write(f"Quality CSV: {input_quality}\n")
        f.write(f"Ratios CSV:  {input_ratios}\n")
        f.write(f"Year used:   {target_year} (auto_max_year={auto_max_year})\n")
        f.write(f"Price metric by sector: {SECTOR_VALUATION_METRIC}\n")
        f.write(f"Price assumed relative: {price_is_relative}\n")
        f.write(f"Filters: rel_quality>={rel_quality_min}, rel_price<={rel_price_max}, qarp_score_min={qarp_score_min}\n")
        f.write(f"Primary score: {primary_score_col}\n")
        f.write(f"Ranked file: {p_ranked}  (rows={len(df)})\n")
        f.write(f"Candidates:  {p_cands}   (rows={len(cand)})\n")
        f.write(f"Plot:        {p_plot}\n\n")
        # Quick top-10 list
        f.write("Top-10 by QARP score:\n")
        top10 = df[[company_col, sector_col, primary_score_col]].head(10)
        for i, row in top10.iterrows():
            f.write(f"  {int(i+1):>2}. {row[company_col]} ({row[sector_col]}): {round(row[primary_score_col], 3)}\n")

    print(f"→ Ranked  -> {p_ranked}")
    print(f"→ Picks    -> {p_cands}")
    print(f"→ Plot     -> {p_plot}")
    print(f"→ Report   -> {p_report}")


if __name__ == "__main__":
    run(
        input_quality="out/quality_scores_2023.csv",
        input_ratios="out/company_vs_sector_ratio_cleaned.csv",
        out_ranked="qarp_ranked_2023.csv",
        out_candidates="qarp_candidates_2023.csv",
        out_plot="qarp_scatter_2023.png",
        out_report="qarp_report_2023.txt",
        year=2023,
        auto_max_year=False,      # set True to always use max available year in ratios file
        price_is_relative=True,   # your ratios are already relative (company ÷ sector median)
        rel_quality_min=1.05,
        rel_price_max=1.00,
        qarp_score_min=None,      # e.g., 0.5 if you want an extra hurdle
        use_residual_model=True,
        label_top_n=12,
        log_x=True,
        percentile_clip_x=99.5,
    )
