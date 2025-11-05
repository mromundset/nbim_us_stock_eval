import os
import pandas as pd
import numpy as np
from typing import Dict, Optional

# -------------------------------------------
# Column name mappings and sector-specific weights
# -------------------------------------------

RATIO_ALIASES: Dict[str, str] = {
    "ROE": "Return On Equity",
    "ROIC": "Return On Invested Capital",
    "OPM": "Operating Profit Margin",
    "CR": "Current Ratio",
}

SECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Technology":          {"ROE": 0.20, "ROIC": 0.35, "Margin": 0.35, "Liquidity": 0.10},
    "Financial Services":  {"ROE": 0.45, "ROIC": 0.30, "Margin": 0.15, "Liquidity": 0.10},
    "Healthcare":          {"ROE": 0.25, "ROIC": 0.30, "Margin": 0.35, "Liquidity": 0.10},
    "Consumer Cyclical":   {"ROE": 0.25, "ROIC": 0.25, "Margin": 0.35, "Liquidity": 0.15},
    "Communication Services": {"ROE": 0.20, "ROIC": 0.25, "Margin": 0.40, "Liquidity": 0.15},
    "Industrials":         {"ROE": 0.25, "ROIC": 0.40, "Margin": 0.25, "Liquidity": 0.10},
    "Consumer Defensive":  {"ROE": 0.30, "ROIC": 0.25, "Margin": 0.35, "Liquidity": 0.10},
    "Energy":              {"ROE": 0.25, "ROIC": 0.40, "Margin": 0.25, "Liquidity": 0.10},
    "Real Estate":         {"ROE": 0.40, "ROIC": 0.30, "Margin": 0.20, "Liquidity": 0.10},
    "Utilities":           {"ROE": 0.40, "ROIC": 0.30, "Margin": 0.20, "Liquidity": 0.10},
    "Basic Materials":     {"ROE": 0.25, "ROIC": 0.35, "Margin": 0.30, "Liquidity": 0.10},
}

DEFAULT_WEIGHTS = {"ROE": 0.25, "ROIC": 0.25, "Margin": 0.25, "Liquidity": 0.25}

# Cap thresholds for each metric (winsorization)
CAPS = {
    "roe_score": 5.0,
    "roic_score": 5.0,
    "margin_score": 3.0,
    "liquidity_score": 3.0,
}

# -------------------------------------------
# Helper functions
# -------------------------------------------

def _resolve_ratio_col(cols, ratio: str) -> str:
    """Find the actual column name in df given short alias."""
    if ratio in cols:
        return ratio
    if ratio in RATIO_ALIASES and RATIO_ALIASES[ratio] in cols:
        return RATIO_ALIASES[ratio]
    rlow = ratio.lower()
    cands = [c for c in cols if rlow in c.lower()]
    if len(cands) == 1:
        return cands[0]
    raise ValueError(f"Could not resolve ratio '{ratio}'. Available: {', '.join(cols)}")

# -------------------------------------------
# Main function
# -------------------------------------------

def run(
    input_data: str = "data\sector_benchmark\sector_performance_ratios_neg_handled.csv",
    out_name: str = "quality_scores_2023.csv",
    report_name: str = "quality_scores_2023_report.txt",
    sector_col: str = "Sector",
    company_col: str = "Company Name",
    year_col: str = "Calendar Year",
    year: int = 2023,
    round_digits: int = 3,
) -> None:
    """
    Compute a quality score for each company in a given year.
    Relativizes ROE, ROIC, Margin, and Liquidity to the sector-year median.
    Sector-specific weights are applied, and sub-scores are winsorized to prevent domination.
    """

    # ---------- Setup ----------
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_csv  = os.path.join(out_dir, out_name)
    out_rpt  = os.path.join(out_dir, report_name)

    # ---------- Load data ----------
    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")
    for c in (sector_col, company_col, year_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {input_data}")

    # Resolve ratio columns
    roe_col  = _resolve_ratio_col(df.columns, "ROE")
    roic_col = _resolve_ratio_col(df.columns, "ROIC")
    marg_col = _resolve_ratio_col(df.columns, "OPM")
    cr_col   = _resolve_ratio_col(df.columns, "CR")

    # Convert to numeric
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    for c in [roe_col, roic_col, marg_col, cr_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter to target year (latest)
    sub = df[df[year_col] == int(year)].copy()
    if sub.empty:
        print(f"No data found for year {year}")
        return

    # ---------- Compute medians by sector ----------
    med = (
        sub.groupby(sector_col)[[roe_col, roic_col, marg_col, cr_col]]
        .median(numeric_only=True)
        .rename(columns={
            roe_col: "med_roe",
            roic_col: "med_roic",
            marg_col: "med_margin",
            cr_col: "med_cr",
        })
    )
    sub = sub.merge(med, left_on=sector_col, right_index=True, how="left")

    # ---------- Relative scoring ----------
    def safe_div(a, b):
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b

    sub["roe_score"]       = [safe_div(a, b) for a, b in zip(sub[roe_col],  sub["med_roe"])]
    sub["roic_score"]      = [safe_div(a, b) for a, b in zip(sub[roic_col], sub["med_roic"])]
    sub["margin_score"]    = [safe_div(a, b) for a, b in zip(sub[marg_col], sub["med_margin"])]
    sub["liquidity_score"] = [safe_div(a, b) for a, b in zip(sub[cr_col],   sub["med_cr"])]

    # ---------- Winsorize sub-scores ----------
    for col, cap in CAPS.items():
        sub[col] = np.clip(sub[col], 0, cap)

    # ---------- Apply sector-specific weights ----------
    total_scores = []
    for _, row in sub.iterrows():
        sector = str(row[sector_col]).strip()
        weights = SECTOR_WEIGHTS.get(sector, DEFAULT_WEIGHTS)

        total = (
            (row["roe_score"]       * weights.get("ROE", 0)) +
            (row["roic_score"]      * weights.get("ROIC", 0)) +
            (row["margin_score"]    * weights.get("Margin", 0)) +
            (row["liquidity_score"] * weights.get("Liquidity", 0))
        )
        total_scores.append(total)

    sub["total_health"] = total_scores

    # For transparency, include the weights used
    sub["roe_w"]       = sub[sector_col].map(lambda s: SECTOR_WEIGHTS.get(s, DEFAULT_WEIGHTS)["ROE"])
    sub["roic_w"]      = sub[sector_col].map(lambda s: SECTOR_WEIGHTS.get(s, DEFAULT_WEIGHTS)["ROIC"])
    sub["margin_w"]    = sub[sector_col].map(lambda s: SECTOR_WEIGHTS.get(s, DEFAULT_WEIGHTS)["Margin"])
    sub["liquidity_w"] = sub[sector_col].map(lambda s: SECTOR_WEIGHTS.get(s, DEFAULT_WEIGHTS)["Liquidity"])

    # ---------- Output ----------
    out = sub[
        [company_col, sector_col, "total_health", "roe_score", "roic_score",
        "margin_score", "liquidity_score", "roe_w", "roic_w", "margin_w", "liquidity_w"]
    ].copy()

    # Sort by total_health descending
    out = out.round(round_digits).sort_values("total_health", ascending=False).reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    # ---------- Report ----------
    with open(out_rpt, "w", encoding="utf-8") as f:
        f.write("# Quality Scoring Framework Report\n")
        f.write(f"Input: {input_data}\nYear: {year}\n\n")
        f.write("Sector-specific weights applied:\n")
        for s, w in SECTOR_WEIGHTS.items():
            f.write(f"  {s}: {w}\n")
        f.write(f"\nCaps applied: {CAPS}\n")
        f.write(f"Output file: {out_csv}\nRows: {len(out)}\n")

    print(f"→ {out_csv}")
    print(f"→ Report -> {out_rpt}")


if __name__ == "__main__":
    run(
        input_data="out/company_vs_sector_ratio_cleaned.csv",
        out_name="quality_scores_2023.csv",
        report_name="quality_scores_2023_report.txt",
        year=2023,
    )
