# commands/plot_sector_ratio_timeseries.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import numpy as np

RATIO_ALIASES: Dict[str, str] = {
    "P/E": "P/E Ratio", "PE": "P/E Ratio",
    "P/S": "P/S Ratio", "PS": "P/S Ratio",
    "ROE": "Return On Equity", "ROIC": "Return On Invested Capital",
    "ROA": "Return On Assets", "ROCE": "Return On Capital Employed",
    "GM": "Gross Profit Margin", "OPM": "Operating Profit Margin",
    "CR": "Current Ratio", "QR": "Quick Ratio", "ICR": "Interest Coverage Ratio",
    "WACC": "WACC", "RFR": "Risk Free Rate", "ERP": "Equity Risk Premium",
    "PreTaxCoD": "Pre-Tax Cost of Debt", "AfterTaxCoD": "After-Tax Cost of Debt",
    "Beta": "Beta",
}

def _resolve_ratio_col(cols, ratio: str) -> str:
    if ratio in cols:
        return ratio
    if ratio in RATIO_ALIASES and RATIO_ALIASES[ratio] in cols:
        return RATIO_ALIASES[ratio]
    rlow = ratio.lower()
    cands = [c for c in cols if rlow in c.lower()]
    if len(cands) == 1:
        return cands[0]
    raise ValueError(f"Could not resolve ratio '{ratio}'. Available: {', '.join(cols)}")

def _eligible_set_for_anchor(df: pd.DataFrame, company_col: str, ratio_col: str,
                             year_col: str, year: int,
                             min_c: Optional[float], max_c: Optional[float],
                             require_positive: bool) -> set:
    """Return companies whose ratio in `year` is within [min_c, max_c] (inclusive).
       If all of min_c/max_c/year are None, returns None to signal 'ignore this anchor'."""
    if year is None:
        return None
    a = df[df[year_col] == int(year)][[company_col, ratio_col]].copy()
    if require_positive:
        a = a[a[ratio_col] > 0]
    if min_c is not None:
        a = a[a[ratio_col] >= float(min_c)]
    if max_c is not None:
        a = a[a[ratio_col] <= float(max_c)]
    return set(a[company_col].dropna().unique())

def run(
    input_data: str = "data/sector_benchmark/sector_performance_ratios_neg_handled.csv",
    sector: str = "Utilities",
    ratio: str = "P/E",
    out_name: str = "utilities_pe.csv",
    plot_name: str = "utilities_pe.png",
    report_name: str = "sector_ratio_timeseries_report.txt",
    sector_col: str = "Sector",
    year_col: str = "Calendar Year",
    company_col: str = "Company Name",
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    log_scale: bool = True,
    line_alpha: float = 0.45,
    line_width: float = 1.2,
    round_digits: int = 3,
    # Anchor A
    min_cutoff: Optional[float] = 0.0,
    max_cutoff: Optional[float] = 1.0,
    anchor_year: int = 2023,
    # Anchor B
    min_cutoff_2: Optional[float] = 0.0,
    max_cutoff_2: Optional[float] = None,
    anchor_year_2: Optional[int] = 2018,
    # Display cap
    y_cap: float = 4.0,
) -> None:
    """
    Plot sector × ratio time series with color-coded legend.
    Include a company iff it satisfies ALL provided anchors:
      - Anchor A: ratio in `anchor_year` ∈ [min_cutoff, max_cutoff] (inclusive)
      - Anchor B: ratio in `anchor_year_2` ∈ [min_cutoff_2, max_cutoff_2] (inclusive)
    Visual cap at y_cap for plotting only.
    """
    out_dir = "out"; os.makedirs(out_dir, exist_ok=True)
    out_csv  = os.path.join(out_dir, out_name)
    out_rpt  = os.path.join(out_dir, report_name)
    out_plot = os.path.join(out_dir, plot_name)

    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")
    for c in (sector_col, year_col, company_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {input_data}")

    # Basic cleaning
    df[sector_col] = df[sector_col].astype(str).str.strip()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    ratio_col = _resolve_ratio_col(df.columns.tolist(), ratio)
    df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")

    # Sector subset (case-insensitive)
    sect_df = df[df[sector_col].str.casefold() == sector.strip().casefold()].copy()

    # ---------- Eligibility via anchors ----------
    require_pos = bool(log_scale)
    eligible_A = _eligible_set_for_anchor(
        sect_df, company_col, ratio_col, year_col,
        anchor_year, min_cutoff, max_cutoff, require_positive=require_pos
    )
    eligible_B = _eligible_set_for_anchor(
        sect_df, company_col, ratio_col, year_col,
        anchor_year_2, min_cutoff_2, max_cutoff_2, require_positive=require_pos
    )

    # Combine criteria: intersection of all non-None sets; if a set is None (anchor omitted), ignore it.
    elig_sets = [s for s in (eligible_A, eligible_B) if s is not None]
    if not elig_sets:
        # If no anchors provided at all, fall back to "everyone in sector"
        eligible = set(sect_df[company_col].dropna().unique())
    else:
        eligible = set.intersection(*elig_sets) if len(elig_sets) > 1 else elig_sets[0]

    if not eligible:
        # Write empty outputs with helpful report
        pd.DataFrame(columns=[company_col, year_col, ratio_col]).to_csv(out_csv, index=False)
        with open(out_rpt, "w", encoding="utf-8") as f:
            f.write("\n".join([
                "# Sector Ratio Time Series (no eligible companies)",
                f"Input: {input_data}",
                f"Sector: {sector}",
                f"Ratio: {ratio} -> '{ratio_col}'",
                f"Anchor A: year={anchor_year}, min={min_cutoff}, max={max_cutoff}",
                f"Anchor B: year={anchor_year_2}, min={min_cutoff_2}, max={max_cutoff_2}",
                "No companies met the anchor criteria.",
            ]))
        print(f"→ {out_csv}"); print(f"Report -> {out_rpt}")
        return

    # ---------- Full history for eligible companies ----------
    sub = sect_df[sect_df[company_col].isin(eligible)].copy()
    if year_min is not None:
        sub = sub[sub[year_col] >= int(year_min)]
    if year_max is not None:
        sub = sub[sub[year_col] <= int(year_max)]
    if log_scale:
        sub = sub[sub[ratio_col] > 0]  # log requirement for plotted points

    # Save full, un-capped values for CSV
    out_df = sub[[company_col, year_col, ratio_col]].dropna(subset=[ratio_col]).copy()
    out_df[ratio_col] = out_df[ratio_col].round(round_digits)
    out_df = out_df.sort_values([company_col, year_col])
    out_df.to_csv(out_csv, index=False)

    # Visual cap for plotting only
    sub["plot_ratio"] = np.where(sub[ratio_col] > y_cap, y_cap, sub[ratio_col])

    # ---------- Plot ----------
    plt.figure(figsize=(11, 6.5))
    unique_companies = list(sub[company_col].unique())
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i % 20) for i, c in enumerate(unique_companies)}

    for name, g in sub.groupby(company_col):
        plt.plot(
            g[year_col],
            g["plot_ratio"],
            color=colors[name],
            label=name,
            marker="o",
            lw=line_width,
            alpha=line_alpha,
        )

    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)

    if log_scale:
        plt.yscale("log")
        ticks = [0.25, 0.5, 1, 2, 4]
        plt.yticks(ticks, [f"{t}×" if t != 1 else "1×" for t in ticks])
    else:
        plt.ylim(0, y_cap * 1.1)

    plt.title(f"{sector} — {ratio_col} over Time (Relative to Sector Median)")
    plt.xlabel("Calendar Year")
    plt.ylabel(f"{ratio_col} (Company ÷ Sector Median)")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Legend below
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        fontsize=7,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_plot, dpi=300)
    plt.close()

    print(f"→ {out_csv}")
    print(f"→ Plot -> {out_plot}")

if __name__ == "__main__":
    run(
        input_data="out/company_vs_sector_ratio_cleaned.csv",
        sector="Technology",
        ratio="P/S",
        out_name="ts_tech_ps.csv",
        plot_name="ts_tech_ps.png",
        report_name="ts_tech_ps_report.txt",
        year_min=2014,
        year_max=2023,
        log_scale=True,
        # Anchor A (e.g., current year screen)
        anchor_year=2023, min_cutoff=0.6, max_cutoff=2.5,
        # Anchor B (e.g., require 2020 also be within a band)
        anchor_year_2=2020, min_cutoff_2=0.5, max_cutoff_2=3.0,
        y_cap=4.0,
    )