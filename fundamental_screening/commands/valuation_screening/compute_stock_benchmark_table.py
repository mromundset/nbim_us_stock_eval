import os
import pandas as pd
import numpy as np
from typing import List

# Keep only these ratio-like metrics
KEEP_COLS = [
    "Ticker", "Company Name", "Sector", "Industry", "Calendar Year",
    "Beta",
    "P/E Ratio", "P/S Ratio", "Price", "Market Cap",
    "Return On Equity", "Return On Invested Capital", "Return On Assets", "Return On Capital Employed",
    "Gross Profit Margin", "Operating Profit Margin",
    "Current Ratio", "Quick Ratio", "Interest Coverage Ratio",
    "WACC", "Risk Free Rate", "Equity Risk Premium",
    "Pre-Tax Cost of Debt", "After-Tax Cost of Debt",
    "Effective Tax Rate"
]

def _numericize(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def run(
    company_yearly_path: str = "data/0_adjusted/company_collapsed_year.csv",
    sector_medians_path: str = "data/sector_medians_2014_2023.csv",
    out_name: str = "company_vs_sector_ratio.csv",
    report_name: str = "company_vs_sector_ratio_report.txt",
    sector_col: str = "Sector",
    year_col: str = "Calendar Year",
    round_digits: int = 3,
) -> None:
    """
    For each numeric metric, divide company-year values by the corresponding
    sector-year median (same metric). Drops non-ratio financial columns.
    """

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, out_name)
    out_rpt = os.path.join(out_dir, report_name)

    comp = pd.read_csv(company_yearly_path, dtype=str, keep_default_na=False, na_filter=False, engine="python")
    sect = pd.read_csv(sector_medians_path, dtype=str, keep_default_na=False, na_filter=False, engine="python")

    # keep only desired columns
    cols_to_keep = [c for c in KEEP_COLS if c in comp.columns]
    comp = comp[cols_to_keep]
    sect = sect[[c for c in KEEP_COLS if c in sect.columns]]

    # numeric cast
    comp[year_col] = pd.to_numeric(comp[year_col], errors="coerce")
    sect[year_col] = pd.to_numeric(sect[year_col], errors="coerce")

    metrics = [c for c in comp.columns if c not in {"Ticker", "Company Name", "Sector", "Industry", "Calendar Year"}]
    _numericize(comp, metrics)
    _numericize(sect, metrics)

    # merge sector medians
    sect_ren = {m: f"Sector Median {m}" for m in metrics}
    sect = sect.rename(columns=sect_ren)
    merged = comp.merge(
        sect[[sector_col, year_col, *sect_ren.values()]],
        on=[sector_col, year_col],
        how="left",
        validate="m:1"
    )

    # compute ratios
    for m in metrics:
        sm = f"Sector Median {m}"
        merged[m] = merged[m].astype(float) / merged[sm].astype(float)
        merged[m] = merged[m].where(merged[sm].notna() & (merged[sm] != 0))
        merged[m] = merged[m].round(round_digits)

    # drop sector median helper columns
    merged = merged[[c for c in KEEP_COLS if c in merged.columns]]

    # save
    merged.to_csv(out_csv, index=False)

    # report
    lines = [
        "# Company vs Sector Ratios (cleaned)",
        f"Company input: {company_yearly_path}",
        f"Sector medians: {sector_medians_path}",
        f"Rows: {len(merged):,}",
        "",
        f"Columns retained ({len(merged.columns)}): {', '.join(merged.columns)}",
        "",
        "Preview (first 10 rows):",
        merged.head(10).to_string(index=False),
        "",
        f"Saved → {out_csv}"
    ]
    with open(out_rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"→ {out_csv}")
    print(f"Report -> {out_rpt}")

if __name__ == "__main__":
    run(
        company_yearly_path="out/company_yearly_averages_2014_2023.csv",
        sector_medians_path="out/sector_yearly_medians_2014_2023.csv",
        out_name="company_vs_sector_ratio.csv",
        report_name="company_vs_sector_ratio_report.txt",
    )
