import os
import pandas as pd
import numpy as np
from typing import List

RATIO_COLS: List[str] = [
    "P/E Ratio", "P/S Ratio", "Price", "Market Cap",
    "Return On Equity", "Return On Invested Capital", "Return On Assets",
    "Return On Capital Employed", "Gross Profit Margin", "Operating Profit Margin",
    "Earnings per Share", "Revenue Per Share", "Book Value Per Share",
    "Current Ratio", "Quick Ratio", "Interest Coverage Ratio", "Beta",
    "WACC", "Risk Free Rate", "Equity Risk Premium",
    "Pre-Tax Cost of Debt", "After-Tax Cost of Debt", "Effective Tax Rate",
]

def _numericize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return df

def _present(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def _latest_per_ticker_year(df: pd.DataFrame, ticker_col: str, year_col: str, price_date_col: str) -> pd.DataFrame:
    """Within each (Ticker, Year), keep the most recent Price Dates."""
    if price_date_col in df.columns:
        df["_price_dt"] = pd.to_datetime(df[price_date_col], errors="coerce", dayfirst=True)
        df = (
            df.sort_values(["Ticker", year_col, "_price_dt"])
              .drop_duplicates(subset=[ticker_col, year_col], keep="last")
              .drop(columns=["_price_dt"])
        )
    else:
        df = (
            df.sort_values([ticker_col, year_col])
              .drop_duplicates(subset=[ticker_col, year_col], keep="last")
        )
    return df

def run(
    input_data: str,
    out_name: str = "sector_yearly_medians.csv",
    report_name: str = "sector_yearly_medians_report.txt",
    sector_col: str = "Sector",
    year_col: str = "Calendar Year",
    ticker_col: str = "Ticker",
    price_date_col: str = "Price Dates",
    year_min: int = 2014,
    year_max: int = 2023,
    replace_zeros_with_nan: bool = True,
    latest_snapshot_per_ticker_year: bool = True,
    round_digits: int = 3,
) -> None:
    """Aggregate medians for 2014–2023 Sector × Year table."""
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, out_name)
    out_rpt = os.path.join(out_dir, report_name)

    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")

    # basic checks
    for col in (sector_col, year_col, ticker_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # clean ids
    df[sector_col] = df[sector_col].astype(str).str.strip()
    df[ticker_col] = df[ticker_col].astype(str).str.strip()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    # --- FILTER YEAR RANGE ---
    pre_rows = len(df)
    df = df[(df[year_col] >= year_min) & (df[year_col] <= year_max)]
    post_rows = len(df)

    # keep only relevant cols
    cols_present = _present(df, RATIO_COLS)
    work = df[[sector_col, year_col, ticker_col, *cols_present]].copy()

    if latest_snapshot_per_ticker_year:
        work = _latest_per_ticker_year(work, ticker_col, year_col, price_date_col)

    work = _numericize(work, cols_present)
    if replace_zeros_with_nan:
        work[cols_present] = work[cols_present].replace(0, np.nan)

    counts = (
        work.groupby([sector_col, year_col])[ticker_col]
        .nunique()
        .rename("Company_Count")
        .reset_index()
    )

    medians = (
        work.groupby([sector_col, year_col])[cols_present]
        .median(numeric_only=True)
        .reset_index()
    )

    out_df = medians.merge(counts, on=[sector_col, year_col], how="left")
    out_df = out_df.sort_values([sector_col, year_col]).reset_index(drop=True)
    out_df[cols_present] = out_df[cols_present].round(round_digits)

    out_df.to_csv(out_csv, index=False)

    # --- REPORT ---
    lines = [
        "# Sector-Year Medians (Filtered 2014–2023)",
        f"Input: {input_data}",
        f"Rows before filter: {pre_rows:,}",
        f"Rows after filter:  {post_rows:,}",
        f"Years kept: {year_min}–{year_max}",
        f"Latest snapshot per (Ticker, Year): {latest_snapshot_per_ticker_year}",
        f"Zeros replaced with NaN: {replace_zeros_with_nan}",
        "",
        f"Metrics included ({len(cols_present)}): {', '.join(cols_present)}",
        "",
        "Preview (first 20 rows):",
        out_df.head(20).to_string(index=False),
        "",
        f"Sectors covered: {out_df[sector_col].nunique()}, Years covered: {out_df[year_col].nunique()}",
        f"Saved → {out_csv}",
    ]
    with open(out_rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"→ {out_csv}")
    print(f"Report -> {out_rpt}")

if __name__ == "__main__":
    run(
        input_data="data/us_stock_valuation_quotefix_pass2.csv",
        out_name="sector_yearly_medians_2014_2023.csv",
        report_name="sector_yearly_medians_report_2014_2023.txt",
        year_min=2014,
        year_max=2023,
    )
