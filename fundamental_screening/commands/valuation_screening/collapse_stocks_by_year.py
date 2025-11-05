import os
import pandas as pd
import numpy as np
from typing import List, Optional

ID_COLS_DEFAULT = ["Ticker", "Company Name", "Sector", "Industry"]

def _numericize(df: pd.DataFrame, skip: List[str]) -> List[str]:
    """Convert all non-ID columns to numeric when possible."""
    numeric_cols = []
    for c in df.columns:
        if c in skip:
            continue
        coerced = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        if coerced.notna().any():
            df[c] = coerced
            numeric_cols.append(c)
    return numeric_cols

def _first_nonnull_mode(series: pd.Series):
    """Return most common non-null value or first non-null."""
    s = series.dropna()
    if s.empty:
        return None
    mode_vals = s.mode(dropna=True)
    return mode_vals.iloc[0] if not mode_vals.empty else s.iloc[0]

def run(
    input_data: str,
    out_name: str = "company_yearly_averages.csv",
    report_name: str = "company_yearly_averages_report.txt",
    ticker_col: str = "Ticker",
    year_col: str = "Calendar Year",
    id_cols: Optional[List[str]] = None,
    year_min: int = 2014,
    year_max: int = 2023,
    round_digits: int = 3,
) -> None:
    """
    Aggregate data to one row per (Ticker, Calendar Year), averaging all numeric columns.
    Assumes 0→NaN handling has already been done.
    """
    id_cols = id_cols or ID_COLS_DEFAULT

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, out_name)
    out_rpt = os.path.join(out_dir, report_name)

    # --- Load ---
    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")

    if ticker_col not in df.columns or year_col not in df.columns:
        raise ValueError(f"Missing required columns: '{ticker_col}' or '{year_col}'")

    # Clean IDs
    for c in [ticker_col] + [col for col in id_cols if col in df.columns]:
        df[c] = df[c].astype(str).str.strip()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    pre_rows = len(df)
    df = df[(df[year_col] >= year_min) & (df[year_col] <= year_max)].copy()
    post_rows = len(df)

    # Convert to numeric
    present_id_cols = [c for c in id_cols if c in df.columns]
    numeric_cols = _numericize(df, skip=present_id_cols + [year_col])

    # --- Aggregate ---
    # identifiers: pick most frequent value per group
    def agg_ids(g: pd.DataFrame) -> pd.Series:
        out = {}
        for c in present_id_cols:
            out[c] = _first_nonnull_mode(g[c])
        return pd.Series(out)

    id_block = (
        df.groupby([ticker_col, year_col], as_index=False)
          .apply(agg_ids)
          .reset_index(drop=True)
    )

    num_block = (
        df.groupby([ticker_col, year_col])[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    out_df = pd.merge(num_block, id_block, on=[ticker_col, year_col], how="left")

    # order columns: IDs → year → numerics
    ordered = [ticker_col] + [c for c in id_cols if c in out_df.columns and c != ticker_col] + [year_col]
    numeric_order = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + numeric_order]

    # Round numeric
    out_df[numeric_cols] = out_df[numeric_cols].round(round_digits)

    # --- Save ---
    out_df.to_csv(out_csv, index=False)

    # --- Report ---
    lines = [
        "# Company-Year Averages",
        f"Input: {input_data}",
        f"Year filter: {year_min}–{year_max}",
        f"Rows before filter: {pre_rows:,}",
        f"Rows after filter:  {post_rows:,}",
        f"Unique company-years out: {len(out_df):,}",
        f"Numeric columns averaged: {len(numeric_cols)}",
        "",
        "Preview (first 15 rows):",
        out_df.head(15).to_string(index=False),
        "",
        f"Saved → {out_csv}",
    ]
    with open(out_rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"→ {out_csv}")
    print(f"Report -> {out_rpt}")

if __name__ == "__main__":
    run(
        input_data="data/us_stock_valuation_quotefix_pass2.csv",
        out_name="company_yearly_averages_2014_2023.csv",
        report_name="company_yearly_averages_report_2014_2023.txt",
        year_min=2014,
        year_max=2023,
    )
