# commands/sector_aggregate.py
import os
from typing import Dict, List, Optional
import pandas as pd

# ---- Bucket definitions (A–E) ----
BUCKETS: Dict[str, List[str]] = {
    "A_ScaleValuation": ["Market Cap", "Price", "P/E Ratio", "P/S Ratio"],
    "B_Profitability": [
        "Return On Equity", "Return On Invested Capital", "Return On Assets",
        "Gross Profit Margin", "Operating Profit Margin",
        "EBITDA", "EBIT", "Net Income", "Revenue"
    ],
    "C_LevLiquidity": [
        "Total Debt", "Long-Term Debt", "Short-Term Debt",
        "Current Ratio", "Quick Ratio", "Interest Coverage Ratio",
        "Total Liabilities", "Total Assets"
    ],
    "D_CashFlow": [
        "Operating Cash Flow", "Free Cash Flow", "Capital Expenditure",
        "Cash And Short Term Investments", "Net Cash", "Stock Based Compensation", "Dividends Paid"
    ],
    "E_CostOfCapital": [
        "WACC", "Risk Free Rate", "Equity Risk Premium", "Effective Tax Rate",
        "Pre-Tax Cost of Debt", "After-Tax Cost of Debt"
    ],
}

def _to_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return out

def _present_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def run(
    input_data: str,
    out_name: str = "sector_aggregates.csv",
    report_name: str = "sector_aggregate_report.txt",
    sector_col: str = "Sector",
    ticker_col: str = "Ticker",
    round_digits: int = 2,
) -> None:
    """
    Build sector-level means grouped into A–E buckets and save as one wide table.
    Report prints each bucket as its own table (Sector × bucket metrics).
    """
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    report_path = os.path.join(out_dir, report_name)

    # Load
    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")

    if sector_col not in df.columns:
        raise ValueError(f"Missing column: {sector_col}")
    if ticker_col not in df.columns:
        raise ValueError(f"Missing column: {ticker_col}")

    # Clean identifiers
    df[sector_col] = df[sector_col].astype(str).str.strip()
    df[ticker_col] = df[ticker_col].astype(str).str.strip()

    # Cast numeric for all present bucket columns
    all_bucket_cols = [c for cols in BUCKETS.values() for c in cols]
    present_numeric_cols = _present_cols(df, all_bucket_cols)
    df_num = _to_numeric_cols(df, present_numeric_cols)

    # Company counts for context
    company_counts = df.groupby(sector_col)[ticker_col].nunique().rename("Company_Count")

    # Compute means per bucket
    bucket_frames = []
    included_by_bucket = {}
    for bucket_key, cols in BUCKETS.items():
        use_cols = _present_cols(df_num, cols)
        included_by_bucket[bucket_key] = use_cols
        if not use_cols:
            continue
        grp = df_num.groupby(sector_col)[use_cols].mean(numeric_only=True)
        grp = grp.rename(columns={c: f"{bucket_key}__{c}" for c in use_cols})
        bucket_frames.append(grp)

    # Merge all buckets + counts into one CSV (wide)
    if bucket_frames:
        wide = pd.concat(bucket_frames, axis=1)
    else:
        wide = pd.DataFrame(index=company_counts.index)
    wide = wide.join(company_counts, how="left")

    # Column order for CSV
    ordered_cols = ["Company_Count"]
    for bucket_key in ["A_ScaleValuation", "B_Profitability", "C_LevLiquidity", "D_CashFlow", "E_CostOfCapital"]:
        pref = f"{bucket_key}__"
        cols = [c for c in wide.columns if c.startswith(pref)]
        ordered_cols.extend(sorted(cols))
    ordered_cols = [c for c in ordered_cols if c in wide.columns]
    wide = wide[ordered_cols].round(round_digits)

    # Save CSV
    wide.to_csv(out_path, index=True)  # index = Sector

    # -------- Report: each bucket as its own table --------
    lines = [
        "# Sector Aggregates (Means) — Bucketed A–E",
        f"Input: {input_data}",
        "",
        f"Sectors covered: {len(wide.index)}",
        f"Metrics present: {len([c for c in wide.columns if '__' in c])}",
        "",
        "Company counts by sector:",
        company_counts.to_frame().to_string(),
        "",
        "------------------------------------------------------------",
    ]

    # Helper to render a bucket table (Sector × metrics in that bucket)
    def render_bucket(bucket_key: str) -> str:
        pref = f"{bucket_key}__"
        cols = [c for c in wide.columns if c.startswith(pref)]
        title = bucket_key.replace("_", " ")
        if not cols:
            return f"## {title}\n(no metrics present)\n"
        # Build a tidy table with original metric names (strip prefix)
        tbl = wide[cols].copy()
        tbl.columns = [c.split("__", 1)[1] for c in cols]
        tbl = tbl.round(round_digits)
        return f"## {title}\n{tbl.to_string()}\n"

    for key in ["A_ScaleValuation", "B_Profitability", "C_LevLiquidity", "D_CashFlow", "E_CostOfCapital"]:
        lines.append(render_bucket(key))
        lines.append("------------------------------------------------------------")

    lines += [
        "",
        f"Saved wide CSV → {out_path}",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"→ {out_path}")
    print(f"Report -> {report_path}")

if __name__ == "__main__":
    run(
        input_data="data/us_stock_valuation_quotefix_pass2.csv",
        out_name="sector_aggregates.csv",
        report_name="sector_aggregate_report.txt",
        sector_col="Sector",
        ticker_col="Ticker",
        round_digits=2,
    )

