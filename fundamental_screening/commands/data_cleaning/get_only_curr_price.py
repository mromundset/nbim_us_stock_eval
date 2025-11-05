# commands/most_recent_price.py
import os
import pandas as pd
from typing import Optional

def run(
    input_data: str,
    out_name: str = "most_recent_price.csv",
    ticker_col: str = "Ticker",
    price_date_col: str = "Price Dates",
    calendar_year: int = 2023,  # set e.g. 2023 to restrict to that year first
    year_col: str = "Calendar Year",
    dayfirst: bool = True,
    report_name: str = "most_recent_price_report.txt",     # optional mini-report
) -> None:
    """
    Produce a CSV with the most recent 'Price Dates' snapshot per company (one row per Ticker).
    If calendar_year is provided, filter to that year first, then pick the latest per Ticker.
    """
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    report_path = os.path.join(out_dir, report_name) if report_name else None

    # Load as strings to preserve formatting
    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")

    if ticker_col not in df.columns:
        raise ValueError(f"Missing column: {ticker_col}")
    if price_date_col not in df.columns:
        raise ValueError(f"Missing column: {price_date_col}")

    # Optional: restrict to a specific year
    pre_n = len(df)
    if calendar_year is not None:
        if year_col not in df.columns:
            raise ValueError(f"calendar_year was provided but '{year_col}' column is missing.")
        yr = pd.to_numeric(df[year_col], errors="coerce")
        df = df[yr == int(calendar_year)].copy()

    # Parse dates and sort: most recent first within each ticker
    df["_price_dt"] = pd.to_datetime(df[price_date_col], errors="coerce", dayfirst=dayfirst)

    # Sort: ticker ASC then price_dt DESC, keep first per ticker
    df_sorted = df.sort_values([ticker_col, "_price_dt"], ascending=[True, False])
    latest = df_sorted.drop_duplicates(subset=[ticker_col], keep="first").drop(columns=["_price_dt"])

    latest.to_csv(out_path, index=False)

    print(f"→ Most-recent-per-ticker saved: {out_path}")
    print(f"Rows in: {pre_n} | rows after filter: {len(df)} | unique tickers out: {len(latest)}")

    # Optional mini-report
    if report_path:
        # per-ticker: latest date kept
        kept_dates = (
            latest[[ticker_col, price_date_col]]
            .rename(columns={ticker_col: "Ticker", price_date_col: "Kept Price Date"})
            .sort_values("Ticker")
            .to_string(index=False)
        )
        lines = [
            "# Most Recent Price Snapshot Report",
            f"Input: {input_data}",
            f"Calendar year filter: {calendar_year if calendar_year is not None else '(none)'}",
            f"Rows before: {pre_n}",
            f"Rows after year filter: {len(df)}",
            f"Unique tickers output: {len(latest)}",
            "",
            "Kept snapshot per ticker:",
            kept_dates,
            "",
            f"Saved CSV → {out_path}",
        ]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Report -> {report_path}")

if __name__ == "__main__":
    run(
        input_data="data/us_stock_valuation_quotefix_pass2.csv",
        out_name="most_recent_price.csv",
        ticker_col="Ticker",
        price_date_col="Price Dates",
        calendar_year=None,     # or e.g. 2023
        report_name="most_recent_price_report.txt",
    )

