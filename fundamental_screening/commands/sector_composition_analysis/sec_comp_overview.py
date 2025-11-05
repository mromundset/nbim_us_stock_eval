import os
import io
import pandas as pd
from typing import Optional

def _to_num(s):
    # robust numeric cast (handles commas, spaces)
    if pd.isna(s): return pd.NA
    return pd.to_numeric(str(s).replace(",", "").strip(), errors="coerce")

def _latest_per_ticker(df: pd.DataFrame, ticker_col: str) -> pd.DataFrame:
    """
    Heuristic: if a date-like column exists, use it to keep the latest row per ticker.
    Otherwise, keep the row with the largest Market Cap per ticker.
    """
    candidates = []
    for c in ["Filing Date", "Price Dates", "Calendar Year", "Period"]:
        if c in df.columns: candidates.append(c)
    df = df.copy()

    # Try parse dates/years
    if "Calendar Year" in candidates:
        df["_order_key"] = pd.to_numeric(df["Calendar Year"], errors="coerce")
    elif "Filing Date" in candidates:
        df["_order_key"] = pd.to_datetime(df["Filing Date"], errors="coerce")
    elif "Price Dates" in candidates:
        df["_order_key"] = pd.to_datetime(df["Price Dates"], errors="coerce")
    else:
        # fallback: use Market Cap
        df["_order_key"] = pd.to_numeric(df["Market Cap"].map(_to_num), errors="coerce")

    # keep the last (max) per ticker
    df = df.sort_values("_order_key").drop_duplicates(subset=[ticker_col], keep="last")
    df = df.drop(columns=["_order_key"])
    return df

def sector_overview(
    df: pd.DataFrame,
    sector_col: str,
    mcap_col: str,
    ticker_col: str
):
    # basic hygiene
    if sector_col not in df.columns: raise ValueError(f"Missing column: {sector_col}")
    if mcap_col not in df.columns:   raise ValueError(f"Missing column: {mcap_col}")
    if ticker_col not in df.columns: raise ValueError(f"Missing column: {ticker_col}")

    df = df.copy()
    df[sector_col] = df[sector_col].astype(str).str.strip()
    df[ticker_col] = df[ticker_col].astype(str).str.strip()
    df[mcap_col]   = df[mcap_col].map(_to_num)

    # ---- 1) Sector by company count (unique tickers) ----
    uniq = df[[ticker_col, sector_col]].dropna().drop_duplicates()
    counts = uniq[sector_col].value_counts().rename("company_count").to_frame()
    counts["company_pct"] = (counts["company_count"] / counts["company_count"].sum() * 100).round(2)
    counts = counts.reset_index().rename(columns={"index": "Sector"}).sort_values("company_pct", ascending=False)

    # ---- 2) Sector by market cap (cap-weighted) ----
    latest = _latest_per_ticker(df[[ticker_col, sector_col, mcap_col]], ticker_col=ticker_col)
    latest = latest.dropna(subset=[mcap_col])
    sector_mcap = latest.groupby(sector_col)[mcap_col].sum().rename("mcap").to_frame().reset_index()
    sector_mcap["weight_pct"] = (sector_mcap["mcap"] / sector_mcap["mcap"].sum() * 100).round(2)
    sector_mcap = sector_mcap.rename(columns={sector_col: "Sector"}).sort_values("weight_pct", ascending=False)

    return counts, sector_mcap

def run(
    input_data: str,
    out_prefix: str = "sector_comp",
    report_name: str = "sector_comp_report.txt",
    sector_col: str = "Sector",
    mcap_col: str = "Market Cap",
    ticker_col: str = "Ticker",
    benchmark_csv: Optional[str] = None  # optional: CSV with columns ["Sector","Benchmark_Weight"]
) -> None:
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    # Load (as strings to preserve)
    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")

    # Compute
    counts, caps = sector_overview(df, sector_col, mcap_col, ticker_col)

    # Optional benchmark merge (cap-weighted comparison)
    comp = caps.copy()
    if benchmark_csv:
        bmk = pd.read_csv(benchmark_csv)
        if not {"Sector","Benchmark_Weight"}.issubset(bmk.columns):
            raise ValueError("benchmark_csv must have columns: Sector,Benchmark_Weight")
        comp = comp.merge(bmk, on="Sector", how="outer").fillna({"mcap": 0})
        comp["diff_vs_benchmark_pct"] = (comp["weight_pct"] - comp["Benchmark_Weight"]).round(2)

    # Write CSVs
    counts_path = os.path.join(out_dir, f"{out_prefix}_by_count.csv")
    caps_path   = os.path.join(out_dir, f"{out_prefix}_by_mcap.csv")
    comp_path   = os.path.join(out_dir, f"{out_prefix}_vs_benchmark.csv") if benchmark_csv else None

    counts.to_csv(counts_path, index=False)
    caps.to_csv(caps_path, index=False)
    if benchmark_csv:
        comp.to_csv(comp_path, index=False)

    # Report
    lines = [
        "# Sector Composition Overview",
        f"Input: {input_data}",
        f"Sectors (by company count):",
        counts.to_string(index=False),
        "",
        f"Sectors (by market cap weight):",
        caps[["Sector","weight_pct"]].to_string(index=False),
    ]
    if benchmark_csv:
        lines += [
            "",
            f"Benchmark compare (cap-weighted):",
            comp[["Sector","weight_pct","Benchmark_Weight","diff_vs_benchmark_pct"]].to_string(index=False),
        ]

    report_path = os.path.join(out_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"→ {counts_path}")
    print(f"→ {caps_path}")
    if benchmark_csv:
        print(f"→ {comp_path}")
    print(f"Report -> {report_path}")