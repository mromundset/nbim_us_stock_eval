import os
import pandas as pd
from typing import Optional, Dict

# Friendly aliases for convenience
RATIO_ALIASES: Dict[str, str] = {
    "P/E": "P/E Ratio",
    "PE": "P/E Ratio",
    "P/S": "P/S Ratio",
    "PS": "P/S Ratio",
    "ROE": "Return On Equity",
    "ROIC": "Return On Invested Capital",
    "ROA": "Return On Assets",
    "ROCE": "Return On Capital Employed",
    "GM": "Gross Profit Margin",
    "OPM": "Operating Profit Margin",
    "CR": "Current Ratio",
    "QR": "Quick Ratio",
    "ICR": "Interest Coverage Ratio",
    "WACC": "WACC",
    "RFR": "Risk Free Rate",
    "ERP": "Equity Risk Premium",
    "PreTaxCoD": "Pre-Tax Cost of Debt",
    "AfterTaxCoD": "After-Tax Cost of Debt",
    "Beta": "Beta",
}

def _resolve_ratio_col(cols, ratio: str) -> str:
    """Resolve user-provided ratio name to actual column name."""
    # Exact match
    if ratio in cols:
        return ratio
    # Alias match
    if ratio in RATIO_ALIASES and RATIO_ALIASES[ratio] in cols:
        return RATIO_ALIASES[ratio]
    # Case-insensitive substring match
    ratio_lower = ratio.lower()
    candidates = [c for c in cols if ratio_lower in c.lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not resolve ratio '{ratio}'. Available columns: {', '.join(cols)}")

def run(
    input_data: str = "data/sector_performance_ratios.csv",                 # path to your ratio table (company_vs_sector_ratio.csv)
    sector: str = "Basic Materials",                     # e.g., "Technology"
    year: int = "2023",                       # e.g., 2023
    ratio: str = "P/E",                      # e.g., "P/E" or "P/E Ratio"
    out_name: str = "basic_materials_pe.csv",
    report_name: Optional[str] = "basic_materials_pe_report.txt",
    sector_col: str = "Sector",
    year_col: str = "Calendar Year",
    ticker_col: str = "Ticker",
    company_col: str = "Company Name",
    ascending: bool = True,          # True = lowest first (potentially undervalued)
    top_n: Optional[int] = None,     # e.g., 25 to only save top 25
    round_digits: int = 3,
) -> None:
    """
    Filter a single sector & year from the ratio table and sort companies
    by a chosen ratio (already relative to sector median).
    Output includes only: Ticker, Company Name, Ratio.
    """
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    rpt_path = os.path.join(out_dir, report_name) if report_name else None

    # Load ratio table
    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")

    # Basic validation
    if year_col not in df.columns or sector_col not in df.columns:
        raise ValueError(f"Missing required columns '{year_col}' or '{sector_col}' in {input_data}.")
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    # Resolve ratio column
    ratio_col = _resolve_ratio_col(df.columns.tolist(), ratio)
    df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")

    # Filter by sector & year
    mask = (df[sector_col] == sector) & (df[year_col] == int(year))
    sub = df.loc[mask, [ticker_col, company_col, ratio_col]].dropna(subset=[ratio_col])

    # Sort by ratio (lowest first = potentially undervalued)
    sub = sub.sort_values(by=ratio_col, ascending=ascending)

    # Optional: keep only top_n
    if top_n is not None and top_n > 0:
        sub = sub.head(top_n)

    # Round ratio
    sub[ratio_col] = sub[ratio_col].round(round_digits)

    # Save
    sub.to_csv(out_path, index=False)

    # Report
    if rpt_path:
        lines = [
            "# Sector Ratio Screen",
            f"Input file: {input_data}",
            f"Sector: {sector} | Year: {year} | Ratio: {ratio} (resolved as '{ratio_col}')",
            f"Sorted {'ascending (lowest first)' if ascending else 'descending (highest first)'}",
            f"Rows output: {len(sub)}",
            "",
            "Top 10 preview:",
            sub.head(100).to_string(index=False),
            "",
            f"Saved → {out_path}",
        ]
        with open(rpt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"→ {out_path}")
    if rpt_path:
        print(f"Report -> {rpt_path}")

if __name__ == "__main__":
    # Example standalone test
    run(
        input_data="out/company_vs_sector_ratio.csv",
        sector="Technology",
        year=2023,
        ratio="P/E",
        out_name="screen_tech_2023_pe.csv",
        report_name="screen_tech_2023_pe_report.txt",
        ascending=True,
        top_n=50,
    )
