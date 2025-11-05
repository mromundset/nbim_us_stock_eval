import os
import pandas as pd
import numpy as np

def run(
    input_data: str,
    out_name: str = "replaced_zeros_nan.csv",
    report_name: str = "zero_replace_report.txt",
    skip_cols: list = ["Ticker", "Company Name", "Sector", "Industry", "Calendar Year", "Period"],
) -> None:
    """
    Reads a CSV, converts numeric columns, and replaces all zeros (0.0) with NaN.
    Non-numeric and identifier columns are left unchanged.

    Parameters
    ----------
    input_data : str
        Path to the CSV file to process.
    out_name : str
        Name for the cleaned output CSV (written to ./out).
    report_name : str
        Name for the summary report (written to ./out).
    skip_cols : list
        Columns that should never be converted to numeric (IDs, names, categories).
    """

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    report_path = os.path.join(out_dir, report_name)

    print(f"Loading {input_data} ...")
    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")
    n_rows, n_cols = df.shape

    # Detect numeric columns
    numeric_cols = [c for c in df.columns if c not in skip_cols]

    # Convert to numeric where possible
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")

    # Count zeros before replacement
    zero_counts = (df[numeric_cols] == 0).sum().sort_values(ascending=False)

    # Replace 0 → NaN
    df[numeric_cols] = df[numeric_cols].replace(0, np.nan)

    # Write cleaned CSV
    df.to_csv(out_path, index=False)

    # Build simple report
    lines = [
        "# Zero Replacement Report",
        f"Input file: {input_data}",
        f"Output file: {out_path}",
        "",
        f"Rows: {n_rows}, Columns: {n_cols}",
        "",
        "Top 20 columns with zero counts before replacement:",
        zero_counts.head(20).to_string(),
        "",
        f"Total columns processed: {len(numeric_cols)}",
        f"Skipped identifier columns: {', '.join(skip_cols)}",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"→ Cleaned file saved to {out_path}")
    print(f"→ Report written to {report_path}")

if __name__ == "__main__":
    run(
        input_data="data/us_stock_valuation_quotefix_pass2.csv",
        out_name="nozeros_us_stock_valuation.csv",
        report_name="zero_replace_report.txt",
    )
