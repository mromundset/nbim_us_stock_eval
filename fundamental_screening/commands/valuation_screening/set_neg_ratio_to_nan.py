import os
import pandas as pd
import numpy as np

def run(
    input_data: str,
    out_name: str = "cleaned_ratios.csv",
    report_name: str = "cleaned_ratios_report.txt",
    pe_col: str = "P/E Ratio",
    ps_col: str = "P/S Ratio",
) -> None:
    """
    Reads a CSV and replaces any negative P/E or P/S ratios with NaN.
    Leaves all other values untouched.
    """

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, out_name)
    out_rpt = os.path.join(out_dir, report_name)

    # --- Load ---
    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")

    # --- Convert to numeric safely ---
    for c in [pe_col, ps_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Count negatives before cleaning ---
    neg_pe = int((df[pe_col] < 0).sum()) if pe_col in df.columns else 0
    neg_ps = int((df[ps_col] < 0).sum()) if ps_col in df.columns else 0

    # --- Replace negatives with NaN ---
    if pe_col in df.columns:
        df.loc[df[pe_col] < 0, pe_col] = np.nan
    if ps_col in df.columns:
        df.loc[df[ps_col] < 0, ps_col] = np.nan

    # --- Save cleaned file ---
    df.to_csv(out_csv, index=False)

    # --- Report ---
    lines = [
        "# Negative P/E and P/S Cleaner",
        f"Input file: {input_data}",
        f"Output file: {out_csv}",
        "",
        f"Negative {pe_col}: {neg_pe}",
        f"Negative {ps_col}: {neg_ps}",
        "",
        f"Rows processed: {len(df):,}",
        "Preview (first 10 rows):",
        df.head(10).to_string(index=False),
    ]
    with open(out_rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"→ Cleaned file saved to {out_csv}")
    print(f"→ Report written to {out_rpt}")

if __name__ == "__main__":
    run(
        input_data="out/company_vs_sector_ratio.csv",
        out_name="company_vs_sector_ratio_cleaned.csv",
        report_name="cleaned_ratios_report.txt",
    )
