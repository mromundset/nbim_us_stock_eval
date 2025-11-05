import os
import pandas as pd
from typing import Optional, List

DEFAULT_NA_TOKENS = {"", " ", "na", "n/a", "none", "null", "nil", "missing", "unknown", "tbd"}

def _na_set(extra: Optional[List[str]]) -> set:
    s = {t.lower() for t in DEFAULT_NA_TOKENS}
    if extra:
        s |= {t.strip().lower() for t in extra}
    return s

def _is_empty(x, na_tokens: set) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower() in na_tokens

def empty_overview(
    df: pd.DataFrame,
    only_columns: Optional[List[str]],
    extra_na_tokens: Optional[List[str]],
    identifier_column: Optional[str],
    sample_limit: int
):
    na_tokens = _na_set(extra_na_tokens)
    cols = only_columns if only_columns else list(df.columns)

    # Boolean mask of empties (future-proof: no applymap)
    mask = df[cols].apply(lambda col: col.map(lambda v: _is_empty(v, na_tokens)))

    total_rows = len(df)
    col_counts = mask.sum(axis=0)
    col_pct = (col_counts / total_rows * 100).round(2) if total_rows > 0 else 0.0

    rows_with_any_bool = mask.any(axis=1)
    rows_with_any = int(rows_with_any_bool.sum())
    rows_with_any_pct = round((rows_with_any / total_rows * 100), 2) if total_rows > 0 else 0.0

    per_col_summary = (
        pd.DataFrame({"empty_count": col_counts, "empty_pct": col_pct})
        .sort_values(["empty_pct", "empty_count"], ascending=False)
    )

    # Identifier distributions (optional)
    id_rows_dist = None
    id_cells_dist = None
    if identifier_column and (identifier_column in df.columns):
        # 1) rows-with-empty per identifier
        id_series = df[identifier_column].astype(str)
        id_rows_dist = (
            rows_with_any_bool.groupby(id_series)
            .sum()
            .rename("rows_with_empty")
            .sort_values(ascending=False)
            .to_frame()
        )
        # 2) total empty cells per identifier
        empty_cells_per_row = mask.sum(axis=1)
        id_cells_dist = (
            empty_cells_per_row.groupby(id_series)
            .sum()
            .rename("empty_cells")
            .sort_values(ascending=False)
            .to_frame()
        )

    # Sample a few source rows that have empties
    sample_rows_idx = rows_with_any_bool[rows_with_any_bool].index.tolist()[:max(0, sample_limit)]
    sample_df = df.loc[sample_rows_idx, cols] if len(sample_rows_idx) else pd.DataFrame(columns=cols)

    return {
        "total_rows": total_rows,
        "rows_with_any": rows_with_any,
        "rows_with_any_pct": rows_with_any_pct,
        "per_col_summary": per_col_summary,
        "id_rows_dist": id_rows_dist,
        "id_cells_dist": id_cells_dist,
        "sample_df": sample_df
    }

def run(
    input_data: str,
    report_name: str = "empty_overview.txt",
    out_name: Optional[str] = None,  # kept for CLI compatibility; not used here
    only_columns: Optional[List[str]] = None,
    extra_na_tokens: Optional[List[str]] = None,
    identifier_column: Optional[str] = None,  # e.g., "TICK" or "ticker"
    sample_limit: int = 5,
) -> None:
    """
    Write an overview of empty values per column to ./out/<report_name>,
    plus identifier distributions and a few source rows with empties.
    """
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, report_name)

    df = pd.read_csv(
        input_data,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="python"
    )

    res = empty_overview(
        df=df,
        only_columns=only_columns,
        extra_na_tokens=extra_na_tokens,
        identifier_column=identifier_column,
        sample_limit=sample_limit
    )

    lines = [
        "# Empty Values Overview",
        f"Input: {input_data}",
        f"Columns scoped: {only_columns if only_columns else 'ALL'}",
        f"Extra NA tokens: {extra_na_tokens if extra_na_tokens else '[]'}",
        f"Identifier column: {identifier_column if identifier_column else 'None'}",
        "",
        f"Total rows: {res['total_rows']}",
        f"Rows with ≥1 empty: {res['rows_with_any']} ({res['rows_with_any_pct']}%)",
        "",
        "## Per-Column Empty Counts",
        res["per_col_summary"].to_string(index=True),
    ]

    # Identifier distributions
    if res["id_rows_dist"] is not None:
        lines += [
            "",
            "## Identifier Distribution — Rows With ≥1 Empty (descending)",
            res["id_rows_dist"].head(25).to_string(),  # top 25 for brevity
        ]
    if res["id_cells_dist"] is not None:
        lines += [
            "",
            "## Identifier Distribution — Total Empty Cells (descending)",
            res["id_cells_dist"].head(25).to_string(),
        ]

    # Sample rows
    if len(res["sample_df"]) > 0:
        lines += [
            "",
            f"## Sample Source Rows Containing Empty Values (up to {sample_limit})",
            res["sample_df"].to_string(index=True),
        ]
    else:
        lines += ["", "## Sample Source Rows Containing Empty Values", "(none)"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report -> {report_path}")

if __name__ == "__main__":
    run(
        input_data="data/us_stock_valuation_quotefix_pass2.csv",
        report_name="empty_check_report.txt",
        out_name=None,
        only_columns=None,             # e.g., ["TICK","Company","Sector"]
        extra_na_tokens=None,
        identifier_column="TICK",      # or "ticker", or None
        sample_limit=5,
    )

