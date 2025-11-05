import os
import pandas as pd

def run(
    input_data: str,
    out_name: str = "sorted_by_price_date.csv",
    ticker_col: str = "Ticker",
    price_date_col: str = "Price Dates",
    dayfirst: bool = True,
) -> None:
    """
    Sorts each company (Ticker) by most-to-least recent 'Price Dates'
    and writes the cleaned CSV to ./out.
    """
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    print(f"Loading {input_data} ...")
    df = pd.read_csv(input_data, dtype=str, keep_default_na=False, na_filter=False, engine="python")
    n_before = len(df)

    if ticker_col not in df.columns or price_date_col not in df.columns:
        raise ValueError(f"Missing required columns: '{ticker_col}' or '{price_date_col}'")

    # Parse and sort
    df["_price_dt"] = pd.to_datetime(df[price_date_col], errors="coerce", dayfirst=dayfirst)

    # Sort by Ticker ascending, Price Dates descending
    df = df.sort_values(by=[ticker_col, "_price_dt"], ascending=[True, False]).drop(columns=["_price_dt"])

    n_after = len(df)

    df.to_csv(out_path, index=False)
    print(f"→ Sorted dataset saved: {out_path}")
    print(f"Rows processed: {n_before} → {n_after}")
    print("Data ordered by most recent Price Dates within each company.")

if __name__ == "__main__":
    run(
        input_data="data/us_stock_valuation_quotefix_pass2.csv",
        out_name="sorted_by_price_date.csv",
        ticker_col="Ticker",
        price_date_col="Price Dates",
        dayfirst=True,
    )
