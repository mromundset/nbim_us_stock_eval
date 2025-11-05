import pandas as pd
import sys
import io
import os

def run(input_data=None, out_name=None):
    if not input_data or not out_name:
        print("Error: --input_data and --out_name arguments are required.")
        return

    # Ensure the 'out' directory exists
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full output path
    full_output_path = os.path.join(output_dir, out_name)

    # Redirect stdout to a string buffer
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        df = pd.read_csv(input_data)
        print(f"Successfully loaded {input_data}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Restore stdout and write buffer to file
        sys.stdout = old_stdout
        with open(full_output_path, 'w') as f:
            f.write(buffer.getvalue())
        return

    print("\n=== Dataframe Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n=== Column Summary ===")
    df.info()

    print("\n=== Missing Values (%)")
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    print(missing_percentages)

    print("\n=== Potential Issues ===")
    # Duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Duplicate column names found: {duplicate_cols}")
    else:
        print("No duplicate column names found.")

    # Constant-value columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"Constant-value columns found: {constant_cols}")
    else:
        print("No constant-value columns found.")

    # Mixed data types
    mixed_type_cols = [col for col in df.columns if df[col].apply(type).nunique() > 1]
    if mixed_type_cols:
        print(f"Mixed data type columns found: {mixed_type_cols}")
    else:
        print("No mixed data type columns found.")

    # Restore stdout
    sys.stdout = old_stdout
    # Get the content from the buffer
    output_str = buffer.getvalue()

    try:
        with open(full_output_path, 'w') as f:
            f.write(output_str)
        print(f"Analysis report saved to {full_output_path}")
    except Exception as e:
        print(f"Error saving report: {e}")

    