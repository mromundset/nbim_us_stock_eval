import csv
import datetime
import io
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

def run(input_data: str, out_name: str, expected_width: int | None = None) -> None:
    """
    Validates the structural integrity of a CSV file, tests hypotheses about its structure,
    and appends a markdown report to the specified output file.
    """
    report_lines = []
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, out_name)

    report_lines.append(f"## Validation Report for `{input_data}`")
    report_lines.append(f"**Timestamp:** {datetime.datetime.now().isoformat()}")
    report_lines.append("-" * 20)

    # --- Check 1: Header & Expected Width ---
    try:
        with open(input_data, 'r', encoding='utf-8') as f:
            header_line = f.readline()
            if expected_width is None:
                expected_width = len(header_line.strip().split(','))
            report_lines.append(f"### 1. Header & Expected Width")
            report_lines.append(f"- **Header Line:** `{header_line.strip()}`")
            report_lines.append(f"- **Expected Column Count:** {expected_width}")
    except Exception as e:
        report_lines.append(f"### 1. Header & Expected Width\n- **ERROR:** Could not read header: {e}")
        with open(full_output_path, 'a') as f:
            f.write("\n".join(report_lines))
        return

    # --- Check 2: Row Length Profile ---
    report_lines.append("\n### 2. Row Length Profile (using `csv.reader`)")
    row_lengths = Counter()
    deviating_lines = []
    try:
        with open(input_data, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, doublequote=True)
            next(reader)  # Skip header
            for i, row in enumerate(reader, start=2):
                row_len = len(row)
                row_lengths[row_len] += 1
                if row_len != expected_width and len(deviating_lines) < 20:
                    deviating_lines.append(i)
        
        report_lines.append("- **Row Length Summary:**")
        for length, count in row_lengths.items():
            report_lines.append(f"  - {count} rows with {length} columns.")
        
        if deviating_lines:
            report_lines.append("- **First 20 Deviating Line Numbers:**")
            report_lines.append(f"  - `{deviating_lines}`")
        else:
            report_lines.append("- No rows with inconsistent column counts found.")
        h1_supported = bool(deviating_lines)
    except Exception as e:
        report_lines.append(f"- **ERROR:** Could not process rows with `csv.reader`: {e}")
        h1_supported = None

    # --- Check 3: Malformed Quote Patterns ---
    report_lines.append("\n### 3. Malformed Quote Pattern Scan")
    pattern_a = re.compile(r'^"([A-Z]{1,6}),""')
    pattern_b = re.compile(r'""[A-Za-z]')
    matches_a = []
    matches_b = []
    try:
        with open(input_data, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                if len(matches_a) < 5 and pattern_a.search(line):
                    matches_a.append((i, line.strip()))
                if len(matches_b) < 5 and pattern_b.search(line):
                    matches_b.append((i, line.strip()))

        report_lines.append(f"- **Pattern A (`^\"([A-Z]{{1,6}}),\"\"`):** Found {len(matches_a)} samples.")
        for i, line in matches_a:
            report_lines.append(f"  - Line {i}: `{line}`")
        
        report_lines.append(f"- **Pattern B (`\"\"[A-Za-z]`):** Found {len(matches_b)} samples.")
        for i, line in matches_b:
            report_lines.append(f"  - Line {i}: `{line}`")
        h2_supported = bool(matches_a or matches_b)
    except Exception as e:
        report_lines.append(f"- **ERROR:** Could not scan for quote patterns: {e}")
        h2_supported = None

    # --- Check 4: Pandas Parse Attempt ---
    report_lines.append("\n### 4. Pandas Parsing & Missingness")
    try:
        df = pd.read_csv(input_data)
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        avg_missing = missing_pct.mean()
        cols_gt_25_missing = (missing_pct > 25).sum()
        
        report_lines.append("- **Pandas Parsing:** Success")
        report_lines.append(f"- **Overall Average Missingness:** {avg_missing:.2f}%")
        report_lines.append(f"- **Columns with >25% Missing Values:** {cols_gt_25_missing} / {len(df.columns)}")
        report_lines.append("- **Top 5 Columns by Missingness:**")
        report_lines.append("```")
        report_lines.append(missing_pct.head().to_string())
        report_lines.append("```")
        h3_supported = avg_missing > 10 # Arbitrary threshold for "inflated" 
    except pd.errors.ParserError as e:
        report_lines.append("- **Pandas Parsing:** FAILED")
        report_lines.append(f"- **ParserError:** `{e}`")
        h3_supported = True
    except Exception as e:
        report_lines.append("- **Pandas Parsing:** FAILED with unexpected error.")
        report_lines.append(f"- **Error:** `{e}`")
        h3_supported = True

    # --- Write Report ---
    try:
        with open(full_output_path, 'a', encoding='utf-8') as f:
            f.write("\n\n" + "\n".join(report_lines) + "\n")
        print(f"Validation report appended to {full_output_path}")
    except Exception as e:
        print(f"Error writing report: {e}")
