import csv
import io
import os
import re
from collections import Counter
from typing import Optional

import pandas as pd


def clean_trailing_quotes(line: str) -> str:
    """Remove trailing quote artifacts like '0.08""' -> '0.08'"""
    # Pattern: value followed by one or more extra quotes at field end
    # Matches: "0.08""" or 0.08"" or similar
    line = re.sub(r'(\d+\.?\d*)""+', r'\1"', line)  # numeric with trailing quotes
    line = re.sub(r'([^,"])""+(?=,|$)', r'\1"', line)  # any value with trailing quotes before delimiter
    return line

def run(
    input_data: str,
    out_name: str,
    report_name: str,
    expected_width: Optional[int] = None,
    dry_run: bool = False,
    post_clean: bool = True,
) -> None:
    """Repair malformed CSV with quote and field grouping issues."""
    
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, out_name)
    report_path = os.path.join(output_dir, report_name)
    
    # Read header to determine expected width
    with open(input_data, "r", encoding="utf-8") as f:
        header = next(csv.reader([f.readline()]))
    expected_width = expected_width or len(header)
    
    # Repair patterns
    patterns = [
        # Fix: "TICK,Company,Sector,Industry"," tail"" -> proper fields
        (re.compile(r'"([A-Z]{1,6}),([^"]+),([^"]+),([^"]+)"\s*,\s*"([^"]*)""+'), 
         lambda m: f'{m.group(1)},"{m.group(2)}","{m.group(3)}","{m.group(4)}, {m.group(5)}"'),
        
        # Fix: "TICK,"" -> TICK,"
        (re.compile(r'"([A-Z]{1,6}),""'), r'\1,"'),
        
        # Fix: "" before letters or delimiters -> "
        (re.compile(r'""(?=[A-Za-z,]|$)'), '"'),
    ]
    
    stats = {"fixed": 0, "unfixed": 0, "unfixed_lines": [], "pre_widths": Counter(), "post_widths": Counter()}
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    
    # Repair pass
    with open(input_data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            stats["pre_widths"][len(next(csv.reader([line])))] += 1
            
            # Apply all patterns
            for pattern, replacement in patterns:
                line = pattern.sub(replacement, line)
            
            # Strip lonely trailing quote if quote count is odd
            if line.rstrip("\n").endswith('"') and line.count('"') % 2 == 1:
                line = line[:-2] + "\n" if line.endswith('"\n') else line[:-1]
            
            # Clean trailing quote artifacts
            if post_clean:
                line = clean_trailing_quotes(line)
            
            # Parse and validate
            try:
                fields = next(csv.reader([line]))
                stats["post_widths"][len(fields)] += 1
                
                if len(fields) == expected_width:
                    stats["fixed"] += 1
                else:
                    stats["unfixed"] += 1
                    if len(stats["unfixed_lines"]) < 20:
                        stats["unfixed_lines"].append((i, len(fields), line.strip()[:200]))
                    
                writer.writerow(fields)
            except Exception as e:
                stats["unfixed"] += 1
                if len(stats["unfixed_lines"]) < 20:
                    stats["unfixed_lines"].append((i, "parse_error", line.strip()[:200]))
                writer.writerow(next(csv.reader([line])))
    
    fixed_content = output.getvalue()
    
    # Validate with pandas
    try:
        df = pd.read_csv(io.StringIO(fixed_content))
        missing_pct = (df.isna().mean() * 100).mean()
        validation_ok = len(stats["post_widths"]) == 1 and stats["unfixed"] == 0
    except Exception as e:
        validation_ok = False
        missing_pct = None
    
    # Write output
    if not dry_run:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(fixed_content)
    
    # Write report
    report = [
        f"# CSV Repair Report",
        f"Input: {input_data}",
        f"Output: {output_path if not dry_run else 'N/A (dry-run)'}",
        f"Expected width: {expected_width}",
        f"",
        f"Pre-repair widths: {dict(stats['pre_widths'])}",
        f"Post-repair widths: {dict(stats['post_widths'])}",
        f"",
        f"Fixed rows: {stats['fixed']}",
        f"Unfixed rows: {stats['unfixed']}",
        f"Missing data: {missing_pct:.2f}%" if missing_pct else "Missing data: N/A",
        f"",
        f"Validation: {'PASS' if validation_ok else 'FAIL'}",
    ]
    
    # Add unfixed rows details
    if stats["unfixed_lines"]:
        report.extend([
            f"",
            f"## Unfixed Rows (showing up to 20)",
        ])
        for line_num, width, snippet in stats["unfixed_lines"]:
            report.append(f"Line {line_num}: width={width} | {snippet}...")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Fixed CSV -> {output_path}")
    print(f"Report -> {report_path}")
    
    if not validation_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    # Example standalone usage
    run(
        input_data="input.csv",
        out_name="repaired.csv",
        report_name="report.txt"
    )
