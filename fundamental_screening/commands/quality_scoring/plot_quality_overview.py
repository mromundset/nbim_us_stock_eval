import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def run(
    input_data: str = "out/quality_scores/quality_scores_2023.csv",
    out_plot: str = "quality_scores_total_health_dist.png",
    out_report: str = "quality_scores_total_health_dist_report.txt",
    value_col: str = "total_health",
    bins: int = 20,
    by_sector: bool = False,
    sector_col: str = "Sector",
    percentile_clip: Optional[float] = None,  # clip right tail at this percentile for visualization
    log_x: bool = False,
    figsize=(10.5, 6.0),
    alpha: float = 0.65
) -> None:
    """
    Plot a probability distribution (density histogram) of total_health from the quality CSV.

    Parameters
    ----------
    input_data : str
        Path to the output CSV from the quality scoring step.
    out_plot : str
        Filename (inside /out) for the PNG plot.
    out_report : str
        Filename (inside /out) for a short text summary.
    value_col : str
        Column to analyze (default: 'total_health').
    bins : int
        Number of histogram bins.
    by_sector : bool
        If True, overlays a density histogram per sector (legend shown).
    sector_col : str
        Sector column name in the CSV.
    percentile_clip : float | None
        If set, clip values above this percentile for *plotting only* (keeps stats unmodified).
    log_x : bool
        If True, uses a log scale on the x-axis.
    """

    # ---------- I/O setup ----------
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, out_plot)
    rpt_path  = os.path.join(out_dir, out_report)

    # ---------- Load ----------
    df = pd.read_csv(input_data)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in {input_data}. Available: {list(df.columns)}")

    vals_all = pd.to_numeric(df[value_col], errors="coerce")
    vals = vals_all.dropna().values

    if vals.size == 0:
        print("No numeric values to plot.")
        return

    # ---------- Stats (on full data, no clipping) ----------
    stats = {
        "count": int(np.isfinite(vals).sum()),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan"),
        "p01": float(np.nanpercentile(vals, 1)),
        "p10": float(np.nanpercentile(vals, 10)),
        "p25": float(np.nanpercentile(vals, 25)),
        "p50": float(np.nanpercentile(vals, 50)),
        "p75": float(np.nanpercentile(vals, 75)),
        "p90": float(np.nanpercentile(vals, 90)),
        "p99": float(np.nanpercentile(vals, 99)),
    }

    # ---------- Optional percentile clipping for plotting only ----------
    if percentile_clip is not None:
        upper = np.nanpercentile(vals, float(percentile_clip))
        vals_plot = np.clip(vals, None, upper)
    else:
        vals_plot = vals

    # ---------- Plot ----------
    plt.figure(figsize=figsize)
    if by_sector and sector_col in df.columns:
        # Overlay per-sector density histograms (normalized)
        sectors = (
            df[[sector_col, value_col]]
            .dropna(subset=[value_col])
            .assign(**{value_col: lambda x: pd.to_numeric(x[value_col], errors="coerce")})
            .dropna(subset=[value_col])
        )
        if percentile_clip is not None:
            upper = np.nanpercentile(sectors[value_col].values, float(percentile_clip))
            sectors[value_col] = np.clip(sectors[value_col].values, None, upper)

        # Limit to sectors with at least 5 data points to avoid noisy curves
        for s, g in sectors.groupby(sector_col):
            gv = g[value_col].values
            if len(gv) < 5:
                continue
            plt.hist(
                gv,
                bins=bins,
                density=True,
                alpha=alpha,
                label=str(s),
                histtype="stepfilled"
            )
        plt.legend(title="Sector", fontsize=8)
        title = f"Probability Distribution of {value_col} by Sector"
    else:
        # Single overall density histogram
        plt.hist(
            vals_plot,
            bins=bins,
            density=True,
            alpha=0.8,
            histtype="stepfilled"
        )
        title = f"Probability Distribution of {value_col}"

    if log_x:
        plt.xscale("log")

    plt.xlabel(value_col)
    plt.ylabel("Probability Density")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # ---------- Report ----------
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("# Total Health Distribution Report\n")
        f.write(f"Input: {input_data}\n")
        f.write(f"Value column: {value_col}\n")
        f.write(f"Bins: {bins}, By-sector: {by_sector}, Percentile clip (plot-only): {percentile_clip}\n\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nPlot saved to: {plot_path}\n")

    print(f"→ Plot -> {plot_path}")
    print(f"→ Report -> {rpt_path}")


if __name__ == "__main__":
    run(
        input_data="out/quality_scores_2023.csv",
        out_plot="quality_scores_total_health_dist.png",
        out_report="quality_scores_total_health_dist_report.txt",
        bins=40,
        by_sector=False,   # set True to overlay per-sector densities
        percentile_clip=99.0,
        log_x=False
    )
