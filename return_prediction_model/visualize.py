# visualizations.py
# Generate evaluation plots from model summary CSV logs

from __future__ import annotations
import os, argparse
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Helper utilities
# ---------------------------

def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def _save(fig: plt.Figure, outdir: str, fname: str) -> str:
    _ensure_outdir(outdir)
    path = os.path.join(outdir, fname)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")
    return path

def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _parse_top_features_series(s: pd.Series) -> List[str]:
    names: List[str] = []
    for cell in s.dropna():
        parts = [p.strip() for p in str(cell).split(";") if p.strip()]
        for p in parts:
            if ":" in p:
                name = p.rsplit(":", 1)[0].strip()
            else:
                name = p.strip()
            if name:
                names.append(name)
    return names

# ---------------------------
# Visualization functions
# ---------------------------

def plot_avg_ic_by_model(csv_path: str, outdir: str = "figures") -> str:
    df = _load(csv_path)
    _require_cols(df, ["model_type", "mean_rank_ic"])
    g = df.groupby("model_type")["mean_rank_ic"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    g.plot(kind="barh", ax=ax)
    ax.set_title("Average Rank IC by Model")
    ax.set_xlabel("Mean Rank IC")
    for i, v in enumerate(g.values):
        ax.text(v, i, f"{v:.3f}", va="center", ha="left", fontsize=9)
    return _save(fig, outdir, "avg_ic_by_model.png")

def plot_normalization_impact(csv_path: str, outdir: str = "figures") -> str:
    df = _load(csv_path)
    _require_cols(df, ["model_type", "normalize", "mean_rank_ic"])
    pivot = df.pivot_table(index="model_type", columns="normalize",
                            values="mean_rank_ic", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Normalization Impact on Rank IC")
    ax.set_ylabel("Mean Rank IC")
    ax.legend(title="Normalize", frameon=False)
    return _save(fig, outdir, "ic_by_model_normalize.png")

def plot_sector_neutral_effect(csv_path: str, outdir: str = "figures") -> str:
    df = _load(csv_path)
    _require_cols(df, ["model_type", "sector_neutral", "mean_rank_ic"])
    pivot = df.pivot_table(index="model_type", columns="sector_neutral",
                            values="mean_rank_ic", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Sector-Neutralization Effect on Rank IC")
    ax.set_ylabel("Mean Rank IC")
    ax.legend(title="Sector Neutral", frameon=False)
    return _save(fig, outdir, "ic_by_model_sector_neutral.png")

def plot_avg_tstat_by_model(csv_path: str, outdir: str = "figures") -> str:
    df = _load(csv_path)
    _require_cols(df, ["model_type", "rank_ic_tstat"])
    g = df.groupby("model_type")["rank_ic_tstat"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    g.plot(kind="bar", ax=ax)
    ax.axhline(2.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Average Rank IC t-stat by Model")
    ax.set_ylabel("t-statistic")
    for i, v in enumerate(g.values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    return _save(fig, outdir, "avg_tstat_by_model.png")

def plot_avg_decile_spread_by_model(csv_path: str, outdir: str = "figures") -> str:
    df = _load(csv_path)
    _require_cols(df, ["model_type", "avg_decile_spread"])
    g = df.groupby("model_type")["avg_decile_spread"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    g.plot(kind="bar", ax=ax)
    ax.set_title("Average Decile Spread by Model")
    ax.set_ylabel("Avg Decile Spread (Top–Bottom)")
    for i, v in enumerate(g.values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    return _save(fig, outdir, "avg_decile_spread_by_model.png")

def plot_ic_heatmap_normalize_sector(csv_path: str, outdir: str = "figures") -> str:
    df = _load(csv_path)
    _require_cols(df, ["normalize", "sector_neutral", "mean_rank_ic"])
    table = df.pivot_table(index="normalize", columns="sector_neutral",
                            values="mean_rank_ic", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5, 3))
    data = table.values
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Rank IC by Normalization × Sector Neutral")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Rank IC")
    return _save(fig, outdir, "ic_heatmap_normalize_sector.png")

def plot_feature_frequency(csv_path: str, outdir: str = "figures", top_k: int = 15) -> str:
    df = _load(csv_path)
    _require_cols(df, ["top_features"])
    names = _parse_top_features_series(df["top_features"])
    s = pd.Series(names).value_counts().sort_values(ascending=True).tail(top_k)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(s))))
    s.plot(kind="barh", ax=ax)
    ax.set_title(f"Top-{min(top_k, len(s))} Features by Frequency")
    ax.set_xlabel("Count in Top-Features")
    for i, v in enumerate(s.values):
        ax.text(v, i, str(v), va="center", ha="left", fontsize=9)
    return _save(fig, outdir, "feature_frequency.png")

def plot_ic_over_time(csv_path: str, outdir: str = "figures") -> str:
    df = _load(csv_path)
    _require_cols(df, ["timestamp", "model_type", "mean_rank_ic"])
    d = df.dropna(subset=["timestamp", "mean_rank_ic"]).sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, g in d.groupby("model_type"):
        ax.plot(g["timestamp"], g["mean_rank_ic"], marker="o", linestyle="-", label=m)
    ax.set_title("Mean Rank IC Over Time by Model Type")
    ax.set_ylabel("Mean Rank IC")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    return _save(fig, outdir, "ic_over_time.png")

# ---------------------------
# Run all
# ---------------------------

def run_all(csv_path: str, outdir: str = "figures") -> Dict[str, str]:
    paths = {}
    paths["avg_ic_by_model"] = plot_avg_ic_by_model(csv_path, outdir)
    paths["ic_by_model_normalize"] = plot_normalization_impact(csv_path, outdir)
    paths["ic_by_model_sector_neutral"] = plot_sector_neutral_effect(csv_path, outdir)
    paths["avg_tstat_by_model"] = plot_avg_tstat_by_model(csv_path, outdir)
    paths["avg_decile_spread_by_model"] = plot_avg_decile_spread_by_model(csv_path, outdir)
    paths["ic_heatmap_normalize_sector"] = plot_ic_heatmap_normalize_sector(csv_path, outdir)
    paths["feature_frequency"] = plot_feature_frequency(csv_path, outdir)
    try:
        paths["ic_over_time"] = plot_ic_over_time(csv_path, outdir)
    except Exception:
        pass
    return paths

# ---------------------------
# CLI wrapper
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualization CLI for Model Evaluation")
    parser.add_argument("--command", required=True,
                        help="Which visualization to run (e.g., run_all, plot_avg_ic_by_model)")
    parser.add_argument("--input", required=True, help="Path to CSV results file")
    parser.add_argument("--output", default="figures", help="Output directory for figures")
    args = parser.parse_args()

    # command dispatcher
    commands = {
        "plot_avg_ic_by_model": plot_avg_ic_by_model,
        "plot_normalization_impact": plot_normalization_impact,
        "plot_sector_neutral_effect": plot_sector_neutral_effect,
        "plot_avg_tstat_by_model": plot_avg_tstat_by_model,
        "plot_avg_decile_spread_by_model": plot_avg_decile_spread_by_model,
        "plot_ic_heatmap_normalize_sector": plot_ic_heatmap_normalize_sector,
        "plot_feature_frequency": plot_feature_frequency,
        "plot_ic_over_time": plot_ic_over_time,
        "run_all": run_all,
    }

    if args.command not in commands:
        raise ValueError(f"Unknown command '{args.command}'. Options: {list(commands.keys())}")

    fn = commands[args.command]
    result = fn(args.input, args.output)
    print("Completed:", args.command)
    if isinstance(result, dict):
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("  Output:", result)
