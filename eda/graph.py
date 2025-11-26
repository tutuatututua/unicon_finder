"""Utility plotting functions for EDA.

Plot two side-by-side graphs of a chosen numeric column from a parquet DataFrame:

- Left: Raw distribution (histogram + KDE), optionally trimmed
- Right: "After binning" distribution (quantile bins bar chart), if n_relevance_bins is provided

Usage (from repo root):

    python -m eda.graph               # saves figure to eda/target_fwd_252d_density.png by default
    python -m eda.graph --show        # also show interactively
    python -m eda.graph --source data/processed/extract_training.parquet --col number_of_stock
    python -m eda.graph --bins 120 --trim 0.01 --col target_fwd_252d --bins-quant 10

Options allow trimming extreme tails (winsor-style) for clearer visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Defaults
DEFAULT_COLUMN = "target_fwd_252d"
DEFAULT_SOURCE = "data/processed/extract_training.parquet"


def plot_target_density(
    df: pd.DataFrame,
    column: str = DEFAULT_COLUMN,
    *,
    bins: int = 100,
    trim: float | None = None,
    show: bool = False,
    save_path: Path | None = None,
    n_relevance_bins: int | None = None,
) -> Path:

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe. Available columns: {list(df.columns)}")

    series = df[column].replace([np.inf, -np.inf], np.nan).dropna()
    original_n = len(series)
    if trim:
        lower = series.quantile(trim)
        upper = series.quantile(1 - trim)
        series = series.clip(lower, upper)
    trimmed_n = len(series)

    # Create two side-by-side plots: raw (left) and binned (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Raw distribution
    ax_raw = axes[0]
    ax_raw.hist(series, bins=bins, density=True, alpha=0.45, edgecolor="white", label="Histogram")
    try:  # KDE using pandas (fallback-safe)
        series.plot(kind="kde", ax=ax_raw, lw=2, label="KDE")
    except Exception:  # pragma: no cover - fallback if KDE fails
        pass
    ax_raw.set_title(f"Raw distribution of '{column}' (n={original_n}, trimmed={trimmed_n})")
    ax_raw.set_xlabel(column)
    ax_raw.set_ylabel("stock")
    ax_raw.grid(alpha=0.2)
    ax_raw.legend()

    # Right: After binning (quantile bins)
    ax_bin = axes[1]
    title_extra = ""
    if n_relevance_bins and n_relevance_bins > 0:
        try:
            q_labels = pd.qcut(series, q=n_relevance_bins, labels=False, duplicates="drop")
            counts = q_labels.value_counts().sort_index()
            # Bar plot of counts by bin index
            ax_bin.bar(range(len(counts)), counts.values, color="C1", alpha=0.7, edgecolor="white")
            ax_bin.set_xlabel("Quantile bin index")
            ax_bin.set_xticks(range(len(counts)))
            ax_bin.set_ylabel("Count")
            title_extra = f"bins={len(counts)}"
            # Optional: overlay edges as text in the title for reference
            _, bin_edges = pd.qcut(series, q=n_relevance_bins, retbins=True, duplicates="drop")
            ax_bin.set_title(f"After binning (qcut, {title_extra})")
            # Annotate counts above bars
            for x, ct in enumerate(counts.values):
                ax_bin.text(x, ct, str(ct), ha="center", va="bottom", fontsize=8)
            ax_bin.grid(alpha=0.2, axis="y")
        except ValueError as e:  # Not enough unique values
            ax_bin.text(0.5, 0.5, f"Binning failed: {e}", ha="center", va="center", transform=ax_bin.transAxes)
            ax_bin.set_axis_off()
    else:
        ax_bin.text(0.5, 0.5, "Binning disabled (provide --bins-quant)", ha="center", va="center", transform=ax_bin.transAxes)
        ax_bin.set_axis_off()

    if save_path is None:
        # default filename close to this script
        save_path = Path(__file__).with_name(f"{column}_density.png")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot density of a chosen numeric column")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Path to processed parquet file (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument("--col", "--column", dest="column", default=DEFAULT_COLUMN, help=f"Column to plot (default: {DEFAULT_COLUMN})")
    parser.add_argument("--trim", type=float, default=False, help="Two-sided quantile to trim (e.g. 0.01 trims 1% tails)")
    parser.add_argument("--show", action="store_true", help="Show interactively")
    parser.add_argument("--no-save", action="store_true", help="Do not save figure (only show if --show)")
    parser.add_argument("--bins-quant", type=int, default=5, help="Number of relevance quantile bins to overlay (0 to disable)")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins for the raw distribution (left plot)")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.source)
    save_path = None if args.no_save else Path(__file__).with_name(f"{args.column}_density.png")
    n_relevance_bins = args.bins_quant if args.bins_quant is not None else None
    out = plot_target_density(
        df,
        column=args.column,
        bins=args.bins,
        trim=args.trim,
        show=args.show,
        save_path=save_path,
        n_relevance_bins=n_relevance_bins,
    )
    if out:
        print(f"Saved density plot to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
