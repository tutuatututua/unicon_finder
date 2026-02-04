
from __future__ import annotations
import os
import argparse
import sys
from typing import List, Tuple
import yaml
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

NUMERIC_EXCLUDE = {"open","high","low","close","adj_close","volume"}
LABEL_COLS = {"label", "fwd_12m_return"}
def load_config(path: str | Path) -> dict:
	"""Load YAML config and return a plain dict.

	The previous implementation wrapped the top level in a SimpleNamespace which
	broke downstream code using cfg.get(...). Returning a dict restores expected
	mapping behavior.
	"""
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Config file not found: {p}")
	with p.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):  # guardrail
		raise ValueError("Config root must be a mapping/dict")
	return data


def ensure_dir(path: str | Path) -> None:
    """Create directory if missing (akin to mkdir -p)."""
    Path(path).mkdir(parents=True, exist_ok=True)


def loading_processed(cfg: dict) -> Tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
	proc_dir = cfg.get("cache", {}).get("processed_dir", "data/processed")
	merged_path = os.path.join(proc_dir, "merged.parquet")
	features_path = os.path.join(proc_dir, "features.parquet")
	labels_path = os.path.join(proc_dir, "labels.parquet")
	panel = feats = labels = None
	if os.path.exists(merged_path):
		panel = pd.read_parquet(merged_path)
	if os.path.exists(features_path):
		feats = pd.read_parquet(features_path)
	if os.path.exists(labels_path):
		labels = pd.read_parquet(labels_path)
	return panel, feats, labels


def _coalesce(panel: pd.DataFrame | None, feats: pd.DataFrame | None, labels: pd.DataFrame | None) -> pd.DataFrame:
	"""Combine available frames prioritizing features->labels->panel."""
	base = None
	for df in [feats, labels, panel]:
		if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
			base = df
			break
	return base if base is not None else pd.DataFrame()


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame()
	idx = df.index
	n_rows = len(df)
	n_cols = len(df.columns)
	tickers = idx.get_level_values("ticker") if isinstance(idx, pd.MultiIndex) and "ticker" in idx.names else df.get("ticker", pd.Series(dtype=str))
	dates = idx.get_level_values("date") if isinstance(idx, pd.MultiIndex) and "date" in idx.names else pd.to_datetime(df.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
	stats = {
		"rows": n_rows,
		"columns": n_cols,
		"tickers": int(pd.Series(tickers).nunique()) if len(tickers) else 0,
		"date_min": getattr(dates.min(), "date", lambda: dates.min())(),
		"date_max": getattr(dates.max(), "date", lambda: dates.max())(),
	}
	return pd.DataFrame([stats])


def feature_summary(df: pd.DataFrame, max_features: int | None = None) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame()
	# Numeric columns only
	num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
	if max_features:
		num_cols = num_cols[:max_features]
	rows = []
	for c in num_cols:
		s = pd.to_numeric(df[c], errors="coerce")
		rows.append({
			"feature": c,
			"count": int(s.count()),
			"na_pct": float(s.isna().mean()*100.0),
			"mean": float(s.mean()) if s.count() else np.nan,
			"std": float(s.std(ddof=0)) if s.count() else np.nan,
			"min": float(s.min()) if s.count() else np.nan,
			"max": float(s.max()) if s.count() else np.nan,
		})
	return pd.DataFrame(rows).sort_values("na_pct")


def missing_matrix(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame()
	miss = df.isna().sum().to_frame("missing")
	miss["missing_pct"] = miss["missing"] / len(df) * 100.0
	return miss.sort_values("missing", ascending=False)


def compute_correlations(df: pd.DataFrame, target: str = "fwd_12m_return", top_n: int = 30) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame()
	numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
	if target not in numeric_cols:
		# Try label col
		if "label" in df.columns and pd.api.types.is_numeric_dtype(df["label"]):
			target = "label"
		else:
			# nothing to do
			return pd.DataFrame()
	# For correlation drop rows where target is nan
	sub = df[numeric_cols].copy()
	sub = sub.loc[sub[target].notna()]
	if sub.empty:
		return pd.DataFrame()
	corr_series = sub.corr(numeric_only=True)[target].drop(target, errors="ignore").dropna()
	top = corr_series.reindex(corr_series.abs().sort_values(ascending=False).head(top_n).index)
	return top.to_frame("correlation")


def plot_distributions(df: pd.DataFrame, out_dir: str, max_plots: int = 25) -> List[str]:
	paths: List[str] = []
	if df.empty:
		return paths
	numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
	# Put target first if exists
	ordered = []
	for tgt in ["fwd_12m_return", "label"]:
		if tgt in numeric_cols:
			ordered.append(tgt)
	ordered += [c for c in numeric_cols if c not in ordered]
	if max_plots:
		ordered = ordered[:max_plots]
	ensure_dir(out_dir)
	for c in ordered:
		plt.figure(figsize=(4,3))
		s = pd.to_numeric(df[c], errors="coerce").dropna()
		if s.empty:
			continue
		plt.hist(s, bins=50, alpha=0.7, color="#2c7fb8")
		plt.title(c)
		plt.tight_layout()
		path = os.path.join(out_dir, f"{c}.png")
		plt.savefig(path)
		plt.close()
		paths.append(path)
	return paths


def plot_missingness(miss_df: pd.DataFrame, out_path: str) -> str | None:
	if miss_df.empty:
		return None
	# Take top 40 columns by missing count to keep plot readable
	top = miss_df.head(40)
	plt.figure(figsize=(10,4))
	plt.bar(top.index, top["missing"], color="#f03b20")
	plt.xticks(rotation=90, fontsize=8)
	plt.ylabel("Missing Count")
	plt.title("Top Missing Columns")
	plt.tight_layout()
	ensure_dir(os.path.dirname(out_path))
	plt.savefig(out_path)
	plt.close()
	return out_path


def plot_corr_heatmap(df: pd.DataFrame, out_path: str, max_features: int = 30) -> str | None:
	if df.empty:
		return None
	numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
	# Remove base price columns to focus on engineered features
	numeric_cols = [c for c in numeric_cols if c not in NUMERIC_EXCLUDE]
	if len(numeric_cols) == 0:
		return None
	# Limit col count
	if len(numeric_cols) > max_features:
		numeric_cols = numeric_cols[:max_features]
	sub = df[numeric_cols].copy()
	# Drop rows where everything is nan
	sub = sub.dropna(how="all")
	if sub.empty:
		return None
	corr = sub.corr(numeric_only=True)
	plt.figure(figsize=(max(6, len(numeric_cols)*0.35), max(6, len(numeric_cols)*0.35)))
	im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
	plt.colorbar(im, fraction=0.046, pad=0.04)
	plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90, fontsize=7)
	plt.yticks(range(len(numeric_cols)), numeric_cols, fontsize=7)
	plt.title("Feature Correlation Heatmap")
	plt.tight_layout()
	ensure_dir(os.path.dirname(out_path))
	plt.savefig(out_path)
	plt.close()
	return out_path


def run_eda(config_path: str, *, max_hist: int = 25) -> None:
	cfg = load_config(config_path)
	panel, feats, labels = loading_processed(cfg)
	df = _coalesce(panel, feats, labels)
	if df.empty:
		print("[eda] No processed data available. Run pipeline first.")
		return

	# Merge features + labels if both present for richer set
	if feats is not None and labels is not None:
		# Both are MultiIndex (date, ticker)
		df = feats.join(labels[[c for c in labels.columns if c not in feats.columns]], how="left")

	out_root = cfg.get("outputs", {}).get("dir", "outputs")
	eda_dir = os.path.join(out_root, "eda")
	ensure_dir(eda_dir)

	# Basic stats
	basic = basic_stats(df)
	basic.to_csv(os.path.join(eda_dir, "basic_stats.csv"), index=False)
	print("[eda] Saved basic_stats.csv")

	# Feature summary
	fsum = feature_summary(df)
	fsum.to_csv(os.path.join(eda_dir, "feature_summary.csv"), index=False)
	print("[eda] Saved feature_summary.csv")

	# Missingness
	miss = missing_matrix(df)
	miss.to_csv(os.path.join(eda_dir, "missing_matrix.csv"))
	print("[eda] Saved missing_matrix.csv")
	plot_missingness(miss, os.path.join(eda_dir, "missingness_bar.png"))

	# Correlations (top absolute with target)
	corr_top = compute_correlations(df)
	if not corr_top.empty:
		corr_top.to_csv(os.path.join(eda_dir, "correlations_topN.csv"))
		print("[eda] Saved correlations_topN.csv")

	# Heatmap
	plot_corr_heatmap(df, os.path.join(eda_dir, "correlations_heatmap.png"))

	# Distributions
	dist_dir = os.path.join(eda_dir, "distributions")
	plot_distributions(df, dist_dir, max_plots=max_hist)
	print("[eda] Saved distributions (subset)")

	print(f"[eda] Complete. Outputs in {eda_dir}")


def main():  # CLI
	parser = argparse.ArgumentParser(description="Run exploratory data analysis on processed data")
	parser.add_argument("--config", type=str, default="config.yaml")
	parser.add_argument("--max-hist", type=int, default=25, help="Maximum histogram plots (to limit runtime)")
	args = parser.parse_args()
	run_eda(args.config, max_hist=args.max_hist)


if __name__ == "__main__":  # pragma: no cover
	main()

