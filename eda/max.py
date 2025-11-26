"""Utility script to convert a Parquet file to CSV.

Usage (from project root):
  python convert.py --input data/processed/dataset.parquet --output data/processed/dataset.csv

Options:
  --input / -i   Path to source parquet file
  --output / -o  Path to destination csv file (will be overwritten if exists)
  --limit / -n   (Optional) Limit number of rows written (useful for sampling / quick inspection)
  --no-header    (Optional) Do not write header row to CSV

Requires pandas + pyarrow (already in requirements).
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
from pathlib import Path


def convert_parquet_to_csv(input_path: str, output_path: str, header: bool = True) -> None:
	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input parquet file not found: {input_path}")

	# Read parquet
	df = pd.read_parquet(input_path)
	df = df[["ticker", "date","target_fwd_252d"]]
	df.sort_values(by=["target_fwd_252d"],ascending=False, inplace=True)


	# Ensure output directory exists
	out_dir = os.path.dirname(output_path)
	if out_dir and not os.path.isdir(out_dir):
		os.makedirs(out_dir, exist_ok=True)

	# Write CSV
	df.to_csv(output_path, index=False, header=header)


def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert a Parquet file to CSV")
	parser.add_argument("--input", "-i", default=r"data\processed\extract_training.parquet", help="Path to input parquet file")
	parser.add_argument("--no-header", action="store_true", help="Do not write header row")
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv if argv is not None else sys.argv[1:])
	in_path = Path(args.input)
	output = Path("data/processed/max_training.csv")
	convert_parquet_to_csv(input_path=args.input, output_path=str(output), header=not args.no_header)
	print(f"Converted {args.input} -> {output}")
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())

