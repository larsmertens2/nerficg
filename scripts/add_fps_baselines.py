#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_TYPE = "BASELINE"


def parse_args():
    parser = ArgumentParser(
        description="Run no-culling FPS baselines and add their average FPS to model_performance_summary.csv."
    )
    parser.add_argument("--csv", type=Path, default=Path("model_performance_summary.csv"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--backup", action="store_true")
    return parser.parse_args()


def read_latest_average_fps(base_dir: Path) -> float | None:
    performance_files = sorted(
        base_dir.glob("performance_*.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for performance_file in performance_files:
        text = performance_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Average FPS:\s+([\d.]+)", text)
        if match:
            return float(match.group(1))
    return None


def build_no_culling_env():
    current_env = os.environ.copy()
    current_env["FASTERGS_SG_PATH"] = ""
    current_env["FASTERGS_SV_PATH"] = ""
    return current_env


def run_baseline_benchmark(base_dir: Path, checkpoint: str, repeats: int) -> tuple[float, list[float]]:
    fps_values = []
    for repeat_index in range(repeats):
        print(f"    Benchmark run {repeat_index + 1}/{repeats}...")
        result = subprocess.run(
            [
                sys.executable,
                "scripts/inference.py",
                "-d",
                str(base_dir),
                "--checkpoint",
                checkpoint,
                "-b",
                "--no-culling",
            ],
            env=build_no_culling_env(),
        )
        if result.returncode != 0:
            print(f"    Benchmark run {repeat_index + 1}/{repeats} failed.")
            continue

        fps = read_latest_average_fps(base_dir)
        if fps is not None:
            fps_values.append(fps)
            print(f"    FPS: {fps:.2f}")

    if not fps_values:
        raise RuntimeError(f"No successful baseline FPS runs for {base_dir}")

    return float(np.mean(fps_values)), fps_values


def update_csv(data: pd.DataFrame, baselines: dict[str, float]) -> pd.DataFrame:
    data = data.copy()
    data["type"] = data["type"].astype(str).str.upper()
    data = data[data["type"] != BASELINE_TYPE].copy()
    data["avg_fps"] = pd.to_numeric(data["avg_fps"], errors="coerce")
    data["baseline_avg_fps"] = data["base_model_dir"].astype(str).map(baselines)

    baseline_rows = []
    for (dataset, base_model_dir), _ in data.groupby(["dataset", "base_model_dir"], dropna=False):
        baseline_fps = baselines[str(base_model_dir)]
        row = {column: pd.NA for column in data.columns}
        row.update(
            {
                "dataset": dataset,
                "type": BASELINE_TYPE,
                "model_file": "no_culling",
                "threshold": 0,
                "lobes": "N/A",
                "sites": "N/A",
                "avg_fps": baseline_fps,
                "render_output_dir": str(Path(str(base_model_dir)) / "baseline_benchmark"),
                "base_model_dir": base_model_dir,
                "mean_culled_gaussians": 0,
                "culled_gaussians_percent": 0,
                "SSIM": 1,
                "FLIP": 0,
                "baseline_avg_fps": baseline_fps,
            }
        )
        baseline_rows.append(row)

    return pd.concat([data, pd.DataFrame(baseline_rows)], ignore_index=True).sort_values(
        ["dataset", "type", "model_file", "threshold"],
        na_position="last",
    )


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    data = pd.read_csv(args.csv)
    required_columns = {"dataset", "type", "avg_fps", "base_model_dir"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {args.csv}: {missing}")

    culling_rows = data[data["type"].astype(str).str.upper() != BASELINE_TYPE]
    models = (
        culling_rows[["dataset", "base_model_dir"]]
        .drop_duplicates()
        .sort_values(["dataset", "base_model_dir"])
    )

    baselines = {}
    for _, row in models.iterrows():
        dataset = row["dataset"]
        base_dir = Path(str(row["base_model_dir"]))
        print(f"\n>>> Baseline FPS for {dataset}: {base_dir}")
        average_fps, fps_values = run_baseline_benchmark(base_dir, args.checkpoint, args.repeats)
        baselines[str(row["base_model_dir"])] = average_fps
        print(f"    Average baseline FPS: {average_fps:.2f} from {fps_values}")

    updated = update_csv(data, baselines)
    if args.backup:
        backup_path = args.csv.with_suffix(args.csv.suffix + ".bak")
        args.csv.replace(backup_path)
        print(f"Backed up original CSV to: {backup_path.resolve()}")

    updated.to_csv(args.csv, index=False)
    print(f"\nUpdated {args.csv.resolve()} with no-culling baseline_avg_fps values.")


if __name__ == "__main__":
    main()
