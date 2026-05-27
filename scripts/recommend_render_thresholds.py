#!/usr/bin/env python3
from __future__ import annotations

import argparse
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SSIM_MIN = 0.99
DEFAULT_FLIP_MAX = 0.005
DEFAULT_TYPES = ("SG", "SV")


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(
        description=(
            "Recommend a general render threshold for SG and SV based on quality "
            "metrics in model_performance_summary.csv."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Input CSV file. If omitted, the script looks for model_performance_summary.csv "
            "or model_performance_summary1.csv in the current directory."
        ),
    )
    parser.add_argument(
        "--ssim-min",
        type=float,
        default=DEFAULT_SSIM_MIN,
        help="Minimum SSIM for a result to count as quality-preserving.",
    )
    parser.add_argument(
        "--flip-max",
        type=float,
        default=DEFAULT_FLIP_MAX,
        help="Maximum FLIP for a result to count as quality-preserving.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=list(DEFAULT_TYPES),
        choices=["SG", "SV", "BASELINE"],
        help="Result types to include in the analysis.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("csv_results") / "threshold_recommendations.csv",
        help="Optional CSV output with the aggregated threshold summary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidate thresholds to print per type.",
    )
    return parser.parse_args()


def resolve_csv_path(csv_path: Path | None) -> Path:
    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        return csv_path

    candidates = [
        Path("model_performance_summary.csv"),
        Path("model_performance_summary1.csv"),
    ]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if existing:
        return existing[0]

    fallback = sorted(Path(".").glob("model_performance_summary*.csv"))
    if fallback:
        return fallback[0]

    raise FileNotFoundError(
        "Could not find a model_performance_summary*.csv file. Pass one explicitly with --csv."
    )


def load_results(csv_path: Path, ssim_min: float, flip_max: float) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required_columns = {
        "dataset",
        "type",
        "threshold",
        "avg_fps",
        "culled_gaussians_percent",
        "SSIM",
        "FLIP",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}"
        )

    data = data.copy()
    for column in ["threshold", "avg_fps", "culled_gaussians_percent", "SSIM", "FLIP"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "mean_culled_gaussians" in data.columns:
        data["mean_culled_gaussians"] = pd.to_numeric(
            data["mean_culled_gaussians"], errors="coerce"
        )

    data["type"] = data["type"].astype(str).str.upper()
    data = data[data["type"].isin(DEFAULT_TYPES)].copy()
    data["quality_preserving"] = (data["SSIM"] >= ssim_min) & (data["FLIP"] <= flip_max)
    return data


def aggregate_thresholds(data: pd.DataFrame, ssim_min: float, flip_max: float) -> pd.DataFrame:
    summary = (
        data.groupby(["type", "threshold"], as_index=False)
        .agg(
            rows=("threshold", "size"),
            datasets=("dataset", "nunique"),
            mean_fps=("avg_fps", "mean"),
            median_fps=("avg_fps", "median"),
            mean_culled_percent=("culled_gaussians_percent", "mean"),
            median_culled_percent=("culled_gaussians_percent", "median"),
            mean_ssim=("SSIM", "mean"),
            median_ssim=("SSIM", "median"),
            mean_flip=("FLIP", "mean"),
            median_flip=("FLIP", "median"),
            quality_pass_rate=("quality_preserving", "mean"),
        )
        .sort_values(["type", "threshold"])
        .reset_index(drop=True)
    )

    summary["quality_gate_pass"] = (
        summary["median_ssim"].ge(ssim_min)
        & summary["median_flip"].le(flip_max)
        & summary["quality_pass_rate"].ge(0.75)
    )
    return summary


def normalize_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    clean = series.astype(float)
    minimum = clean.min()
    maximum = clean.max()
    if pd.isna(minimum) or pd.isna(maximum) or np.isclose(maximum, minimum):
        return pd.Series(np.zeros(len(clean)), index=clean.index)
    normalized = (clean - minimum) / (maximum - minimum)
    return normalized if higher_is_better else 1.0 - normalized


def score_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    scored["fps_score"] = normalize_series(scored["median_fps"], higher_is_better=True)
    scored["ssim_score"] = normalize_series(scored["median_ssim"], higher_is_better=True)
    scored["flip_score"] = normalize_series(scored["median_flip"], higher_is_better=False)
    scored["composite_score"] = (
        0.55 * scored["fps_score"]
        + 0.30 * scored["ssim_score"]
        + 0.15 * scored["flip_score"]
    )
    return scored


def choose_recommendation(type_summary: pd.DataFrame) -> pd.Series:
    gated = type_summary[type_summary["quality_gate_pass"]].copy()
    if not gated.empty:
        return gated.sort_values(
            [
                "median_fps",
                "mean_fps",
                "quality_pass_rate",
                "threshold",
            ],
            ascending=[False, False, False, True],
        ).iloc[0]

    return type_summary.sort_values(
        ["composite_score", "median_fps", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]


def format_threshold(value: float) -> str:
    return f"{value:g}"


def print_type_summary(type_name: str, summary: pd.DataFrame, top_k: int) -> None:
    type_summary = summary[summary["type"].eq(type_name)].copy()
    if type_summary.empty:
        print(f"{type_name}: no rows found")
        return

    recommendation = choose_recommendation(type_summary)
    print(f"\n{type_name} recommendation")
    print(
        f"  threshold: {format_threshold(float(recommendation['threshold']))} | "
        f"median speed: {recommendation['median_fps']:.2f} fps | "
        f"median SSIM: {recommendation['median_ssim']:.3f} | "
        f"median FLIP: {recommendation['median_flip']:.5f} | "
        f"quality pass rate: {recommendation['quality_pass_rate']:.1%}"
    )
    if bool(recommendation["quality_gate_pass"]):
        print("  selection: quality gate passed, then highest speed")
    else:
        print("  selection: no threshold met the quality gate, used quality/speed composite fallback")

    print(f"\n  top {top_k} candidates:")
    for _, row in type_summary.sort_values(
        ["quality_gate_pass", "median_fps", "threshold"],
        ascending=[False, False, True],
    ).head(top_k).iterrows():
        print(
            f"    t={format_threshold(float(row['threshold']))}: "
            f"fps={row['median_fps']:.2f} | cull={row['median_culled_percent']:.2f}% | "
            f"ssim={row['median_ssim']:.3f} | flip={row['median_flip']:.5f} | "
            f"pass={row['quality_pass_rate']:.1%}"
        )


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    data = load_results(csv_path, args.ssim_min, args.flip_max)
    summary = aggregate_thresholds(data, args.ssim_min, args.flip_max)
    scored_summary = score_thresholds(summary)

    print(f"Input CSV: {csv_path}")
    print(f"Quality gate: SSIM >= {args.ssim_min:.3f}, FLIP <= {args.flip_max:.5f}")
    for type_name in args.types:
        if type_name == "BASELINE":
            continue
        print_type_summary(type_name, scored_summary, args.top_k)

    output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_summary.to_csv(output_path, index=False)
    print(f"\nWrote aggregated threshold summary to: {output_path}")


if __name__ == "__main__":
    main()
