#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUALITY_SSIM_MIN = 0.99
QUALITY_FLIP_MAX = 0.005


METHOD_COLORS = {
    "SG": "#2563eb",
    "SV": "#dc2626",
    "BASELINE": "#111827",
}


def parse_args():
    parser = ArgumentParser(
        description="Create CSV summaries and graphs comparing SG and SV culling results."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("model_performance_summary.csv"),
        help="Input CSV with benchmark results.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("csv_results"),
        help="Output directory for graphs and summary CSV files.",
    )
    parser.add_argument(
        "--ssim-min",
        type=float,
        default=QUALITY_SSIM_MIN,
        help="Minimum SSIM for the quality-preserving summary.",
    )
    parser.add_argument(
        "--flip-max",
        type=float,
        default=QUALITY_FLIP_MAX,
        help="Maximum FLIP for the quality-preserving summary.",
    )
    return parser.parse_args()


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    required_columns = {
        "dataset",
        "type",
        "threshold",
        "lobes",
        "sites",
        "avg_fps",
        "mean_culled_gaussians",
        "culled_gaussians_percent",
        "SSIM",
        "FLIP",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    data = data.copy()
    numeric_columns = [
        "threshold",
        "lobes",
        "sites",
        "avg_fps",
        "mean_culled_gaussians",
        "culled_gaussians_percent",
        "SSIM",
        "FLIP",
    ]
    optional_numeric_columns = [
        "baseline_avg_fps",
        "baseline_fps_std",
        "fps_delta_vs_baseline",
        "fps_speedup_vs_baseline",
        "fps_percent_change_vs_baseline",
    ]
    numeric_columns.extend(
        column for column in optional_numeric_columns if column in data.columns
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "baseline_avg_fps" in data.columns:
        if "fps_delta_vs_baseline" not in data.columns:
            data["fps_delta_vs_baseline"] = data["avg_fps"] - data["baseline_avg_fps"]
        if "fps_speedup_vs_baseline" not in data.columns:
            data["fps_speedup_vs_baseline"] = data["avg_fps"] / data["baseline_avg_fps"]
        if "fps_percent_change_vs_baseline" not in data.columns:
            data["fps_percent_change_vs_baseline"] = (
                data["fps_speedup_vs_baseline"] - 1
            ) * 100

    data["type"] = data["type"].astype(str).str.upper()
    data["config_value"] = np.select(
        [data["type"].eq("SG"), data["type"].eq("SV")],
        [data["lobes"], data["sites"]],
        default=0,
    )
    data["config_label"] = np.select(
        [data["type"].eq("SG"), data["type"].eq("SV")],
        [
            "lobes=" + data["lobes"].fillna(-1).astype(int).astype(str),
            "sites=" + data["sites"].fillna(-1).astype(int).astype(str),
        ],
        default="no culling",
    )
    data["quality_preserving"] = (
        data["SSIM"].ge(QUALITY_SSIM_MIN) & data["FLIP"].le(QUALITY_FLIP_MAX)
    )
    data["utility_score"] = (
        data["avg_fps"].rank(pct=True)
        + data["culled_gaussians_percent"].rank(pct=True)
        + data["SSIM"].rank(pct=True)
        + (1 - data["FLIP"].rank(pct=True))
    )
    return data.sort_values(["dataset", "type", "config_value", "threshold"])


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#374151")


def method_sort_key(method: str) -> tuple[int, str]:
    order = {"BASELINE": 0, "SG": 1, "SV": 2}
    return order.get(method, 99), method


def plot_metric_vs_culled(data: pd.DataFrame, output_dir: Path, metric: str, ylabel: str) -> None:
    datasets = sorted(data["dataset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    axes = axes.ravel()

    for axis, dataset in zip(axes, datasets):
        subset = data[data["dataset"].eq(dataset)]
        for method in sorted(subset["type"].unique(), key=method_sort_key):
            method_subset = subset[subset["type"].eq(method)]
            grouped = (
                method_subset.groupby("threshold", as_index=False)
                .agg(
                    culled_gaussians_percent=("culled_gaussians_percent", "mean"),
                    metric=(metric, "mean"),
                )
                .sort_values("culled_gaussians_percent")
            )
            axis.plot(
                grouped["culled_gaussians_percent"],
                grouped["metric"],
                marker="o",
                linewidth=2,
                color=method_color(method),
                label=method,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Culled Gaussians (%)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)

    for axis in axes[len(datasets) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"{ylabel} vs Culling Amount", y=1.03, fontsize=15)
    save_figure(output_dir / f"{metric.lower()}_vs_culled_percent.png")


def plot_fps_vs_quality(data: pd.DataFrame, output_dir: Path) -> None:
    datasets = sorted(data["dataset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    for axis, dataset in zip(axes, datasets):
        subset = data[data["dataset"].eq(dataset)]
        for method in sorted(subset["type"].unique(), key=method_sort_key):
            method_subset = subset[subset["type"].eq(method)]
            axis.scatter(
                method_subset["FLIP"],
                method_subset["avg_fps"],
                s=25 + method_subset["culled_gaussians_percent"] * 1.8,
                alpha=0.7,
                color=method_color(method),
                edgecolor="white",
                linewidth=0.6,
                label=method,
            )
        axis.axvline(QUALITY_FLIP_MAX, color="#111827", linestyle="--", linewidth=1)
        axis.set_title(dataset)
        axis.set_xlabel("FLIP (lower is better)")
        axis.set_ylabel("Average FPS")
        axis.grid(True, alpha=0.25)

    for axis in axes[len(datasets) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Speed vs Visual Error (bubble size = culled %)", y=1.03, fontsize=15)
    save_figure(output_dir / "fps_vs_flip_tradeoff.png")


def plot_speedup_vs_quality(data: pd.DataFrame, output_dir: Path) -> None:
    if "fps_speedup_vs_baseline" not in data.columns:
        return

    datasets = sorted(data["dataset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    for axis, dataset in zip(axes, datasets):
        subset = data[data["dataset"].eq(dataset)]
        for method in sorted(subset["type"].unique(), key=method_sort_key):
            method_subset = subset[subset["type"].eq(method)]
            axis.scatter(
                method_subset["FLIP"],
                method_subset["fps_speedup_vs_baseline"],
                s=35 + method_subset["culled_gaussians_percent"] * 1.8,
                alpha=0.75,
                color=method_color(method),
                edgecolor="white",
                linewidth=0.6,
                label=method,
            )
        axis.axhline(1.0, color="#111827", linestyle="--", linewidth=1)
        axis.axvline(QUALITY_FLIP_MAX, color="#6b7280", linestyle=":", linewidth=1)
        axis.set_title(dataset)
        axis.set_xlabel("FLIP (lower is better)")
        axis.set_ylabel("FPS speedup vs no culling")
        axis.grid(True, alpha=0.25)

    for axis in axes[len(datasets) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Speedup vs Visual Error (baseline = 1.0)", y=1.03, fontsize=15)
    save_figure(output_dir / "speedup_vs_flip_tradeoff.png")


def plot_threshold_curves(data: pd.DataFrame, output_dir: Path) -> None:
    datasets = sorted(data["dataset"].unique())
    for dataset in datasets:
        subset = data[data["dataset"].eq(dataset) & data["type"].isin(["SG", "SV"])]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        for method in sorted(subset["type"].unique(), key=method_sort_key):
            method_subset = subset[subset["type"].eq(method)]
            for config_label, config_subset in method_subset.groupby("config_label"):
                label = f"{method} {config_label}"
                alpha = 0.8 if method == "SG" else 0.65
                axes[0].plot(
                    config_subset["threshold"],
                    config_subset["avg_fps"],
                    marker="o",
                    linewidth=1.8,
                    label=label,
                    color=method_color(method),
                    alpha=alpha,
                )
                axes[1].plot(
                    config_subset["threshold"],
                    config_subset["FLIP"],
                    marker="o",
                    linewidth=1.8,
                    label=label,
                    color=method_color(method),
                    alpha=alpha,
                )

        axes[0].set_title("FPS by threshold")
        axes[0].set_ylabel("Average FPS")
        axes[1].set_title("FLIP by threshold")
        axes[1].set_ylabel("FLIP (lower is better)")
        for axis in axes:
            axis.set_xlabel("Inference threshold")
            axis.grid(True, alpha=0.25)
        axes[1].axhline(QUALITY_FLIP_MAX, color="#111827", linestyle="--", linewidth=1)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(f"{dataset}: Threshold Sensitivity", y=1.08, fontsize=15)
        save_figure(output_dir / f"{dataset}_threshold_sensitivity.png")


def plot_quality_preserving_best(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    filtered = data[data["quality_preserving"]].copy()
    best = (
        filtered.sort_values(
            ["dataset", "type", "avg_fps", "culled_gaussians_percent", "SSIM"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["dataset", "type"], as_index=False)
        .head(1)
        .sort_values(["dataset", "type"])
    )

    best.to_csv(output_dir / "best_quality_preserving_configs.csv", index=False)

    if best.empty:
        return best

    labels = sorted(best["dataset"].unique())
    x_positions = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for offset, method in [(-width / 2, "SG"), (width / 2, "SV")]:
        method_best = best[best["type"].eq(method)].set_index("dataset").reindex(labels)
        axes[0].bar(
            x_positions + offset,
            method_best["avg_fps"],
            width,
            label=method,
            color=method_color(method),
        )
        axes[1].bar(
            x_positions + offset,
            method_best["culled_gaussians_percent"],
            width,
            label=method,
            color=method_color(method),
        )

    axes[0].set_title("Fastest config that keeps quality")
    axes[0].set_ylabel("Average FPS")
    axes[1].set_title("Culling reached while keeping quality")
    axes[1].set_ylabel("Culled Gaussians (%)")
    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)

    fig.suptitle(
        f"Quality-preserving configs (SSIM >= {QUALITY_SSIM_MIN}, FLIP <= {QUALITY_FLIP_MAX})",
        y=1.03,
        fontsize=15,
    )
    save_figure(output_dir / "best_quality_preserving_configs.png")
    return best


def pareto_front(group: pd.DataFrame) -> pd.DataFrame:
    candidates = group.sort_values(
        ["FLIP", "avg_fps", "culled_gaussians_percent"],
        ascending=[True, False, False],
    )
    front_rows = []
    best_fps = -np.inf
    best_culled = -np.inf
    for _, row in candidates.iterrows():
        if row["avg_fps"] > best_fps or row["culled_gaussians_percent"] > best_culled:
            front_rows.append(row)
            best_fps = max(best_fps, row["avg_fps"])
            best_culled = max(best_culled, row["culled_gaussians_percent"])
    if not front_rows:
        return candidates.head(0)
    return pd.DataFrame(front_rows)


def write_pareto_outputs(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    fronts = []
    for (dataset, method), group in data.groupby(["dataset", "type"]):
        front = pareto_front(group).copy()
        front["dataset"] = dataset
        front["type"] = method
        fronts.append(front)
    pareto = pd.concat(fronts, ignore_index=True) if fronts else data.head(0)
    pareto.to_csv(output_dir / "pareto_front_configs.csv", index=False)

    datasets = sorted(data["dataset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for axis, dataset in zip(axes, datasets):
        subset = pareto[pareto["dataset"].eq(dataset)]
        for method in sorted(subset["type"].unique()):
            method_subset = subset[subset["type"].eq(method)].sort_values("FLIP")
            axis.plot(
                method_subset["FLIP"],
                method_subset["avg_fps"],
                marker="o",
                linewidth=2,
                color=method_color(method),
                label=method,
            )
        axis.set_title(dataset)
        axis.set_xlabel("FLIP (lower is better)")
        axis.set_ylabel("Average FPS")
        axis.grid(True, alpha=0.25)

    for axis in axes[len(datasets) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Pareto Front: Best Speed / Error Trade-offs", y=1.03, fontsize=15)
    save_figure(output_dir / "pareto_front_fps_vs_flip.png")
    return pareto


def write_overall_summaries(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary_aggs = {
        "rows": ("type", "size"),
        "mean_fps": ("avg_fps", "mean"),
        "max_fps": ("avg_fps", "max"),
        "mean_culled_percent": ("culled_gaussians_percent", "mean"),
        "max_culled_percent": ("culled_gaussians_percent", "max"),
        "mean_ssim": ("SSIM", "mean"),
        "min_ssim": ("SSIM", "min"),
        "mean_flip": ("FLIP", "mean"),
        "max_flip": ("FLIP", "max"),
        "quality_preserving_rows": ("quality_preserving", "sum"),
    }
    overall_aggs = {
        "rows": ("type", "size"),
        "mean_fps": ("avg_fps", "mean"),
        "max_fps": ("avg_fps", "max"),
        "mean_culled_percent": ("culled_gaussians_percent", "mean"),
        "max_culled_percent": ("culled_gaussians_percent", "max"),
        "mean_ssim": ("SSIM", "mean"),
        "mean_flip": ("FLIP", "mean"),
        "quality_preserving_rows": ("quality_preserving", "sum"),
    }
    if "fps_speedup_vs_baseline" in data.columns:
        summary_aggs["mean_speedup_vs_baseline"] = ("fps_speedup_vs_baseline", "mean")
        summary_aggs["max_speedup_vs_baseline"] = ("fps_speedup_vs_baseline", "max")
        overall_aggs["mean_speedup_vs_baseline"] = ("fps_speedup_vs_baseline", "mean")
        overall_aggs["max_speedup_vs_baseline"] = ("fps_speedup_vs_baseline", "max")

    summary = (
        data.groupby(["dataset", "type"], as_index=False)
        .agg(**summary_aggs)
        .sort_values(["dataset", "type"])
    )
    summary.to_csv(output_dir / "method_summary_by_dataset.csv", index=False)

    overall = (
        data.groupby("type", as_index=False)
        .agg(**overall_aggs)
        .sort_values("type")
    )
    overall.to_csv(output_dir / "method_summary_overall.csv", index=False)

    metric_specs = [
        ("mean_fps", "Mean speed", "Average FPS"),
        ("mean_culled_percent", "Mean culling", "Culled Gaussians (%)"),
        ("mean_flip", "Mean visual error", "FLIP"),
    ]
    if "mean_speedup_vs_baseline" in overall.columns:
        metric_specs.insert(1, ("mean_speedup_vs_baseline", "Mean speedup", "FPS / baseline"))

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(4.8 * len(metric_specs), 5))
    axes = np.atleast_1d(axes)
    for axis, (metric, title, ylabel) in zip(axes, metric_specs):
        axis.bar(
            overall["type"],
            overall[metric],
            color=[method_color(method) for method in overall["type"]],
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Overall SG vs SV Summary", y=1.03, fontsize=15)
    save_figure(output_dir / "overall_method_summary.png")
    return summary


def write_recommendations(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    culling_data = data[data["type"].isin(["SG", "SV"])].copy()
    for dataset, dataset_group in culling_data.groupby("dataset"):
        quality_group = dataset_group[dataset_group["quality_preserving"]]
        source = quality_group if not quality_group.empty else dataset_group
        for method, method_group in source.groupby("type"):
            best = method_group.sort_values(
                ["utility_score", "avg_fps", "culled_gaussians_percent"],
                ascending=[False, False, False],
            ).iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "type": method,
                    "threshold": best["threshold"],
                    "config_label": best["config_label"],
                    "avg_fps": best["avg_fps"],
                    "culled_gaussians_percent": best["culled_gaussians_percent"],
                    "SSIM": best["SSIM"],
                    "FLIP": best["FLIP"],
                    "quality_preserving": bool(best["quality_preserving"]),
                    "utility_score": best["utility_score"],
                    "baseline_avg_fps": best.get("baseline_avg_fps", np.nan),
                    "fps_speedup_vs_baseline": best.get("fps_speedup_vs_baseline", np.nan),
                    "fps_percent_change_vs_baseline": best.get("fps_percent_change_vs_baseline", np.nan),
                }
            )

    recommendations = pd.DataFrame(rows).sort_values(["dataset", "type"])
    recommendations.to_csv(output_dir / "recommended_configs.csv", index=False)
    return recommendations


def main() -> None:
    args = parse_args()
    global QUALITY_SSIM_MIN, QUALITY_FLIP_MAX
    QUALITY_SSIM_MIN = args.ssim_min
    QUALITY_FLIP_MAX = args.flip_max

    ensure_output_dir(args.out)
    data = load_results(args.csv)
    data.to_csv(args.out / "cleaned_model_performance_summary.csv", index=False)

    write_overall_summaries(data, args.out)
    plot_metric_vs_culled(data, args.out, "avg_fps", "Average FPS")
    if "fps_speedup_vs_baseline" in data.columns:
        plot_metric_vs_culled(
            data,
            args.out,
            "fps_speedup_vs_baseline",
            "FPS speedup vs no culling",
        )
    plot_metric_vs_culled(data, args.out, "SSIM", "SSIM (higher is better)")
    plot_metric_vs_culled(data, args.out, "FLIP", "FLIP (lower is better)")
    plot_fps_vs_quality(data, args.out)
    plot_speedup_vs_quality(data, args.out)
    plot_threshold_curves(data, args.out)
    plot_quality_preserving_best(data, args.out)
    write_pareto_outputs(data, args.out)
    write_recommendations(data, args.out)

    print(f"Wrote culling analysis results to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
