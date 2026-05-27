#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Minimal, focused analysis: produce clean FLIP vs Speedup lines (one line per threshold)
# and grouped bar charts (Speedup / FLIP / Culling) per method (SG, SV).


def parse_args() -> ArgumentParser:
    p = ArgumentParser(description="Clean culling analysis: focused plots per method.")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV with results")
    p.add_argument("--out", type=Path, default=Path("output/culling_plots"), help="Output directory for plots")
    # Veranderd naar --linear-flip omdat logaritmisch nu de betere standaardkeuze is
    p.add_argument("--linear-flip", action="store_true", help="Plot FLIP axis on a linear scale instead of log scale")
    return p.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    # normalize expected column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df


def compute_speedup_series(df: pd.DataFrame) -> pd.Series:
    if "fps_speedup_vs_baseline" in df.columns:
        return df["fps_speedup_vs_baseline"].astype(float)
    if "baseline_avg_fps" in df.columns and "avg_fps" in df.columns:
        return (df["avg_fps"].astype(float) / df["baseline_avg_fps"].astype(float)).replace([np.inf, -np.inf], np.nan)
    # fallback: normalize by dataset mean to get a relative speed measure
    if "avg_fps" in df.columns and "dataset" in df.columns:
        return df.groupby("dataset")["avg_fps"].transform(lambda x: x.astype(float) / x.astype(float).mean())
    # last fallback: use avg_fps raw if present
    if "avg_fps" in df.columns:
        return df["avg_fps"].astype(float)
    # otherwise return NaNs
    return pd.Series(np.nan, index=df.index)


def aggregate_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speedup_plot"] = compute_speedup_series(df)
    # ensure common columns exist
    for col in ["type", "threshold", "config_label", "config_value", "FLIP", "culled_gaussians_percent"]:
        if col not in df.columns:
            df[col] = np.nan
    # derive config_value from lobes/sites when missing
    if df["config_value"].isna().all():
        # SG uses 'lobes', SV uses 'sites'
        if "type" in df.columns and df["type"].dropna().size > 0:
            if "lobes" in df.columns:
                df.loc[df["type"].str.upper() == "SG", "config_value"] = pd.to_numeric(df.loc[df["type"].str.upper() == "SG", "lobes"], errors="coerce")
            if "sites" in df.columns:
                df.loc[df["type"].str.upper() == "SV", "config_value"] = pd.to_numeric(df.loc[df["type"].str.upper() == "SV", "sites"], errors="coerce")

    df["config_value"] = pd.to_numeric(df["config_value"], errors="coerce")
    # aggregate mean per method/amount/threshold
    agg = (
        df.groupby(["type", "config_value", "threshold"], as_index=False)
        .agg(speedup=("speedup_plot", "mean"), FLIP=("FLIP", "mean"), culling=("culled_gaussians_percent", "mean"))
    )
    return agg


def plot_flip_vs_speedup(agg: pd.DataFrame, out_dir: Path, log_flip: bool = True) -> None:
    methods = [m for m in sorted(agg["type"].dropna().unique()) if str(m).upper() in {"SG", "SV"}]
    for method in methods:
        mdf = agg[agg["type"].eq(method)].copy()
        if mdf.empty:
            continue
        thresholds = sorted(mdf["threshold"].dropna().unique())
        amounts = sorted(mdf["config_value"].dropna().unique())
        if len(thresholds) == 0 or len(amounts) == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.get_cmap("plasma")
        colors = cmap(np.linspace(0, 1, len(thresholds)))

        # vertical offset scale to separate overlapping lines slightly
        y_min = mdf["speedup"].min()
        y_max = mdf["speedup"].max()
        y_span = max(y_max - y_min, 1e-9)
        offset_step = 0.02 * y_span

        # prepare FLIP offset if log scale requested: avoid zeros by adding small eps
        if log_flip:
            nonzero_min = mdf["FLIP"][mdf["FLIP"] > 0].min()
            if np.isnan(nonzero_min) or nonzero_min <= 0:
                eps = 1e-6
            else:
                eps = float(nonzero_min) / 10.0
            # set log scale on x axis
            ax.set_xscale("log")

        for i, thr in enumerate(thresholds):
            row = mdf[mdf["threshold"].eq(thr)].set_index("config_value").reindex(amounts)
            x = row["FLIP"].values.astype(float)
            if log_flip:
                x = x + eps
            y = row["speedup"].values.astype(float)
            # small offset
            y = y + (i - (len(thresholds) - 1) / 2) * offset_step
            valid = ~np.isnan(x) & ~np.isnan(y)
            if not np.any(valid):
                continue
            xi = x[valid]; yi = y[valid]
            order = np.argsort(xi)
            xi = xi[order]; yi = yi[order]
            ax.plot(xi, yi, marker="o", color=colors[i], linewidth=2, label=f"t={thr:g}")
            # annotate amounts (verkleind van 7 naar 6 voor minder overlap)
            amt_list = np.array(amounts)[valid][order]
            for xv, yv, amt in zip(xi, yi, amt_list):
                ax.annotate(str(int(amt)), (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=8)

        ax.set_xlabel("FLIP")
        ax.set_ylabel("Speedup (scene-independent)")
        # adding which="both" zorgt voor een overzichtelijk grid bij log-schaal
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(title="threshold", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.suptitle("")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"clean_flip_vs_speedup_{str(method).lower()}.png"
        plt.tight_layout(rect=[0, 0, 0.92, 1])
        plt.savefig(save_path, dpi=180)
        plt.close(fig)


def plot_grouped_bars(agg: pd.DataFrame, out_dir: Path) -> None:
    methods = [m for m in sorted(agg["type"].dropna().unique()) if str(m).upper() in {"SG", "SV"}]
    for method in methods:
        mdf = agg[agg["type"].eq(method)].copy()
        if mdf.empty:
            continue
        amounts = sorted(mdf["config_value"].dropna().unique())
        thresholds = sorted(mdf["threshold"].dropna().unique())
        if not amounts or not thresholds:
            continue

        x = np.arange(len(amounts))
        width = 0.8 / max(1, len(thresholds))
        cmap = plt.get_cmap("plasma")
        ax_label = "lobes" if str(method).upper() == "SG" else "sites"
        metric_specs = [
            ("speedup", "Speedup", "{:.2f}"),
            ("FLIP", "FLIP", "{:.5f}"),
            ("culling", "Culling (%)", "{:.1f}"),
        ]
        for metric_key, ylabel, fmt in metric_specs:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars_all = []
            for j, thr in enumerate(thresholds):
                vals = []
                for amt in amounts:
                    row = mdf[(mdf["config_value"].eq(amt)) & (mdf["threshold"].eq(thr))]
                    vals.append(np.nan if row.empty else float(row[metric_key].mean()))
                color = cmap(j / max(1, len(thresholds) - 1))
                bars_all.append(ax.bar(x + j * width, vals, width=width, color=color, label=f"t={thr:g}"))

            ax.set_xticks(x + width * (len(thresholds) - 1) / 2)
            ax.set_xticklabels([str(int(a)) for a in amounts])
            ax.set_xlabel(ax_label)
            ax.set_ylabel(ylabel)
            ax.set_title("")
            ax.grid(axis="y", alpha=0.25)

            # --- FIX: Force Vertical Y-axis to 100% for culling ---
            if metric_key == "culling":
                ax.set_ylim(0, 100)
            # ------------------------------------------------------

            rects = [r for bc in bars_all for r in bc]
            _annotate_bars_generic(ax, rects, fmt, fontsize=8)
            ax.legend(frameon=False)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"clean_bar_{metric_key}_{str(method).lower()}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=180)
            plt.close(fig)


def plot_culling_vs_scores(agg: pd.DataFrame, out_dir: Path, log_culling: bool = False) -> None:
    methods = [m for m in sorted(agg["type"].dropna().unique()) if str(m).upper() in {"SG", "SV"}]
    for method in methods:
        mdf = agg[agg["type"].eq(method)].copy()
        if mdf.empty:
            continue

        thresholds = sorted(mdf["threshold"].dropna().unique())
        amounts = sorted(mdf["config_value"].dropna().unique())
        if len(thresholds) == 0 or len(amounts) == 0:
            continue

        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(thresholds)))

        # optional log scale for culling values if they cluster near zero
        if log_culling:
            nonzero_min = mdf["culling"][mdf["culling"] > 0].min()
            eps = 1e-6 if np.isnan(nonzero_min) or nonzero_min <= 0 else float(nonzero_min) / 10.0
            ax_s.set_xscale("log")
            ax_f.set_xscale("log")
        else:
            eps = 0.0
        metric_specs = [
            ("speedup", "Speedup (scene-independent)", f"clean_culling_vs_speedup_{str(method).lower()}.png"),
            ("FLIP", "FLIP", f"clean_culling_vs_flip_{str(method).lower()}.png"),
        ]
        for metric_key, ylabel, filename in metric_specs:
            fig, ax = plt.subplots(figsize=(7.5, 5))
            for i, thr in enumerate(thresholds):
                thr_df = mdf[mdf["threshold"].eq(thr)].set_index("config_value").reindex(amounts)
                x = thr_df["culling"].values.astype(float)
                if log_culling:
                    x = x + eps
                y = thr_df[metric_key].values.astype(float)
                valid = ~np.isnan(x) & ~np.isnan(y)
                if not np.any(valid):
                    continue
                xv = x[valid]
                yv = y[valid]
                order = np.argsort(xv)
                xv = xv[order]
                yv = yv[order]
                ax.plot(xv, yv, marker="o", color=colors[i], linewidth=2, label=f"t={thr:g}")
                amt = np.array(amounts)[valid][order]
                for xx, yy, aa in zip(xv, yv, amt):
                    ax.annotate(str(int(aa)), (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=8)

            ax.set_xlabel("Culling (%)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25, which="both")
            ax.legend(title="threshold", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.suptitle("")
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / filename
            plt.tight_layout(rect=[0, 0, 0.92, 1])
            plt.savefig(save_path, dpi=180)
            plt.close(fig)


def _annotate_bars_generic(ax, rects, fmt="{:.2f}", fontsize=8):
    for rect in rects:
        try:
            h = rect.get_height()
        except Exception:
            continue
        if np.isnan(h):
            continue
        va = "bottom" if h >= 0 else "top"
        ax.text(rect.get_x() + rect.get_width() / 2, h, fmt.format(h), ha="center", va=va, fontsize=fontsize)


def main():
    args = parse_args()
    out_dir = args.out
    df = load_data(args.csv)
    agg = aggregate_for_plotting(df)
    
    # Bug opgelost: log_flip wordt nu correct berekend en doorgegeven
    log_flip = not args.linear_flip
    plot_flip_vs_speedup(agg, out_dir, log_flip=log_flip)
    plot_grouped_bars(agg, out_dir)
    plot_culling_vs_scores(agg, out_dir, log_culling=False)
    print(f"Wrote plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()