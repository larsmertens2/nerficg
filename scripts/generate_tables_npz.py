#! /usr/bin/env python3

"""Generate quality tables from SG/SV NPZ inference folders without moving images."""

import warnings
import csv
import os
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean

import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CSV_KEY_FIELDS = ["dataset", "type", "model_file", "threshold", "lobes", "sites"]


def contains_images(path: Path) -> bool:
    return path.exists() and any(child.suffix.lower() in IMAGE_SUFFIXES for child in path.iterdir() if child.is_file())


def threshold_label(name: str) -> str:
    return name.removeprefix("threshold_")


def threshold_csv_value(name: str) -> str:
    return f"{float(threshold_label(name).replace('_', '.')):g}"


def method_name_from_inference_dir(scene_dir: Path, inference_dir: Path) -> str | None:
    parts = inference_dir.relative_to(scene_dir).parts
    if len(parts) < 5 or parts[-2] != "inference":
        return None

    test_type = parts[0].upper()
    npz_threshold = threshold_label(parts[1])
    variant = parts[2]
    render_threshold = threshold_label(parts[-1])
    return f"{test_type}_{variant}_npz_{npz_threshold}_render_{render_threshold}"


def csv_key_from_inference_dir(scene_dir: Path, inference_dir: Path) -> tuple[str, str, str, str, str, str] | None:
    parts = inference_dir.relative_to(scene_dir).parts
    if len(parts) < 5 or parts[-2] != "inference":
        return None

    test_type = parts[0].upper()
    variant_dir = scene_dir.joinpath(*parts[:3])
    npz_files = sorted(path for path in variant_dir.glob("*.npz") if path.is_file())
    if not npz_files:
        return None

    variant = parts[2]
    lobes = variant.removeprefix("lobes_") if variant.lower().startswith("lobes_") else "N/A"
    sites = variant.removeprefix("sites_") if variant.lower().startswith("sites_") else "N/A"
    return (
        scene_dir.name,
        test_type,
        npz_files[0].name,
        threshold_csv_value(parts[-1]),
        lobes,
        sites,
    )


def discover_npz_methods(scene_dir: Path) -> dict[str, Path]:
    methods = {}
    for inference_root in scene_dir.rglob("inference"):
        for inference_dir in sorted(path for path in inference_root.iterdir() if path.is_dir()):
            if not contains_images(inference_dir):
                continue
            method_name = method_name_from_inference_dir(scene_dir, inference_dir)
            if method_name:
                methods[method_name] = inference_dir
    return methods


def find_fastergs_scene_dir(scene_name: str, fastergs_root: Path) -> Path | None:
    if not fastergs_root.exists():
        return None

    candidates = [
        path for path in fastergs_root.iterdir()
        if path.is_dir() and scene_name.lower() in path.name.lower()
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def find_test_dir(scene_name: str, fastergs_root: Path, gt_step: str, gt_subdir: str) -> Path | None:
    scene_dir = find_fastergs_scene_dir(scene_name, fastergs_root)
    if scene_dir is None:
        return None

    preferred = scene_dir / f"test_{gt_step}"
    test_dirs = []
    if preferred.exists():
        test_dirs.append(preferred)
    test_dirs.extend(
        sorted(
            (path for path in scene_dir.glob("test_*") if path.is_dir() and path != preferred),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )

    for test_dir in test_dirs:
        if contains_images(test_dir / gt_subdir):
            return test_dir
    return None


def discover_scenes(npz_root: Path, fastergs_root: Path, gt_step: str, gt_subdir: str):
    scenes = {}
    for scene_dir in sorted(path for path in npz_root.iterdir() if path.is_dir()):
        test_dir = find_test_dir(scene_dir.name, fastergs_root, gt_step, gt_subdir)
        if test_dir is None:
            print(f"Skipping {scene_dir.name}: no FasterGS test {gt_subdir} folder found")
            continue

        methods = discover_npz_methods(scene_dir)
        if not methods:
            print(f"Skipping {scene_dir.name}: no NPZ inference image folders found")
            continue

        scenes[scene_dir.name] = {
            "gt": test_dir / gt_subdir,
            "methods": methods,
            "source_root": scene_dir,
        }
    return scenes


def sorted_method_names(scenes: dict) -> list[str]:
    method_names = {
        method_name
        for scene in scenes.values()
        for method_name in scene["methods"]
    }
    return sorted(method_names)


def load_csv_rows(csv_path: Path):
    if not csv_path.exists():
        return [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def ensure_columns(fieldnames: list[str], required_columns: list[str]) -> list[str]:
    updated_fieldnames = list(fieldnames)
    for column in required_columns:
        if column not in updated_fieldnames:
            updated_fieldnames.append(column)
    return updated_fieldnames


def write_csv_rows(csv_path: Path, fieldnames: list[str], rows: list[dict]):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def load_performance_csv(csv_path: Path, metric_names: list[str]):
    rows, fieldnames = load_csv_rows(csv_path)
    fieldnames = ensure_columns(fieldnames, CSV_KEY_FIELDS + metric_names)
    rows_by_key = {
        tuple(row.get(field, "") for field in CSV_KEY_FIELDS): row
        for row in rows
    }
    write_csv_rows(csv_path, fieldnames, rows)
    return rows, fieldnames, rows_by_key


def metric_values_exist(row: dict | None, metric_names: list[str]) -> bool:
    return row is not None and all(str(row.get(metric_name, "")).strip() for metric_name in metric_names)


def update_performance_csv(csv_path: Path, fieldnames: list[str], rows: list[dict], rows_by_key: dict, row_key: tuple, values: dict):
    row = rows_by_key.get(row_key)
    if row is None:
        row = dict(zip(CSV_KEY_FIELDS, row_key))
        rows.append(row)
        rows_by_key[row_key] = row

    row.update(values)
    write_csv_rows(csv_path, fieldnames, rows)


def image_count(path: Path) -> int:
    return sum(1 for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES)


def write_sources_config(output_dir: Path, metric_names: list[str], scene_names: list[str], method_names: list[str], scenes: dict):
    sources = {
        scene_name: {
            "gt": str(scenes[scene_name]["gt"].resolve()),
            "methods": {
                method_name: str(method_path.resolve())
                for method_name, method_path in scenes[scene_name]["methods"].items()
            },
        }
        for scene_name in scene_names
    }
    config = {
        "METRICS": metric_names,
        "SCENES": scene_names,
        "METHODS": method_names,
        "SOURCES": sources,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=4, canonical=False, sort_keys=False)


def compute_tables(output_dir: Path, metric_names: list[str], scenes: dict, method_names: list[str], csv_path: Path):
    from tabulate import tabulate

    import generate_tables

    for metric_name in metric_names:
        if metric_name not in generate_tables.known_metrics:
            raise ValueError(f"Unknown metric: {metric_name}")

    metric_functions = [generate_tables.known_metrics[metric_name][0] for metric_name in metric_names]
    metric_formatting = [generate_tables.known_metrics[metric_name][1] for metric_name in metric_names]
    metric_requires_mask = [generate_tables.known_metrics[metric_name][2] for metric_name in metric_names]
    results = {
        metric_name: [[] for _ in method_names]
        for metric_name in metric_names
    }
    csv_rows, csv_fieldnames, csv_rows_by_key = load_performance_csv(csv_path, metric_names)

    for scene_name, scene in scenes.items():
        print(f"Processing scene {scene_name}")
        gt_path = scene["gt"]
        gt_files = [
            str(gt_path / name)
            for name in generate_tables.list_sorted_files(gt_path)
            if Path(name).suffix.lower() in IMAGE_SUFFIXES
        ]
        gt_images = generate_tables.load_images(gt_files, scale_factor=None, num_threads=1, desc="loading reference images")[0]
        mask_images = [None] * len(gt_images)

        for method_name in method_names:
            method_path = scene["methods"].get(method_name)
            method_index = method_names.index(method_name)
            if method_path is None:
                print(f"  Missing {method_name} for {scene_name}; filling zeros")
                for metric_name in metric_names:
                    results[metric_name][method_index].append(0.0)
                continue

            csv_key = csv_key_from_inference_dir(scene["source_root"], method_path)
            existing_csv_row = csv_rows_by_key.get(csv_key) if csv_key else None
            if metric_values_exist(existing_csv_row, metric_names):
                print(f"  Skipping {method_name} for {scene_name}; metrics already in {csv_path}")
                for metric_name in metric_names:
                    results[metric_name][method_index].append(float(existing_csv_row[metric_name]))
                continue

            if image_count(method_path) != len(gt_images):
                print(f"  Warning: {method_name} has {image_count(method_path)} images, gt has {len(gt_images)}")

            scene_metrics = generate_tables.compute_metrics(
                method_path,
                gt_images,
                mask_images,
                metric_functions,
                metric_requires_mask,
            )
            csv_values = {
                metric_name: generate_tables.known_metrics[metric_name][1](metric_value)
                for metric_name, metric_value in zip(metric_names, scene_metrics)
            }
            if csv_key:
                update_performance_csv(csv_path, csv_fieldnames, csv_rows, csv_rows_by_key, csv_key, csv_values)
                print(f"  Updated {csv_path} for {scene_name} / {method_name}")
            for metric_name, metric_value in zip(metric_names, scene_metrics):
                results[metric_name][method_index].append(metric_value)

    scene_names = list(scenes)
    headers = ["method"] + scene_names + ["mean"]
    metric_tables = {
        metric_name: [
            [method_name]
            + [
                metric_formatting[metric_names.index(metric_name)](scene_result)
                for scene_result in results[metric_name][method_names.index(method_name)]
            ]
            + [
                metric_formatting[metric_names.index(metric_name)](
                    mean(results[metric_name][method_names.index(method_name)])
                )
            ]
            for method_name in method_names
        ]
        for metric_name in metric_names
    }

    with open(output_dir / "metrics.txt", "w") as f:
        f.write("\n\n".join(
            f"{metric_name}\n{tabulate(metric_table, headers, colalign=['left'] + ['center'] * (len(metric_table[0]) - 1), disable_numparse=True)}"
            for metric_name, metric_table in metric_tables.items()
        ))
    with open(output_dir / "latex_tables.txt", "w") as f:
        f.write("\n\n".join(
            f"% {metric_name} (format: {table_format})\n{tabulate(metric_table, headers, colalign=['left'] + ['center'] * (len(metric_table[0]) - 1), disable_numparse=True, tablefmt=table_format)}"
            for table_format in ["latex", "latex_raw", "latex_booktabs", "latex_longtable", "plain"]
            for metric_name, metric_table in metric_tables.items()
        ))


def main():
    warnings.filterwarnings("ignore")
    parser = ArgumentParser(
        prog="generate_tables_npz.py",
        description="Generate metrics tables directly from SG_test/npz_files inference outputs.",
    )
    parser.add_argument("--npz-root", default="SG_test/npz_files", type=Path)
    parser.add_argument("--fastergs-root", default="output/FasterGS", type=Path)
    parser.add_argument("-o", "--output-dir", default="Tabel_Root/npz_tables", type=Path)
    parser.add_argument("--gt-step", default="3000", help="Preferred FasterGS test folder step, e.g. 3000 or 30000.")
    parser.add_argument("--gt-subdir", default="rgb", help="Subfolder inside test_* to use as ground truth.")
    parser.add_argument("--metrics", nargs="+", default=["SSIM", "FLIP"])
    parser.add_argument("--csv", default="model_performance_summary.csv", type=Path, help="CSV file to update live.")
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()

    scenes = discover_scenes(args.npz_root, args.fastergs_root, args.gt_step, args.gt_subdir)
    if not scenes:
        raise RuntimeError("No scenes found with both FasterGS ground truth images and NPZ inference folders.")

    scene_names = list(scenes)
    method_names = sorted_method_names(scenes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_sources_config(args.output_dir, args.metrics, scene_names, method_names, scenes)
    print(f"Wrote discovered source config to {args.output_dir / 'config.yaml'}")

    if args.config_only:
        return

    import generate_tables

    generate_tables.Logger.set_mode(generate_tables.Logger.MODE_VERBOSE)
    compute_tables(args.output_dir, args.metrics, scenes, method_names, args.csv)
    print(f"Wrote tables to {args.output_dir / 'metrics.txt'} and {args.output_dir / 'latex_tables.txt'}")


if __name__ == "__main__":
    main()
