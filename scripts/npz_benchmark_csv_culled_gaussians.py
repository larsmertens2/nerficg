#! /usr/bin/env python3

"""npz_benchmark_csv_culled_gaussians.py: Export culled gaussian statistics for NPZ thresholds."""

import csv
import os
import re
from pathlib import Path

import numpy as np

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Logging import Logger


def find_matching_config(dataset_name, output_root="output/FasterGS"):
    output_path = Path(output_root)
    if not output_path.exists():
        return None

    candidates = []
    for folder in output_path.iterdir():
        if folder.is_dir() and dataset_name.lower() in folder.name.lower():
            config_file = folder / "training_config.yaml"
            if config_file.exists():
                candidates.append(config_file)

    if not candidates:
        return None

    candidates.sort(key=lambda path: path.parent.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_lobes_and_sites(npz_path: Path):
    path_str = str(npz_path)
    lobes_match = re.search(r"lobes_(\d+)", path_str, flags=re.IGNORECASE)
    sites_match = re.search(r"sites_(\d+)", path_str, flags=re.IGNORECASE)
    return (
        lobes_match.group(1) if lobes_match else "N/A",
        sites_match.group(1) if sites_match else "N/A",
    )


def read_existing_keys(csv_path: Path, key_fields):
    if not csv_path.exists():
        return set()

    keys = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(tuple(row.get(field, "") for field in key_fields))
    return keys


def load_csv_rows(csv_path: Path):
    if not csv_path.exists():
        return [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv_rows(csv_path: Path, fieldnames, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def ensure_columns(fieldnames, required_columns):
    updated_fieldnames = list(fieldnames)
    for column in required_columns:
        if column not in updated_fieldnames:
            updated_fieldnames.append(column)
    return updated_fieldnames


def make_row_key(row, key_fields):
    return tuple(row.get(field, "") for field in key_fields)


def has_values(row, columns):
    return all(str(row.get(column, "")).strip() for column in columns)


def benchmark_thresholds():
    return [0.4]


def threshold_dir_name(threshold_value: float):
    return f"threshold_{str(threshold_value).replace('.', '_')}"


def configure_env_for_npz(npz_path: Path, test_type: str):
    abs_npz_path = str(npz_path.resolve())
    if test_type.upper() == "SG":
        os.environ["FASTERGS_SG_PATH"] = abs_npz_path
        os.environ.pop("FASTERGS_SV_PATH", None)
    else:
        os.environ["FASTERGS_SV_PATH"] = abs_npz_path
        os.environ.pop("FASTERGS_SG_PATH", None)


def load_benchmark_context(base_dir: Path, npz_path: Path = None, test_type: str = None):
    Framework.setup(config_path=str(base_dir / "training_config.yaml"), require_custom_config=True)
    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH,
    )
    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / "checkpoints" / "final.pt"),
    ).eval()
    
    # Reload SG/SV values after checkpoint loading (checkpoint overwrites them)
    if npz_path and test_type:
        if test_type.upper() == "SG":
            model.load_SG_Values(str(npz_path.resolve()))
        else:
            model.load_SV_Values(str(npz_path.resolve()))
    
    renderer = MI.get_renderer(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        model=model,
    )
    return dataset, model, renderer


def get_benchmark_views(dataset):
    dataset.test()
    views = list(dataset)
    if views:
        return views, "test"

    dataset.train()
    views = list(dataset)
    if views:
        Logger.log_warning("No test images found, falling back to the training set for benchmarking.")
        return views, "train"

    raise Framework.InferenceError("No images found for benchmarking.")


def compute_mean_culled_gaussians(renderer, views):
    culled_values = []
    for view in views:
        outputs = renderer.render_image_inference(view=view)
        culled_tensor = outputs.get("gaussians_culled")
        if culled_tensor is None:
            continue
        culled_values.append(float(culled_tensor.detach().reshape(-1)[0].cpu().item()))

    return float(np.mean(culled_values)) if culled_values else 0.0


def main():
    root_test_dir = Path("SG_test/npz_files")
    output_csv = Path("model_performance_summary.csv")
    base_columns = [
        "dataset",
        "type",
        "model_file",
        "threshold",
        "lobes",
        "sites",
        "avg_fps",
        "render_output_dir",
        "base_model_dir",
    ]
    culling_columns = ["mean_culled_gaussians", "culled_gaussians_percent"]
    
    # 1. Laad bestaande rijen en headers in
    rows, fieldnames = load_csv_rows(output_csv)
    
    # Zorg dat de master-fieldnames alle mogelijke kolommen bevatten voor NIEUWE rijen
    extended_fieldnames = ensure_columns(fieldnames, base_columns + culling_columns)
    
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        write_csv_rows(output_csv, extended_fieldnames, rows)
        print(f">>> Nieuw CSV bestand aangemaakt met headers: {output_csv}")

    # Groepeer bestaande rijen op basis van model-eigenschappen
    npz_groups = {}
    for row in rows:
        npz_key = (row.get("dataset"), row.get("type"), row.get("model_file"), str(row.get("lobes")), str(row.get("sites")))
        if npz_key not in npz_groups:
            npz_groups[npz_key] = []
        npz_groups[npz_key].append(row)

    default_thresholds = benchmark_thresholds()

    # Alle NPZ bestanden verzamelen
    all_npz = []
    for p in root_test_dir.rglob("*.npz"):
        parts = [part.upper() for part in p.parts]
        if ("SG" in parts or "SV" in parts) and "camera_data" not in p.name:
            all_npz.append(p)

    for npz_path in all_npz:
        dataset_name = npz_path.relative_to(root_test_dir).parts[0]
        test_type = next((part for part in npz_path.parts if part.upper() in {"SG", "SV"}), None)
        
        if not test_type:
            continue

        lobes, sites = parse_lobes_and_sites(npz_path)
        model_dir = npz_path.parent
        
        npz_key = (dataset_name, test_type.upper(), npz_path.name, str(lobes), str(sites))
        existing_rows_for_npz = npz_groups.get(npz_key, [])

        thresholds_to_run = []
        if existing_rows_for_npz:
            for r in existing_rows_for_npz:
                try:
                    t_val = float(r.get("threshold", 0.4))
                    thresholds_to_run.append((t_val, r))
                except ValueError:
                    continue
        else:
            for t_val in default_thresholds:
                thresholds_to_run.append((t_val, None))

        context_loaded = False
        dataset, model, renderer, benchmark_views, total_gaussians = None, None, None, None, 0

        for threshold_value, existing_row in thresholds_to_run:
            threshold_csv_value = f"{threshold_value:g}"
            
            # Controleer scherp of culling kolommen écht tekst bevatten (en niet leeg/Spaties zijn)
            if existing_row and has_values(existing_row, culling_columns):
                print(f"    Slaan over: Threshold {threshold_csv_value} voor {npz_path.name} heeft al culling-resultaten.")
                continue

            print(f"\n>>> VERWERKEN: {npz_path.name} (Threshold: {threshold_csv_value})")

            config_source = find_matching_config(dataset_name)
            if not config_source:
                print(f"    Skipping: Geen basis model gevonden voor {dataset_name}")
                continue
                
            base_dir = config_source.parent

            if not context_loaded:
                configure_env_for_npz(npz_path, test_type)
                dataset, model, renderer = load_benchmark_context(base_dir, npz_path, test_type)
                benchmark_views, _ = get_benchmark_views(dataset)
                total_gaussians = int(model.gaussians.means.shape[0]) if model.gaussians is not None else 0
                context_loaded = True

            renderer.SG_THRESHOLD = threshold_value
            renderer.SV_THRESHOLD = threshold_value
            renderer.USE_SG = (test_type.upper() == "SG")
            renderer.USE_SV = (test_type.upper() == "SV")

            print(f"    Berekenen culling...")
            mean_culled = compute_mean_culled_gaussians(renderer, benchmark_views)
            percent_culled = (mean_culled / total_gaussians * 100.0) if total_gaussians > 0 else 0.0

            new_row = {
                "dataset": dataset_name,
                "type": test_type.upper(),
                "model_file": npz_path.name,
                "threshold": threshold_csv_value,
                "lobes": lobes,
                "sites": sites,
                "render_output_dir": str((model_dir / "inference" / threshold_dir_name(threshold_value)).resolve()),
                "base_model_dir": str(base_dir.resolve()),
                "mean_culled_gaussians": f"{mean_culled:.2f}",
                "culled_gaussians_percent": f"{percent_culled:.2f}"
            }

            if existing_row:
                existing_row.update(new_row)
                print(f"    [OK] Bestaande CSV-rij bijgewerkt.")
            else:
                rows.append(new_row)
                print(f"    [OK] Nieuwe CSV-rij toegevoegd.")
            
            # Schrijf de update direct weg met de complete header-set
            write_csv_rows(output_csv, extended_fieldnames, rows)

    print(f"\nKlaar! Alle resultaten zijn opgeslagen in {output_csv}")

if __name__ == "__main__":
    main()
