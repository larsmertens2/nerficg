import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

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


def parse_threshold_from_path(npz_path: Path):
    for part in npz_path.parts:
        match = re.fullmatch(r"threshold_([0-9_]+)", part, flags=re.IGNORECASE)
        if match:
            return float(match.group(1).replace("_", "."))
    return None


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


def append_csv_row(csv_path: Path, fieldnames, row):
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def extract_average_fps(command_result):
    combined_output = "\n".join(filter(None, [command_result.stdout, command_result.stderr]))
    match = re.search(r"Average FPS:\s+([\d.]+)", combined_output)
    return float(match.group(1)) if match else None


def read_latest_average_fps(base_dir: Path):
    performance_files = sorted(base_dir.glob("performance_*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for perf_file in performance_files:
        with open(perf_file, "r") as f:
            match = re.search(r"Average FPS:\s+([\d.]+)", f.read())
            if match:
                return float(match.group(1))
    return None


def build_threshold_args(test_type: str, threshold: float | None):
    if threshold is None:
        return []
    if test_type.upper() == "SG":
        return ["--sg-threshold", str(threshold)]
    return ["--sv-threshold", str(threshold)]


def build_renderer_mode_args(test_type: str):
    if test_type.upper() == "SG":
        return ["--use-sg"]
    return ["--use-sv"]


def build_env_for_npz(npz_path: Path, test_type: str):
    current_env = os.environ.copy()
    abs_npz_path = str(npz_path.resolve())
    if test_type.upper() == "SG":
        current_env["FASTERGS_SG_PATH"] = abs_npz_path
        current_env["FASTERGS_SV_PATH"] = ""
    else:
        current_env["FASTERGS_SV_PATH"] = abs_npz_path
        current_env["FASTERGS_SG_PATH"] = ""
    return current_env


def benchmark_thresholds():
    return [round(step / 10, 1) for step in range(1, 10)]


def threshold_dir_name(threshold_value: float):
    return f"threshold_{str(threshold_value).replace('.', '_')}"

def main():
    root_test_dir = Path("SG_test/npz_files")
    output_csv = Path("model_performance_summary.csv")
    fieldnames = [
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
    processed_keys = read_existing_keys(output_csv, ["dataset", "type", "model_file", "threshold", "lobes", "sites"])
    threshold_values = benchmark_thresholds()

    # 1. Strenge filter: Alleen .npz bestanden die in een 'SG' of 'SV' submap zitten
    all_npz = []
    for p in root_test_dir.rglob("*.npz"):
        parts = [part.upper() for part in p.parts]
        if "SG" in parts or "SV" in parts:
            if "camera_data" not in p.name:
                all_npz.append(p)

    for npz_path in all_npz:
        dataset_name = npz_path.relative_to(root_test_dir).parts[0]
        test_type = next((part for part in npz_path.parts if part.upper() in {"SG", "SV"}), None)
        if not test_type:
            print(f"\n>>> Overslaan: kan SG/SV type niet bepalen voor {npz_path}")
            continue

        lobes, sites = parse_lobes_and_sites(npz_path)

        # De specifieke map van de npz (bijv. lobes_4)
        model_dir = npz_path.parent

        print(f"\n>>> VERWERKEN: {npz_path.name}")
        print(f"    Pad: {npz_path}")

        # Zoek basis config
        config_source = find_matching_config(dataset_name)
        if not config_source:
            print(f"    Skipping: Geen basis model gevonden in output/FasterGS voor {dataset_name}")
            continue
        base_dir = config_source.parent

        current_env = build_env_for_npz(npz_path, test_type)

        for threshold_value in threshold_values:
            threshold_csv_value = f"{threshold_value:g}"
            threshold_folder = model_dir / "inference" / threshold_dir_name(threshold_value)
            row_key = (dataset_name, test_type.upper(), npz_path.name, threshold_csv_value, lobes, sites)
            if row_key in processed_keys:
                print(f"    Overslaan threshold {threshold_csv_value}: al verwerkt")
                continue

            threshold_args = build_threshold_args(test_type, threshold_value)
            renderer_mode_args = build_renderer_mode_args(test_type)
            print(f"    Rendering naar: {threshold_folder}")
            render_result = subprocess.run([
                sys.executable, "scripts/inference.py",
                "-d", str(base_dir),
                "--checkpoint", "final.pt",
                "-s", "test",
                "--output-dir", str(threshold_folder.absolute()),
                "--flat-output",
                *renderer_mode_args,
                *threshold_args,
            ], env=current_env, capture_output=True, text=True)
            if render_result.returncode != 0:
                print(f"    Render failed voor {npz_path.name} at threshold {threshold_csv_value}")
                if render_result.stderr:
                    print(render_result.stderr)
                continue

            fps_list = []
            for i in range(2):
                print(f"    Benchmark run {i+1}/3 voor threshold {threshold_csv_value}...")
                benchmark_result = subprocess.run([
                    sys.executable, "scripts/inference.py",
                    "-d", str(base_dir),
                    "--checkpoint", "final.pt",
                    "-b",
                    *renderer_mode_args,
                    *threshold_args,
                ], env=current_env)
                if benchmark_result.returncode != 0:
                    print(f"    Benchmark run {i+1}/3 failed voor {npz_path.name} at threshold {threshold_csv_value}")
                    continue

                fps = read_latest_average_fps(base_dir)
                if fps is not None:
                    fps_list.append(fps)

            avg_fps = np.mean(fps_list) if fps_list else 0

            row = {
                "dataset": dataset_name,
                "type": test_type.upper(),
                "model_file": npz_path.name,
                "threshold": threshold_csv_value,
                "lobes": lobes,
                "sites": sites,
                "avg_fps": f"{avg_fps:.2f}",
                "render_output_dir": str(threshold_folder.resolve()),
                "base_model_dir": str(base_dir.resolve()),
            }
            append_csv_row(output_csv, fieldnames, row)
            processed_keys.add(row_key)
            print(f"    Opgeslagen in CSV voor threshold {threshold_csv_value}: {output_csv}")

    print(f"\nKlaar! Resultaten worden per run opgeslagen in {output_csv}")

if __name__ == "__main__":
    main()