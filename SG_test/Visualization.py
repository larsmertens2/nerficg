import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


def _load_camera_positions(cam_data):
    if "camera_positions" in cam_data:
        return cam_data["camera_positions"]
    if "camera_c2w" in cam_data:
        return cam_data["camera_c2w"][:, :3, 3]
    raise KeyError("camera_data.npz mist camera_positions en camera_c2w")


def _to_lon_lat(vectors):
    vectors = np.asarray(vectors)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = vectors / norms
    lon = np.degrees(np.arctan2(normalized[..., 1], normalized[..., 0]))
    lat = np.degrees(np.arcsin(np.clip(normalized[..., 2], -1.0, 1.0)))
    return lon, lat


def _lon_lat_grid(res_lat=500, res_lon=1000):
    lat_range = np.linspace(-np.pi / 2, np.pi / 2, res_lat)
    lon_range = np.linspace(-np.pi, np.pi, res_lon)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)
    return torch.tensor(np.stack([x, y, z], axis=-1).reshape(-1, 3), dtype=torch.float32)


def _predict_sg_grid(model_data, gaussian_index, omega, res_lat, res_lon, threshold_model):
    axis = torch.tensor(model_data["axis"][gaussian_index], dtype=torch.float32)
    sharpness = torch.tensor(model_data["sharpness"][gaussian_index], dtype=torch.float32)
    amplitude = torch.tensor(model_data["amplitude"][gaussian_index], dtype=torch.float32)
    bias = float(model_data["bias"]) if "bias" in model_data else 4.6

    with torch.no_grad():
        axis = torch.nn.functional.normalize(axis, dim=-1)
        dot = torch.matmul(omega, axis.T)
        logits = torch.sum(amplitude * torch.exp(sharpness * (dot - 1.0)), dim=-1)
        preds = torch.sigmoid(logits - bias)

    grid_binary = (preds > threshold_model).float().reshape(res_lat, res_lon).numpy()
    overlay_points = _to_lon_lat(axis.detach().cpu().numpy())
    overlay_sizes = 90.0 + 160.0 * (
        model_data["amplitude"][gaussian_index] / (np.max(model_data["amplitude"][gaussian_index]) + 1e-8)
    )
    return grid_binary, overlay_points, overlay_sizes, "SG Lobes", f"Spherical Gaussian Prediction (Model Value > {threshold_model})"


def _predict_sv_grid(model_data, gaussian_index, omega, res_lat, res_lon, threshold_model):
    sites = torch.tensor(model_data["sites"][gaussian_index], dtype=torch.float32)
    values = torch.tensor(model_data["values"][gaussian_index], dtype=torch.float32)

    with torch.no_grad():
        logits = torch.matmul(omega, sites.T)
        weights = torch.nn.functional.softmax(logits, dim=-1)
        preds = torch.sum(weights * values, dim=-1)

    grid_binary = (preds > threshold_model).float().reshape(res_lat, res_lon).numpy()
    overlay_points = _to_lon_lat(sites.detach().cpu().numpy())
    return grid_binary, overlay_points, 200, "Voronoi Sites", f"Spherical Voronoi Prediction (Model Value > {threshold_model})"


def _load_optional_npz(path):
    if path is None:
        return None
    resolved_path = _resolve_path(path)
    if not resolved_path.exists():
        print(f"Waarschuwing: Bestand niet gevonden, overslaan: {resolved_path}")
        return None
    try:
        return np.load(resolved_path, allow_pickle=True)
    except OSError as exc:
        print(f"Waarschuwing: Bestand kon niet worden geladen, overslaan: {resolved_path} ({exc})")
        return None


def _npz_has_keys(npz_data, required_keys):
    return npz_data is not None and all(key in npz_data for key in required_keys)


def compare_visibility_binary_grids(
    file_cam,
    file_gauss,
    gaussian_index,
    file_sg=None,
    file_sv=None,
    threshold_cam=0.1,
    threshold_sg=0.5,
    threshold_sv=0.5,
):
    try:
        cam_data = np.load(_resolve_path(file_cam), allow_pickle=True)
        gauss_data = np.load(_resolve_path(file_gauss))
    except FileNotFoundError as exc:
        print(f"Fout: Kon een of meer bestanden niet vinden: {exc}")
        return

    sg_data = _load_optional_npz(file_sg)
    sv_data = _load_optional_npz(file_sv)

    target_pos = gauss_data["means"][gaussian_index]
    cam_positions = _load_camera_positions(cam_data)
    contributions = cam_data["contributions"][:, gaussian_index]

    directions_cam = target_pos - cam_positions
    dist = np.linalg.norm(directions_cam, axis=1)
    valid_mask = dist > 0
    norm_dir = directions_cam[valid_mask] / dist[valid_mask, np.newaxis]
    cam_lon, cam_lat = _to_lon_lat(norm_dir)
    cam_binary = (contributions[valid_mask] > threshold_cam).astype(float)

    res_lat, res_lon = 500, 1000
    grid_x, grid_y = np.mgrid[-180:180:1000j, -90:90:500j]
    grid_cam_binary = griddata((cam_lon, cam_lat), cam_binary, (grid_x, grid_y), method="nearest")
    grid_cam_binary = np.nan_to_num(grid_cam_binary, nan=0.0)

    omega = _lon_lat_grid(res_lat=res_lat, res_lon=res_lon)
    model_panels = []
    if _npz_has_keys(sg_data, ("axis", "sharpness", "amplitude")):
        model_panels.append(
            _predict_sg_grid(sg_data, gaussian_index, omega, res_lat, res_lon, threshold_sg)
        )
    elif sg_data is not None:
        print("Waarschuwing: SG-bestand mist axis/sharpness/amplitude; SG-paneel wordt overgeslagen.")

    if _npz_has_keys(sv_data, ("sites", "values")):
        model_panels.append(
            _predict_sv_grid(sv_data, gaussian_index, omega, res_lat, res_lon, threshold_sv)
        )
    elif sv_data is not None:
        print("Waarschuwing: SV-bestand mist sites/values; SV-paneel wordt overgeslagen.")

    if not model_panels:
        print("Fout: SG- en SV-bestanden ontbreken allebei; er is niets om te visualiseren.")
        return

    fig, axes = plt.subplots(1 + len(model_panels), 1, figsize=(16, 7 * (1 + len(model_panels))), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    red_green_map = ListedColormap(["#e74c3c", "#2ecc71"])

    ax1 = axes[0]
    ax1.imshow(
        grid_cam_binary.T,
        extent=[-180, 180, -90, 90],
        origin="lower",
        cmap=red_green_map,
        aspect="auto",
        interpolation="none",
    )
    ax1.set_title(
        f"Ground Truth Binary Heatmap (Camera Contribution > {threshold_cam})\n(Interpolated: Nearest Neighbor)",
        fontsize=14,
    )

    for ax, (grid_binary, overlay, sizes, label, title) in zip(axes[1:], model_panels):
        ax.imshow(
            grid_binary,
            extent=[-180, 180, -90, 90],
            origin="lower",
            cmap=red_green_map,
            aspect="auto",
            interpolation="none",
        )
        x_vals, y_vals = overlay
        ax.scatter(x_vals, y_vals, c="white", edgecolors="black", s=sizes, marker="*", label=label)
        ax.set_title(title + "\n(Hard Boundaries)", fontsize=14)
        ax.legend()

    for ax in axes:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_ylabel("Latitude (Degrees)", fontsize=12)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.axhline(0, color="black", lw=1, alpha=0.5)
        ax.axvline(0, color="black", lw=1, alpha=0.5)
        ax.set_xticks(np.arange(-180, 181, 45))
        ax.set_yticks(np.arange(-90, 91, 30))

    axes[-1].set_xlabel("Longitude (Degrees)", fontsize=12)
    plt.tight_layout()
    plt.show()


compare_visibility_binary_grids(
    file_cam=BASE_DIR / "npz_files/camera_data.npz",
    file_gauss=BASE_DIR / "npz_files/gaussians_atlas.npz",
    file_sg=BASE_DIR / "npz_files/SV/threshold_0_1/sites_10/sv_s10_t0_1_temp10.npz",
    file_sv=BASE_DIR / "npz_files/SV/threshold_0_01/sites_10/sv_s10_t0_01_temp10.npz",
    gaussian_index=160,
    threshold_cam=0.01,
    threshold_sg=0.5,
    threshold_sv=0.5,
)