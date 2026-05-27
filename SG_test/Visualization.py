import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from scipy.interpolate import griddata
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

def _predict_sg_grid(model_data, gaussian_index, omega, res_lat, res_lon, user_threshold=0.1):
    axis = torch.tensor(model_data["axis"][gaussian_index], dtype=torch.float32)
    sharpness = torch.tensor(model_data["sharpness"][gaussian_index], dtype=torch.float32)
    amplitude = torch.tensor(model_data["amplitude"][gaussian_index], dtype=torch.float32)

    with torch.no_grad():
        dot = torch.matmul(omega, axis.T)
        exponent = sharpness * (dot - 1.0)
        raw_sum = torch.sum(amplitude * torch.exp(exponent), dim=-1)
        
        probabilities = torch.sigmoid(raw_sum)
        grid_visibility = (probabilities > user_threshold).float().reshape(res_lat, res_lon).cpu().numpy()

    overlay_points = _to_lon_lat(axis.detach().cpu().numpy())
    
    amp_np = model_data["amplitude"][gaussian_index]
    sharp_np = model_data["sharpness"][gaussian_index]
    
    return grid_visibility, overlay_points, sharp_np, amp_np, "SG Lobes"

def _predict_sv_grid(model_data, gaussian_index, omega, res_lat, res_lon, user_threshold=0.1):
    sites = torch.tensor(model_data["sites"][gaussian_index], dtype=torch.float32)
    c_k = torch.tensor(model_data["values"][gaussian_index], dtype=torch.float32)

    with torch.no_grad():
        logits = torch.matmul(omega, sites.T)
        weights = torch.nn.functional.softmax(logits, dim=-1)
        visibility_prob = torch.sum(weights * c_k, dim=-1)
        grid_visibility = (visibility_prob > user_threshold).float().reshape(res_lat, res_lon).numpy()
    
    overlay_points = _to_lon_lat(sites.detach().cpu().numpy())
    return grid_visibility, overlay_points, "Voronoi Sites"

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
    threshold_sg=0.4,
    threshold_sv=0.4
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
    cam_binary = (contributions[valid_mask] > 0.0).astype(float)

    res_lat, res_lon = 500, 1000
    
    grid_lon, grid_lat = np.meshgrid(np.linspace(-180, 180, res_lon), np.linspace(-90, 90, res_lat))
    grid_cam_binary = griddata((cam_lon, cam_lat), cam_binary, (grid_lon, grid_lat), method="nearest")
    grid_cam_binary = np.nan_to_num(grid_cam_binary, nan=0.0)

    omega = _lon_lat_grid(res_lat=res_lat, res_lon=res_lon)
    
    has_sv = False
    has_sg = False

    if _npz_has_keys(sv_data, ("sites", "values")):
        sv_grid, sv_overlay, sv_label = _predict_sv_grid(sv_data, gaussian_index, omega, res_lat, res_lon, threshold_sv)
        has_sv = True
    elif sv_data is not None:
        print("Waarschuwing: SV-bestand mist sites/values; SV-paneel wordt overgeslagen.")

    if _npz_has_keys(sg_data, ("axis", "sharpness", "amplitude")):
        sg_grid, sg_overlay, sg_sharpness, sg_amplitude, sg_label = _predict_sg_grid(sg_data, gaussian_index, omega, res_lat, res_lon, threshold_sg)
        has_sg = True
    elif sg_data is not None:
        print("Waarschuwing: SG-bestand mist axis/sharpness/amplitude; SG-paneel wordt overgeslagen.")

    num_panels = 1 + int(has_sv) + int(has_sg)
    if num_panels == 1:
        print("Fout: SG- en SV-bestanden ontbreken allebei; er is niets om te visualiseren.")
        return

    fig, axes = plt.subplots(num_panels, 1, figsize=(14, 5 * num_panels))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    visibility_cmap = mcolors.ListedColormap(["#d63031", "#2ed573"])
    current_ax_idx = 0

    # ==========================================
    # Panel 1: Ground Truth
    # ==========================================
    ax1 = axes[current_ax_idx]
    ax1.imshow(
        grid_cam_binary,
        extent=[-180, 180, -90, 90],
        origin="lower",
        cmap=visibility_cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="none",
    )
    ax1.set_title(f"GT (Original Camera Ground Truth, Gaussian Index: {gaussian_index})", fontsize=20, pad=10)
    current_ax_idx += 1

    # ==========================================
    # Panel 2: Spherical Voronoi
    # ==========================================
    if has_sv:
        ax = axes[current_ax_idx]
        ax.imshow(
            sv_grid,
            extent=[-180, 180, -90, 90],
            origin="lower",
            cmap=visibility_cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="none",
        )
        
        ax.set_title(f"SV Prediction (Threshold Map @ > {threshold_sv})", fontsize=20, pad=10)
        current_ax_idx += 1

    # ==========================================
    # Panel 3: Spherical Gaussians
    # ==========================================
    if has_sg:
        ax = axes[current_ax_idx]
        ax.imshow(
            sg_grid,
            extent=[-180, 180, -90, 90],
            origin="lower",
            cmap=visibility_cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="none",
        )
        
        ax.set_title(f"SG Prediction (Threshold Map @ > {threshold_sg})", fontsize=20, pad=10)

    # Apply global axis formatting rules
    for ax in axes:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (Degrees)", fontsize=18)
        ax.set_ylabel("Latitude (Degrees)", fontsize=18)
        ax.set_xticks(np.arange(-180, 181, 45))
        ax.set_yticks(np.arange(-90, 91, 30))

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent.parent / "csv_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"visibility_comparison_idx{gaussian_index}_sg{threshold_sg}_sv{threshold_sv}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    compare_visibility_binary_grids(
        file_cam=BASE_DIR / "npz_files/lego/camera_data.npz",
        file_gauss=BASE_DIR / "npz_files/lego/gaussians_atlas.npz",
        file_sg=BASE_DIR / r"npz_files\lego\SG\threshold_0_0\lobes_9\sg_l9_t0_0.npz",
        file_sv=BASE_DIR / r"npz_files\lego\SV\threshold_0\sites_9\sv_s9_t0_temp5.npz",
        gaussian_index=800,
        threshold_sg=0.7,
        threshold_sv=0.7
    )
