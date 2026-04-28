import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

def compare_visibility_binary_grids(file_cam, file_gauss, file_voronoi, gaussian_index, threshold_cam=0.1, threshold_voronoi=0.5):
    # --- 1. Data Laden ---
    try:
        cam_data = np.load(file_cam, allow_pickle=True)
        gauss_data = np.load(file_gauss)
        voronoi_data = np.load(file_voronoi)
    except FileNotFoundError:
        print("Fout: Kon een of meer bestanden niet vinden.")
        return

    target_pos = gauss_data['means'][gaussian_index]
    cam_positions = cam_data['camera_positions']
    contributions = cam_data['contributions'][:, gaussian_index]
    
    sites = torch.tensor(voronoi_data['sites'][gaussian_index])
    values = torch.tensor(voronoi_data['values'][gaussian_index])

    # --- 2. Camera Data verwerken naar Binair Grid (Ground Truth) ---
    
    # BELANGRIJK: Gebruik DEZELFDE vector als in de training: Camera -> Gaussian
    directions_cam = target_pos - cam_positions 
    dist = np.linalg.norm(directions_cam, axis=1)
    # Voorkom delen door nul
    valid_mask = dist > 0
    norm_dir = directions_cam[valid_mask] / dist[valid_mask, np.newaxis]
    
    # Sferische coördinaten in graden berekenen
    cam_lon = np.degrees(np.arctan2(norm_dir[:, 1], norm_dir[:, 0]))
    cam_lat = np.degrees(np.arcsin(norm_dir[:, 2]))
    
    # Pas de threshold toe: 1.0 (Groen/Zichtbaar) of 0.0 (Rood/Geculd)
    cam_binair = (contributions[valid_mask] > threshold_cam).astype(float)

    # Definieer het grid voor de visualisatie (resolutie kan verhoogd worden)
    res_lat, res_lon = 500, 1000
    grid_x, grid_y = np.mgrid[-180:180:1000j, -90:90:500j]

    # Interpoleer de losse punten naar het grid met 'nearest' methode voor harde grenzen
    grid_cam_binair = griddata((cam_lon, cam_lat), cam_binair, (grid_x, grid_y), method='nearest')
    
    # Gaten opvullen (bijv. bij de polen waar geen camera's zijn) - we vullen ze met rood (geculd)
    grid_cam_binair = np.nan_to_num(grid_cam_binair, nan=0.0)

    # --- 3. Voronoi Data verwerken naar Binair Grid (Model Prediction) ---
    
    # Maak sferische grid-vectoren (omega) die overeenkomen met het visualisatie-grid
    lat_range = np.linspace(-np.pi/2, np.pi/2, res_lat)
    lon_range = np.linspace(-np.pi, np.pi, res_lon)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

    # omega is de richting van cam naar gaussian
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)
    
    omega = torch.tensor(np.stack([x, y, z], axis=-1).reshape(-1, 3), dtype=torch.float32)
    
    with torch.no_grad():
        logits = torch.matmul(omega, sites.T)
        weights = torch.nn.functional.softmax(logits, dim=-1)
        preds = torch.sum(weights * values, dim=-1)
    
    # Pas Voronoi threshold toe
    grid_voronoi_binair = (preds > threshold_voronoi).float().reshape(res_lat, res_lon).numpy()

    # --- 4. Plotten ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True, sharey=True)
    
    # Definieer de rood-groene colormap
    red_green_map = ListedColormap(['#e74c3c', '#2ecc71']) # [Rood, Groen]

    # Subplot 1: Interpolated Ground Truth
    im1 = ax1.imshow(grid_cam_binair.T, extent=[-180, 180, -90, 90], origin='lower', 
                     cmap=red_green_map, aspect='auto', interpolation='none')
    ax1.set_title(f"Ground Truth Binaire Heatmap (Cameras Contribution > {threshold_cam})\n(Interpolated: Nearest Neighbor)", fontsize=14)
    
    # Optioneel: Plot de originele camerapunten er héél klein overheen ter controle
    # ax1.scatter(cam_lon, cam_lat, c='black', s=1, alpha=0.1)

    # Subplot 2: Voronoi Prediction
    im2 = ax2.imshow(grid_voronoi_binair, extent=[-180, 180, -90, 90], origin='lower', 
                     cmap=red_green_map, aspect='auto', interpolation='none')
    
    # Sites terug naar lon/lat voor visualisatie
    site_lon = np.degrees(np.arctan2(sites[:, 1], sites[:, 0]))
    site_lat = np.degrees(np.arcsin(sites[:, 2] / torch.norm(sites, dim=-1)))
    ax2.scatter(site_lon, site_lat, c='white', edgecolors='black', s=200, marker='*', label='Voronoi Sites')
    
    ax2.set_title(f"Spherical Voronoi Prediction (Model Value > {threshold_voronoi})\n(Hard Boundaries)", fontsize=14)
    ax2.legend()

    # Algemene opmaak
    for ax in [ax1, ax2]:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_ylabel("Latitude (Degrees)", fontsize=12)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.axhline(0, color='black', lw=1, alpha=0.5) # Evenaar
        ax.axvline(0, color='black', lw=1, alpha=0.5) # Nulmeridiaan
        ax.set_xticks(np.arange(-180, 181, 45))
        ax.set_yticks(np.arange(-90, 91, 30))

    ax2.set_xlabel("Longitude (Degrees)", fontsize=12)
    plt.tight_layout()
    plt.show()

# Run de vergelijking met binaire grids
compare_visibility_binary_grids(
    file_cam="npz_files/camera_data.npz", 
    file_gauss="npz_files/gaussians_atlas.npz", 
    file_voronoi="npz_files/sv_s8_t0_1_temp5.npz",
    gaussian_index=250, 
    threshold_cam=0.1,         # Drempel voor grondwaarheid
    threshold_voronoi=0.5     # Drempel voor modelvoorspelling
)