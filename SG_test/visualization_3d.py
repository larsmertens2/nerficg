import numpy as np
import open3d as o3d
import os
import sys

def visualize_gaussian_atlas(npz_path):
    if not os.path.exists(npz_path):
        print(f"[Fout] Kan het bestand '{npz_path}' niet vinden.")
        print("Zorg dat je het juiste pad naar 'gaussians_atlas.npz' opgeeft.")
        return

    print(f"[-] Laden van {npz_path}...")
    # Laad de gecomprimeerde numpy array
    data = np.load(npz_path)
    
    if 'means' not in data.files:
        print("[Fout] De key 'means' werd niet gevonden in het .npz bestand.")
        print(f"Beschikbare keys: {data.files}")
        return

    # Haal de xyz posities van de Gaussians op
    points = data['means']
    num_points = points.shape[0]
    print(f"[+] Succesvol {num_points:,} Gaussians geladen.")

    print("[-] 3D puntenwolk opbouwen...")
    # Maak een Open3D PointCloud object aan
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Omdat er geen kleur in de atlas zit, genereren we een kleurverloop op basis van de Z-as (hoogte)
    # Dit geeft een veel beter 3D-dieptebeeld dan een egale kleur.
    z_coords = points[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # Voorkom delen door nul als alle punten op dezelfde hoogte liggen
    if z_max - z_min > 0:
        normalized_z = (z_coords - z_min) / (z_max - z_min)
    else:
        normalized_z = np.zeros(num_points)

    # Maak een mooie blauw-naar-rode gradient
    colors = np.zeros((num_points, 3))
    colors[:, 0] = normalized_z          # Rood neemt toe met de hoogte
    colors[:, 2] = 1.0 - normalized_z    # Blauw neemt af met de hoogte
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("[+] Visualisatie starten. Gebruik je muis om rond te draaien en te zoomen.")
    print("    - Linkermuisknop: Draaien")
    # Open de interactieve 3D viewer
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="NVGSViewer - 3D Gaussian Atlas",
        width=1024,
        height=768,
        left=50,
        top=50
    )

if __name__ == "__main__":
    # Pas dit pad aan naar de locatie van jouw geëxporteerde bestand
    target_file = "output/FasterGS/fused_export/view_contributions/gaussians_atlas.npz"
    
    visualize_gaussian_atlas(target_file)