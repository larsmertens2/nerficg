import numpy as np
import open3d as o3d

def visualize_visibility_normalized(file_path_camera, file_path_gaussian, gaussian_index):
    # 1. Data laden
    cam_data = np.load(file_path_camera, allow_pickle=True)
    gauss_data = np.load(file_path_gaussian)

    target_pos = gauss_data['means'][gaussian_index]
    cam_positions = cam_data['camera_positions']
    contributions = cam_data['contributions']
    
    points = [target_pos] # Index 0
    lines = []
    colors = []

    # 2. Lijnen berekenen met normalisatie (lengte 1)
    for i in range(len(cam_positions)):
        # Bereken de vector van de Gaussian naar de camera
        direction = cam_positions[i] - target_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Normaliseer de vector (lengte 1) en tel op bij target_pos
            normalized_endpoint = target_pos + (direction / distance)
            
            points.append(normalized_endpoint)
            point_idx = len(points) - 1 # De index van het zojuist toegevoegde punt
            
            lines.append([0, point_idx])
            
            # Zichtbaarheid check
            is_visible = contributions[i, gaussian_index] > 0
            colors.append([0, 1, 0] if is_visible else [1, 0, 0])

    # 3. Open3D objecten maken
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

    # De Gaussian zelf als een punt (geel)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector([target_pos])
    point_cloud.colors = o3d.utility.Vector3dVector([[1, 1, 0]])

    # 4. Visualisatie
    print(f"Visualizing Gaussian Index: {gaussian_index}")
    print(f"All lines normalized to length 1.0")
    o3d.visualization.draw_geometries([line_set, point_cloud])

# Uitvoeren
visualize_visibility_normalized(
    file_path_camera="npz_files/camera_data.npz", 
    file_path_gaussian="npz_files/gaussians_atlas.npz", 
    gaussian_index=100
)