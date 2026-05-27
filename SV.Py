import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# --- 1. Grid Setup ---
res = 1000  # High resolution for ultra-smooth curve extraction
x = np.linspace(0, 10, res)
y = np.linspace(0, 10, res)
X, Y = np.meshgrid(x, y)

# --- 2. Define the Shared Anchor Center ---
cx, cy = 5.0, 5.0 

# --- 3. Define Site Vectors (Angles and Extents) ---
sites = [
    {'angle_deg': 160, 'extent': 1.6, 'color': '#ff7f0e'},  # Orange
    {'angle_deg': 195, 'extent': 1.4, 'color': '#5b6ee1'},  # Blue
    {'angle_deg': 45,  'extent': 1.3, 'color': '#b39ddb'},  # Purple
    {'angle_deg': -10, 'extent': 0.5, 'color': '#e31a1c'},  # Red
    {'angle_deg': -85, 'extent': 2.1, 'color': '#2ca02c'}   # Green
]

# Calculate relative grid coordinates from the center
dX = X - cx
dY = Y - cy
grid_angles = np.arctan2(dY, dX)     # Grid angles in radians [-pi, pi]
grid_radius = np.sqrt(dX**2 + dY**2) # Radial distance from center

# --- 4. Math: Continuous Peak-Anchored Angular Interpolation ---
site_angles_rad = np.array([np.deg2rad(s['angle_deg']) for s in sites])
site_extents = np.array([s['extent'] for s in sites])

# Calculate angular distance to every site vector, accounting for 2*pi wrapping
angle_diffs = np.abs(grid_angles[..., np.newaxis] - site_angles_rad)
angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)

# Global interpolation inverse angular distance kernel
power = 6.0  
weights = 1.0 / (angle_diffs + 0.05)**power
sum_weights = np.sum(weights, axis=-1, keepdims=True)
sum_weights[sum_weights == 0] = 1.0 
normalized_weights = weights / sum_weights

# Compute the continuous smooth radius limit across all angles
smooth_boundary_radius = np.sum(normalized_weights * site_extents, axis=-1)

# Generate a continuous algebraic field (1.0 at center, 0.0 exactly on the envelope boundary)
Z_field = np.maximum(0, 1.0 - (grid_radius / smooth_boundary_radius))

# --- 5. Extract the Smooth Envelope Contour ---
contours = measure.find_contours(Z_field, 0.001)

# --- 6. Plotting (Enforcing Pure White Background & Compact Viewport) ---
fig, ax = plt.subplots(figsize=(6, 6), dpi=120, facecolor='white')
ax.set_facecolor('white')

# Keep viewport bound cleanly around our compact shapes
ax.set_xlim(2.4, 7.6)
ax.set_ylim(2.4, 7.6)
ax.set_aspect('equal')
ax.axis('off')

# NOTE: Gray line section using ndimage.laplace has been completely deleted.

# Plot the fluidly smooth black envelope contour line
for contour in contours:
    contour_x = x[0] + contour[:, 1] * (x[-1] - x[0]) / (res - 1)
    contour_y = y[0] + contour[:, 0] * (y[-1] - y[0]) / (res - 1)
    ax.plot(contour_x, contour_y, color='#1c1c1c', linewidth=2.5, solid_capstyle='round', zorder=2)

# Plot the core Site Vectors radiating from the center as arrows
for s in sites:
    rad = np.deg2rad(s['angle_deg'])
    vx = s['extent'] * np.cos(rad)
    vy = s['extent'] * np.sin(rad)
    
    # Draw directional arrow using annotate for clean, reliable scaling
    ax.annotate(
        '', 
        xy=(cx + vx, cy + vy), 
        xytext=(cx, cy),
        arrowprops=dict(
            arrowstyle="-|>", 
            color=s['color'], 
            lw=2.0, 
            mutation_scale=12,  # Controls the head size of the arrow
            zorder=3
        )
    )
    
    # Terminal point right on the crest of the envelope line
    ax.scatter([cx + vx], [cy + vy], color=s['color'], s=15, zorder=4)

# Plot the root origin center point
ax.scatter([cx], [cy], color='#1c1c1c', s=20, zorder=5)

# Save with crisp transparency-free white backgrounds
plt.tight_layout()
plt.savefig('smooth_voronoi_arrows.png', facecolor='white', edgecolor='none', bbox_inches='tight', pad_inches=0.05)
plt.show()