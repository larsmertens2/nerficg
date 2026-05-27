import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.ndimage as ndimage
from skimage import measure

# --- 1. Grid Setup for Crisp Rendering ---
res = 800  # High resolution for ultra-smooth lines
x = np.linspace(0, 10, res)
y = np.linspace(0, 10, res)
X, Y = np.meshgrid(x, y)
Z_field = np.zeros_like(X, dtype=float)

# --- 2. Define the Shared Anchor Center ---
cx, cy = 5.0, 5.0 

# --- 3. Define Ellipse Configurations (Scaled Down / Smaller Ovals) ---
ellipses_data = [
    {'a': 0.59, 'b': 0.24, 'angle': 160, 'color': '#ff7f0e'},  # Orange
    {'a': 0.66, 'b': 0.18, 'angle': 195, 'color': '#5b6ee1'},  # Blue
    {'a': 0.68, 'b': 0.35, 'angle': 45,  'color': '#b39ddb'},  # Purple
    {'a': 0.75, 'b': 0.16, 'angle': -10, 'color': '#e31a1c'},  # Red
    {'a': 0.79, 'b': 0.18, 'angle': -85, 'color': '#2ca02c'}   # Green
]

computed_ellipses = []

# --- 4. Math: Generate a Smooth Field Matrix ---
for e in ellipses_data:
    rad = np.deg2rad(e['angle'])
    
    # Anchor the base of the lobe exactly at the shared center (cx, cy)
    ellipse_center_x = cx + e['a'] * np.cos(rad)
    ellipse_center_y = cy + e['a'] * np.sin(rad)
    
    computed_ellipses.append({
        'center': (ellipse_center_x, ellipse_center_y),
        'width': e['a'] * 2,
        'height': e['b'] * 2,
        'angle': e['angle'],
        'color': e['color']
    })
    
    # Transform grid to ellipse local space
    x_rot = (X - ellipse_center_x) * np.cos(rad) + (Y - ellipse_center_y) * np.sin(rad)
    y_rot = -(X - ellipse_center_x) * np.sin(rad) + (Y - ellipse_center_y) * np.cos(rad)
    
    # Calculate continuous algebraic distance field
    dist_field = np.maximum(0, 1.0 - ((x_rot**2 / e['a']**2) + (y_rot**2 / e['b']**2)))
    Z_field = np.maximum(Z_field, dist_field)

# --- 5. Tighter Blur and Threshold for a Closer Envelope ---
# Lower sigma lets the line dip deeper into the valleys between ovals
blurred_field = ndimage.gaussian_filter(Z_field, sigma=5)

# A slightly higher contour level pulls the line closer to the ovals
contours = measure.find_contours(blurred_field, 0.001)

# --- 6. Plotting (Compact View, Pure White Background) ---
fig, ax = plt.subplots(figsize=(6, 6), dpi=120, facecolor='white')
ax.set_facecolor('white')

# Set tight boundaries around our smaller ovals
ax.set_xlim(2.0, 8.0)
ax.set_ylim(2.0, 8.0)
ax.set_aspect('equal')
ax.axis('off')

# Plot the tightly tracking black envelope contour line
for contour in contours:
    contour_x = x[0] + contour[:, 1] * (x[-1] - x[0]) / (res - 1)
    contour_y = y[0] + contour[:, 0] * (y[-1] - y[0]) / (res - 1)
    ax.plot(contour_x, contour_y, color='#1c1c1c', linewidth=2.2, solid_capstyle='round', zorder=1)

# Plot the individual colored lobes cleanly on top
for ce in computed_ellipses:
    patch = Ellipse(
        xy=ce['center'],
        width=ce['width'],
        height=ce['height'],
        angle=ce['angle'],
        edgecolor=ce['color'],
        facecolor='none',
        linewidth=1.8,
        zorder=2
    )
    ax.add_patch(patch)

# Plot the root origin center point
ax.scatter([cx], [cy], color='#1c1c1c', s=18, zorder=3)

# Save and layout configuration
plt.tight_layout()
plt.savefig('tighter_spherical_envelope.png', facecolor='white', edgecolor='none', bbox_inches='tight', pad_inches=0.05)
plt.show()