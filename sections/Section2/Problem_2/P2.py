import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define surfaces
def surface1(x, y):
    return 2*x**2 + 2*y**2  # Bottom surface (parabolic)
def surface2(x, y):
    return 2*np.exp(-x**2 - y**2)  # Top surface (exponential)

# Generate grid points
n = 15
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, n)
x_grid, y_grid = np.meshgrid(x, y)
x_flat = x_grid.ravel()
y_flat = y_grid.ravel()

# Calculate z-values
z_bottom = surface1(x_flat, y_flat)
z_top = surface2(x_flat, y_flat)

# Combine points with unique vertex IDs
bottom_points = np.column_stack((x_flat, y_flat, z_bottom))
top_points = np.column_stack((x_flat, y_flat, z_top))
all_points = np.vstack((bottom_points, top_points))

# Delaunay triangulation for each surface
tri_bottom = Delaunay(bottom_points[:, :2])
tri_top = Delaunay(top_points[:, :2])

# Adjust top triangle indices (offset by bottom point count)
N = len(bottom_points)
triangles_bottom = tri_bottom.simplices
triangles_top = tri_top.simplices + N

# Identify boundary points using ConvexHull
hull = ConvexHull(bottom_points[:, :2])
boundary_indices = hull.vertices

# Create side triangles (quadrilaterals split into two triangles)
side_triangles = []
for i in range(len(boundary_indices)):
    current = boundary_indices[i]
    next_idx = boundary_indices[(i+1) % len(boundary_indices)]
    
    # Bottom-to-top connections
    side_triangles.append([current, next_idx, current + N])
    side_triangles.append([next_idx, current + N, next_idx + N])

# Combine all triangles
all_triangles = np.vstack([
    triangles_bottom,
    triangles_top,
    np.array(side_triangles)
])

# Visualize
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot with consistent parameters
ax.plot_trisurf(
    all_points[:, 0], 
    all_points[:, 1], 
    all_points[:, 2], 
    triangles=all_triangles,
    cmap='viridis',
    edgecolor='k',  # Show edges for clarity
    alpha=0.8,
    linewidth=0.3
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=25, azim=-45)  # Adjust viewing angle
plt.tight_layout()
plt.savefig("3d_surface_plot.png", dpi=300)
plt.show()