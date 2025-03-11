import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import fsolve

# a) 

# --- Define the surfaces ---
def surface1(x, y):
    """Bottom surface (parabolic)."""
    return 2 * x**2 + 2 * y**2

def surface2(x, y):
    """Top surface (exponential)."""
    return 2 * np.exp(-x**2 - y**2)

# --- Find the boundary radius R where the two surfaces meet ---
def find_boundary_radius():
    # We need to solve: r^2 = exp(-r^2)
    # Define function f(r) = r^2 - exp(-r^2)
    func = lambda r: r**2 - np.exp(-r**2)
    # Use an initial guess (e.g., 0.7)
    r_boundary = fsolve(func, 0.7)[0]
    return r_boundary

# --- Generate grid points inside a disk of radius R ---
def generate_grid_points(R, num_points=50):
    # Create a grid over the square [-R, R] x [-R, R]
    x = np.linspace(-R, R, num_points)
    y = np.linspace(-R, R, num_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    # Keep only points within the circle x^2 + y^2 <= R^2
    mask = xv**2 + yv**2 <= R**2
    xv = xv[mask]
    yv = yv[mask]
    points = np.vstack([xv, yv]).T
    return points

# --- Generate the point clouds for both surfaces ---
def generate_surface_point_cloud(points):
    # For a set of (x,y) points, compute z for top and bottom surfaces
    z_top = surface2(points[:, 0], points[:, 1])
    z_bottom = surface1(points[:, 0], points[:, 1])
    top_points = np.hstack([points, z_top[:, None]])
    bottom_points = np.hstack([points, z_bottom[:, None]])
    return top_points, bottom_points

# --- Simple helper: check if a point is (approximately) on the boundary ---
def is_boundary_point(point, R, tol=1e-3):
    r = np.sqrt(point[0]**2 + point[1]**2)
    return np.isclose(r, R, atol=tol)

#

# --- Main function to build and visualize the closed surface ---
def main():
    # 1. Determine the boundary radius R
    R = find_boundary_radius()
    print("Boundary radius R =", R)
    
    # 2. Generate a grid of (x,y) points in the disk of radius R
    grid_points = generate_grid_points(R, num_points=50)
    
    # 3. Compute the 3D points for the top and bottom surfaces
    top_points, bottom_points = generate_surface_point_cloud(grid_points)
    
    # 4. Compute Delaunay triangulation for the grid (x,y) -- same for both surfaces.
    tri_top = Delaunay(grid_points)
    tri_bottom = Delaunay(grid_points)
    
    # 5. Identify boundary vertices (they lie on the circle r ≈ R)
    boundary_indices = []
    for i, pt in enumerate(grid_points):
        if is_boundary_point(pt, R, tol=1e-3):
            boundary_indices.append(i)
    boundary_indices = np.array(boundary_indices)
    print("Found", len(boundary_indices), "boundary points.")
    
    # 6. Build a mapping for the bottom surface vertices:
    #    If a bottom vertex is on the boundary, map it to the corresponding top vertex index.
    #    Otherwise, assign a new (offset) index.
    num_top = len(top_points)
    bottom_mapping = {}
    bottom_interior_points = []
    for i, pt in enumerate(bottom_points):
        if is_boundary_point(grid_points[i], R, tol=1e-3):
            bottom_mapping[i] = i  # use the top patch vertex (they share same (x,y))
        else:
            bottom_mapping[i] = num_top + len(bottom_interior_points)
            bottom_interior_points.append(pt)
    bottom_interior_points = np.array(bottom_interior_points)
    
    # 7. Combine vertices:
    #    - Top surface: all vertices (from grid_points with z_top)
    #    - Bottom surface: only the interior vertices (non-boundary) are added separately
    combined_vertices = np.vstack([top_points, bottom_interior_points])
    
    # 8. Reassign the bottom triangles to use the new vertex indices.
    bottom_triangles = []
    for tri in tri_bottom.simplices:
        # Map each vertex in the triangle using the bottom_mapping dictionary.
        new_tri = [bottom_mapping[idx] for idx in tri]
        bottom_triangles.append(new_tri)
    bottom_triangles = np.array(bottom_triangles)
    
    # 9. The top triangles remain the same (they use the indices of top_points)
    top_triangles = tri_top.simplices
    
    # 10. Concatenate the triangles from both surfaces.
    combined_triangles = np.vstack([top_triangles, bottom_triangles])
    
    # 11. Visualization using matplotlib's plot_trisurf.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        combined_vertices[:, 0],
        combined_vertices[:, 1],
        combined_vertices[:, 2],
        triangles=combined_triangles,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8
    )
    ax.set_title("Closed Surface Triangulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

if __name__ == '__main__':
    main()