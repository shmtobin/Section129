#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Create a directory for saving plots if it doesn't exist.
if not os.path.exists("plots"):
    os.makedirs("plots")

# ----------------------------
# Part a:
# ----------------------------

# --- Define the surfaces ---
def surface1(x, y):
    """Bottom surface (parabolic)."""
    return 2 * x**2 + 2 * y**2

def surface2(x, y):
    """Top surface (exponential)."""
    return 2 * np.exp(-x**2 - y**2)

# --- Find the boundary radius R where the two surfaces meet ---
def find_boundary_radius():
    func = lambda r: r**2 - np.exp(-r**2)
    r_boundary = fsolve(func, 0.7)[0]
    return r_boundary

# --- Generate grid points inside a disk of radius R ---
def generate_grid_points(R, num_points=50):
    x = np.linspace(-R, R, num_points)
    y = np.linspace(-R, R, num_points)
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.flatten(), yv.flatten()
    mask = xv**2 + yv**2 <= R**2
    return np.vstack([xv[mask], yv[mask]]).T

# --- Generate the point clouds for both surfaces ---
def generate_surface_point_cloud(points):
    z_top = surface2(points[:, 0], points[:, 1])
    z_bottom = surface1(points[:, 0], points[:, 1])
    return np.hstack([points, z_top[:, None]]), np.hstack([points, z_bottom[:, None]])

# ----------------------------
# Part b:
# ----------------------------
def delaunay_triangulation():
    # Find the boundary radius and generate grid points.
    R = find_boundary_radius()
    grid_points = generate_grid_points(R, num_points=50)
    
    # Create point clouds for the top and bottom surfaces.
    top_points, bottom_points = generate_surface_point_cloud(grid_points)
    n = grid_points.shape[0]  # Total number of grid points
    
    # Perform Delaunay triangulation on the (x,y) projection.
    tri = Delaunay(grid_points)
    
    # Identify boundary indices (points on the disk edge, where r is approximately R).
    boundary_indices = [i for i, pt in enumerate(grid_points)
                        if np.isclose(np.sqrt(pt[0]**2 + pt[1]**2), R, atol=1e-3)]
    boundary_set = set(boundary_indices)
    
    # Identify interior indices (points not on the boundary) for the bottom surface.
    interior_indices = [i for i in range(n) if i not in boundary_set]
    
    # Build the combined vertex list:
    #   - Use all top surface points (indices 0 to n-1).
    #   - Append only the interior bottom surface points, avoiding duplicate boundary vertices.
    vertices_top = top_points
    vertices_bottom_interior = bottom_points[interior_indices]
    combined_vertices = np.vstack((vertices_top, vertices_bottom_interior))
    
    # Create a mapping for bottom surface vertices:
    #   - For boundary points, use the same index as in the top surface.
    #   - For interior points, assign a new index starting from n.
    bottom_mapping = {}
    for j, i in enumerate(interior_indices):
        bottom_mapping[i] = n + j
    for i in boundary_indices:
        bottom_mapping[i] = i
    
    # Re-index the triangles from the bottom surface using the mapping.
    bottom_triangles = []
    for triangle in tri.simplices:
        new_triangle = [bottom_mapping[i] for i in triangle]
        bottom_triangles.append(new_triangle)
    bottom_triangles = np.array(bottom_triangles)
    
    # Top surface triangles use the original triangulation indices.
    top_triangles = tri.simplices
    
    # Combine the triangles from both surfaces.
    combined_triangles = np.vstack((top_triangles, bottom_triangles))
    
    # Visualize the closed surface mesh.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(combined_vertices[:, 0], combined_vertices[:, 1],
                    combined_vertices[:, 2],
                    triangles=combined_triangles, cmap='viridis',
                    edgecolor='none', alpha=0.8)
    ax.set_title("Closed Surface Triangulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("plots/full_triangulation.png", dpi=300)
    plt.show()

# ----------------------------
# Part c:
# ----------------------------
def volume_mesh_triangulation():
    """
    Generate a volume mesh using Delaunay triangulation.
    For each (x,y) point in a grid within the disk of radius R, sample points
    between the bottom and top surfaces to form a 3D point cloud.
    Then, perform Delaunay triangulation on the 3D points to create tetrahedra,
    extract the boundary faces, and visualize the volume mesh.
    """
    # Find boundary radius.
    R = find_boundary_radius()
    
    # Use a lower resolution grid for (x,y) for volume mesh visualization.
    num_xy = 15
    xy_points = generate_grid_points(R, num_points=num_xy)
    
    # Build 3D point cloud by sampling along z between bottom and top surfaces for each (x,y).
    volume_points = []
    num_z = 4  # number of levels in the z-direction
    for pt in xy_points:
        x, y = pt
        z_bottom = surface1(x, y)
        z_top = surface2(x, y)
        z_values = np.linspace(z_bottom, z_top, num_z)
        for z in z_values:
            volume_points.append([x, y, z])
    volume_points = np.array(volume_points)
    
    # Perform 3D Delaunay triangulation on the volume points.
    tri_3d = Delaunay(volume_points)
    
    # Extract tetrahedra (simplices).
    tetrahedra = tri_3d.simplices
    
    # Extract faces from tetrahedra.
    faces = {}
    for tet in tetrahedra:
        # Each tetrahedron has 4 faces (combinations of 3 vertices).
        for face in [(tet[0], tet[1], tet[2]),
                     (tet[0], tet[1], tet[3]),
                     (tet[0], tet[2], tet[3]),
                     (tet[1], tet[2], tet[3])]:
            face_sorted = tuple(sorted(face))
            faces[face_sorted] = faces.get(face_sorted, 0) + 1
    
    # Boundary faces appear only once (an interior face is shared by 2 tetrahedra).
    boundary_faces = [face for face, count in faces.items() if count == 1]
    
    # Visualize the volume mesh's boundary surface using Poly3DCollection.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a collection for the boundary triangles.
    triangles = [volume_points[list(face)] for face in boundary_faces]
    mesh = Poly3DCollection(triangles, facecolor='cyan', edgecolor='gray', alpha=0.5)
    ax.add_collection3d(mesh)
    
    # Optionally, scatter plot the volume points.
    ax.scatter(volume_points[:, 0], volume_points[:, 1], volume_points[:, 2], color='red', s=10)
    
    ax.set_title("Volume Mesh Triangulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Set axis limits for better visualization.
    ax.set_xlim(np.min(volume_points[:, 0]), np.max(volume_points[:, 0]))
    ax.set_ylim(np.min(volume_points[:, 1]), np.max(volume_points[:, 1]))
    ax.set_zlim(np.min(volume_points[:, 2]), np.max(volume_points[:, 2]))
    
    plt.savefig("plots/volume_mesh.png", dpi=300)
    plt.show()

# ----------------------------
# Part d:
# ----------------------------
def surface_mesh_from_volume_mesh():
    """
    Generate a surface mesh from the volume mesh and visualize it.
    
    The steps are as follows:
      1. Generate a 3D volume point cloud by sampling (x,y) points inside the disk 
         and interpolating z between the bottom and top surfaces (same as in Part c).
      2. Perform a 3D Delaunay triangulation on these volume points to obtain tetrahedra.
      3. Extract the boundary faces from the tetrahedra:
         - Each tetrahedron contributes 4 faces.
         - A face shared by two tetrahedra is an interior face (it appears twice).
         - A face appearing only once is a boundary face.
      4. Visualize the surface mesh using plot_trisurf.
      
    Comparison with Part b:
      - In Part b, the surface mesh is generated directly from a 2D Delaunay triangulation 
        on the (x,y) projection of the surfaces. 
      - Here, the surface mesh is derived from the volume mesh tetrahedralization, so the 
        connectivity may differ. This approach can capture the full 3D boundary of a volume, 
        whereas Part b only considers the two explicit surfaces.
    """
    # Find boundary radius.
    R = find_boundary_radius()
    
    # Use a lower resolution grid for (x,y) to create the volume points.
    num_xy = 15
    xy_points = generate_grid_points(R, num_points=num_xy)
    
    # Build 3D point cloud by sampling along z between bottom and top surfaces.
    volume_points = []
    num_z = 4  # number of sampling levels along z
    for pt in xy_points:
        x, y = pt
        z_bottom = surface1(x, y)
        z_top = surface2(x, y)
        z_values = np.linspace(z_bottom, z_top, num_z)
        for z in z_values:
            volume_points.append([x, y, z])
    volume_points = np.array(volume_points)
    
    # Perform 3D Delaunay triangulation to obtain tetrahedra.
    tri_3d = Delaunay(volume_points)
    tetrahedra = tri_3d.simplices
    
    # Extract faces from the tetrahedra.
    face_count = {}
    for tet in tetrahedra:
        # Each tetrahedron has 4 faces.
        for face in [(tet[0], tet[1], tet[2]),
                     (tet[0], tet[1], tet[3]),
                     (tet[0], tet[2], tet[3]),
                     (tet[1], tet[2], tet[3])]:
            face_sorted = tuple(sorted(face))
            face_count[face_sorted] = face_count.get(face_sorted, 0) + 1
    
    # Boundary faces occur only once (interior faces are shared by two tetrahedra).
    boundary_faces = [face for face, count in face_count.items() if count == 1]
    boundary_faces = np.array(boundary_faces)
    
    # Visualize the extracted surface mesh using plot_trisurf.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot_trisurf expects the vertices and a triangles array that contains indices into the vertices.
    # Here, volume_points are our vertices, and boundary_faces provides the connectivity.
    ax.plot_trisurf(volume_points[:, 0], volume_points[:, 1], volume_points[:, 2],
                    triangles=boundary_faces, cmap='plasma', edgecolor='none', alpha=0.8)
    
    ax.set_title("Surface Mesh Extracted from Volume Mesh")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.savefig("plots/surface_from_volume_mesh.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    # Run Part b: Closed Surface Triangulation.
    delaunay_triangulation()
    
    # Run Part c: Volume Mesh Triangulation.
    volume_mesh_triangulation()
    
    # Run Part d: Surface Mesh Derived from Volume Mesh.
    surface_mesh_from_volume_mesh()