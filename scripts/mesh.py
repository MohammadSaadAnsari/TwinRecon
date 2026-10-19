"""Automatic, bounded Poisson surface reconstruction and consistency map."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def create_mesh(cloud, output_path, depth=8, max_input_points=180000):
    if len(cloud.points) < 1000:
        raise RuntimeError("Poisson meshing needs at least 1,000 object points")
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if len(points) > max_input_points:
        indices = np.linspace(0, len(points)-1, max_input_points, dtype=int)
        points, colors = points[indices], colors[indices]
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points)
    source.colors = o3d.utility.Vector3dVector(colors)
    spacing = float(np.median(cKDTree(points).query(points[::max(1, len(points)//20000)], k=2)[0][:, 1]))
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=spacing * 8, max_nn=30))
    source.orient_normals_consistent_tangent_plane(12)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source, depth=depth, n_threads=2)
    density = np.asarray(densities)
    # Poisson fills unseen regions. Trim weakly supported skirt triangles.
    low_density = density < np.quantile(density, .04)
    bbox_min = np.percentile(points, .1, axis=0) - spacing*6
    bbox_max = np.percentile(points, 99.9, axis=0) + spacing*6
    vertices = np.asarray(mesh.vertices)
    outside = np.any((vertices < bbox_min) | (vertices > bbox_max), axis=1)
    mesh.remove_vertices_by_mask(low_density | outside)
    if len(mesh.triangles) == 0:
        raise RuntimeError("Poisson mesh trimming removed every triangle")
    nearest = cKDTree(points)
    color_ids = nearest.query(np.asarray(mesh.vertices), k=1)[1]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[color_ids])
    mesh.compute_vertex_normals()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False):
        raise RuntimeError(f"Could not write Poisson mesh to {output_path}")
    return mesh, {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
                  "poisson_depth": depth, "input_points": len(points)}


def consistency_map(cloud, mesh, output_path, spacing=None):
    """Mark points distant from the Poisson surface fitted to this same cloud.

    This is a diagnostic for recapture. It does not claim missing surfaces are
    recovered or that the Poisson surface is ground truth.
    """
    points = np.asarray(cloud.points)
    if spacing is None:
        tree = cKDTree(points)
        spacing = float(np.median(tree.query(points[::max(1, len(points)//20000)], k=2)[0][:, 1]))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), 50000):
        query = o3d.core.Tensor(points[start:start+50000].astype(np.float32))
        distances[start:start+len(query)] = scene.compute_distance(query).numpy()
    threshold = max(spacing * 2.5, 1e-6)
    bad = distances > threshold
    confidence = np.clip(distances / max(threshold, 1e-8), 0, 1)
    diagnostic = o3d.geometry.PointCloud()
    diagnostic.points = o3d.utility.Vector3dVector(points)
    diagnostic.colors = o3d.utility.Vector3dVector(np.column_stack((confidence, 1-confidence, np.zeros(len(points)))))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), diagnostic, write_ascii=False)
    fraction = float(np.mean(bad))
    return {"metric": "point_to_poisson_mesh_disagreement",
            "ground_truth_accuracy_available": False,
            "distance_units": "sfm_scene_units",
            "distance_threshold": float(threshold),
            "median_distance": float(np.median(distances)),
            "mesh_disagreement_fraction": fraction,
            "unreliable_fraction": fraction,  # Legacy report readers.
            "unreliable_points": int(np.sum(bad))}


def main():
    parser = argparse.ArgumentParser(description="Create a Poisson mesh from a TwinRecon point cloud")
    parser.add_argument("pointcloud", type=Path, help="Input dense_point_cloud.ply")
    parser.add_argument("--depth", type=int, default=8, choices=range(6, 11))
    args = parser.parse_args()
    cloud = o3d.io.read_point_cloud(str(args.pointcloud))
    output = args.pointcloud.with_name("model.ply")
    _, stats = create_mesh(cloud, output, depth=args.depth)
    print(json.dumps({"mesh": str(output), **stats}, indent=2))


if __name__ == "__main__":
    main()
