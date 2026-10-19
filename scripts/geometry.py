"""CPU geometry cleanup and bounded-memory Gaussian PLY export."""

from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


SH_C0 = 0.28209479177387814


def _cloud(points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def clean_object_cloud(fused_path, output_path, views=None, masks_dir=None,
                       max_points=None):
    """Keep multi-view foreground, remove isolated points, and preserve soles.

    No percentile bounding-box crop is used: it can delete dark extremities.
    A RANSAC plane is removed only if it is significantly broader than the
    non-planar foreground. This rejects tables but preserves sole faces.
    """
    cloud = o3d.io.read_point_cloud(str(fused_path))
    if not cloud.has_points():
        raise RuntimeError("Fused MVS point cloud is empty")
    raw_count = len(cloud.points)
    mask_stats = {"raw_points": raw_count}
    if views is not None and masks_dir is not None:
        from segmentation import filter_points_by_masks
        keep, support = filter_points_by_masks(np.asarray(cloud.points), views, masks_dir)
        if keep.sum() < 1000 or keep.mean() < .01:
            raise RuntimeError("Multi-view masks rejected almost all MVS points; inspect segmentation/preview.jpg")
        cloud = cloud.select_by_index(np.flatnonzero(keep))
        mask_stats["mask_kept_points"] = len(cloud.points)
        mask_stats["mean_foreground_views"] = round(float(support[keep].mean()), 2)

    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.2)
    if len(cloud.points) < 1000:
        raise RuntimeError("Too few points remain after outlier removal")
    mask_stats["outlier_kept_points"] = len(cloud.points)

    points = np.asarray(cloud.points)
    sample = points[::max(1, len(points) // 30000)]
    spacing = float(np.median(cKDTree(points).query(sample, k=2)[0][:, 1]))
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    plane_threshold = max(2 * spacing, diagonal * .002)
    removed_plane = False
    if len(cloud.points) > 4000:
        plane, inliers = cloud.segment_plane(distance_threshold=plane_threshold,
                                              ransac_n=3, num_iterations=350)
        fraction = len(inliers) / len(cloud.points)
        if .12 < fraction < .8 and len(inliers) > 1000:
            normal = np.asarray(plane[:3], dtype=float)
            normal /= np.linalg.norm(normal)
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            tangent = np.cross(normal, axis)
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(normal, tangent)
            plane_xy = points[inliers] @ np.stack((tangent, bitangent), axis=1)
            remaining = np.ones(len(points), dtype=bool)
            remaining[inliers] = False
            object_xy = points[remaining] @ np.stack((tangent, bitangent), axis=1)
            plane_size = np.prod(np.ptp(np.percentile(plane_xy, [2, 98], axis=0), axis=0))
            object_size = np.prod(np.ptp(np.percentile(object_xy, [2, 98], axis=0), axis=0))
            # A shoe sole is roughly as wide as the shoe. A table is broader.
            if plane_size > 2.2 * max(object_size, 1e-12):
                cloud = cloud.select_by_index(inliers, invert=True)
                removed_plane = True
    mask_stats["ground_plane_removed"] = removed_plane
    mask_stats["plane_distance_threshold"] = round(plane_threshold, 6)

    if max_points and len(cloud.points) > max_points:
        # Deterministic, uniform thinning affects only export, never masks.
        stride = int(np.ceil(len(cloud.points) / max_points))
        cloud = cloud.select_by_index(np.arange(0, len(cloud.points), stride))
    center = np.median(np.asarray(cloud.points), axis=0)
    centered = np.asarray(cloud.points) - center
    result = _cloud(centered, np.asarray(cloud.colors))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), result, write_ascii=False):
        raise RuntimeError(f"Could not write point cloud to {output_path}")
    mask_stats["final_points"] = len(result.points)
    mask_stats["center_world"] = center.tolist()
    mask_stats["median_spacing"] = spacing
    return result, center, mask_stats


def _ellipsoid_parameters(points, chunk_size=50000):
    """Estimate local tangent ellipsoids in bounded CPU chunks."""
    tree = cKDTree(points)
    for start in range(0, len(points), chunk_size):
        neighborhood = tree.query(points[start:start+chunk_size], k=min(9, len(points)))[1]
        neighbors = points[neighborhood]
        offsets = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", offsets, offsets) / max(1, neighbors.shape[1]-1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # np.linalg.eigh returns ascending eigenvalues: first axis is the thin normal.
        eigenvectors[:, :, 0] *= np.linalg.det(eigenvectors)[:, None]
        spacing = np.linalg.norm(neighbors[:, 1] - neighbors[:, 0], axis=1)
        spacing = np.maximum(spacing, 1e-6)
        scales = np.sqrt(np.maximum(eigenvalues, 0))
        scales[:, 0] = np.clip(scales[:, 0], spacing*.22, spacing*.65)
        scales[:, 1:] = np.clip(scales[:, 1:], spacing[:, None]*.5, spacing[:, None]*2.2)
        quats_xyzw = Rotation.from_matrix(eigenvectors).as_quat().astype(np.float32)
        quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
        yield start, np.log(scales).astype(np.float32), quats_wxyz


def export_gaussians(cloud, output_path, max_splats=450000, sh_degree=3):
    """Write SuperSplat-compatible ellipsoids. Higher SH coefficients are zero.

    View-dependent coefficients require calibrated per-view radiance and
    visibility optimization; inventing them from a single fused RGB value
    produces color shifts. Zero terms preserve the measured MVS color.
    """
    if sh_degree not in (0, 3):
        raise ValueError("sh_degree must be 0 or 3")
    points = np.asarray(cloud.points)
    colors = np.clip(np.asarray(cloud.colors), 0, 1)
    if len(points) > max_splats:
        indices = np.linspace(0, len(points)-1, max_splats, dtype=int)
        points, colors = points[indices], colors[indices]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    if sh_degree == 3:
        names += [f"f_rest_{i}" for i in range(45)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    vertices = np.zeros(len(points), dtype=[(name, "<f4") for name in names])
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
    dc = (colors - .5) / SH_C0
    for axis in range(3):
        vertices[f"f_dc_{axis}"] = dc[:, axis]
    vertices["opacity"] = 3.5
    for start, scales, quats in _ellipsoid_parameters(points):
        stop = start + len(scales)
        for axis in range(3):
            vertices[f"scale_{axis}"][start:stop] = scales[:, axis]
        for axis in range(4):
            vertices[f"rot_{axis}"][start:stop] = quats[:, axis]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(output_path))
    return len(vertices)
