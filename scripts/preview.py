"""Prepare centered, upright binary previews for the local WebGL viewer."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation


SH_C0 = 0.28209479177387814


def _positions(vertices, indices=None):
    data = vertices if indices is None else vertices[indices]
    return np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)


@lru_cache(maxsize=16)
def display_frame(job_folder):
    """Find visual up from registered camera rolls and the long horizontal axis.

    SfM world orientation is arbitrary. Camera image-up vectors are coherent
    when the operator holds the camera upright, so they define a useful display
    frame without changing any exported reconstruction coordinates.
    """
    folder = Path(job_folder)
    source = folder / "pointcloud" / "dense_point_cloud.ply"
    cloud = PlyData.read(str(source))["vertex"].data
    indices = np.linspace(0, len(cloud)-1, min(50000, len(cloud)), dtype=int)
    points = _positions(cloud, indices).astype(np.float64)
    center = np.median(points, axis=0)
    cameras_file = folder / "cameras.json"
    camera_positions = []
    up = None
    if cameras_file.exists():
        cameras = json.loads(cameras_file.read_text()).get("cameras", [])
        if cameras:
            matrices = [np.asarray(camera["camera_to_world"], dtype=float) for camera in cameras]
            ups = np.asarray([-matrix[:3, 1] for matrix in matrices])
            candidate = ups.mean(axis=0)
            if np.linalg.norm(candidate) > .35:
                up = candidate / np.linalg.norm(candidate)
            camera_positions = [matrix[:3, 3] for matrix in matrices]
    if up is None:
        # Older jobs have no exported cameras. Their orientation cannot be
        # recovered reliably, so use a stable Y-up display frame.
        up = np.array([0., 1., 0.])
    horizontal = points - center
    horizontal -= (horizontal @ up)[:, None] * up
    _, axes = np.linalg.eigh(np.cov(horizontal.T))
    right = axes[:, -1]
    right -= np.dot(right, up) * up
    right /= max(np.linalg.norm(right), 1e-8)
    forward = np.cross(right, up)
    if camera_positions and np.dot(camera_positions[0] - center, forward) < 0:
        right, forward = -right, -forward
    basis = np.column_stack((right, up, forward))
    radius = float(np.quantile(np.linalg.norm((points-center) @ basis, axis=1), .985))
    return center.astype(np.float32), basis.astype(np.float32), max(radius, 1e-6)


def point_preview(job_folder, ply_path, limit=150000):
    cloud = PlyData.read(str(ply_path))["vertex"].data
    if not len(cloud):
        raise ValueError("Point cloud is empty")
    ids = np.linspace(0, len(cloud)-1, min(limit, len(cloud)), dtype=int)
    sample = cloud[ids]
    center, basis, radius = display_frame(str(job_folder))
    positions = (_positions(sample)-center) @ basis / radius
    colors = np.column_stack((sample["red"], sample["green"], sample["blue"])).astype(np.float32) / 255.
    return np.column_stack((positions, colors)).astype("<f4", copy=False).tobytes()


def splat_preview(job_folder, ply_path, limit=160000):
    """Pack position, RGB, opacity and six symmetric covariance terms.

    Stride: 13 little-endian float32 values. Covariance is in the same upright,
    normalized frame as the point preview. The browser projects it to an
    elliptical Gaussian and sorts splats before alpha compositing.
    """
    vertices = PlyData.read(str(ply_path))["vertex"].data
    if not len(vertices):
        raise ValueError("Gaussian PLY is empty")
    ids = np.linspace(0, len(vertices)-1, min(limit, len(vertices)), dtype=int)
    center, basis, radius = display_frame(str(job_folder))
    output = np.empty((len(ids), 13), dtype="<f4")
    for start in range(0, len(ids), 40000):
        stop = min(start+40000, len(ids))
        batch = vertices[ids[start:stop]]
        output[start:stop, :3] = (_positions(batch)-center) @ basis / radius
        dc = np.column_stack((batch["f_dc_0"], batch["f_dc_1"], batch["f_dc_2"]))
        output[start:stop, 3:6] = np.clip(.5 + SH_C0*dc, 0, 1)
        opacity = np.asarray(batch["opacity"], dtype=np.float32)
        output[start:stop, 6] = 1 / (1 + np.exp(-np.clip(opacity, -20, 20)))
        quats = np.column_stack((batch["rot_1"], batch["rot_2"], batch["rot_3"], batch["rot_0"]))
        quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
        rotation = Rotation.from_quat(quats).as_matrix().astype(np.float32)
        oriented = np.einsum("ij,njk->nik", basis.T, rotation)
        scales_sq = np.exp(2*np.column_stack((batch["scale_0"], batch["scale_1"], batch["scale_2"]))) / radius**2
        covariance = np.einsum("nia,na,nja->nij", oriented, scales_sq, oriented)
        output[start:stop, 7:] = np.column_stack((covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2],
                                                  covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2]))
    return output.tobytes()
