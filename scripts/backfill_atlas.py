"""Build an atlas for a completed scan whose temporary COLMAP work was removed."""

import argparse
import json
from pathlib import Path
import tempfile

import open3d as o3d
import numpy as np

from atlas import build_confidence_atlas
from run_pipeline import extract_frames
from segmentation import segment_images


def backfill(job_dir, refine=False):
    job_dir = Path(job_dir).resolve()
    report_file = job_dir / "reconstruction.json"
    report = json.loads(report_file.read_text())
    if report.get("status") != "complete":
        raise RuntimeError("Backfill requires a completed scan")
    cameras = json.loads((job_dir / report["outputs"]["cameras"]).read_text())["cameras"]
    views = []
    for camera in cameras:
        c2w = np.asarray(camera["camera_to_world"], dtype=float)
        rotation = c2w[:3, :3].T
        translation = -rotation @ c2w[:3, 3]
        record = (camera["model"], camera["width"], camera["height"],
                  np.asarray(camera["intrinsics"], dtype=float))
        views.append((camera["image"], record, rotation, translation))
    cloud = o3d.io.read_point_cloud(str(job_dir / report["outputs"]["point_cloud"]))
    mesh = o3d.io.read_triangle_mesh(str(job_dir / report["outputs"]["poisson_mesh"]))
    if not cloud.has_points() or not mesh.has_triangles():
        raise RuntimeError("Cloud or mesh is missing")
    with tempfile.TemporaryDirectory(prefix="twinrecon_atlas_") as temp:
        images, masks = Path(temp) / "images", Path(temp) / "masks"
        extract_frames(report["input_video"], images, report["frames_processed"],
                       max(max(camera["width"], camera["height"]) for camera in cameras))
        segment_images(images, masks, Path(temp) / "preview.jpg",
                       backend=report.get("stages", {}).get("segmentation", {}).get("backend", "grabcut"))
        if refine:
            from refinement import refine_splats
            report["stages"]["refinement"] = refine_splats(
                job_dir / report["outputs"]["gaussian_splat"], views, np.zeros(3), images, masks)
            report["sh_coefficients_estimated"] = report["stages"]["refinement"]["sh_accepted"]
        atlas_file = job_dir / "diagnostics" / "confidence_atlas.ply"
        atlas_json = job_dir / "diagnostics" / "confidence_atlas.json"
        report["stages"]["atlas"] = build_confidence_atlas(
            cloud, mesh, views, np.zeros(3), images, masks, atlas_file, atlas_json)
    report["outputs"]["confidence_atlas"] = str(atlas_file.relative_to(job_dir))
    report["outputs"]["atlas_data"] = str(atlas_json.relative_to(job_dir))
    temporary = report_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(report_file)
    return report["stages"]["atlas"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--refine", action="store_true", help="Also attempt held-out gsplat refinement")
    args = parser.parse_args()
    print(json.dumps(backfill(args.job_dir, args.refine), indent=2))
