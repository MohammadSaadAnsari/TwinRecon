"""TwinRecon: single-object video to cloud, splat, mesh and diagnostics."""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from geometry import clean_object_cloud, export_gaussians
from mesh import consistency_map, create_mesh
from segmentation import read_colmap_text, segment_images


LOG = logging.getLogger("twinrecon")


def extract_frames(video_path, output_dir, target_frames=60, max_dim=960):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = np.linspace(0, max(0, total-1), min(target_frames, total), dtype=int) if total > 0 else None
    wanted = set(selected.tolist()) if selected is not None else None
    count, index = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if wanted is None or index in wanted:
                height, width = frame.shape[:2]
                ratio = min(1., max_dim / max(height, width))
                if ratio < 1:
                    frame = cv2.resize(frame, (round(width*ratio), round(height*ratio)), interpolation=cv2.INTER_AREA)
                path = output_dir / f"frame_{count:04d}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not write frame {path}")
                count += 1
                if count >= target_frames:
                    break
            index += 1
    finally:
        cap.release()
    if count < 3:
        raise RuntimeError("Video needs at least three readable frames")
    LOG.info("Stage 1 | Extracted %d frames at <= %d px", count, max_dim)
    return count


def _run_colmap(colmap, args, log_file):
    LOG.info("COLMAP | %s", args[0])
    with open(log_file, "a") as stream:
        stream.write("\n$ " + " ".join([str(colmap), *map(str, args)]) + "\n")
        stream.flush()
        result = subprocess.run([str(colmap), *map(str, args)], stdout=stream,
                                stderr=subprocess.STDOUT, check=False,
                                cwd=Path(log_file).parent / "work")
    if result.returncode:
        tail = Path(log_file).read_text(errors="replace")[-5000:]
        if "no CUDA-capable device is detected" in tail or "CUDA driver version is insufficient" in tail:
            raise RuntimeError("COLMAP cannot access the NVIDIA GPU/CUDA driver; verify nvidia-smi, then see " + str(log_file))
        raise RuntimeError(f"COLMAP {args[0]} failed (exit {result.returncode}); see {log_file}")


def run_colmap(images_dir, masks_dir, work_dir, log_file):
    colmap = Path(sys.executable).parent / "colmap"
    if not colmap.exists():
        executable = shutil.which("colmap")
        if not executable:
            raise RuntimeError("COLMAP is not installed or not on PATH")
        colmap = Path(executable)
    db = work_dir / "database.db"
    sparse = work_dir / "sparse"
    mvs = work_dir / "mvs"
    sparse.mkdir(parents=True, exist_ok=True)
    mvs.mkdir(parents=True, exist_ok=True)

    _run_colmap(colmap, ["feature_extractor", "--database_path", db, "--image_path", images_dir,
                         "--ImageReader.mask_path", masks_dir, "--ImageReader.single_camera", "1",
                         "--ImageReader.camera_model", "SIMPLE_RADIAL",
                         "--SiftExtraction.max_num_features", "12000",
                         "--SiftExtraction.peak_threshold", "0.002"], log_file)
    _run_colmap(colmap, ["exhaustive_matcher", "--database_path", db], log_file)
    _run_colmap(colmap, ["sequential_matcher", "--database_path", db], log_file)
    _run_colmap(colmap, ["mapper", "--database_path", db, "--image_path", images_dir,
                         "--output_path", sparse], log_file)
    models = [p for p in sparse.iterdir() if p.is_dir() and (p / "points3D.bin").exists()]
    if not models:
        raise RuntimeError("COLMAP could not register a sparse object model")
    best = max(models, key=lambda p: (p / "images.bin").stat().st_size)
    sparse_text = work_dir / "sparse_text"
    sparse_text.mkdir()
    _run_colmap(colmap, ["model_converter", "--input_path", best, "--output_path", sparse_text,
                         "--output_type", "TXT"], log_file)
    _run_colmap(colmap, ["image_undistorter", "--image_path", images_dir,
                         "--input_path", best, "--output_path", mvs,
                         "--output_type", "COLMAP", "--max_image_size", "960"], log_file)
    _run_colmap(colmap, ["patch_match_stereo", "--workspace_path", mvs,
                         "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", "960",
                         "--PatchMatchStereo.gpu_index", "0", "--PatchMatchStereo.cache_size", "1",
                         "--PatchMatchStereo.window_radius", "7", "--PatchMatchStereo.num_iterations", "4",
                         "--PatchMatchStereo.filter_min_ncc", "0.02"], log_file)
    fused = mvs / "fused.ply"
    _run_colmap(colmap, ["stereo_fusion", "--workspace_path", mvs, "--workspace_format", "COLMAP",
                         "--input_type", "geometric", "--StereoFusion.cache_size", "1",
                         "--StereoFusion.min_num_pixels", "2", "--output_path", fused], log_file)
    if not fused.exists() or fused.stat().st_size == 0:
        raise RuntimeError("COLMAP fusion produced no point cloud")
    return fused, sparse_text


def export_cameras(views, center, output_path):
    records = []
    for name, (model, width, height, params), rotation, translation in views:
        c2w = np.eye(4)
        c2w[:3, :3] = rotation.T
        c2w[:3, 3] = -rotation.T @ translation - center
        records.append({"image": name, "model": model, "width": width, "height": height,
                        "intrinsics": params.tolist(), "camera_to_world": c2w.tolist()})
    Path(output_path).write_text(json.dumps({"coordinates": "centered COLMAP world, OpenCV camera convention",
                                             "cameras": records}, indent=2))
    return len(records)


def run(args):
    start = time.time()
    root = Path(args.output_root).resolve() if args.output_root else Path("outputs") / f"job_{int(start*1000)}"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    work = root / "work"
    images = work / "images"
    masks = work / "masks"
    log_file = root / "pipeline.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    report = {"job_id": root.name, "input_video": str(Path(args.video).resolve()),
              "status": "running", "outputs": {}, "stages": {}}
    report_file = root / "reconstruction.json"

    def save():
        report_file.write_text(json.dumps(report, indent=2))

    save()
    try:
        report["frames_processed"] = extract_frames(args.video, images, args.frames, args.max_dim)
        preview = root / "segmentation" / "preview.jpg"
        report["stages"]["segmentation"] = segment_images(images, masks, preview, args.object_box, args.segmenter)
        LOG.info("Stage 1.5 | Segmented %d frames (mean foreground %.1f%%)",
                 report["frames_processed"], report["stages"]["segmentation"]["mean_foreground_fraction"] * 100)
        report["outputs"]["segmentation_preview"] = str(preview.relative_to(root))
        save()
        LOG.info("Stage 2 | Building masked SfM and dense MVS cloud")
        fused, sparse_text = run_colmap(images, masks, work, log_file)
        views = read_colmap_text(sparse_text)
        LOG.info("Stage 2 | %d registered cameras", len(views))
        cloud_file = root / "pointcloud" / "dense_point_cloud.ply"
        cloud, center, cleanup = clean_object_cloud(fused, cloud_file, views, masks)
        report["stages"]["cleanup"] = cleanup
        report["outputs"]["point_cloud"] = str(cloud_file.relative_to(root))
        cameras = root / "cameras.json"
        report["registered_cameras"] = export_cameras(views, center, cameras)
        report["outputs"]["cameras"] = str(cameras.relative_to(root))
        save()

        splat_file = root / "gaussian_splat" / "gaussian_splat.ply"
        LOG.info("Stage 3 | Writing bounded-memory ellipsoid splats")
        report["splat_count"] = export_gaussians(cloud, splat_file, args.max_splats, args.sh_degree)
        report["sh_degree"] = args.sh_degree
        report["sh_coefficients_estimated"] = False
        report["outputs"]["gaussian_splat"] = str(splat_file.relative_to(root))
        save()

        if not args.no_mesh:
            LOG.info("Stage 4 | Fitting Poisson mesh")
            mesh_file = root / "mesh" / "poisson_mesh.ply"
            mesh, mesh_stats = create_mesh(cloud, mesh_file, depth=args.mesh_depth)
            report["stages"]["mesh"] = mesh_stats
            report["outputs"]["poisson_mesh"] = str(mesh_file.relative_to(root))
            save()
            LOG.info("Stage 5 | Measuring cloud-to-mesh consistency")
            diagnostic_file = root / "diagnostics" / "consistency.ply"
            report["stages"]["consistency"] = consistency_map(cloud, mesh, diagnostic_file)
            report["outputs"]["consistency_map"] = str(diagnostic_file.relative_to(root))

        report["points_count"] = len(cloud.points)
        report["total_time_seconds"] = round(time.time() - start, 2)
        report["status"] = "complete"
        save()
        if not args.keep_work:
            shutil.rmtree(work)
        LOG.info("Complete | %s | %d points | %d splats", root, report["points_count"], report["splat_count"])
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["total_time_seconds"] = round(time.time() - start, 2)
        save()
        LOG.exception("Reconstruction failed; work files kept for diagnosis")
        raise


def main():
    parser = argparse.ArgumentParser(description="TwinRecon single-object reconstruction")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--max-dim", type=int, default=960)
    parser.add_argument("--object-box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--segmenter", choices=("grabcut", "rembg"), default="grabcut")
    parser.add_argument("--max-splats", type=int, default=450000)
    parser.add_argument("--sh-degree", type=int, choices=(0, 3), default=3)
    parser.add_argument("--mesh-depth", type=int, choices=range(6, 11), default=8)
    parser.add_argument("--no-mesh", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    if args.frames < 3 or args.max_dim < 128 or args.max_splats < 1000:
        parser.error("frames >= 3, max-dim >= 128 and max-splats >= 1000 are required")
    run(args)


if __name__ == "__main__":
    main()
