import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import torch
import numpy as np
import cv2
import open3d as o3d
from plyfile import PlyData, PlyElement

def extract_frames(video_path, output_dir, target_frames=50, max_dim=960):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // target_frames)
    count, idx = 0, 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0 and count < target_frames:
            h, w = frame.shape[:2]
            scale = min(1.0, max_dim / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(output_dir, f"frame_{count:04d}.jpg"), frame)
            count += 1
        idx += 1
    cap.release()
    print(f"[Stage 1] Extracted {count} high-resolution frames.")
    return count

def run_colmap_mvs(images_dir, workspace_dir):
    db = os.path.join(workspace_dir, "database.db")
    sparse = os.path.join(workspace_dir, "sparse")
    mvs_dir = os.path.join(workspace_dir, "mvs")
    fused_ply = os.path.join(mvs_dir, "fused.ply")
    
    os.makedirs(sparse, exist_ok=True)
    os.makedirs(mvs_dir, exist_ok=True)
    
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", db,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--SiftExtraction.max_num_features", "12000"
    ], check=True)
    
    subprocess.run(["colmap", "exhaustive_matcher", "--database_path", db], check=True)
    subprocess.run(["colmap", "sequential_matcher", "--database_path", db], check=True)
    subprocess.run(["colmap", "mapper", "--database_path", db, "--image_path", images_dir, "--output_path", sparse], check=True)
    
    subdirs = [d for d in os.listdir(sparse) if os.path.isdir(os.path.join(sparse, d))]
    if not subdirs:
        raise RuntimeError("COLMAP SfM failed to build sparse model.")
        
    best_subdir = max(subdirs, key=lambda sd: os.path.getsize(os.path.join(sparse, sd, "points3D.bin")) if os.path.exists(os.path.join(sparse, sd, "points3D.bin")) else 0)
    sparse_model_path = os.path.join(sparse, best_subdir)

    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", images_dir,
        "--input_path", sparse_model_path,
        "--output_path", mvs_dir,
        "--output_type", "COLMAP",
        "--max_image_size", "800"
    ], check=True)

    subprocess.run([
        "colmap", "patch_match_stereo",
        "--workspace_path", mvs_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.max_image_size", "800",
        "--PatchMatchStereo.window_radius", "5",
        "--PatchMatchStereo.num_iterations", "3"
    ], check=True)

    fused_success = False
    for cmd in ["stereo_fusion", "dense_fuser", "stereo_fuser"]:
        try:
            subprocess.run([
                "colmap", cmd,
                "--workspace_path", mvs_dir,
                "--workspace_format", "COLMAP",
                "--input_type", "geometric",
                "--output_path", fused_ply
            ], check=True)
            fused_success = True
            break
        except Exception:
            continue

    if not fused_success:
        raise RuntimeError("Failed to run COLMAP depth map fusion.")

    return fused_ply

def convert_mvs_to_gaussian_splat(fused_ply_path, output_ply_path, remove_ground=True):
    pcd = o3d.io.read_point_cloud(fused_ply_path)
    if len(pcd.points) == 0:
        raise RuntimeError("Stereo fuser produced 0 points.")

    # Statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)

    # RANSAC Plane Segmentation (Remove Floor / Table)
    if remove_ground and len(pcd.points) > 1000:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        pcd = pcd.select_by_index(inliers, invert=True)

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    center = np.median(pts, axis=0)
    pts = pts - center

    pcd_centered = o3d.geometry.PointCloud()
    pcd_centered.points = o3d.utility.Vector3dVector(pts)
    pcd_centered.colors = o3d.utility.Vector3dVector(cols)
    distances = np.asarray(pcd_centered.compute_nearest_neighbor_distance())
    distances = np.maximum(distances, 0.0005)

    scales = np.log(np.tile(distances[:, None] * 0.45, (1, 3))).astype(np.float32)

    N = len(pts)
    normals = np.zeros((N, 3), dtype=np.float32)
    C0 = 0.28209479177387814
    f_dc = (cols - 0.5) / C0

    opacities = np.full((N, 1), 3.0, dtype=np.float32)
    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    data = np.concatenate([pts, normals, f_dc, opacities, scales, quats], axis=1)

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]

    elements = np.empty(N, dtype=dtype)
    for idx, name in enumerate([d[0] for d in dtype]):
        elements[name] = data[:, idx]

    PlyData([PlyElement.describe(elements, 'vertex')], text=False).write(output_ply_path)
    o3d.io.write_point_cloud(fused_ply_path.replace("fused.ply", "centered_dense.ply"), pcd_centered)
    print(f"[Stage 3] Exported {N:,} isolated surface Gaussians (floor auto-cropped) to {output_ply_path}")
    return N

def main():
    parser = argparse.ArgumentParser(description="TwinRecon MVS Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    args = parser.parse_args()
    
    start_time = time.time()
    job_id = f"job_{int(start_time)}"
    
    output_root = os.path.abspath(f"outputs/{job_id}")
    work_dir = os.path.join(output_root, "work")
    images_dir = os.path.join(work_dir, "images")
    
    pointcloud_dir = os.path.join(output_root, "pointcloud")
    splat_dir = os.path.join(output_root, "gaussian_splat")
    
    os.makedirs(pointcloud_dir, exist_ok=True)
    os.makedirs(splat_dir, exist_ok=True)
    
    final_dense_ply = os.path.join(pointcloud_dir, "dense_point_cloud.ply")
    final_splat_ply = os.path.join(splat_dir, "gaussian_splat.ply")
    report_json = os.path.join(output_root, "reconstruction.json")
    
    num_frames = extract_frames(args.video, images_dir, target_frames=50, max_dim=960)
    
    try:
        fused_mvs_ply = run_colmap_mvs(images_dir, work_dir)
        num_points = convert_mvs_to_gaussian_splat(fused_mvs_ply, final_splat_ply, remove_ground=True)
        shutil.copyfile(fused_mvs_ply.replace("fused.ply", "centered_dense.ply"), final_dense_ply)
    except Exception as err:
        print(f"\n[ERROR] Dense reconstruction failed: {err}")
        sys.exit(1)
        
    shutil.rmtree(work_dir, ignore_errors=True)
    elapsed = round(time.time() - start_time, 2)
    
    report = {
        "job_id": job_id,
        "input_video": os.path.abspath(args.video),
        "total_time_seconds": elapsed,
        "frames_processed": num_frames,
        "points_count": num_points,
        "outputs": {
            "point_cloud": os.path.relpath(final_dense_ply, output_root),
            "gaussian_splat": os.path.relpath(final_splat_ply, output_root)
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_limit_gb": 4.0
        }
    }
    
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*50)
    print("TWINRECON DENSE RECONSTRUCTION COMPLETE")
    print(f"Job Output Root : {output_root}")
    print(f"Dense Pointcloud: {final_dense_ply}")
    print(f"Gaussian Splat  : {final_splat_ply}")
    print(f"Total Points    : {num_points:,}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
