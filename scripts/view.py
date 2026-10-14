import os
import sys
import glob
import open3d as o3d

def main():
    # Find latest output job
    jobs = sorted(glob.glob("outputs/job_*"))
    if not jobs:
        print("No job outputs found in outputs/")
        sys.exit(1)
        
    latest_job = jobs[-1]
    
    # Check for dense point cloud or gaussian splat
    ply_path = os.path.join(latest_job, "pointcloud", "dense_point_cloud.ply")
    if not os.path.exists(ply_path):
        ply_path = os.path.join(latest_job, "gaussian_splat", "gaussian_splat.ply")
        
    if not os.path.exists(ply_path):
        print(f"No PLY files found in {latest_job}")
        sys.exit(1)

    print(f"Loading latest reconstruction: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd], window_name=f"TwinRecon Viewer - {os.path.basename(latest_job)}", width=1280, height=720)

if __name__ == "__main__":
    main()
