import glob
import sys
import open3d as o3d

jobs = sorted(glob.glob("outputs/job_*"))
if not jobs:
    sys.exit("No jobs found.")

latest = jobs[-1]
ply_path = f"{latest}/pointcloud/dense_point_cloud.ply"
out_mesh = f"{latest}/pointcloud/model.ply"

print(f"Generating 3D Mesh from: {ply_path}")
pcd = o3d.io.read_point_cloud(ply_path)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(10)

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
o3d.io.write_triangle_mesh(out_mesh, mesh)
print(f"Saved 3D Mesh to: {out_mesh}")
