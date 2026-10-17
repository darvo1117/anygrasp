"""Review one env after batch_detect.py: the cloud AnyGrasp received + ALL grasps (red) + kept grasps (green).

    ~/miniforge3/envs/anygrasp/bin/python view_grasps.py ~/isaacsim_anygrasp/runs/batch_0008 0
    ... ~/isaacsim_anygrasp/runs/batch_0008 0 sam2     # grasps filtered with masks/sam2/
    ... --cloud_only                                    # just the cropped cloud, no grasps
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("batch_dir")
ap.add_argument("env", type=int)
ap.add_argument("source", nargs="?", default="gt", help="mask source subfolder of grasps/")
ap.add_argument("--cloud_only", action="store_true")
args = ap.parse_args()

gdir = Path(args.batch_dir).expanduser() / "grasps" / args.source
name = f"env_{args.env:03d}"

c = np.load(gdir / f"{name}_cloud.npz")  # saved by batch_detect.py: cropped + voxelised
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c["points"].astype(np.float64)))
pcd.colors = o3d.utility.Vector3dVector(c["colors"] / 255.0)
geoms = [pcd]
print(f"{name}: {len(c['points'])} points in the AnyGrasp input")

if not args.cloud_only:
    from graspnetAPI import GraspGroup
    all_g = GraspGroup(np.load(gdir / f"{name}_all.npy").astype(np.float64))
    kept_t = np.load(gdir / f"{name}.npz")["translations"]
    for g, mesh in zip(all_g, all_g.to_open3d_geometry_list()):
        is_kept = len(kept_t) and np.min(np.linalg.norm(kept_t - g.translation, axis=1)) < 1e-6
        mesh.paint_uniform_color([0, 1, 0] if is_kept else [1, 0, 0])
        geoms.append(mesh)
    print(f"{len(all_g)} grasps, {len(kept_t)} kept")

o3d.visualization.draw_geometries(geoms, window_name=f"{name}: AnyGrasp input, all grasps (red), kept (green)")
