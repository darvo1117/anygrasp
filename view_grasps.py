"""Review one env after batch_detect.py: the cloud AnyGrasp received + its grasps.

Colours: red = all grasps (not on the target), green = kept (on the target), yellow = best kept grasp.

    ~/miniforge3/envs/anygrasp/bin/python anygrasp/view_grasps.py runs/batch_0012 0
    ... runs/batch_0012 0 sam2          # grasps filtered with masks/sam2/
    ... --kept_only                     # hide the red ones
    ... --best_only                     # only the best grasp
    ... --cloud_only                    # just the cropped cloud, no grasps
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

RED, GREEN, YELLOW = [1, 0, 0], [0, 0.8, 0], [1, 0.85, 0]

ap = argparse.ArgumentParser()
ap.add_argument("batch_dir")
ap.add_argument("env", type=int)
ap.add_argument("source", nargs="?", default="gt", help="mask source subfolder of grasps/")
ap.add_argument("--kept_only", action="store_true", help="hide grasps that are not on the target")
ap.add_argument("--best_only", action="store_true", help="show only the best kept grasp")
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
    kept = np.load(gdir / f"{name}.npz")
    kept_t = kept["translations"]
    best_t = kept_t[0] if len(kept_t) else None  # kept grasps are sorted by score

    best_mesh = None
    for g, mesh in zip(all_g, all_g.to_open3d_geometry_list()):
        is_kept = len(kept_t) and np.min(np.linalg.norm(kept_t - g.translation, axis=1)) < 1e-6
        is_best = best_t is not None and np.linalg.norm(best_t - g.translation) < 1e-6
        if is_best:
            mesh.paint_uniform_color(YELLOW)
            best_mesh = mesh          # added last, so it is drawn on top
            continue
        if args.best_only or (args.kept_only and not is_kept):
            continue
        mesh.paint_uniform_color(GREEN if is_kept else RED)
        geoms.append(mesh)
    if best_mesh is not None:
        geoms.append(best_mesh)

    print(f"{len(all_g)} grasps, {len(kept_t)} kept")
    if best_t is not None:
        s, w = kept["scores"][0], kept["widths"][0]
        print(f"best (yellow): score {s:.3f} | width {w * 100:.1f} cm | camera frame "
              f"({best_t[0]:+.3f}, {best_t[1]:+.3f}, {best_t[2]:+.3f}) m")

o3d.visualization.draw_geometries(geoms, window_name=f"{name}: red = all, green = kept, yellow = best")
