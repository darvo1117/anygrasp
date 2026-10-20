"""Review one env after batch_detect.py: the cloud AnyGrasp received + its grasps, with the tilt filter applied.

Colours:
  red     not on the target (outside the mask)
  orange  on the target, but tilted more than --max_tilt from the camera axis (filtered out)
  green   on the target and within --max_tilt (passes filter_object_grasps.py)
  yellow  the first grasp that gets executed (best score among the green ones)
Tilt = angle between the grasp approach (x) and the camera optical axis (+z), same as filter_object_grasps.py.

    ~/miniforge3/envs/anygrasp/bin/python anygrasp/view_grasps.py runs/batch_0015 0
    ... runs/batch_0015 0 --max_tilt 25        # what a looser filter would let through
    ... runs/batch_0015 0 --passed_only        # only green + yellow
    ... runs/batch_0015 0 --kept_only          # hide the red ones
    ... runs/batch_0015 0 sam2                 # grasps filtered with masks/sam2/
    ... runs/batch_0015 0 --cloud_only         # just the cropped cloud, no grasps
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

RED, ORANGE, GREEN, YELLOW = [1, 0, 0], [1, 0.5, 0], [0, 0.8, 0], [1, 0.85, 0]

ap = argparse.ArgumentParser()
ap.add_argument("batch_dir")
ap.add_argument("env", type=int)
ap.add_argument("source", nargs="?", default="gt", help="mask source subfolder of grasps/")
ap.add_argument("--max_tilt", type=float, default=20.0, help="deg, same as filter_object_grasps.MAX_TILT_DEG")
ap.add_argument("--num_select", type=int, default=5, help="same as filter_object_grasps.NUM_SELECT")
ap.add_argument("--kept_only", action="store_true", help="hide grasps that are not on the target")
ap.add_argument("--passed_only", action="store_true", help="show only grasps that pass the tilt filter")
ap.add_argument("--cloud_only", action="store_true")
args = ap.parse_args()

gdir = Path(args.batch_dir).expanduser() / "grasps" / args.source
name = f"env_{args.env:03d}"


def tilt_deg(rot):
    """(k, 3, 3) camera-frame rotations -> (k,) angle between approach (x) and the optical axis (+z)."""
    return np.degrees(np.arccos(np.clip(rot[:, 2, 0], -1.0, 1.0)))


c = np.load(gdir / f"{name}_cloud.npz")  # saved by batch_detect.py: cropped + voxelised
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c["points"].astype(np.float64)))
pcd.colors = o3d.utility.Vector3dVector(c["colors"] / 255.0)
geoms = [pcd]
print(f"{name}: {len(c['points'])} points in the AnyGrasp input")

if not args.cloud_only:
    from graspnetAPI import GraspGroup

    all_g = GraspGroup(np.load(gdir / f"{name}_all.npy").astype(np.float64))
    kept = np.load(gdir / f"{name}.npz")  # on the target, best score first
    kept_t = kept["translations"].reshape(-1, 3)
    kept_tilt = tilt_deg(kept["rotation_matrices"].reshape(-1, 3, 3))
    passed = np.nonzero(kept_tilt <= args.max_tilt)[0]          # kept indices, still best score first
    selected = passed[: args.num_select]

    def kept_index(g):
        if not len(kept_t):
            return None
        d = np.linalg.norm(kept_t - g.translation, axis=1)
        return int(np.argmin(d)) if d.min() < 1e-6 else None

    first = None
    for g, mesh in zip(all_g, all_g.to_open3d_geometry_list()):
        i = kept_index(g)
        if i is None:
            colour = RED
        elif kept_tilt[i] > args.max_tilt:
            colour = ORANGE
        else:
            colour = GREEN
        if i is not None and len(selected) and i == selected[0]:
            mesh.paint_uniform_color(YELLOW)
            first = mesh              # added last, so it is drawn on top
            continue
        if (args.kept_only and colour == RED) or (args.passed_only and colour != GREEN):
            continue
        mesh.paint_uniform_color(colour)
        geoms.append(mesh)
    if first is not None:
        geoms.append(first)

    print(f"{len(all_g)} detected | {len(kept_t)} on the target | {len(passed)} within {args.max_tilt:.0f} deg "
          f"| {len(selected)} get executed")
    print("  index  score   width   tilt   status")
    for i in range(len(kept_t)):
        if i in selected:
            status = f"executed as k={int(np.nonzero(selected == i)[0][0])}"
        elif kept_tilt[i] <= args.max_tilt:
            status = "passes, beyond num_select"
        else:
            status = "too tilted"
        print(f"  {i:5d}  {kept['scores'][i]:.3f}  {kept['widths'][i] * 1000:5.1f}mm  {kept_tilt[i]:4.1f}  {status}")

o3d.visualization.draw_geometries(
    geoms, window_name=f"{name}: red = off target, orange = too tilted, green = passes, yellow = first executed")
