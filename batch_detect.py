"""Stage 2: run AnyGrasp on every cloud of a batch and keep only the grasps on the target object.

Reads   <batch>/clouds/env_XXX.npz           (from scripts/capture.py in Isaac Lab)
        <batch>/masks/<source>/env_XXX.png   target mask, white = object ("gt" now, e.g. "sam2" later; .npy also works)
Writes  <batch>/grasps/<source>/env_XXX.npz  top-k grasps on the target, camera OpenCV optical frame
        <batch>/grasps/<source>/env_XXX_all.npy  ALL grasps after NMS (Nx17 graspnetAPI array, unfiltered)
        <batch>/grasps/<source>/env_XXX_cloud.npz  the cropped + voxelised cloud AnyGrasp received
        <batch>/grasps/<source>/summary.json

Mask filter, as in detect_grasps.py: each grasp centre is projected into the image with the camera K and
kept if it falls on the mask (dilated by --dilate px, since centres of thin parts sit on the mask edge).
The mask is extruded along the camera rays, so a grasp centre inside the pipe still counts.

AnyGrasp always sees the whole (cropped) cloud, so its collision check knows the bin walls; the mask
only decides which grasps are kept. The detector is created once and reused for every env.

Environment: anygrasp. No activation needed when called with the env's own python:
    ~/miniforge3/envs/anygrasp/bin/python batch_detect.py --batch_dir ~/isaacsim_anygrasp/runs/batch_0006
    ... --vis 0              # show env 0: cloud (dimmed outside the mask) + kept grasps
    ... --mask_source sam2   # use masks/sam2/ instead of masks/gt/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

GRASP_DIR = Path.home() / "anygrasp_sdk" / "grasp_detection"
CWD = Path.cwd()  # before chdir: relative --batch_dir paths are resolved from where the script was launched

# gsnet.so, the license and the checkpoint live in grasp_detection:
# import from there and use that folder as the working directory.
sys.path.insert(0, str(GRASP_DIR))
os.chdir(GRASP_DIR)
from gsnet import create_detector  # noqa: E402

WORKSPACE = [(-0.4, 0.4), (-0.4, 0.4), (0.05, 1.0)]  # x, y, z in metres (camera frame), as on the real robot
VOXEL = 0.005
MAX_GRIPPER_WIDTH = 0.16   # m: OnRobot RG6 maximum opening (check against your gripper)
GRIPPER_HEIGHT = 0.03


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------
def load_mask(path_without_suffix: Path) -> np.ndarray:
    """(H, W) bool from env_XXX.png (white = object) or env_XXX.npy."""
    png, npy = path_without_suffix.with_suffix(".png"), path_without_suffix.with_suffix(".npy")
    if png.exists():
        from PIL import Image
        img = np.asarray(Image.open(png))
        if img.ndim == 3:
            img = img[..., 0]
        return img > 127
    if npy.exists():
        return np.load(npy).astype(bool)
    raise SystemExit(f"no mask {png} or {npy}")


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Square dilation by px pixels (separable, numpy only)."""
    if px <= 0:
        return mask
    out = mask.copy()
    for axis in (0, 1):
        src, out = out, out.copy()
        for s in range(1, px + 1):
            out |= np.roll(src, s, axis=axis) | np.roll(src, -s, axis=axis)
    return out


def points_in_mask(points: np.ndarray, mask: np.ndarray, k: np.ndarray, invert: bool = False) -> np.ndarray:
    """True for points (camera OpenCV frame) whose projection lands on the mask."""
    h, w = mask.shape
    z = points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(k[0, 0] * points[:, 0] / z + k[0, 2]).astype(np.int64)
        v = np.round(k[1, 1] * points[:, 1] / z + k[1, 2]).astype(np.int64)
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    hit = np.zeros(len(points), dtype=bool)
    hit[inside] = mask[v[inside], u[inside]]
    return ~hit if invert else hit


# ---------------------------------------------------------------------------
# Cloud and grasps
# ---------------------------------------------------------------------------
def crop_and_voxelise(xyz, rgb):
    keep = np.ones(len(xyz), dtype=bool)
    for i, (lo, hi) in enumerate(WORKSPACE):
        keep &= (xyz[:, i] > lo) & (xyz[:, i] < hi)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[keep].astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(rgb[keep].astype(np.float64))
    return pcd.voxel_down_sample(VOXEL)


def grasps_to_array(grasps):
    """GraspGroup -> Nx17 array (graspnetAPI layout)."""
    arr = getattr(grasps, "grasp_group_array", None)
    if arr is not None:
        return np.asarray(arr, dtype=np.float32)
    rows = [np.concatenate([[g.score, g.width, getattr(g, "height", 0.02), g.depth],
                            np.asarray(g.rotation_matrix).reshape(-1),
                            np.asarray(g.translation).reshape(-1),
                            [getattr(g, "object_id", -1)]]) for g in grasps]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 17)


def save_grasps(path: Path, grasps, extra: dict):
    if grasps is None or len(grasps) == 0:
        arrays = dict(scores=np.zeros(0, np.float32), widths=np.zeros(0, np.float32),
                      depths=np.zeros(0, np.float32), translations=np.zeros((0, 3), np.float32),
                      rotation_matrices=np.zeros((0, 3, 3), np.float32))
    else:
        arrays = dict(
            scores=np.array([g.score for g in grasps], dtype=np.float32),
            widths=np.array([g.width for g in grasps], dtype=np.float32),
            depths=np.array([g.depth for g in grasps], dtype=np.float32),
            translations=np.array([g.translation for g in grasps], dtype=np.float32),
            rotation_matrices=np.array([g.rotation_matrix for g in grasps], dtype=np.float32),
        )
    np.savez(path, **arrays, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_dir", required=True)
    ap.add_argument("--mask_source", default="gt", help="subfolder of masks/: gt, sam2, ...")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--dilate", type=int, default=8,
                    help="mask margin in px (grasp centres on the edge of thin parts)")
    ap.add_argument("--invert", action="store_true", help="the mask is an EXCLUSION zone")
    ap.add_argument("--checkpoint_path", default=str(GRASP_DIR / "log" / "checkpoint_detection.tar"))
    ap.add_argument("--vis", type=int, default=None, help="env index to show after processing")
    args = ap.parse_args()

    batch = Path(args.batch_dir).expanduser()
    batch = (batch if batch.is_absolute() else CWD / batch).resolve()
    clouds = sorted((batch / "clouds").glob("env_*.npz"))
    if not clouds:
        raise SystemExit(f"no clouds in {batch / 'clouds'}")
    mask_dir = batch / "masks" / args.mask_source
    out_dir = batch / "grasps" / args.mask_source
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    detector = create_detector(argparse.Namespace(
        checkpoint_path=args.checkpoint_path,
        max_gripper_width=MAX_GRIPPER_WIDTH,
        gripper_height=GRIPPER_HEIGHT,
    ))
    if detector is None:
        raise SystemExit("Could not create the detector (check the license).")
    print(f"detector ready in {time.perf_counter() - t0:.2f} s | {len(clouds)} envs | "
          f"mask: {args.mask_source}{' (inverted)' if args.invert else ''}, margin {args.dilate} px")

    summary, shown = {}, None
    for path in clouds:
        t = time.perf_counter()
        name = path.stem
        d = np.load(path)
        k = d["K"].astype(np.float64)
        pcd = crop_and_voxelise(d["points"].astype(np.float32), d["colors"].astype(np.float32) / 255.0)
        mask = dilate(load_mask(mask_dir / name), args.dilate)

        np.savez_compressed(out_dir / f"{name}_cloud.npz",
                            points=np.asarray(pcd.points, dtype=np.float32),
                            colors=(np.asarray(pcd.colors) * 255).astype(np.uint8))
        grasps = detector.get_grasp(np.asarray(pcd.points, dtype=np.float32),
                                    {"dense_grasp": False, "collision_detection": True})
        n_all, n_on, kept = 0, 0, None
        if grasps is not None and len(grasps) > 0:
            grasps = grasps.nms().sort_by_score()
            n_all = len(grasps)
            np.save(out_dir / f"{name}_all.npy", grasps_to_array(grasps))
            centres = np.array([g.translation for g in grasps], dtype=np.float64)
            on_target = np.flatnonzero(points_in_mask(centres, mask, k, invert=args.invert))
            n_on = len(on_target)
            if len(on_target):
                kept = grasps[on_target[:args.top_k].tolist()]
        n_kept = 0 if kept is None else len(kept)

        save_grasps(out_dir / f"{name}.npz", kept, {"n_detected": n_all, "n_on_target": n_on, "frame": "camera_opencv"})
        summary[name] = {"detected": n_all, "on_target": n_on, "kept": n_kept,
                         "best_score": float(kept[0].score) if n_kept else None}
        print(f"{name}: {len(pcd.points):6d} pts | {n_all:4d} grasps -> {n_on} on target, {n_kept} kept "
              f"| {time.perf_counter() - t:.2f} s")
        if args.vis is not None and name == f"env_{args.vis:03d}":
            shown = (pcd, kept, mask, k)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    empty = [k for k, v in summary.items() if v["kept"] == 0]
    print(f"done in {time.perf_counter() - t0:.1f} s -> {out_dir}")
    if empty:
        print(f"[WARN] no grasp on the target for: {', '.join(empty)}")

    if shown is not None:
        pcd, kept, mask, k = shown
        inside = points_in_mask(np.asarray(pcd.points), mask, k, invert=args.invert)
        cols = np.asarray(pcd.colors).copy()
        cols[~inside] *= 0.25                          # dim the cloud outside the zone of interest
        pcd.colors = o3d.utility.Vector3dVector(cols)
        geoms = [pcd]
        if kept is not None:
            grippers = kept.to_open3d_geometry_list()
            grippers[0].paint_uniform_color([0.0, 1.0, 0.0])  # best grasp in green
            geoms += grippers
        o3d.visualization.draw_geometries(geoms, window_name=f"env_{args.vis:03d}: kept grasps (green = best)")


if __name__ == "__main__":
    main()
