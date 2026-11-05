"""AnyGrasp tracking on a sequence recorded by scripts/capture_sequence.py, fed frame by frame as if it were live.

Approach cone
  A grasp is valid when its approach direction lies inside a cone that points down: tip on the grasp point,
  axis = camera optical axis (--cone_axis camera, default) or gravity (--cone_axis world), half-angle --max_tilt
  (default 30 deg), the same in every direction.

Frame 0: the tracker detects grasps on the whole cloud. The grasps to track are those whose centre projects onto
the (dilated) GT target mask and that are inside the cone, best score first, --num_track of them.
Frames 1..T-1: tracker.update(points, colors, grasp_ids) with the ids it returned for the previous frame, as
anygrasp_sdk/grasp_tracking/demo.py does. Then the cone is checked again: a tracked grasp that left it (or, with
--max_jump_mm, jumped too far) is replaced by the valid grasp of this frame closest to where it was (distance of
the AnyGrasp paper, eq. 5), and that id is passed on. If no valid grasp exists it is kept and flagged.
--no_cone_gate checks the cone on frame 0 only (the tracker's raw output), to compare.

Modes
  default      every frame is processed: how good is the tracking itself
  --realtime   frames arrive at the recorded fps. A frame that arrives while the tracker is still busy is skipped,
               and the next one processed is the newest available, as with a live camera or a remote server.
               It uses the measured compute time, so it runs as fast as the GPU allows and never sleeps.

Metrics (static scene: a perfect tracker keeps every grasp still in the object's frame)
  per tracked grasp: position jitter (std, mm), max position / rotation deviation from frame 0 (mm, deg),
  width std (mm). Grasps are expressed in the target object's frame with the ground-truth poses, so the same
  metrics stay meaningful once the camera or the object moves (level c).
  Latency of build-cloud + update (mean / p95 / max ms) -> the frame rate the tracker could sustain live.

Output: <seq>/tracking/<filter>[_rt]/tracks.npz + metrics.json

Environment: anygrasp. No activation needed when called with the env's own python:
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/track_replay.py --seq runs/seq_0000
    ... --filter kalman          # oneeuro (default) / kalman / none
    ... --realtime
    ... --vis
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

TRACK_DIR = Path.home() / "anygrasp_sdk" / "grasp_tracking"
CWD = Path.cwd()  # before chdir: relative --seq paths are resolved from where the script was launched

# tracker.so, the license and the checkpoint live in grasp_tracking: import from there and work in that folder.
sys.path.insert(0, str(TRACK_DIR))
os.chdir(TRACK_DIR)
from tracker import create_tracker  # noqa: E402

WORKSPACE = [(-0.4, 0.4), (-0.4, 0.4), (0.05, 1.0)]  # x, y, z in metres (camera frame), as in batch_detect.py


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def load_frame(frames: Path, i: int, k: np.ndarray, scale: float):
    """color_XXX.png + depth_XXX.npy -> points (M, 3) float32 camera OpenCV frame, colors (M, 3) float32 in [0, 1]."""
    colors = np.asarray(Image.open(frames / f"color_{i:03d}.png"), dtype=np.float32)[..., :3] / 255.0
    depth = np.load(frames / f"depth_{i:03d}.npy").astype(np.float32) / scale
    return colors, depth


def build_cloud(colors, depth, k):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - k[0, 2]) * depth / k[0, 0]
    y = (v - k[1, 2]) * depth / k[1, 1]
    pts = np.stack([x, y, depth], axis=-1)
    keep = depth > 0
    for a, (lo, hi) in enumerate(WORKSPACE):
        keep &= (pts[..., a] > lo) & (pts[..., a] < hi)
    return pts[keep].astype(np.float32), colors[keep].astype(np.float32)


def load_mask(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path))
    return (img[..., 0] if img.ndim == 3 else img) > 127


# same as batch_detect.py
def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    out = mask.copy()
    for axis in (0, 1):
        src, out = out, out.copy()
        for s in range(1, px + 1):
            out |= np.roll(src, s, axis=axis) | np.roll(src, -s, axis=axis)
    return out


def points_in_mask(points: np.ndarray, mask: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    z = points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(k[0, 0] * points[:, 0] / z + k[0, 2]).astype(np.int64)
        v = np.round(k[1, 1] * points[:, 1] / z + k[1, 2]).astype(np.int64)
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    hit = np.zeros(len(points), dtype=bool)
    hit[inside] = mask[v[inside], u[inside]]
    return hit


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def quat_to_mat(q):
    """(4,) wxyz -> (3, 3)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def angle_deg(r_a, r_b):
    """Angle between rotations, treating the two parallel-gripper poses (180 deg about the approach) as equal."""
    flip = np.diag([1.0, -1.0, -1.0])
    out = []
    for rb in (r_b, r_b @ flip):
        tr = np.einsum("...ij,...ij->...", r_a, rb)  # trace(r_a^T r_b)
        out.append(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))
    return np.minimum(*out)


def cone_tilt_deg(rots_c: np.ndarray, r_wc: np.ndarray, axis: str) -> np.ndarray:
    """(k, 3, 3) grasp rotations in the camera frame -> (k,) angle between the approach (column 0) and the cone axis.

    The cone has its tip on the grasp point and opens upwards around the cone axis, so a grasp passes when its
    approach points down the axis within --max_tilt, the same in every direction:
      camera: axis = camera optical axis (+z, the direction the camera looks)
      world:  axis = gravity (world -z), independent of how the camera is tilted
    """
    a = rots_c[:, :, 0]
    cos = a[:, 2] if axis == "camera" else -(a @ r_wc.T)[:, 2]
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def grasp_distance(t0, r0, t, r) -> np.ndarray:
    """Grasp distance of the AnyGrasp paper (eq. 5): dt / 0.01 m + 0.1 * dR / pi, one grasp vs (k,) grasps."""
    return np.linalg.norm(t - t0, axis=-1) / 0.01 + 0.1 * angle_deg(r0[None], r) / 180.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="runs/seq_XXXX from capture_sequence.py")
    ap.add_argument("--checkpoint_path", default=str(TRACK_DIR / "log" / "checkpoint_tracking.tar"))
    ap.add_argument("--filter", default="oneeuro", help="smoothing of rotation, width, depth: oneeuro / kalman / none")
    ap.add_argument("--num_track", type=int, default=5, help="grasps to track (picked on frame 0)")
    ap.add_argument("--max_tilt", type=float, default=30.0,
                    help="half-angle of the downward cone the approach must stay in, deg (frame 0 and every frame)")
    ap.add_argument("--cone_axis", choices=["camera", "world"], default="camera",
                    help="cone axis: camera optical axis, or gravity (world down)")
    ap.add_argument("--no_cone_gate", action="store_true",
                    help="only use the cone on frame 0 (old behaviour): keep whatever the tracker returns afterwards")
    ap.add_argument("--max_jump_mm", type=float, default=0.0,
                    help="also reject a tracked grasp that moved more than this since the last frame. 0 = off")
    ap.add_argument("--dilate", type=int, default=8, help="mask margin in px")
    ap.add_argument("--realtime", action="store_true", help="skip frames that arrive while the tracker is busy")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    seq = Path(args.seq).expanduser()
    seq = (seq if seq.is_absolute() else CWD / seq).resolve()
    meta = json.loads((seq / "meta.json").read_text())
    poses = np.load(seq / "poses.npz")
    frames = seq / "frames"
    k = np.asarray(meta["K"], dtype=np.float32)
    fps, n_frames, scale = meta["fps"], meta["n_frames"], meta.get("depth_scale", 1000.0)
    obj = meta.get("target_object", "pipe_0")
    t_rec = poses["t"] - poses["t"][0]

    gate = not args.no_cone_gate
    tag = (f"{args.filter}_{args.cone_axis}{args.max_tilt:.0f}" + ("_nogate" if not gate else "")
           + (f"_jump{args.max_jump_mm:.0f}" if args.max_jump_mm > 0 else "") + ("_rt" if args.realtime else ""))
    out_dir = seq / "tracking" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    tracker = create_tracker(argparse.Namespace(checkpoint_path=args.checkpoint_path, filter=args.filter,
                                                vis=args.vis))
    if not tracker:
        raise SystemExit("Could not create the tracker (check tracker.so for this Python and the license).")
    print(f"tracker ready in {time.perf_counter() - t0:.2f} s | {seq.name}: {n_frames} frames at {fps} fps, "
          f"level {meta.get('level')} | filter {args.filter}{' | realtime' if args.realtime else ''}")
    print(f"cone: {args.max_tilt:.0f} deg around the {args.cone_axis} axis, "
          f"{'enforced on every frame' if gate else 'frame 0 only'}"
          f"{f' | max jump {args.max_jump_mm:.0f} mm' if args.max_jump_mm > 0 else ''}")

    vis = None
    if args.vis:
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=720, width=1280, window_name=f"{seq.name} tracking")
        flip = np.diag([1.0, 1.0, -1.0, 1.0])

    rec = {key: [] for key in ("frame", "trans_c", "rot_c", "widths", "depths", "scores", "ids", "tilt_deg",
                               "violated", "replaced", "t_build_ms", "t_update_ms")}
    grasp_ids, n_track, f, clock, checked_ids = [0], None, 0, 0.0, False
    prev_t = prev_r = None
    while f < n_frames:
        colors, depth = load_frame(frames, f, k, scale)  # disk read: a replay artefact, not timed
        ta = time.perf_counter()
        points, cols = build_cloud(colors, depth, k)
        tb = time.perf_counter()
        target_gg, curr_gg, target_ids, _ = tracker.update(points, cols, grasp_ids)
        tc = time.perf_counter()

        if curr_gg is None or len(curr_gg) == 0:
            raise SystemExit(f"frame {f}: the tracker found no grasps at all")
        all_t = np.asarray(curr_gg.translations, dtype=np.float64)
        all_r = np.asarray(curr_gg.rotation_matrices, dtype=np.float64)
        all_tilt = cone_tilt_deg(all_r, quat_to_mat(poses["cam_quat_w_ros"][f]), args.cone_axis)
        in_cone = all_tilt <= args.max_tilt

        if f == 0:  # choose what to track: on the target, inside the cone, best score first
            on = points_in_mask(all_t, dilate(load_mask(frames / "mask_000.png"), args.dilate), k)
            cand = np.flatnonzero(on & in_cone)
            print(f"frame 0: {len(curr_gg)} grasps, {int(on.sum())} on '{obj}', {len(cand)} inside the "
                  f"{args.max_tilt:.0f} deg cone")
            if len(cand) == 0:
                raise SystemExit("no grasp on the target to track (try a larger --dilate / --max_tilt)")
            grasp_ids = cand[np.argsort(-np.asarray(curr_gg.scores)[cand])][:args.num_track]
            target_gg = curr_gg[grasp_ids]
            n_track = len(grasp_ids)
            violated = replaced = np.zeros(n_track, dtype=bool)
        else:
            grasp_ids = np.asarray(target_ids).reshape(-1).astype(np.int64)[:n_track].copy()
            if not checked_ids and len(grasp_ids):  # the gate relies on ids being indices into curr_gg
                checked_ids = True
                gap = np.abs(all_t[grasp_ids] - np.asarray(target_gg.translations)[:len(grasp_ids)]).max()
                if gap > 1e-4:
                    print(f"[WARN] target_grasp_ids do not index curr_gg (gap {gap * 1000:.1f} mm): "
                          f"the cone gate may pick wrong replacements")
            jump = np.linalg.norm(all_t[grasp_ids] - prev_t[:len(grasp_ids)], axis=-1) * 1000.0
            violated = ~in_cone[grasp_ids]
            if args.max_jump_mm > 0:
                violated |= jump > args.max_jump_mm
            replaced = np.zeros(len(grasp_ids), dtype=bool)
            if gate and violated.any():
                allowed = in_cone.copy()
                for j in np.flatnonzero(violated):
                    # the valid grasp of this frame closest to where this grasp was (paper's distance)
                    d = grasp_distance(prev_t[j], prev_r[j], all_t, all_r)
                    ok = allowed.copy()
                    if args.max_jump_mm > 0:
                        ok &= np.linalg.norm(all_t - prev_t[j], axis=-1) * 1000.0 <= args.max_jump_mm
                    d[~ok] = np.inf
                    best = int(np.argmin(d))
                    if np.isfinite(d[best]):
                        grasp_ids[j], replaced[j] = best, True
                if replaced.any():  # replacements come unfiltered: the tracker resets its filter on new ids
                    target_gg = curr_gg[grasp_ids]
            if violated.any():
                what = "left the cone" + (" or jumped" if args.max_jump_mm > 0 else "")
                if gate:
                    stuck = int((violated & ~replaced).sum())
                    result = f"{int(replaced.sum())} replaced" + (f", {stuck} kept (no valid alternative)" if stuck else "")
                else:
                    result = "kept (gate off)"
                print(f"frame {f:03d}: {int(violated.sum())} grasp(s) {what} -> {result}")

        # record (pad if the tracker ever returns fewer grasps than it was given)
        def take(arr, shape):
            a = np.full((n_track, *shape), np.nan, dtype=np.float64)
            arr = np.asarray(arr, dtype=np.float64).reshape(-1, *shape)[:n_track]
            a[:len(arr)] = arr
            return a
        n_got = 0 if target_gg is None else len(target_gg)
        if n_got != n_track:
            print(f"[WARN] frame {f}: tracker returned {n_got} of {n_track} grasps")
        rec["frame"].append(f)
        rec["trans_c"].append(take(target_gg.translations if n_got else [], (3,)))
        rec["rot_c"].append(take(target_gg.rotation_matrices if n_got else [], (3, 3)))
        rec["widths"].append(take(target_gg.widths if n_got else [], ()))
        rec["depths"].append(take(target_gg.depths if n_got else [], ()))
        rec["scores"].append(take(target_gg.scores if n_got else [], ()))
        rec["ids"].append(take(np.asarray(grasp_ids).reshape(-1)[:n_track], ()))
        rec["tilt_deg"].append(take(all_tilt[np.asarray(grasp_ids).reshape(-1)[:n_track]], ()))
        rec["violated"].append(take(violated, ()))
        rec["replaced"].append(take(replaced, ()))
        prev_t, prev_r = rec["trans_c"][-1], rec["rot_c"][-1]
        rec["t_build_ms"].append((tb - ta) * 1000.0)
        rec["t_update_ms"].append((tc - tb) * 1000.0)

        if vis is not None:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
            cloud.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
            cloud.transform(flip)
            grippers = target_gg.to_open3d_geometry_list() if n_got else []
            for j, g in enumerate(grippers):  # green: tracked, orange: replaced by the gate, red: outside cone
                if j < len(violated) and replaced[j]:
                    g.paint_uniform_color([1.0, 0.55, 0.0])
                elif j < len(violated) and violated[j]:
                    g.paint_uniform_color([1.0, 0.0, 0.0])
                else:
                    g.paint_uniform_color([0.0, 0.8, 0.0])
                g.transform(flip)
            for g in [cloud, *grippers]:
                vis.add_geometry(g, reset_bounding_box=(f == 0))
            vis.poll_events()
            vis.update_renderer()
            for g in [cloud, *grippers]:
                vis.remove_geometry(g, reset_bounding_box=False)

        if f % 10 == 0:
            print(f"frame {f:03d}: {len(points):7d} pts | build {rec['t_build_ms'][-1]:6.1f} ms | "
                  f"update {rec['t_update_ms'][-1]:6.1f} ms")

        # next frame
        if args.realtime:
            clock = max(clock, t_rec[f]) + (tc - ta)          # can't start before the frame has arrived
            f = max(f + 1, int(np.floor(clock * fps + 1e-9)))  # newest frame available when the tracker is free
        else:
            f += 1

    # ------------------------------------------------------------------ metrics
    idx = np.asarray(rec["frame"])
    trans_c, rot_c = np.stack(rec["trans_c"]), np.stack(rec["rot_c"])     # (F, n, 3), (F, n, 3, 3)
    pos_o, rot_o = np.full_like(trans_c, np.nan), np.full_like(rot_c, np.nan)
    for j, fi in enumerate(idx):  # camera -> world -> target object frame, with the ground truth
        r_wc, p_wc = quat_to_mat(poses["cam_quat_w_ros"][fi]), poses["cam_pos_w"][fi]
        r_wo, p_wo = quat_to_mat(poses[f"{obj}_quat_w"][fi]), poses[f"{obj}_pos_w"][fi]
        p_w = trans_c[j] @ r_wc.T + p_wc
        pos_o[j] = (p_w - p_wo) @ r_wo
        rot_o[j] = r_wo.T @ r_wc @ rot_c[j]

    tilt, viol, repl = (np.stack(rec[k_]) for k_ in ("tilt_deg", "violated", "replaced"))
    per_grasp = []
    for g in range(n_track):
        ok = np.isfinite(pos_o[:, g, 0])
        p, r, wdt = pos_o[ok, g], rot_o[ok, g], np.stack(rec["widths"])[ok, g]
        dev = np.linalg.norm(p - p[0], axis=-1) * 1000.0
        rdev = angle_deg(r[0][None], r)
        per_grasp.append({
            "grasp": g,
            "frames_tracked": int(ok.sum()),
            "score_frame0": round(float(rec["scores"][0][g]), 4),
            "max_tilt_deg": round(float(np.nanmax(tilt[:, g])), 1),
            "frames_outside_cone_by_tracker": int(np.nansum(viol[:, g])),
            "frames_replaced": int(np.nansum(repl[:, g])),
            "jitter_mm": round(float(np.sqrt((p.std(axis=0) ** 2).sum()) * 1000.0), 2),
            "max_dev_mm": round(float(dev.max()), 2),
            "final_dev_mm": round(float(dev[-1]), 2),
            "max_rot_dev_deg": round(float(rdev.max()), 2),
            "width_std_mm": round(float(wdt.std() * 1000.0), 2),
        })

    total = np.asarray(rec["t_build_ms"]) + np.asarray(rec["t_update_ms"])
    upd = np.asarray(rec["t_update_ms"][1:]) if len(idx) > 1 else np.asarray(rec["t_update_ms"])
    p95 = float(np.percentile(total[1:] if len(total) > 1 else total, 95))
    metrics = {
        "seq": str(seq),
        "level": meta.get("level"),
        "filter": args.filter,
        "realtime": args.realtime,
        "fps_recorded": fps,
        "frames_total": n_frames,
        "frames_processed": len(idx),
        "frames_skipped": n_frames - len(idx),
        "num_track": n_track,
        "cone": {"axis": args.cone_axis, "max_tilt_deg": args.max_tilt, "enforced_every_frame": gate,
                 "max_jump_mm": args.max_jump_mm or None},
        "latency_ms": {  # frame 0 left out of the stats: it includes the first detection / warm-up
            "frame0_total": round(float(total[0]), 1),
            "update_mean": round(float(upd.mean()), 1),
            "total_mean": round(float(total[1:].mean() if len(total) > 1 else total.mean()), 1),
            "total_p95": round(p95, 1),
            "total_max": round(float(total.max()), 1),
        },
        "sustainable_fps_p95": round(1000.0 / p95, 1) if p95 > 0 else None,
        "summary": {
            "jitter_mm_mean": round(float(np.mean([g["jitter_mm"] for g in per_grasp])), 2),
            "max_dev_mm": round(float(np.max([g["max_dev_mm"] for g in per_grasp])), 2),
            "max_rot_dev_deg": round(float(np.max([g["max_rot_dev_deg"] for g in per_grasp])), 2),
            "max_tilt_deg": round(float(np.nanmax(tilt)), 1),
            "violations": int(np.nansum(viol)),
            "replacements": int(np.nansum(repl)),
        },
        "per_grasp": per_grasp,
    }
    np.savez(out_dir / "tracks.npz", frame=idx, t=t_rec[idx], trans_c=trans_c, rot_c=rot_c, pos_obj=pos_o,
             rot_obj=rot_o, **{k_: np.asarray(rec[k_]) for k_ in ("widths", "depths", "scores", "ids", "tilt_deg",
                                                                   "violated", "replaced", "t_build_ms",
                                                                   "t_update_ms")})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n[SUMMARY] {len(idx)}/{n_frames} frames processed | {n_track} grasps tracked")
    for g in per_grasp:
        print(f"  grasp {g['grasp']}: jitter {g['jitter_mm']:5.2f} mm | max dev {g['max_dev_mm']:5.2f} mm, "
              f"{g['max_rot_dev_deg']:5.2f} deg | width std {g['width_std_mm']:4.2f} mm | "
              f"max tilt {g['max_tilt_deg']:4.1f} deg | left cone {g['frames_outside_cone_by_tracker']}x, "
              f"replaced {g['frames_replaced']}x")
    lat = metrics["latency_ms"]
    print(f"  latency: update {lat['update_mean']} ms, total {lat['total_mean']} ms mean / {lat['total_p95']} ms p95"
          f" -> ~{metrics['sustainable_fps_p95']} fps sustainable (recorded at {fps})")
    print(f"[INFO] saved -> {out_dir}")
    if vis is not None:
        vis.destroy_window()


if __name__ == "__main__":
    main()
