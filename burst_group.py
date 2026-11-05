"""Grouping of ALL detector grasps over a burst / sequence, with optional PNG mask and optional traceability.

Works with real camera or simulation data. The input folder is detected automatically:
  clouds    cloud_XXX.npy                          (N, 3) / (N, 6) or organised (H, W, 3|6); metres or mm
  sim seq   meta.json + frames/color_XXX.png + frames/depth_XXX.npy        (capture_sequence.py; K from meta.json)
  rgbd      color_XXX.png + depth_XXX.npy|png      (depth uint16 = mm, float = m; K from --calib or a camera.json)
  api       *.zip with v01/color.png + v01/depth.npy + v01/camera.json, or sub-folders with those three files
            (the pipeFit API's native scene format; K from each camera.json)

Steps
  1. detect     AnyGrasp detector on every frame (or --detections: reuse a detections.npz saved by a previous run)
  2. mask       optional: keep only grasps (or points) inside a PNG mask - one png for all frames, or a folder with
                mask_XXX.png per frame (for a sim sequence: <seq>/frames). White = keep.
  3. group      detect_burst.py rules: best-first, nearest running centroid within group_mm and group_deg
  4. trace      optional (--trace / --no_trace): for every grasp of the start frame, share of the other frames with a
                re-detection within match_mm / match_deg; per group the mean of its start grasps and the trace of
                the group representative over all frames
Output (<input>/burst_group/ or --out): groups.csv, grasps.csv, detections.npz (reuse), run.json (config + timing)

Environment: anygrasp. Examples (project root):
    PY=~/miniforge3/envs/anygrasp/bin/python
    $PY anygrasp/burst_group.py --input ~/Time_consistency/burst                                    # real, whole scene
    $PY anygrasp/burst_group.py --input ~/Time_consistency/burst --calib camera_calibration_1280x720.json \
        --mask tubes_mask.png                                                                      # real, masked
    $PY anygrasp/burst_group.py --input runs/multi/seed_100/seq_0000 --mask runs/multi/seed_100/seq_0000/frames
    ... --no_trace                                    # grouping only (compare time and results)
    ... --detections <out>/detections.npz            # skip the detector, regroup / retrace with other settings
Needs group_and_trace.py next to this script (shared geometry, detector, grouping and trace code).
"""

# =====================================================================================================
# CONFIG: edit here
# =====================================================================================================
CONFIG = {
    "DETECTOR": {
        "workspace": [(-0.40, 0.40), (-0.40, 0.40), (0.05, 1.00)],  # crop, metres, camera frame
        "voxel": 0.005,                    # voxel size in metres (0 = none)
        "max_gripper_width": 0.10,         # SDK valid range (0, 0.1]
        "gripper_height": 0.03,
        "collision_detection": True,
        "nms": True,
        "top_k": 0,                        # grasps kept per frame, best first; 0 = all
        "max_frames": 0,                   # 0 = all frames
    },
    "MASK": {
        "apply_to": "grasps",              # "grasps": detect on the whole scene (keeps collision context), keep grasps
                                           #           whose centre falls in the mask
                                           # "points": remove the points outside the mask BEFORE detection
        "dilate_px": 8,                    # margin around the mask, pixels
    },
    "GROUPING": {
        "group_mm": 20.0,
        "group_deg": 45.0,                 # None = position only
        "rank_mode": "penalized",          # "avg" | "penalized"
        "min_support": 0.0,
    },
    "TRACE": {
        "enabled": True,                   # default; --trace / --no_trace override
        "start_frame": 0,
        "match_mm": 20.0,
        "match_deg": 30.0,
    },
}
# =====================================================================================================

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import group_and_trace as gt  # noqa: E402


# --------------------------------------------------------------- intrinsics
def read_k(path: Path):
    """(K, (W, H) or None) from camera_calibration.json, meta.json or camera.json."""
    d = json.loads(Path(path).read_text())
    cam = d.get("camera", d) if isinstance(d.get("camera"), dict) else d
    k = np.asarray(cam["K"], float)
    res = cam.get("resolution") or ([cam["width"], cam["height"]] if "width" in cam else None)
    return k, (tuple(res) if res else None)


def scale_k(k, res, shape_hw):
    """K for an image of shape_hw, from K given at resolution res (same sensor)."""
    if res is None or (res[1], res[0]) == tuple(shape_hw):
        return k
    sx, sy = shape_hw[1] / res[0], shape_hw[0] / res[1]
    k2 = k.copy()
    k2[0, 0] *= sx; k2[1, 1] *= sy
    k2[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
    k2[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
    return k2


# --------------------------------------------------------------- input
def read_image(src):
    from PIL import Image
    return np.asarray(Image.open(src))


def depth_to_metres(d):
    d = np.asarray(d)
    return d.astype(np.float64) / 1000.0 if np.issubdtype(d.dtype, np.integer) or np.nanmax(d) > 20 else \
        d.astype(np.float64)


def backproject(depth_m, rgb, k):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    xyz = np.stack([(u - k[0, 2]) * depth_m / k[0, 0], (v - k[1, 2]) * depth_m / k[1, 1], depth_m], axis=-1)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if rgb is None:
        col = np.full((valid.sum(), 3), 0.6)
    else:
        col = rgb[..., :3].astype(np.float64)[valid] / 255.0
    return xyz[valid], col


def load_input(folder: Path, calib):
    """(kind, frames [(f, xyz, rgb)], K or None, K resolution or None)."""
    calib_k, calib_res = read_k(calib) if calib else (None, None)

    clouds = sorted(folder.glob("cloud_*.npy"))
    if clouds:
        frames = []
        for f, p in enumerate(clouds):
            d = np.load(p).astype(np.float64)
            d = d.reshape(-1, d.shape[-1])
            xyz = d[:, :3] / (1000.0 if np.nanmax(np.abs(d[:, :3])) > 20 else 1.0)
            rgb = d[:, 3:6] if d.shape[1] >= 6 else np.full((len(d), 3), 0.6)
            rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
            keep = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
            frames.append((f, xyz[keep], rgb[keep]))
        return "clouds", frames, calib_k, calib_res

    if (folder / "meta.json").exists() and (folder / "frames").is_dir():
        frames, k = gt.frames_from_seq(folder)
        meta = json.loads((folder / "meta.json").read_text())
        res = None
        c0 = folder / "frames" / "color_000.png"
        if c0.exists():
            h, w = read_image(c0).shape[:2]
            res = (w, h)
        return "sim seq", frames, k, res

    colors = sorted(folder.glob("color_*.png"))
    if colors:
        k, res = calib_k, calib_res
        if k is None:
            for name in ("camera.json", "meta.json"):
                if (folder / name).exists():
                    k, res = read_k(folder / name)
                    break
        if k is None:
            raise SystemExit("rgbd input needs --calib (or a camera.json / meta.json in the folder) for K")
        frames = []
        for f, c in enumerate(colors):
            stem = c.stem.replace("color_", "")
            dp = next((p for p in (folder / f"depth_{stem}.npy", folder / f"depth_{stem}.png") if p.exists()), None)
            if dp is None:
                raise SystemExit(f"no depth_{stem}.npy / .png for {c.name}")
            depth = depth_to_metres(np.load(dp) if dp.suffix == ".npy" else read_image(dp))
            rgb = read_image(c)
            kk = scale_k(k, res, depth.shape)
            frames.append((f, *backproject(depth, rgb, kk)))
        return "rgbd", frames, k, res if res else (depth.shape[1], depth.shape[0])

    scenes = sorted(folder.glob("*.zip")) or sorted(p for p in folder.iterdir()
                                                     if p.is_dir() and any(p.rglob("camera.json")))
    if scenes:
        frames, k, res = [], None, None
        for f, s in enumerate(scenes):
            if s.suffix == ".zip":
                z = zipfile.ZipFile(s)
                names = z.namelist()
                pick = lambda end: next(n for n in names if n.endswith(end))  # noqa: E731
                cam = json.loads(z.read(pick("camera.json")))
                depth = depth_to_metres(np.load(io.BytesIO(z.read(pick("depth.npy")))))
                rgb = read_image(io.BytesIO(z.read(pick("color.png"))))
            else:
                cj = next(s.rglob("camera.json"))
                cam = json.loads(cj.read_text())
                depth = depth_to_metres(np.load(cj.parent / "depth.npy"))
                rgb = read_image(cj.parent / "color.png")
            k = np.asarray(cam["K"], float)
            res = (depth.shape[1], depth.shape[0])
            frames.append((f, *backproject(depth, rgb, k)))
        return "api scenes", frames, k, res

    raise SystemExit(f"{folder}: no cloud_XXX.npy, sim sequence, color_XXX.png + depth_XXX, or API scenes found")


# --------------------------------------------------------------- mask
class PngMask:
    """White = keep. One png for every frame, or a folder with mask_XXX.png."""

    def __init__(self, src: Path, k, res, dilate_px):
        if k is None:
            raise SystemExit("a PNG mask needs the intrinsics: pass --calib (camera_calibration.json) for clouds")
        self.src, self.k, self.res, self.px, self.cache = src, k, res, dilate_px, {}

    def image(self, f):
        if f not in self.cache:
            p = self.src / f"mask_{f:03d}.png" if self.src.is_dir() else self.src
            if not p.exists():
                raise SystemExit(f"mask not found: {p}")
            m = gt.dilate(gt.load_png_mask(p), self.px) if self.px else gt.load_png_mask(p)
            self.cache[f] = (m, scale_k(self.k, self.res, m.shape))
        return self.cache[f]

    def inside(self, f, xyz):
        m, k = self.image(f)
        xyz = np.atleast_2d(xyz)
        z = np.where(xyz[:, 2] > 0, xyz[:, 2], np.nan)
        with np.errstate(invalid="ignore"):
            u = np.rint(k[0, 0] * xyz[:, 0] / z + k[0, 2])
            v = np.rint(k[1, 1] * xyz[:, 1] / z + k[1, 2])
        ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
        out = np.zeros(len(xyz), bool)
        out[ok] = m[v[ok].astype(int), u[ok].astype(int)]
        return out


# --------------------------------------------------------------- detections cache
def save_detections(path, cands, n_frames):
    np.savez_compressed(path, frame=np.array([c["frame"] for c in cands], int),
                        rank=np.array([c["rank"] for c in cands], int),
                        score=np.array([c["score"] for c in cands]), width=np.array([c["width"] for c in cands]),
                        depth=np.array([c["depth"] for c in cands]),
                        pos=np.array([c["pos"] for c in cands]).reshape(-1, 3),
                        R=np.array([c["R"] for c in cands]).reshape(-1, 3, 3), n_frames=n_frames)


def load_detections(path):
    z = np.load(path)
    cands = [{"frame": int(z["frame"][i]), "rank": int(z["rank"][i]), "score": float(z["score"][i]),
              "width": float(z["width"][i]), "depth": float(z["depth"][i]), "pos": z["pos"][i], "R": z["R"][i]}
             for i in range(len(z["frame"]))]
    return cands, int(z["n_frames"])


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="grouping of all grasps of a burst (+ optional mask and traceability)")
    ap.add_argument("--input", required=True, help="folder: clouds, sim sequence, color+depth pngs/npys or API scenes")
    ap.add_argument("--calib", help="json with K (camera_calibration.json / meta.json / camera.json)")
    ap.add_argument("--mask", help="PNG mask (white = keep) or a folder with mask_XXX.png; omit = whole scene")
    ap.add_argument("--mask_to", choices=["grasps", "points"], help="override CONFIG MASK apply_to")
    tr = ap.add_mutually_exclusive_group()
    tr.add_argument("--trace", dest="trace", action="store_true", default=None, help="compute traceability")
    tr.add_argument("--no_trace", dest="trace", action="store_false", help="grouping only")
    ap.add_argument("--detections", help="detections.npz of a previous run: no detector")
    ap.add_argument("--out", help="output folder (default <input>/burst_group)")
    args = ap.parse_args()

    cfg = CONFIG
    do_trace = cfg["TRACE"]["enabled"] if args.trace is None else args.trace
    mask_to = args.mask_to or cfg["MASK"]["apply_to"]
    folder = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else folder / "burst_group"
    out.mkdir(parents=True, exist_ok=True)
    timing = {}

    # ---- 1. input + detection
    t0 = time.perf_counter()
    need_frames = not args.detections or (args.mask and mask_to == "points")
    k = res = None
    kind = "detections.npz"
    if need_frames or args.mask:
        kind, frames, k, res = load_input(folder, args.calib)
        if cfg["DETECTOR"]["max_frames"]:
            frames = frames[:cfg["DETECTOR"]["max_frames"]]
        print(f"[input] {kind}: {len(frames)} frames, {int(np.mean([len(x) for _, x, _ in frames]))} points/frame"
              + (f", K fx {k[0, 0]:.1f} at {res}" if k is not None else ""))
    timing["load_s"] = time.perf_counter() - t0

    mask = PngMask(Path(args.mask).expanduser().resolve(), k, res, cfg["MASK"]["dilate_px"]) if args.mask else None
    if mask and mask_to == "points":
        if args.detections:
            raise SystemExit("--mask_to points changes the detections: run without --detections")
        frames = [(f, x[mask.inside(f, x)], c[mask.inside(f, x)]) for f, x, c in frames]
        print(f"[mask] points: {int(np.mean([len(x) for _, x, _ in frames]))} points/frame left inside the mask")

    t0 = time.perf_counter()
    if args.detections:
        cands, n_frames = load_detections(Path(args.detections).expanduser())
        print(f"[detect] reused {args.detections}: {len(cands)} grasps over {n_frames} frames")
    else:
        n_frames = len(frames)
        print(f"[detect] AnyGrasp on {n_frames} frames (collision check "
              f"{'on' if cfg['DETECTOR']['collision_detection'] else 'OFF'})")
        cands = gt.run_detector(frames, cfg["DETECTOR"])
        save_detections(out / "detections.npz", cands, n_frames)
    timing["detect_s"] = time.perf_counter() - t0

    # ---- 2. mask on grasp centres
    n_all = len(cands)
    if mask and mask_to == "grasps":
        keep = np.zeros(len(cands), bool)
        for f in sorted({c["frame"] for c in cands}):
            idx = [i for i, c in enumerate(cands) if c["frame"] == f]
            keep[idx] = mask.inside(f, np.array([cands[i]["pos"] for i in idx]))
        cands = [c for c, kp in zip(cands, keep) if kp]
    by_frame = {}
    for c in cands:
        by_frame.setdefault(c["frame"], []).append(c)
    for fr in by_frame.values():
        for r, c in enumerate(sorted(fr, key=lambda c: c["rank"])):
            c["rank"] = r + 1
    print(f"[mask] {('png on ' + mask_to) if mask else 'none (whole scene)'}: {len(cands)}/{n_all} grasps kept")
    if not cands:
        raise SystemExit("no grasp left")

    # ---- 3. grouping
    t0 = time.perf_counter()
    groups = gt.group(cands, n_frames, cfg["GROUPING"])
    timing["group_s"] = time.perf_counter() - t0

    # ---- 4. traceability
    t_cfg = cfg["TRACE"]
    s = t_cfg["start_frame"]
    start = sorted(by_frame.get(s, []), key=lambda c: c["rank"])
    t0 = time.perf_counter()
    grasp_rows = []
    for i, c in enumerate(start):
        g, _ = gt.find_group(groups, c["pos"], c["R"], cfg["GROUPING"]["group_mm"], cfg["GROUPING"]["group_deg"])
        row = {"grasp": i, "rank": c["rank"], "score": round(c["score"], 4), "group": g["group"] if g else None,
               "group_support": round(g["support"], 3) if g else None}
        if do_trace:
            t, mm, _ = gt.trace(c["pos"], c["R"], by_frame, [f for f in range(n_frames) if f != s], t_cfg)
            row.update(trace=round(t, 3), match_mm=None if mm is None else round(mm, 2))
        row.update(width_mm=round(c["width"] * 1000, 1), tilt_deg=round(gt.tilt_deg(c["R"]), 1),
                   x=round(float(c["pos"][0]), 4), y=round(float(c["pos"][1]), 4), z=round(float(c["pos"][2]), 4))
        grasp_rows.append(row)
    group_rows = []
    for g in groups:
        row = {"group": g["group"], "members": g["members"], "frames_hit": g["frames_hit"],
               "support": round(g["support"], 3), "avg_rank": round(g["avg_rank"], 2),
               "penalized_rank": round(g["penalized_rank"], 2), "avg_score": round(g["avg_score"], 4),
               "max_score": round(g["max_score"], 4), "spread_mm": round(g["spread_mm"], 1),
               "rot_spread_deg": round(g["rot_spread_deg"], 1)}
        if do_trace:
            st = [r["trace"] for r in grasp_rows if r["group"] == g["group"]]
            rep, _, _ = gt.trace(g["centre"], g["R"], by_frame, list(range(n_frames)), t_cfg)
            row.update(start_grasps=len(st), trace_mean=round(float(np.mean(st)), 3) if st else None,
                       trace_max=round(float(np.max(st)), 3) if st else None, rep_trace=round(rep, 3))
        row.update(width_mm=round(g["width"] * 1000, 1), tilt_deg=round(g["tilt_deg"], 1),
                   x=round(float(g["centre"][0]), 4), y=round(float(g["centre"][1]), 4),
                   z=round(float(g["centre"][2]), 4), rotation_matrix=json.dumps(np.round(g["R"], 5).tolist()))
        group_rows.append(row)
    timing["trace_s"] = (time.perf_counter() - t0) if do_trace else 0.0

    # ---- report
    print(f"\nGROUPS: {len(groups)} (radius {cfg['GROUPING']['group_mm']:.0f} mm, angle "
          f"{cfg['GROUPING']['group_deg']}, rank {cfg['GROUPING']['rank_mode']}) - top 15")
    hdr = f"  {'grp':>4} {'memb':>5} {'frames':>7} {'supp':>5} {'pen rk':>6} {'score':>6} {'spread':>7} {'rot':>4}"
    print(hdr + ("  trace  rep_tr" if do_trace else ""))
    for r in group_rows[:15]:
        line = (f"  {r['group']:>4} {r['members']:>5} {r['frames_hit']:>3}/{n_frames:<3} {r['support']:5.2f} "
                f"{r['penalized_rank']:6.1f} {r['avg_score']:6.3f} {r['spread_mm']:5.1f}mm {r['rot_spread_deg']:3.0f}°")
        if do_trace:
            line += f"  {('%.2f' % r['trace_mean']) if r['trace_mean'] is not None else '   -':>5}  {r['rep_trace']:6.2f}"
        print(line)
    if do_trace and grasp_rows:
        trs = np.array([r["trace"] for r in grasp_rows])
        sc = np.array([r["score"] for r in grasp_rows])
        print(f"\nTRACE of the {len(grasp_rows)} grasps of frame {s}: mean {trs.mean():.2f}, "
              f"{(trs >= 0.5).mean() * 100:.0f}% >= 0.5, {(trs >= 0.8).mean() * 100:.0f}% >= 0.8"
              + (f", corr(score, trace) {np.corrcoef(sc, trs)[0, 1]:+.2f}" if len(trs) > 2 and trs.std() > 0 else ""))
    tot = sum(timing.values())
    print("\nTIME  " + "  ".join(f"{k[:-2]} {v:.2f} s" for k, v in timing.items()) + f"  | total {tot:.2f} s"
          + (f"  (trace = {100 * timing['trace_s'] / tot:.0f}% of it)" if do_trace and tot else ""))

    write_csv(out / "groups.csv", group_rows)
    write_csv(out / "grasps.csv", grasp_rows)
    (out / "run.json").write_text(json.dumps({
        "input": str(folder), "kind": kind, "frames": n_frames, "mask": args.mask, "mask_to": mask_to if args.mask
        else None, "trace": do_trace, "detections_reused": args.detections, "grasps_total": n_all,
        "grasps_kept": len(cands), "groups": len(groups), "timing_s": {k: round(v, 3) for k, v in timing.items()},
        "config": cfg}, indent=2, default=str))
    print(f"\n[out] {out}: groups.csv, grasps.csv, detections.npz, run.json")


if __name__ == "__main__":
    main()
