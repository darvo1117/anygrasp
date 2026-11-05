"""Grouping + traceability of ALL AnyGrasp detector grasps over a sequence of frames, real or sim.

No API, no robot, no camera pose: frames in -> detector on every frame -> groups (+ trace) -> two CSV tables.

INPUT (--input, one folder; the format is detected by itself):
  a) clouds        cloud_XXX.npy     (N, 3) or (N, 6) xyz [+ rgb]  (capture_burst.py; metres or mm)
                                     (H, W, 3) or (H, W, 6)        organised cloud (one point per pixel)
  b) images        color_XXX.png + depth_XXX.npy|png  (also inside a frames/ subfolder, as capture_sequence.py)
                   K from meta.json / camera.json next to them (needed to turn depth into points)
  c) API scenes    *.zip or sub-folders holding color.png + depth.npy + camera.json   (one scene per frame)

MASK (optional, --mask PNG or a folder with mask_XXX.png; white = keep; without it: the whole scene):
  apply_to "grasps"  the detector sees the whole scene (its collision check is not affected); only grasps whose
                     centre falls in the mask are kept                                            (default)
  apply_to "points"  the cloud is cut to the mask before detection (faster, but the collision check no longer
                     sees what is outside the mask)
  Which pixel a point / grasp falls on: automatic for images, API scenes and organised clouds; (N, 6) clouds need
  a K: --K file.json (any json with "K" or camera.K, e.g. camera_calibration_1280x720.json), scaled to the mask size

TRACKING (on by default; --no_track = grouping only, to compare): geometric traceability. Every grasp is searched
  for in every OTHER frame: a detection there within MATCH_MM and MATCH_DEG of its pose counts as re-detected.

OUTPUT (--out, default <input>/seq_group_track/): two tables, overwritten on every run (use --out per experiment)
  grasps.csv   one row per grasp (every frame, after the mask)
    grasp_id        row id
    frame, rank     frame it was detected in; its place in that frame (1 = best score)
    score           AnyGrasp score
    group           group it joined (= groups.csv group); empty if its group was dropped by min_support
    group_best      True for the grasp that defines its group (the group's orientation is this grasp's)
    trace           [tracking] share of the other frames in which it is re-detected (0..1)
    trace_hits      [tracking] number of those frames
    width_mm, depth_mm, tilt_deg    gripper opening, insertion depth, approach angle vs the camera axis
    x, y, z         grasp centre, metres, camera frame (OpenCV: z forward, x right, y down)
    r00 .. r22      rotation matrix (column 0 = approach, column 1 = closing direction)
  groups.csv   one row per group, best first (penalized_rank, then support)
    group           id
    n_grasps        grasps in the group, all frames together
    frames_hit, support   frames with at least one of its grasps; that share of all frames (0..1)
    avg_rank, penalized_rank   mean rank of its grasps; the same with missing frames counted as the worst rank
    avg_score, max_score  AnyGrasp score of its grasps
    spread_mm, rot_spread_deg   distance of its grasps to the centre (rms) / angle to the best grasp (mean)
    trace_rep       [tracking] share of all frames in which the group's representative pose is detected
    trace_mean, trace_max   [tracking] mean / max trace of its grasps
    width_mm, depth_mm, tilt_deg, x, y, z, r00..r22   representative grasp: the centre of its grasps with the
                    orientation of its best grasp
  (<out>/.cache/detections.npz: the detector's output, reused when input + detector settings are the same)

Environment: anygrasp. Needs group_and_trace.py next to it (and compare_burst_tracking.py for --vis).
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/seq_group_track.py --input ~/Time_consistency/burst
    ... --input runs/multi/seed_100/seq_0000 --mask runs/multi/seed_100/seq_0000/frames --out results/sim_mask
    ... --input ~/Time_consistency/burst --mask tubes.png --K ~/Time_consistency/camera_calibration_1280x720.json
    ... --no_track          grouping only (timings printed at the end: compare with the tracking run)
"""

# =====================================================================================================
# CONFIG: edit here
# =====================================================================================================
CONFIG = {
    "INPUT": {
        "cloud_glob": "cloud_*.npy",       # clouds: which files (sorted by name = frame order)
        "max_frames": 0,                   # 0 = all
        "depth_scale": None,               # images: depth units per metre; None = auto (uint16 / >20 -> mm, else m)
    },
    "DETECTOR": {
        "workspace": [(-0.4, 0.4), (-0.4, 0.4), (0.05, 1.0)],  # x, y, z crop in metres, camera frame
        "voxel": 0.005,                    # m, voxel downsampling before detection
        "max_gripper_width": 0.10,         # m; the SDK accepts (0, 0.1]
        "gripper_height": 0.03,
        "collision_detection": True,
        "nms": True,
        "top_k": 0,                        # grasps kept per frame (best first); 0 = all
        "reuse_cache": True,               # reuse <out>/.cache/detections.npz if same input + settings
    },
    "MASK": {
        "apply_to": "grasps",              # "grasps" (filter grasp centres) | "points" (cut the cloud first)
        "dilate_px": 8,                    # margin around the mask, pixels
        "K": None,                         # (N, 6) clouds only: 3x3 intrinsics, e.g. [[fx,0,cx],[0,fy,cy],[0,0,1]]
        "K_size": None,                    # (W, H) that K belongs to; None = the mask size
    },
    "GROUPING": {
        "group_mm": 20.0,                  # max distance from a grasp to a group's centre
        "group_deg": 45.0,                 # max angle to the group's first (best) member; None = position only
        "rank_mode": "penalized",          # "avg" | "penalized" (missing frames count as the worst rank)
        "min_support": 0.0,                # drop groups seen in fewer than this share of frames
    },
    "TRACKING": {
        "enabled": True,                   # --no_track / --track override
        "match_mm": 20.0,                  # re-detected in a frame = a detection there within this distance
        "match_deg": 30.0,                 # ... AND this angle (180 deg flip about the approach = same grasp)
    },
    "SUPER": {                             # super candidates (terminal only, not in the CSV)
        "n": 10,                           # how many: one per group, groups taken by frequency
        "max_tilt_deg": 30.0,              # approach vs the camera's z axis (0 = straight along the view)
        "min_trace": 0.5,                  # and re-detected in at least this share of the other frames
    },                                     # (0 = no trace condition; ignored with --no_track)
                                           # a group whose best grasp fails goes to its 2nd, 3rd ... best grasp;
                                           # a group with none that passes is skipped
    "VIS": {"show_groups": 10, "members": True, "gripper_mm": 3.0, "point_mm": 4.0},
}
# =====================================================================================================

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import group_and_trace as gat  # noqa: E402  (detector and grouping rules shared with it)


# --------------------------------------------------------------- input
def read_image(data):
    from PIL import Image
    return np.asarray(Image.open(data if isinstance(data, (str, Path)) else io.BytesIO(data)))


def depth_to_m(d, scale):
    d = np.asarray(d)
    if scale is None:
        scale = 1000.0 if (np.issubdtype(d.dtype, np.integer) or np.nanmax(d) > 20) else 1.0
    return d.astype(np.float64) / scale


def k_from_json(path: Path):
    d = json.loads(Path(path).read_text())
    k = d.get("K") or (d.get("camera") or {}).get("K")
    return None if k is None else np.asarray(k, float)


def find_k(*folders):
    for f in folders:
        for name in ("meta.json", "camera.json", "camera_info.json"):
            if (f / name).exists():
                k = k_from_json(f / name)
                if k is not None:
                    return k
    return None


def frame_from_depth(f, rgb, depth_m, k):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    xyz = np.stack([(u - k[0, 2]) * depth_m / k[0, 0], (v - k[1, 2]) * depth_m / k[1, 1], depth_m], -1)
    ok = np.isfinite(depth_m) & (depth_m > 0)
    col = rgb[..., :3].astype(np.float64) / 255.0 if rgb is not None else np.full((h, w, 3), 0.6)
    return {"frame": f, "xyz": xyz[ok], "rgb": col[ok], "pix": np.stack([u[ok], v[ok]], 1), "size": (w, h), "K": k}


def load_frames(inp: Path, cfg):
    """[{frame, xyz (N,3) m, rgb (N,3) 0..1, pix (N,2) or None, size (W,H) or None, K or None}], kind."""
    icfg = cfg["INPUT"]
    clouds = sorted(inp.glob(icfg["cloud_glob"]))
    img_dir = inp / "frames" if (inp / "frames").is_dir() else inp
    colors = sorted(img_dir.glob("color_*.png"))
    scenes = sorted(p for p in inp.iterdir() if p.suffix == ".zip" or (p.is_dir() and (p / "depth.npy").exists()))
    out = []
    if clouds:
        kind = "clouds"
        for f, p in enumerate(clouds):
            d = np.load(p).astype(np.float64)
            if d.ndim == 3:                                   # organised (H, W, C)
                h, w = d.shape[:2]
                u, v = np.meshgrid(np.arange(w), np.arange(h))
                d, pix, size = d.reshape(-1, d.shape[2]), np.stack([u.ravel(), v.ravel()], 1), (w, h)
                kind = "organised clouds"
            else:
                pix, size = None, None
            xyz = d[:, :3] / (1000.0 if np.nanmax(np.abs(d[:, :3])) > 20 else 1.0)
            rgb = d[:, 3:6] if d.shape[1] >= 6 else np.full((len(d), 3), 0.6)
            rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
            ok = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
            fr = {"frame": f, "xyz": xyz[ok], "rgb": rgb[ok], "pix": None if pix is None else pix[ok], "size": size,
                  "K": None}
            if pix is not None:                               # K of an organised cloud: fit u = fx x/z + cx
                a = np.c_[fr["xyz"][:, 0] / fr["xyz"][:, 2], np.ones(ok.sum())]
                fx, cx = np.linalg.lstsq(a, fr["pix"][:, 0], rcond=None)[0]
                b = np.c_[fr["xyz"][:, 1] / fr["xyz"][:, 2], np.ones(ok.sum())]
                fy, cy = np.linalg.lstsq(b, fr["pix"][:, 1], rcond=None)[0]
                fr["K"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            out.append(fr)
    elif colors:
        kind = "images"
        k = find_k(img_dir, inp, inp.parent)
        if k is None:
            raise SystemExit(f"images need a K (meta.json / camera.json with \"K\") in {img_dir} or {inp}")
        for cp in colors:
            idx = cp.stem.split("_")[-1]
            dp = next((q for q in (img_dir / f"depth_{idx}.npy", img_dir / f"depth_{idx}.png") if q.exists()), None)
            if dp is None:
                print(f"[WARN] no depth for {cp.name}, skipped")
                continue
            depth = np.load(dp) if dp.suffix == ".npy" else read_image(dp)
            out.append(frame_from_depth(len(out), read_image(cp), depth_to_m(depth, icfg["depth_scale"]), k))
    elif scenes:
        kind = "API scenes"
        for f, p in enumerate(scenes):
            if p.suffix == ".zip":
                z = zipfile.ZipFile(p)
                name = lambda n: next(x for x in z.namelist() if x.endswith(n))  # noqa: E731
                rgb, depth = read_image(z.read(name("color.png"))), np.load(io.BytesIO(z.read(name("depth.npy"))))
                k = np.asarray(json.loads(z.read(name("camera.json")))["K"], float)
            else:
                rgb, depth, k = read_image(p / "color.png"), np.load(p / "depth.npy"), k_from_json(p / "camera.json")
            out.append(frame_from_depth(f, rgb, depth_to_m(depth, icfg["depth_scale"]), k))
    else:
        raise SystemExit(f"nothing I can read in {inp}: no {icfg['cloud_glob']}, no color_XXX.png + depth_XXX, "
                         "no API scenes (*.zip / folders with color.png + depth.npy + camera.json)")
    if icfg["max_frames"]:
        out = out[:icfg["max_frames"]]
    return out, kind


# --------------------------------------------------------------- mask
class PngMask:
    """White = keep. One PNG for every frame, or a folder with mask_XXX.png (one per frame)."""

    def __init__(self, path: Path, cfg, k_override):
        self.path, self.cfg, self.cache = path, cfg, {}
        self.k_cfg = k_override if k_override is not None else (
            None if cfg["K"] is None else np.asarray(cfg["K"], float))

    def image(self, f):
        if f not in self.cache:
            p = self.path / f"mask_{f:03d}.png" if self.path.is_dir() else self.path
            if not p.exists():
                raise SystemExit(f"mask {p} not found")
            m = read_image(p)
            m = (m[..., 0] if m.ndim == 3 else m) > 127
            self.cache[f] = gat.dilate(m, self.cfg["dilate_px"]) if self.cfg["dilate_px"] else m
        return self.cache[f]

    def k_for(self, fr, m):
        """K scaled to the mask's size."""
        h, w = m.shape
        if fr["K"] is not None:
            k, (w0, h0) = fr["K"], fr["size"]
        elif self.k_cfg is not None:
            k = self.k_cfg
            w0, h0 = self.cfg["K_size"] or (w, h)
        else:
            raise SystemExit("this input is an unorganised (N, 6) cloud: the mask needs a K to know which pixel a "
                             "point falls on. Give --K camera_calibration_WxH.json or CONFIG MASK 'K'.")
        if (w0, h0) == (w, h):
            return k
        sx, sy = w / w0, h / h0
        return np.array([[k[0, 0] * sx, 0, (k[0, 2] + 0.5) * sx - 0.5],
                         [0, k[1, 1] * sy, (k[1, 2] + 0.5) * sy - 0.5], [0, 0, 1.0]])

    def hit_xyz(self, fr, xyz):
        m = self.image(fr["frame"])
        k = self.k_for(fr, m)
        z = np.where(xyz[:, 2] > 0, xyz[:, 2], np.nan)
        u = np.rint(k[0, 0] * xyz[:, 0] / z + k[0, 2])
        v = np.rint(k[1, 1] * xyz[:, 1] / z + k[1, 2])
        ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
        hit = np.zeros(len(xyz), bool)
        hit[ok] = m[v[ok].astype(int), u[ok].astype(int)]
        return hit

    def hit_points(self, fr):
        m = self.image(fr["frame"])
        if fr["pix"] is not None and fr["size"] == (m.shape[1], m.shape[0]):
            return m[fr["pix"][:, 1], fr["pix"][:, 0]]
        return self.hit_xyz(fr, fr["xyz"])


# --------------------------------------------------------------- detections (+ cache)
def cache_key(inp, frames, cfg, mask_path):
    pts = cfg["MASK"]["apply_to"] == "points"
    h = hashlib.sha1(json.dumps({"in": str(inp), "n": len(frames), "pts": [len(f["xyz"]) for f in frames],
                                 "det": cfg["DETECTOR"], "mask": str(mask_path) if pts else None,
                                 "mcfg": cfg["MASK"] if pts else None}, default=str, sort_keys=True).encode())
    return h.hexdigest()[:16]


def save_cache(path, key, cands):
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")   # atomic: a parallel run never reads half a file
    np.savez_compressed(tmp, key=key, frame=[c["frame"] for c in cands], rank=[c["rank"] for c in cands],
                        score=[c["score"] for c in cands], width=[c["width"] for c in cands],
                        depth=[c["depth"] for c in cands], pos=np.array([c["pos"] for c in cands]).reshape(-1, 3),
                        R=np.array([c["R"] for c in cands]).reshape(-1, 3, 3))
    os.replace(tmp, path)


def load_cache(path, key):
    if not path.exists():
        return None
    with np.load(path) as zf:                 # read every array ONCE (z["x"] on an npz decompresses it again)
        z = {k: zf[k] for k in zf.files}
    if str(z["key"]) != key:
        return None
    return [{"frame": int(z["frame"][i]), "rank": int(z["rank"][i]), "score": float(z["score"][i]),
             "width": float(z["width"][i]), "depth": float(z["depth"][i]), "pos": z["pos"][i], "R": z["R"][i]}
            for i in range(len(z["frame"]))]


# --------------------------------------------------------------- tracking
def redetected(Pq, Rq, P, R, F, frame_ids, cfg):
    """(len(Pq), len(frame_ids)) bool: frame f has a detection within match_mm AND match_deg of query q.
    Same rule as group_and_trace.trace (angle flip-aware: 180 deg about the approach = same grasp), vectorised."""
    out = np.zeros((len(Pq), len(frame_ids)), bool)
    cos_min = np.cos(np.radians(cfg["match_deg"]))
    for k, f in enumerate(frame_ids):
        idx = np.where(F == f)[0]
        if not len(idx) or not len(Pq):
            continue
        d = np.linalg.norm(Pq[:, None, :] - P[idx][None, :, :], axis=2) * 1000.0
        qi, j = np.nonzero(d <= cfg["match_mm"])
        if not len(qi):
            continue
        col = np.einsum("pkj,pkj->pj", Rq[qi], R[idx[j]])            # column-wise dot products
        tr = np.maximum(col.sum(1), col[:, 0] - col[:, 1] - col[:, 2])  # trace(A'B), trace(A'B flip)
        ok = (tr - 1.0) / 2.0 >= cos_min - 1e-9
        out[qi[ok], k] = True
    return out


# --------------------------------------------------------------- main
def fmt(x, nd=3):
    return None if x is None else round(float(x), nd)


def quat_xyzw(r):
    """Rotation matrix -> quaternion (x, y, z, w), w >= 0."""
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x, y, z = np.copysign(x, r[2, 1] - r[1, 2]), np.copysign(y, r[0, 2] - r[2, 0]), np.copysign(z, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def super_candidates(groups, cands, ids, cfg, tracking):
    """One grasp per group, groups by frequency (frames_hit, then n_grasps): the group's best grasp if its
    approach is within max_tilt_deg of the camera z axis (and its trace >= min_trace), else its next best, ..."""
    lim, want = cfg["max_tilt_deg"], cfg["n"]
    tmin = cfg["min_trace"] if tracking else 0.0
    if cfg["min_trace"] and not tracking:
        print("\n[WARN] --no_track: no trace computed, the min_trace condition is ignored")
    cond = f"approach <= {lim:.0f} deg from the camera z axis" + (f", trace >= {tmin:.2f}" if tmin else "")
    order = sorted(groups, key=lambda g: (-g["frames_hit"], -g["members"], g["penalized_rank"]))
    picked, skipped = [], []
    for g in order:
        if len(picked) == want:
            break
        for k, i in enumerate(g["member_ids"]):          # member_ids are in best-first order (rank, -score)
            c = cands[i]
            if gat.tilt_deg(c["R"]) <= lim and (not tmin or c["trace"] >= tmin):
                picked.append((g, k + 1, i))
                break
        else:
            skipped.append(g["group"])
    print(f"\nSUPER CANDIDATES: {len(picked)} (groups by frequency; per group the best grasp with {cond})")
    print(f"  {'#':>2} {'grp':>4} {'frames':>6} {'cand':>4} {'grasp_id':>8} {'fr':>3} {'rk':>3} {'score':>6} "
          + (f"{'trace':>5} " if tracking else "") + f"{'tilt':>5} {'width':>6}   {'x':>7} {'y':>7} {'z':>6}   "
          f"{'approach (col 0)':>20}   {'closing (col 1)':>20}   quaternion x, y, z, w")
    for n, (g, k, i) in enumerate(picked, 1):
        c = cands[i]
        a, b, q = c["R"][:, 0], c["R"][:, 1], quat_xyzw(c["R"])
        v = lambda u: "(" + ", ".join(f"{e:+.2f}" for e in u) + ")"  # noqa: E731
        print(f"  {n:>2} {g['group']:>4} {g['frames_hit']:>3}/{g['n_frames']:<2} {k:>4} {ids[i]:>8} {c['frame']:>3} "
              f"{c['rank']:>3} {c['score']:6.3f} " + (f"{c['trace']:5.2f} " if tracking else "")
              + f"{gat.tilt_deg(c['R']):4.0f}° {c['width'] * 1000:4.0f}mm   {c['pos'][0]:+7.3f} {c['pos'][1]:+7.3f} "
              f"{c['pos'][2]:6.3f}   {v(a):>20}   {v(b):>20}   ({', '.join(f'{e:+.3f}' for e in q)})")
    if skipped:
        print(f"  skipped (no grasp with {cond}): groups {skipped}")
    if len(picked) < want:
        print(f"  [WARN] only {len(picked)} groups have a grasp with {cond}")
    print("  cand = which grasp of its group (1 = the group's best); pose in the camera frame (m)")


def rot_cols(r):
    return {f"r{a}{b}": fmt(r[a, b], 5) for a in range(3) for b in range(3)}


def main():
    ap = argparse.ArgumentParser(description="grouping + traceability of all detector grasps over a sequence")
    ap.add_argument("--input", required=True, help="folder: cloud_XXX.npy | color_XXX.png+depth_XXX | API scenes")
    ap.add_argument("--mask", help="PNG (white = keep) or a folder with mask_XXX.png; none = the whole scene")
    ap.add_argument("--K", help="(N, 6) clouds + mask: a json with K (\"K\" or camera.K)")
    tr = ap.add_mutually_exclusive_group()
    tr.add_argument("--no_track", action="store_true", help="grouping only")
    tr.add_argument("--track", action="store_true", help="grouping + traceability (CONFIG default)")
    ap.add_argument("--no_cache", action="store_true", help="run the detector even if cached detections match")
    ap.add_argument("--out", help="output folder (default <input>/seq_group_track)")
    ap.add_argument("--min_trace", type=float, help="super candidates: override CONFIG SUPER min_trace (0 = off)")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    cfg = CONFIG
    if args.no_track:
        cfg["TRACKING"]["enabled"] = False
    if args.track:
        cfg["TRACKING"]["enabled"] = True
    tracking = cfg["TRACKING"]["enabled"]
    if args.min_trace is not None:
        cfg["SUPER"]["min_trace"] = args.min_trace
    inp = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else inp / "seq_group_track"
    (out / ".cache").mkdir(parents=True, exist_ok=True)
    t = {}
    t0 = time.perf_counter()

    # ---- 1. frames
    frames, kind = load_frames(inp, cfg)
    n = len(frames)
    t["load_s"] = time.perf_counter() - t0
    print(f"[input] {kind}: {n} frames from {inp}  ({np.mean([len(f['xyz']) for f in frames]):.0f} points/frame)")

    mask = None
    if args.mask:
        mask = PngMask(Path(args.mask).expanduser().resolve(), cfg["MASK"],
                       None if args.K is None else k_from_json(Path(args.K).expanduser()))
        if cfg["MASK"]["apply_to"] == "points":
            ta = time.perf_counter()
            for fr in frames:
                keep = mask.hit_points(fr)
                fr["xyz"], fr["rgb"] = fr["xyz"][keep], fr["rgb"][keep]
            t["mask_s"] = time.perf_counter() - ta
            print(f"[mask] cloud cut to the mask before detection: {np.mean([len(f['xyz']) for f in frames]):.0f} "
                  "points/frame left")

    # ---- 2. detector (cached)
    ta = time.perf_counter()
    cache, key = out / ".cache" / "detections.npz", cache_key(inp, frames, cfg, args.mask)
    cands = None if (args.no_cache or not cfg["DETECTOR"]["reuse_cache"]) else load_cache(cache, key)
    if cands is not None:
        print(f"[detect] reusing cached detections ({len(cands)} grasps): detector not run")
        t["detect_cached"] = True
    else:
        print(f"[detect] AnyGrasp detector on {n} frames (collision check "
              f"{'on' if cfg['DETECTOR']['collision_detection'] else 'OFF'})")
        cands = gat.run_detector([(f["frame"], f["xyz"], f["rgb"]) for f in frames], cfg["DETECTOR"])
        t["detect_cached"] = False
        save_cache(cache, key, cands)
    t["detect_s"] = time.perf_counter() - ta
    n_raw = len(cands)

    # ---- 3. mask on grasp centres
    if mask is not None and cfg["MASK"]["apply_to"] == "grasps":
        ta = time.perf_counter()
        by_f = {fr["frame"]: fr for fr in frames}
        keep = np.zeros(len(cands), bool)
        for f in sorted({c["frame"] for c in cands}):
            idx = [i for i, c in enumerate(cands) if c["frame"] == f]
            keep[idx] = mask.hit_xyz(by_f[f], np.array([cands[i]["pos"] for i in idx]))
        cands = [c for c, k in zip(cands, keep) if k]
        t["mask_s"] = t.get("mask_s", 0.0) + time.perf_counter() - ta
    by_frame = {}
    for c in cands:
        by_frame.setdefault(c["frame"], []).append(c)
    for fr in by_frame.values():                             # ranks 1..n again inside the mask
        for r, c in enumerate(sorted(fr, key=lambda c: c["rank"])):
            c["rank"] = r + 1
    print(f"[mask] {'none (whole scene)' if mask is None else args.mask + ' on ' + cfg['MASK']['apply_to']}: "
          f"{len(cands)}/{n_raw} grasps kept")
    if not cands:
        raise SystemExit("no grasp left")

    # ---- 4. grouping
    ta = time.perf_counter()
    g_cfg = cfg["GROUPING"]
    groups = gat.group(cands, n, g_cfg)
    for g in groups:
        g["n_frames"] = n
    t["group_s"] = time.perf_counter() - ta

    # ---- 5. tracking: every grasp searched for in every other frame
    if tracking:
        ta = time.perf_counter()
        t_cfg = cfg["TRACKING"]
        frame_ids = [fr["frame"] for fr in frames]
        P = np.array([c["pos"] for c in cands])
        R = np.array([c["R"] for c in cands])
        F = np.array([c["frame"] for c in cands])
        hit = redetected(P, R, P, R, F, frame_ids, t_cfg)
        hit &= F[:, None] != np.array(frame_ids)[None, :]     # its own frame does not count
        hits = hit.sum(1)
        for c, h in zip(cands, hits):
            c["trace_hits"], c["trace"] = int(h), h / max(n - 1, 1)
        if groups:
            rep = redetected(np.array([g["centre"] for g in groups]), np.array([g["R"] for g in groups]),
                             P, R, F, frame_ids, t_cfg)
            for g, h in zip(groups, rep):
                tr = [cands[i]["trace"] for i in g["member_ids"]]
                g["trace_rep"], g["trace_mean"], g["trace_max"] = h.sum() / n, float(np.mean(tr)), float(np.max(tr))
        t["track_s"] = time.perf_counter() - ta
    t["total_s"] = time.perf_counter() - t0

    # ---- 6. the two tables
    member_of = {}
    for g in groups:
        for k, i in enumerate(g["member_ids"]):
            member_of[i] = (g["group"], k == 0)
    grasp_rows, grasp_id = [], {}
    for i, c in sorted(enumerate(cands), key=lambda ic: (ic[1]["frame"], ic[1]["rank"])):
        grp, first = member_of.get(i, (None, False))
        grasp_id[i] = len(grasp_rows)
        row = {"grasp_id": len(grasp_rows), "frame": c["frame"], "rank": c["rank"], "score": fmt(c["score"], 4),
               "group": grp, "group_best": first}
        if tracking:
            row.update({"trace": fmt(c["trace"]), "trace_hits": c["trace_hits"]})
        row.update({"width_mm": fmt(c["width"] * 1000, 1), "depth_mm": fmt(c["depth"] * 1000, 1),
                    "tilt_deg": fmt(gat.tilt_deg(c["R"]), 1),
                    "x": fmt(c["pos"][0], 4), "y": fmt(c["pos"][1], 4), "z": fmt(c["pos"][2], 4), **rot_cols(c["R"])})
        grasp_rows.append(row)
    group_rows = []
    for g in groups:
        r = {"group": g["group"], "n_grasps": g["members"], "frames_hit": g["frames_hit"],
             "support": fmt(g["support"]), "avg_rank": fmt(g["avg_rank"], 2),
             "penalized_rank": fmt(g["penalized_rank"], 2), "avg_score": fmt(g["avg_score"], 4),
             "max_score": fmt(g["max_score"], 4), "spread_mm": fmt(g["spread_mm"], 1),
             "rot_spread_deg": fmt(g["rot_spread_deg"], 1)}
        if tracking:
            r.update({"trace_rep": fmt(g["trace_rep"]), "trace_mean": fmt(g["trace_mean"]),
                      "trace_max": fmt(g["trace_max"])})
        r.update({"width_mm": fmt(g["width"] * 1000, 1), "depth_mm": fmt(g["depth"] * 1000, 1),
                  "tilt_deg": fmt(g["tilt_deg"], 1), "x": fmt(g["centre"][0], 4), "y": fmt(g["centre"][1], 4),
                  "z": fmt(g["centre"][2], 4), **rot_cols(g["R"])})
        group_rows.append(r)
    for old in ("groups_track.csv", "groups_notrack.csv", "grasps_track.csv", "all_grasps_track.csv",
                "all_grasps_notrack.csv", "timing.csv"):     # outputs of earlier versions of this script
        if (out / old).exists():
            (out / old).unlink()
    if (out / "detections.npz").exists():
        os.replace(out / "detections.npz", out / ".cache" / "detections.npz")
    gat.write_csv(out / "grasps.csv", grasp_rows)
    gat.write_csv(out / "groups.csv", group_rows)

    # ---- 7. print
    print(f"\nGROUPS: {len(groups)} (radius {g_cfg['group_mm']:.0f} mm, angle {g_cfg['group_deg']}, "
          f"rank {g_cfg['rank_mode']}) — top 15")
    head = f"  {'grp':>3} {'grasps':>6} {'frames':>7} {'supp':>5} {'pen rk':>6} {'score':>6} {'spread':>7} {'rot':>4}"
    print(head + (f" {'tr_rep':>6} {'tr_mean':>7}" if tracking else "") + f" {'tilt':>4}")
    for r in group_rows[:15]:
        line = (f"  {r['group']:>3} {r['n_grasps']:>6} {r['frames_hit']:>3}/{n:<3} {r['support']:5.2f} "
                f"{r['penalized_rank']:6.1f} {r['avg_score']:6.3f} {r['spread_mm']:5.1f}mm {r['rot_spread_deg']:3.0f}°")
        if tracking:
            line += f" {r['trace_rep']:6.2f} {r['trace_mean']:7.2f}"
        print(line + f" {r['tilt_deg']:3.0f}°")
    if tracking:
        tr_all = np.array([c["trace"] for c in cands])
        print(f"\n  trace of all {len(cands)} grasps: mean {tr_all.mean():.2f}, >= 0.8 in {100 * (tr_all >= 0.8).mean():.0f} %")
        sup, trp = np.array([r["support"] for r in group_rows]), np.array([r["trace_rep"] for r in group_rows])
        if len(group_rows) > 2 and sup.std() > 0 and trp.std() > 0:
            print(f"  support vs trace_rep over all groups: r = {np.corrcoef(sup, trp)[0, 1]:+.2f}")
    det = "cached" if t["detect_cached"] else f"{1000 * t['detect_s'] / max(n, 1):.0f} ms/frame"
    print(f"\nTIME  load {t['load_s']:.2f} s | detect {t['detect_s']:.2f} s ({det}) | mask {t.get('mask_s', 0.0):.2f} s"
          f" | grouping {t['group_s']:.2f} s | tracking {t.get('track_s', 0.0):.2f} s | total {t['total_s']:.2f} s")
    print(f"[out] {out}: grasps.csv ({len(grasp_rows)} rows), groups.csv ({len(group_rows)} rows)")
    super_candidates(groups, cands, grasp_id, cfg["SUPER"], tracking)

    if args.vis:
        fr = frames[0]
        sel = np.random.default_rng(0).choice(len(fr["xyz"]), min(len(fr["xyz"]), 300000), replace=False)
        for g in groups:
            g.setdefault("trace_mean", None)
        gat.visualise(fr["xyz"][sel], fr["rgb"][sel], groups, cands, [], [], [], cfg,
                      f"{inp.name}: {len(groups)} groups" + (" + trace" if tracking else ""))


if __name__ == "__main__":
    main()
