"""Grouping + traceability of AnyGrasp DETECTOR grasps over a burst of frames, with an optional mask, and an
evaluation of external grasps against both.

  1. detections  AnyGrasp detector on every frame (with its collision check), or read from a burst_result.json
  2. mask        keep only grasps whose centre lies in the mask (none / sim / image / box); the detector itself
                 always sees the whole scene, so its collision check is not affected
  3. grouping    all frames pooled, best-first (rank, then score), a grasp joins the nearest group whose centre is
                 within GROUP_MM and whose first (best) member is within GROUP_DEG (180 deg flip = same grasp):
                 the same rules as detect_burst.py. Per group: members, frames hit, support, avg / penalized rank,
                 mean / max score, spread (mm, deg), representative grasp
  4. tracking    geometric traceability, as detector_traceability.py: every grasp of START_FRAME, anchored to its
                 start pose, is searched for in every other frame (within MATCH_MM and MATCH_DEG):
                 trace = share of frames in which it is detected again
  5. external    grasps from a file: the group each one falls in (or none) and its trace over ALL frames
                 (0 if it is never detected), plus how close the nearest detection comes

All settings are in the CONFIG block below: edit them there. The command line only chooses the input / output
(and can override a few common settings).

Inputs (one of):
  --seq    runs/.../seq_XXXX       sim sequence (color + depth + K); masks from Isaac available (MASK "sim")
  --clouds ~/<folder>/burst        real burst: cloud_XXX.npy (N, 6) in metres (capture_burst.py)
  --detections .../burst_result.json   detections already computed by detect_burst.py (no detector run;
                                        use DETECTOR top_k 1024 there so it kept every grasp)
External grasps (optional):  --external grasps.json | grasps.npz      (camera frame of START_FRAME)

Environment: anygrasp. Examples (project root):
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/group_and_trace.py --seq runs/multi/seed_100/seq_0000 --vis
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/group_and_trace.py --clouds ~/Time_consistency/burst --mask box
    ... --external my_grasps.json
Output folder (default <input>/group_and_trace/): groups.csv, grasps.csv, external.csv, result.json
"""

# =====================================================================================================
# CONFIG: edit here
# =====================================================================================================
CONFIG = {
    # ---- detector (used with --seq / --clouds; ignored with --detections) -------------------------
    "DETECTOR": {
        "workspace": [(-0.4, 0.4), (-0.4, 0.4), (0.05, 1.0)],  # x, y, z crop in metres, camera frame
        "voxel": 0.005,                    # m, voxel downsampling before detection (as detect_grasps.py)
        "max_gripper_width": 0.10,         # m; the SDK accepts (0, 0.1]
        "gripper_height": 0.03,            # m, finger height used by the collision check
        "collision_detection": True,       # the SDK's collision check (keep True)
        "nms": True,                       # remove near-duplicates in every frame
        "top_k": 0,                        # grasps kept per frame after NMS, best score first; 0 = all
        "max_frames": 0,                   # 0 = every frame of the input
    },
    # ---- mask: which grasps count (applied to grasp centres, in every frame) ----------------------
    "MASK": {
        "mode": "none",                    # "none" | "sim" | "image" | "box"
        "dilate_px": 8,                    # sim / image: margin in pixels (thin parts)
        "image": None,                     # image mode: a PNG (white = keep), or a folder with mask_XXX.png
        "K": None,                         # image mode: 3x3 intrinsics of that image, e.g. [[fx,0,cx],[0,fy,cy],[0,0,1]]
        "box": [(-0.10, 0.10), (-0.10, 0.10), (0.30, 0.70)],  # box mode: x, y, z limits in metres, camera frame
    },
    # ---- grouping ---------------------------------------------------------------------------------
    "GROUPING": {
        "group_mm": 20.0,                  # max distance from a grasp to a group's centre
        "group_deg": 45.0,                 # max angle to the group's first (best) member; None = position only
        "rank_mode": "penalized",          # "avg" or "penalized" (missing frames count as the worst rank)
        "min_support": 0.0,                # drop groups seen in fewer than this share of frames
    },
    # ---- tracking (geometric traceability) --------------------------------------------------------
    "TRACKING": {
        "start_frame": 0,                  # frame whose grasps are scored (external grasps are in this frame)
        "match_mm": 20.0,                  # a re-detection within this distance of the start pose ...
        "match_deg": 30.0,                 # ... and this angle counts
    },
    # ---- external grasps --------------------------------------------------------------------------
    "EXTERNAL": {
        "group_mm": None,                  # None = GROUPING group_mm: max distance to a group centre
        "group_deg": None,                 # None = GROUPING group_deg
    },
    # ---- visualisation (--vis) --------------------------------------------------------------------
    "VIS": {
        "show_groups": 10,                 # groups drawn (best first)
        "members": True,                   # draw every member of those groups as small dots
        "gripper_mm": 3.0,
        "point_mm": 4.0,
    },
}
# =====================================================================================================

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRASP_DIR = Path.home() / "anygrasp_sdk" / "grasp_detection"
FLIP_X = np.diag([1.0, -1.0, -1.0])
PALETTE = [(0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16), (0.58, 0.40, 0.74),
           (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81),
           (0.68, 0.78, 0.91), (1.00, 0.73, 0.47), (0.60, 0.87, 0.54), (1.00, 0.60, 0.59), (0.77, 0.69, 0.84)]


# --------------------------------------------------------------- geometry
def angle_deg(ra, rb):
    """Angle between two grasp rotations, flip-aware (180 deg about the approach = same parallel-jaw grasp)."""
    tr = max(np.trace(ra.T @ rb), np.trace(ra.T @ rb @ FLIP_X))
    return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))


def tilt_deg(r):
    """Approach (x column) vs the optical axis (+z), as detect_grasps.py."""
    return float(np.degrees(np.arccos(np.clip(r[2, 0], -1.0, 1.0))))


def dilate(mask, px):
    out = mask.copy()
    for axis in (0, 1):
        src, out = out, out.copy()
        for s in range(1, px + 1):
            out |= np.roll(src, s, axis=axis) | np.roll(src, -s, axis=axis)
    return out


def load_png_mask(path: Path):
    from PIL import Image
    img = np.asarray(Image.open(path))
    return (img[..., 0] if img.ndim == 3 else img) > 127


# --------------------------------------------------------------- input frames
def frames_from_seq(seq: Path):
    """[(frame, xyz (N, 3) m, rgb (N, 3) 0..1)] from a sim sequence + its K."""
    from PIL import Image
    meta = json.loads((seq / "meta.json").read_text())
    k = np.asarray(meta["K"], dtype=np.float64)
    out = []
    for f in range(meta["n_frames"]):
        rgb = np.asarray(Image.open(seq / "frames" / f"color_{f:03d}.png"), dtype=np.float64)[..., :3] / 255.0
        z = np.load(seq / "frames" / f"depth_{f:03d}.npy").astype(np.float64) / meta.get("depth_scale", 1000.0)
        h, w = z.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        xyz = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)
        valid = z > 0
        out.append((f, xyz[valid], rgb[valid]))
    return out, k


def frames_from_clouds(folder: Path):
    """[(frame, xyz, rgb)] from cloud_XXX.npy files (N, 3) / (N, 6), metres (mm converted)."""
    out = []
    for f, path in enumerate(sorted(folder.glob("cloud_*.npy"))):
        d = np.load(path).astype(np.float64)
        xyz = d[:, :3] / (1000.0 if np.nanmax(np.abs(d[:, :3])) > 20 else 1.0)
        rgb = d[:, 3:6] if d.shape[1] >= 6 else np.full((len(d), 3), 0.6)
        rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
        keep = np.isfinite(xyz).all(axis=1) & ~np.all(xyz == 0, axis=1)
        out.append((f, xyz[keep], rgb[keep]))
    return out


def crop_voxel(xyz, rgb, cfg):
    import open3d as o3d
    keep = np.ones(len(xyz), dtype=bool)
    for a, (lo, hi) in enumerate(cfg["workspace"]):
        keep &= (xyz[:, a] > lo) & (xyz[:, a] < hi)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[keep]))
    pcd.colors = o3d.utility.Vector3dVector(rgb[keep])
    return pcd.voxel_down_sample(cfg["voxel"]) if cfg["voxel"] else pcd


# --------------------------------------------------------------- detections
def run_detector(frames, cfg):
    """[{frame, rank, score, width, depth, pos, R}] for every frame (camera frame)."""
    sys.path.insert(0, str(GRASP_DIR))
    cwd = os.getcwd()
    os.chdir(GRASP_DIR)                      # gsnet.so, its license and checkpoint live there
    from gsnet import create_detector
    det = create_detector(argparse.Namespace(checkpoint_path=str(GRASP_DIR / "log" / "checkpoint_detection.tar"),
                                             max_gripper_width=cfg["max_gripper_width"],
                                             gripper_height=cfg["gripper_height"]))
    os.chdir(cwd)
    if det is None:
        raise SystemExit("could not create the detector (license)")
    cands = []
    for f, xyz, rgb in frames:
        pcd = crop_voxel(xyz, rgb, cfg)
        gg = det.get_grasp(np.asarray(pcd.points, dtype=np.float32),
                           {"dense_grasp": False, "collision_detection": cfg["collision_detection"]})
        n = 0
        if gg is not None and len(gg):
            if cfg["nms"]:
                gg = gg.nms()
            gg = gg.sort_by_score()
            if cfg["top_k"]:
                gg = gg[:cfg["top_k"]]
            t, r = np.asarray(gg.translations, float), np.asarray(gg.rotation_matrices, float)
            s, w, d = (np.asarray(getattr(gg, a), float) for a in ("scores", "widths", "depths"))
            n = len(s)
            for i in range(n):
                cands.append({"frame": f, "rank": i + 1, "score": float(s[i]), "width": float(w[i]),
                              "depth": float(d[i]), "pos": t[i], "R": r[i]})
        print(f"  frame {f:3d}: {len(pcd.points):6d} points -> {n:4d} grasps")
    return cands


def read_detections(path: Path):
    res = json.loads(path.read_text())
    cands = [{"frame": c["frame"], "rank": c["rank"], "score": c["score"], "width": c["width"], "depth": c["depth"],
              "pos": np.array([c["x"], c["y"], c["z"]], float), "R": np.asarray(c["rotation_matrix"], float)}
             for c in res["candidates"]]
    tk = res.get("settings", {}).get("top_k")
    if tk is not None and tk < 200:
        print(f"[WARN] {path.name} kept only top_k {tk} grasps per frame: lower-ranked grasps are missing")
    return cands, res.get("burst", {}).get("seq"), len(res.get("frames", [])) or None


# --------------------------------------------------------------- mask
class Mask:
    def __init__(self, cfg, seq: Path | None, k_seq):
        self.mode, self.cfg = cfg["mode"], cfg
        self.cache = {}
        if self.mode == "sim":
            if seq is None:
                raise SystemExit("MASK mode 'sim' needs a sim sequence (--seq, or a burst_result.json made from one)")
            self.dir, self.k = seq / "frames", np.asarray(k_seq, float)
        elif self.mode == "image":
            if cfg["image"] is None or cfg["K"] is None:
                raise SystemExit("MASK mode 'image' needs MASK['image'] (png or folder) and MASK['K']")
            self.img, self.k = Path(cfg["image"]).expanduser(), np.asarray(cfg["K"], float)
        elif self.mode == "box":
            self.box = cfg["box"]
        elif self.mode != "none":
            raise SystemExit(f"unknown MASK mode {self.mode!r}")

    def _image(self, f):
        if f not in self.cache:
            if self.mode == "sim":
                path = self.dir / f"mask_{f:03d}.png"
            else:
                path = self.img / f"mask_{f:03d}.png" if self.img.is_dir() else self.img
            self.cache[f] = dilate(load_png_mask(path), self.cfg["dilate_px"])
        return self.cache[f]

    def hit(self, f, p):
        if self.mode == "none":
            return True
        if self.mode == "box":
            return all(lo <= p[a] <= hi for a, (lo, hi) in enumerate(self.box))
        if p[2] <= 0:
            return False
        m = self._image(f)
        u = int(round(self.k[0, 0] * p[0] / p[2] + self.k[0, 2]))
        v = int(round(self.k[1, 1] * p[1] / p[2] + self.k[1, 2]))
        return 0 <= v < m.shape[0] and 0 <= u < m.shape[1] and bool(m[v, u])


# --------------------------------------------------------------- grouping (detect_burst.py rules)
def group(cands, n_frames, cfg):
    thr, deg = cfg["group_mm"] / 1000.0, cfg["group_deg"]
    groups = []
    for i in sorted(range(len(cands)), key=lambda i: (cands[i]["rank"], -cands[i]["score"])):
        c = cands[i]
        best, best_d = None, np.inf
        for g in groups:
            d = np.linalg.norm(c["pos"] - g["sum"] / len(g["members"]))
            if d > thr or d >= best_d:
                continue
            if deg is not None and angle_deg(cands[g["members"][0]]["R"], c["R"]) > deg:
                continue
            best, best_d = g, d
        if best is None:
            groups.append({"members": [i], "sum": c["pos"].copy()})
        else:
            best["members"].append(i)
            best["sum"] += c["pos"]
    max_rank = max((c["rank"] for c in cands), default=1)
    out = []
    for g in groups:
        mem = [cands[i] for i in g["members"]]
        pos = np.array([m["pos"] for m in mem])
        centre = pos.mean(0)
        best = mem[0]                                     # first visited = best (rank, -score)
        frames = sorted({m["frame"] for m in mem})
        best_rank = {}
        for m in mem:
            best_rank[m["frame"]] = min(best_rank.get(m["frame"], 1e9), m["rank"])
        dist = np.linalg.norm(pos - centre, axis=1) * 1000.0
        angs = [angle_deg(best["R"], m["R"]) for m in mem]
        out.append({
            "members": len(mem), "frames_hit": len(frames), "support": len(frames) / n_frames,
            "avg_rank": float(np.mean([m["rank"] for m in mem])),
            "penalized_rank": float((sum(best_rank.values()) + (n_frames - len(frames)) * (max_rank + 1)) / n_frames),
            "avg_score": float(np.mean([m["score"] for m in mem])), "max_score": float(max(m["score"] for m in mem)),
            "spread_mm": float(np.sqrt((dist ** 2).mean())), "spread_max_mm": float(dist.max()),
            "rot_spread_deg": float(np.mean(angs)), "rot_spread_max_deg": float(np.max(angs)),
            "centre": centre, "R": best["R"], "width": best["width"], "depth": best["depth"],
            "tilt_deg": tilt_deg(best["R"]), "member_ids": list(g["members"]),
        })
    key = "penalized_rank" if cfg["rank_mode"] == "penalized" else "avg_rank"
    out.sort(key=lambda g: (g[key], -g["support"], -g["avg_score"]))
    out = [g for g in out if g["support"] >= cfg["min_support"]]
    for r, g in enumerate(out):
        g["group"] = r + 1
    return out


def find_group(groups, pos, rot, max_mm, max_deg):
    """Nearest group centre within max_mm (and max_deg of its representative): (group, dist mm) or (None, None)."""
    best, best_d = None, np.inf
    for g in groups:
        d = np.linalg.norm(g["centre"] - pos) * 1000.0
        if d <= max_mm and d < best_d and (max_deg is None or angle_deg(g["R"], rot) <= max_deg):
            best, best_d = g, d
    return (best, round(best_d, 2)) if best is not None else (None, None)


# --------------------------------------------------------------- tracking (geometric)
def trace(pos, rot, by_frame, frames, cfg):
    """Share of `frames` with a detection within match_mm / match_deg of (pos, rot), and the match distances."""
    hits, dists, nearest = 0, [], []
    for f in frames:
        cs = by_frame.get(f, [])
        if not cs:
            continue
        p = np.array([c["pos"] for c in cs])
        d = np.linalg.norm(p - pos, axis=1) * 1000.0
        nearest.append(float(d.min()))
        for j in np.argsort(d):
            if d[j] > cfg["match_mm"]:
                break
            if angle_deg(rot, cs[j]["R"]) <= cfg["match_deg"]:
                hits += 1
                dists.append(float(d[j]))
                break
    n = len(frames)
    return (hits / n if n else 0.0), (float(np.mean(dists)) if dists else None), \
        (float(np.median(nearest)) if nearest else None)


# --------------------------------------------------------------- external grasps
def read_external(path: Path):
    """[{pos, R, width, depth, score, name}] from a json (list / {candidates|grasps|groups: [...]} / single grasp)
    or an npz (translations, rotation_matrices, [widths, depths, scores])."""
    if path.suffix == ".npz":
        z = np.load(path)
        t, r = np.asarray(z["translations"], float).reshape(-1, 3), np.asarray(z["rotation_matrices"], float)
        n = len(t)
        get = lambda k, dflt: np.asarray(z[k], float) if k in z.files else np.full(n, dflt)  # noqa: E731
        w, d, s = get("widths", np.nan), get("depths", np.nan), get("scores", np.nan)
        return [{"name": f"ext_{i}", "pos": t[i], "R": r[i].reshape(3, 3), "width": float(w[i]), "depth": float(d[i]),
                 "score": float(s[i])} for i in range(n)]
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "part_id" in data:
            print("[WARN] this looks like part_detection.json: its rotation_matrix is the robot's yaw-only rotation, "
                  "NOT AnyGrasp's grasp rotation. Angles will not compare; use latest_grasps.npz instead.")
        for key in ("candidates", "grasps", "groups"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]
    out = []
    for i, g in enumerate(data):
        if "grasp" in g and isinstance(g["grasp"], dict):          # a group of burst_result.json
            g = {**g["grasp"], "score": g.get("max_score")}
        pos = np.asarray(g["translation"], float) if "translation" in g else np.array([g["x"], g["y"], g["z"]], float)
        out.append({"name": str(g.get("name", g.get("part_id", g.get("id", f"ext_{i}")))), "pos": pos,
                    "R": np.asarray(g["rotation_matrix"], float).reshape(3, 3),
                    "width": float(g.get("width", np.nan)), "depth": float(g.get("depth", np.nan)),
                    "score": float(g["score"]) if g.get("score") is not None else np.nan})
    return out


# --------------------------------------------------------------- visualisation
def visualise(frame_xyz, frame_rgb, groups, cands, grasps, external, ext_rows, cfg, title):
    import open3d as o3d
    sys.path.insert(0, str(HERE))
    import compare_burst_tracking as cbt
    v = cfg["VIS"]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame_xyz))
    pcd.colors = o3d.utility.Vector3dVector(frame_rgb)
    pcd = pcd.voxel_down_sample(0.004)
    geoms, labels = [("cloud", pcd)], []
    for g in groups[:v["show_groups"]]:
        col = PALETTE[(g["group"] - 1) % len(PALETTE)]
        name = f"G{g['group']}"
        geoms.append((f"{name} gripper", cbt.gripper_mesh(g["centre"], g["R"], g["width"], g["depth"], col,
                                                         v["gripper_mm"] / 1000.0)))
        geoms.append((f"{name} point", cbt.spheres([g["centre"]], v["point_mm"] / 1000.0, col)))
        if v["members"]:
            geoms.append((f"{name} members", cbt.spheres([cands[i]["pos"] for i in g["member_ids"]],
                                                         v["point_mm"] / 2500.0, col)))
        tr = f" tr {g['trace_mean']:.2f}" if g.get("trace_mean") is not None else ""
        labels.append((g["centre"] + [0, 0, -0.03], f"G{g['group']} {g['frames_hit']}/{g['n_frames']}{tr}"))
    for e, row in zip(external, ext_rows):
        w = e["width"] if np.isfinite(e["width"]) else 0.04
        d = e["depth"] if np.isfinite(e["depth"]) else 0.02
        geoms.append((f"EXT {e['name']}", cbt.gripper_mesh(e["pos"], e["R"], w, d, (1.0, 0.0, 1.0),
                                                           1.5 * v["gripper_mm"] / 1000.0)))
        labels.append((e["pos"] + [0, 0, -0.05],
                       f"{e['name']}: {('G' + str(row['group'])) if row['group'] else 'no group'} tr {row['trace']:.2f}"))
    try:
        import open3d.visualization.gui as gui
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer(title, 1400, 900)
        for name, geom in geoms:
            vis.add_geometry(name, geom)
        for pos, text in labels:
            vis.add_3d_label(pos, text)
        vis.point_size = 2
        vis.show_settings = True
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()
    except Exception as e:
        print(f"[vis] labelled viewer unavailable ({e}); colours only")
        o3d.visualization.draw_geometries([g for _, g in geoms], window_name=title)


# --------------------------------------------------------------- main
def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="grouping + traceability of detector grasps (settings: CONFIG block)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", help="sim sequence folder (capture_sequence.py)")
    src.add_argument("--clouds", help="folder with cloud_XXX.npy (capture_burst.py)")
    src.add_argument("--detections", help="burst_result.json (detect_burst.py), no detector run")
    ap.add_argument("--external", help="grasps to evaluate: .json or .npz, camera frame of the start frame")
    ap.add_argument("--mask", choices=["none", "sim", "image", "box"], help="override CONFIG MASK mode")
    ap.add_argument("--out", help="output folder (default <input>/group_and_trace)")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    cfg = CONFIG
    if args.mask:
        cfg["MASK"]["mode"] = args.mask
    det_cfg, g_cfg, t_cfg = cfg["DETECTOR"], cfg["GROUPING"], cfg["TRACKING"]

    # ---- 1. detections
    seq, k_seq, frames, n_frames = None, None, None, None
    if args.seq:
        seq = Path(args.seq).expanduser().resolve()
        frames, k_seq = frames_from_seq(seq)
    elif args.clouds:
        frames = frames_from_clouds(Path(args.clouds).expanduser().resolve())
    if frames is not None:
        if det_cfg["max_frames"]:
            frames = frames[:det_cfg["max_frames"]]
        if not frames:
            raise SystemExit("no frames found")
        n_frames = len(frames)
        print(f"running the detector on {n_frames} frames (collision check {'on' if det_cfg['collision_detection'] else 'OFF'})")
        cands = run_detector(frames, det_cfg)
        in_root = seq if seq else Path(args.clouds).expanduser().resolve()
    else:
        det_path = Path(args.detections).expanduser().resolve()
        cands, seq_path, n_frames = read_detections(det_path)
        n_frames = n_frames or (max(c["frame"] for c in cands) + 1)
        if seq_path and Path(seq_path).exists():
            seq = Path(seq_path)
            k_seq = json.loads((seq / "meta.json").read_text())["K"]
        in_root = det_path.parent
    out = Path(args.out).expanduser().resolve() if args.out else in_root / "group_and_trace"
    out.mkdir(parents=True, exist_ok=True)

    # ---- 2. mask
    mask = Mask(cfg["MASK"], seq, k_seq)
    n_all = len(cands)
    cands = [c for c in cands if mask.hit(c["frame"], c["pos"])]
    by_frame = {}
    for c in cands:
        by_frame.setdefault(c["frame"], []).append(c)
    for fr in by_frame.values():                         # ranks 1..n again inside the mask
        for r, c in enumerate(sorted(fr, key=lambda c: c["rank"])):
            c["rank"] = r + 1
    print(f"mask '{mask.mode}': {len(cands)}/{n_all} grasps kept over {n_frames} frames")
    if not cands:
        raise SystemExit("no grasp left after the mask")

    # ---- 3. grouping
    groups = group(cands, n_frames, g_cfg)
    for g in groups:
        g["n_frames"] = n_frames

    # ---- 4. tracking: grasps of the start frame
    s = t_cfg["start_frame"]
    others = [f for f in range(n_frames) if f != s]
    grasp_rows = []
    for i, c in enumerate(sorted(by_frame.get(s, []), key=lambda c: c["rank"])):
        tr, mm, _ = trace(c["pos"], c["R"], by_frame, others, t_cfg)
        g, gd = find_group(groups, c["pos"], c["R"], g_cfg["group_mm"], g_cfg["group_deg"])
        grasp_rows.append({"grasp": i, "rank": c["rank"], "score": round(c["score"], 4), "trace": round(tr, 3),
                           "match_mm": None if mm is None else round(mm, 2), "group": g["group"] if g else None,
                           "group_support": round(g["support"], 3) if g else None,
                           "width_mm": round(c["width"] * 1000, 1), "depth_mm": round(c["depth"] * 1000, 1),
                           "tilt_deg": round(tilt_deg(c["R"]), 1), "x": round(float(c["pos"][0]), 4),
                           "y": round(float(c["pos"][1]), 4), "z": round(float(c["pos"][2]), 4)})
    for g in groups:                                     # group traceability = mean trace of its start grasps
        tr = [r["trace"] for r in grasp_rows if r["group"] == g["group"]]
        g["trace_mean"] = float(np.mean(tr)) if tr else None
        g["trace_max"] = float(np.max(tr)) if tr else None
        g["start_grasps"] = len(tr)

    # ---- 5. external grasps
    external, ext_rows = [], []
    if args.external:
        e_cfg = cfg["EXTERNAL"]
        emm = e_cfg["group_mm"] or g_cfg["group_mm"]
        edeg = e_cfg["group_deg"] if e_cfg["group_deg"] is not None else g_cfg["group_deg"]
        external = read_external(Path(args.external).expanduser())
        for e in external:
            tr, mm, near = trace(e["pos"], e["R"], by_frame, list(range(n_frames)), t_cfg)
            g, gd = find_group(groups, e["pos"], e["R"], emm, edeg)
            ext_rows.append({"name": e["name"], "in_mask": mask.hit(s, e["pos"]),
                             "score": None if not np.isfinite(e["score"]) else round(e["score"], 4),
                             "group": g["group"] if g else None, "group_dist_mm": gd,
                             "group_support": round(g["support"], 3) if g else None,
                             "group_avg_rank": round(g["avg_rank"], 2) if g else None,
                             "group_trace_mean": None if not g or g["trace_mean"] is None else round(g["trace_mean"], 3),
                             "trace": round(tr, 3), "match_mm": None if mm is None else round(mm, 2),
                             "nearest_detection_mm": None if near is None else round(near, 1),
                             "x": round(float(e["pos"][0]), 4), "y": round(float(e["pos"][1]), 4),
                             "z": round(float(e["pos"][2]), 4)})

    # ---- report + files
    group_rows = [{"group": g["group"], "members": g["members"], "frames_hit": g["frames_hit"],
                   "support": round(g["support"], 3), "avg_rank": round(g["avg_rank"], 2),
                   "penalized_rank": round(g["penalized_rank"], 2), "avg_score": round(g["avg_score"], 4),
                   "max_score": round(g["max_score"], 4), "spread_mm": round(g["spread_mm"], 1),
                   "spread_max_mm": round(g["spread_max_mm"], 1), "rot_spread_deg": round(g["rot_spread_deg"], 1),
                   "rot_spread_max_deg": round(g["rot_spread_max_deg"], 1),
                   "start_grasps": g["start_grasps"],
                   "trace_mean": None if g["trace_mean"] is None else round(g["trace_mean"], 3),
                   "trace_max": None if g["trace_max"] is None else round(g["trace_max"], 3),
                   "width_mm": round(g["width"] * 1000, 1), "tilt_deg": round(g["tilt_deg"], 1),
                   "x": round(float(g["centre"][0]), 4), "y": round(float(g["centre"][1]), 4),
                   "z": round(float(g["centre"][2]), 4)} for g in groups]
    print(f"\nGROUPS ({len(groups)}; radius {g_cfg['group_mm']:.0f} mm, angle {g_cfg['group_deg']}, "
          f"rank {g_cfg['rank_mode']}) — top 15")
    print(f"  {'grp':>3} {'members':>7} {'frames':>7} {'support':>7} {'avg rk':>6} {'pen rk':>6} {'score':>6} "
          f"{'spread':>7} {'rot':>5} {'trace':>5} {'tilt':>4}")
    for g in group_rows[:15]:
        print(f"  {g['group']:>3} {g['members']:>7} {g['frames_hit']:>3}/{n_frames:<3} {g['support']:7.2f} "
              f"{g['avg_rank']:6.1f} {g['penalized_rank']:6.1f} {g['avg_score']:6.3f} {g['spread_mm']:5.1f}mm "
              f"{g['rot_spread_deg']:4.0f}° {('%.2f' % g['trace_mean']) if g['trace_mean'] is not None else '   -':>5} "
              f"{g['tilt_deg']:3.0f}°")
    print(f"\nTRACEABILITY of the {len(grasp_rows)} grasps of frame {s} (<= {t_cfg['match_mm']:.0f} mm, "
          f"{t_cfg['match_deg']:.0f} deg, over {len(others)} other frames) — top 10")
    for r in sorted(grasp_rows, key=lambda r: (-r["trace"], -r["score"]))[:10]:
        print(f"  grasp {r['grasp']:>3} rank {r['rank']:>3} score {r['score']:.3f} trace {r['trace']:.2f} "
              f"group {r['group'] or '-'}")
    if ext_rows:
        print(f"\nEXTERNAL grasps ({len(ext_rows)}), trace over all {n_frames} frames")
        for r in ext_rows:
            if r["group"]:
                grp = f"G{r['group']} ({r['group_dist_mm']:.1f} mm away, support {r['group_support']:.2f})"
            else:
                grp = "none"
            near = r["nearest_detection_mm"] if r["nearest_detection_mm"] is not None else "-"
            print(f"  {r['name']:<14} {'in' if r['in_mask'] else 'OUT of'} mask | group {grp} | "
                  f"trace {r['trace']:.2f} | nearest detection {near} mm")

    write_csv(out / "groups.csv", group_rows)
    write_csv(out / "grasps.csv", grasp_rows)
    write_csv(out / "external.csv", ext_rows)
    (out / "result.json").write_text(json.dumps({
        "input": {"seq": args.seq, "clouds": args.clouds, "detections": args.detections, "external": args.external},
        "frames": n_frames, "config": cfg, "groups": group_rows, "grasps": grasp_rows, "external": ext_rows},
        indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"\n[INFO] {out}: groups.csv, grasps.csv, external.csv, result.json")

    if args.vis:
        if frames is not None:
            _, xyz, rgb = next(fr for fr in frames if fr[0] == s)
        elif seq is not None:
            (_, xyz, rgb), = [fr for fr in frames_from_seq(seq)[0] if fr[0] == s]
        else:
            raise SystemExit("--vis with --detections needs the sequence (burst_result.json made from a sim seq)")
        keep = (xyz[:, 2] > 0) & (xyz[:, 2] < 1.2)
        visualise(xyz[keep], rgb[keep], groups, cands, grasp_rows, external, ext_rows, cfg,
                  f"groups (G#  frames  trace) + external grasps (magenta)")


if __name__ == "__main__":
    main()
