"""Compare the burst method (grouping) with AnyGrasp tracking on the SAME frames, and show both in one 3D view.

Three runs on one sequence (all in the camera optical frame, metres):
  A  burst     detect_burst.py on the sequence (via seq_to_burst.py): detector (gsnet) + grouping
  B  tracking  track_replay.py: the tracker follows the frame-0 top-K grasps through the frames
  C  burst on the tracker's grasps: detect_burst.py's OWN grouping code applied to the top-K grasps the tracker
               detects in every frame (saved by track_replay.py --save_topk)
  B vs C: same model, different method (follow vs group)    -> does tracking measure what grouping measures?
  A vs C: same method, different model (detector vs tracker) -> do the two AnyGrasp models agree?

Per tracked grasp (B) it reports:
  survival   share of frame-to-frame steps that stay within group_mm (and group_deg if set): "time consistency"
  jitter     std of the position over the frames; drift = max distance from its frame-0 position; jumps
and the group it started in: C group (exact: with --select_nms, B's grasps ARE C's frame-0 top-K) and the nearest
A group. Spearman correlation between C support and B survival says whether the two variables agree.

Run track_replay.py with the burst's settings first:
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/track_replay.py --seq runs/seq_0004 \
        --no_mask --no_cone --num_track 10 --select_nms --save_topk 10
Then:
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/compare_burst_tracking.py --burst_root ~/<folder>_sim \
        --track runs/seq_0004/tracking/oneeuro_nocone_n10_nomask_nms --vis
    ... --view ABC        which runs to draw: any of A, B, C (default AB)
Grouping settings (top_k, group_mm, group_deg, rank_mode) are read from the burst_result.json of run A.
Output: <track>/compare.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

FLIP_X = np.diag([1.0, -1.0, -1.0])


def angle_deg(ra, rb):
    """Flip-aware rotation angle between two grasps (180 deg about the approach = same parallel-jaw grasp)."""
    def ang(a, b):
        return np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))
    return float(min(ang(ra, rb), ang(ra, rb @ FLIP_X)))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    def rank(v):  # average ranks, so ties (e.g. many groups with support 1.0) get equal ranks
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        return (start + (cnt - 1) / 2.0)[inv]
    rx, ry = rank(x[ok]), rank(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------- mask (same rules as batch_detect.py)
def load_mask(path: Path) -> np.ndarray:
    from PIL import Image
    img = np.asarray(Image.open(path))
    return (img[..., 0] if img.ndim == 3 else img) > 127


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    out = mask.copy()
    for axis in (0, 1):
        src, out = out, out.copy()
        for s in range(1, px + 1):
            out |= np.roll(src, s, axis=axis) | np.roll(src, -s, axis=axis)
    return out


def on_mask(p: np.ndarray, mask: np.ndarray, k: np.ndarray) -> bool:
    """Grasp centre (camera OpenCV frame) projects onto the mask."""
    if p[2] <= 0:
        return False
    u = int(round(k[0, 0] * p[0] / p[2] + k[0, 2]))
    v = int(round(k[1, 1] * p[1] / p[2] + k[1, 2]))
    h, w = mask.shape
    return 0 <= u < w and 0 <= v < h and bool(mask[v, u])


class Masks:
    """GT target mask of every frame of the sequence (dilated), loaded on demand."""
    def __init__(self, seq: Path, px: int):
        self.dir, self.px, self.cache = seq / "frames", px, {}
        self.k = np.asarray(json.loads((seq / "meta.json").read_text())["K"], dtype=np.float64)

    def hit(self, frame: int, p: np.ndarray) -> bool:
        if frame not in self.cache:
            self.cache[frame] = dilate(load_mask(self.dir / f"mask_{frame:03d}.png"), self.px)
        return on_mask(p, self.cache[frame], self.k)


def pipe_surface(seq: Path, frame: int = 0) -> np.ndarray:
    """(M, 3) camera-frame points of the target (pipe) in one frame: depth pixels inside the GT mask (not dilated).
    The baseline for "how much of the pipe do the A zones cover"."""
    meta = json.loads((seq / "meta.json").read_text())
    k = np.asarray(meta["K"], dtype=np.float64)
    z = np.load(seq / "frames" / f"depth_{frame:03d}.npy").astype(np.float64) / meta.get("depth_scale", 1000.0)
    m = load_mask(seq / "frames" / f"mask_{frame:03d}.png") & (z > 0)
    v, u = np.nonzero(m)
    zz = z[v, u]
    return np.stack([(u - k[0, 2]) * zz / k[0, 0], (v - k[1, 2]) * zz / k[1, 1], zz], axis=1)


def rerank(cands):
    """Ranks 1..n per frame again, in the original order, after some candidates were removed."""
    by_frame = {}
    for c in sorted(cands, key=lambda c: (c["frame"], c["rank"])):
        by_frame.setdefault(c["frame"], []).append(c)
    for fr in by_frame.values():
        for r, c in enumerate(fr):
            c["rank"] = r + 1
    for i, c in enumerate(cands):
        c["id"] = i
    return cands


def nearest_group(groups, pos, rot, max_mm, max_deg):
    """Group whose grasp point (centroid of its members) is closest to (pos, rot), within max_mm / max_deg.
    Returns (group, dist mm, angle deg) or (None, None, None)."""
    best, best_d, best_a = None, np.inf, None
    for g in groups:
        gr = g["grasp"]
        d = np.linalg.norm(np.array([gr["x"], gr["y"], gr["z"]]) - pos) * 1000.0
        if d > max_mm or d >= best_d:
            continue
        a = angle_deg(rot, np.asarray(gr["rotation_matrix"]))
        if max_deg is not None and a > max_deg:
            continue
        best, best_d, best_a = g, d, a
    if best is None:
        return None, None, None
    return best, round(best_d, 2), round(best_a, 1)


def nearest_candidate(cands, frame, pos, rot, max_mm, max_deg):
    """Closest candidate of `frame` to (pos, rot): (candidate, dist mm, angle deg) or (None, ...)."""
    best, best_d, best_a = None, np.inf, None
    for c in cands:
        if c["frame"] != frame:
            continue
        d = np.linalg.norm(c["pos"] - pos) * 1000.0
        a = angle_deg(rot, c["R"])
        if d <= max_mm and (max_deg is None or a <= max_deg) and d < best_d:
            best, best_d, best_a = c, d, a
    return best, (round(best_d, 2) if best is not None else None), (round(best_a, 1) if best is not None else None)


# --------------------------------------------------------------- visualisation
def gripper_mesh(pos, rot, width, depth, color, thick=0.004):
    """Same shape as graspnetAPI's plot_gripper_pro_max (two fingers, base, tail; x = approach, y = closing),
    but with the finger thickness `thick` in metres instead of its fixed 4 mm."""
    import open3d as o3d

    tail_length, depth_base = 0.04, 0.02
    boxes = [  # (size x, y, z), offset of the box corner, in the grasp frame
        ((depth + depth_base + thick, thick, thick), (-(depth_base + thick), -(width / 2 + thick), -thick / 2)),
        ((depth + depth_base + thick, thick, thick), (-(depth_base + thick), width / 2, -thick / 2)),
        ((thick, width, thick), (-(thick + depth_base), -width / 2, -thick / 2)),
        ((tail_length, thick, thick), (-(tail_length + thick + depth_base), -thick / 2, -thick / 2)),
    ]
    mesh = o3d.geometry.TriangleMesh()
    for (sx, sy, sz), off in boxes:
        b = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
        b.translate(off)
        mesh += b
    mesh.rotate(np.asarray(rot, dtype=np.float64), center=(0, 0, 0))
    mesh.translate(np.asarray(pos, dtype=np.float64))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def spheres(points, radius, color):
    """All points as one mesh of small spheres."""
    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    for q in points:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
        s.translate(np.asarray(q, dtype=np.float64))
        m += s
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


def visualise(cloud_path, view, groups_a, groups_c, cands_a, cands_c, tracks, palette, show, voxel,
              thick, point_r, members):
    import open3d as o3d

    data = np.load(cloud_path)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data[:, :3].astype(np.float64)))
    if data.shape[1] >= 6:
        rgb = data[:, 3:6]
        pcd.colors = o3d.utility.Vector3dVector((rgb / 255.0 if rgb.max() > 1 else rgb).astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel)
    geoms, labels = [("cloud", pcd)], []

    def add_groups(tag, groups, cands):
        for g in groups[:show]:
            gr = g["grasp"]
            col = palette[(g["group_rank"] - 1) % len(palette)]
            pos = np.array([gr["x"], gr["y"], gr["z"]])
            name = f"{tag} #{g['group_rank']}"
            geoms.append((f"{name} gripper",
                          gripper_mesh(pos, gr["rotation_matrix"], gr["width"], gr["depth"], col, thick)))
            geoms.append((f"{name} point", spheres([pos], point_r, col)))   # the group's grasp point
            if members:  # every member of the group (one per frame it was seen in), as in detect_burst.py
                pts = [cands[i]["pos"] for i in g.get("member_ids", []) if i < len(cands)]
                if pts:
                    geoms.append((f"{name} members", spheres(pts, point_r * 0.6, col)))
            labels.append((pos + [0, 0, -0.03 if tag == "A" else -0.045],
                           f"{tag}#{g['group_rank']} {g['frames_hit']}/{g['n_frames']}"))

    if "A" in view:
        add_groups("A", groups_a, cands_a)
    if "C" in view:
        add_groups("C", groups_c, cands_c)
    if "B" in view:
        for t in tracks[:show]:
            col = palette[(t["c_group"] - 1) % len(palette)] if t["c_group"] else (0.2, 0.2, 0.2)
            p = t["traj"]
            ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(p),
                                      o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(p) - 1)]))
            ls.paint_uniform_color(col)
            dots = spheres(p, point_r * 0.5, col)  # one per frame: a still grasp is one blob, a jumping one a trail
            # gripper at the START pose (frame 0): where the grasp was chosen. The trail shows where it went.
            geoms += [(f"B track {t['track']}", ls), (f"B dots {t['track']}", dots),
                      (f"B start point {t['track']}", spheres([p[0]], point_r, col)),
                      (f"B start {t['track']}", gripper_mesh(p[0], t["rot_first"], t["width_first"],
                                                              t["depth_first"], col, thick))]
            labels.append((p[0] + [0, 0, -0.06], f"B{t['track']} surv {t['survival']:.2f}"))

    try:
        import open3d.visualization.gui as gui
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer(f"Burst vs tracking ({view})", 1400, 900)
        for name, geom in geoms:
            vis.add_geometry(name, geom)
        for pos, text in labels:
            vis.add_3d_label(pos, text)
        vis.point_size = 2
        vis.show_settings = True  # the settings panel lists every geometry with a checkbox
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()
    except Exception as e:
        print(f"[vis] labelled viewer unavailable ({e}); colours only")
        o3d.visualization.draw_geometries([g for _, g in geoms], window_name=f"Burst vs tracking ({view})")


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst_root", required=True, help="the _sim copy of the burst folder (run A)")
    ap.add_argument("--track", required=True, help="runs/seq_XXXX/tracking/<tag> from track_replay.py")
    ap.add_argument("--show", type=int, default=10, help="groups / tracks drawn")
    ap.add_argument("--view", default="AB", help="runs to draw: any of A, B, C")
    ap.add_argument("--match_mm", type=float, default=None,
                    help="max distance from a B start point to an A group's grasp point (default: the group radius)")
    ap.add_argument("--group_mm", type=float, default=None,
                    help="regroup A and C with this radius instead of the one run A used (no need to rerun A)")
    ap.add_argument("--group_deg", type=float, default=None,
                    help="regroup A and C with this orientation limit (0 = position only). Default: run A's")
    ap.add_argument("--survival_mm", type=float, default=10.0,
                    help="B survival: a frame-to-frame step larger than this counts as a jump (kept apart from "
                         "--group_mm, so changing the groups does not change B)")
    ap.add_argument("--zone_top", type=int, default=10, help="A zones = grasp points of the top-N A groups")
    ap.add_argument("--zone_mm", type=float, default=None, help="radius of an A zone (default: the group radius)")
    ap.add_argument("--burst_result", default=None,
                    help="burst_result.json of run A (default: <burst_root>/burst_result.json). The multi-seed runner "
                         "keeps one per sequence")
    ap.add_argument("--anchor_mm", type=float, default=20.0,
                    help="anchored stability: a frame counts as 'same grasp' if within this distance of the frame-0 pose")
    ap.add_argument("--anchor_deg", type=float, default=None,
                    help="... and within this angle of the frame-0 orientation (default: run A's group angle, 30)")
    ap.add_argument("--anchored_min", type=float, default=0.9,
                    help="a track is 'stable (anchored)' if at least this share of its frames is anchored")
    ap.add_argument("--mask", action="store_true",
                    help="keep only grasps whose centre lies on the GT target mask (the pipe) in A and C, then "
                         "group again. The detector still saw the whole scene (collision check intact)")
    ap.add_argument("--dilate", type=int, default=8, help="mask margin in px (as batch_detect.py)")
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--gripper_mm", type=float, default=4.0,
                    help="finger thickness of the drawn grippers in mm (graspnetAPI draws 4)")
    ap.add_argument("--point_mm", type=float, default=4.0, help="radius of the grasp-point spheres in mm")
    ap.add_argument("--no_members", action="store_true", help="do not draw the members of each group")
    args = ap.parse_args()

    root = Path(args.burst_root).expanduser().resolve()
    track_dir = Path(args.track).expanduser().resolve()
    sys.path.insert(0, str(root))
    import detect_burst as db  # noqa: E402  (grouping functions only: nothing runs on import)

    res_a = json.loads((Path(args.burst_result).expanduser() if args.burst_result
                        else root / "burst_result.json").read_text())
    st = res_a["settings"]
    top_k, group_mm, group_deg, rank_mode = st["top_k"], st["group_mm"], st["group_deg"], st["rank_mode"]
    surv_deg = group_deg                           # B survival keeps run A's angle and --survival_mm
    if args.group_mm is not None:
        group_mm = args.group_mm
    if args.group_deg is not None:
        group_deg = None if args.group_deg <= 0 else args.group_deg
    regroup = args.group_mm is not None or args.group_deg is not None
    match_mm = args.match_mm or group_mm
    zone_mm = args.zone_mm or group_mm
    tr = np.load(track_dir / "tracks.npz")
    tmeta = json.loads((track_dir / "metrics.json").read_text())
    if "det_trans" not in tr:
        raise SystemExit("tracks.npz has no per-frame detections: rerun track_replay.py with --save_topk 10")

    # same frames?
    src = res_a.get("burst", {}).get("seq")
    if src and Path(src).resolve() != Path(tmeta["seq"]).resolve():
        print(f"[WARN] run A used {src}, the tracking used {tmeta['seq']}: not the same frames")
    n_a = len(res_a["frames"])
    n_frames = len(tr["frame"])
    if n_a != n_frames:
        print(f"[WARN] A has {n_a} frames, the tracking processed {n_frames} (realtime mode skips frames)")

    seq = Path(tmeta["seq"])
    masks = Masks(seq, args.dilate) if args.mask else None
    if args.mask and tmeta.get("no_mask", False):
        print("[WARN] --mask, but the tracking (B) was run with --no_mask: B still follows grasps outside the pipe. "
              "Rerun track_replay.py without --no_mask for a pipe-only comparison")
    if args.mask and top_k < 200:
        print(f"[WARN] run A kept only top_k {top_k} per frame: pipe grasps outside it are missing. Use --top_k 1024")

    # ---- A: candidates from burst_result.json (regrouped when masked)
    cands_a = [{"frame": c["frame"], "rank": c["rank"], "score": c["score"], "width": c["width"], "depth": c["depth"],
                "pos": np.array([c["x"], c["y"], c["z"]]), "R": np.asarray(c["rotation_matrix"]),
                "group_rank": c.get("group_rank")} for c in res_a["candidates"]]
    n_a_all = len(cands_a)
    if masks is not None or regroup:
        if masks is not None:
            cands_a = [c for c in cands_a if masks.hit(c["frame"], c["pos"])]
        cands_a = rerank(cands_a)
        groups_a = db.summarise_groups(db.group_candidates(cands_a, group_mm, group_deg), cands_a, n_a, top_k,
                                       rank_mode) if cands_a else []
    else:
        groups_a = res_a["groups"]
    for g in groups_a:
        g["n_frames"] = n_a

    # ---- C: detect_burst's grouping on the tracker's per-frame detections
    cands_c = []
    for fi in range(n_frames):
        for r in range(int(tr["det_n"][fi])):
            if r >= top_k:
                break
            cands_c.append({"frame": fi, "rank": r + 1, "score": float(tr["det_score"][fi, r]),
                            "width": float(tr["det_width"][fi, r]), "depth": float(tr["det_depth"][fi, r]),
                            "pos": tr["det_trans"][fi, r].astype(float), "R": tr["det_rot"][fi, r].astype(float)})
    n_c_all = len(cands_c)
    if masks is not None:
        cands_c = [c for c in cands_c if masks.hit(int(tr["frame"][c["frame"]]), c["pos"])]
    cands_c = rerank(cands_c)
    groups_c = db.summarise_groups(db.group_candidates(cands_c, group_mm, group_deg), cands_c, n_frames, top_k,
                                   rank_mode) if cands_c else []
    for g in groups_c:
        g["n_frames"] = n_frames
    if masks is not None:
        print(f"mask on: A keeps {len(cands_a)}/{n_a_all} candidates on the pipe, C keeps {len(cands_c)}/{n_c_all}")

    # ---- B: one record per tracked grasp
    traj, rots = tr["trans_c"], tr["rot_c"]                 # (F, n, 3), (F, n, 3, 3)
    zones = groups_a[:args.zone_top]                        # the "persistent zones" of the burst method
    zone_pts = np.array([[g["grasp"]["x"], g["grasp"]["y"], g["grasp"]["z"]] for g in zones]).reshape(-1, 3)
    tracks = []
    for j in range(traj.shape[1]):
        p, r = traj[:, j].astype(float), rots[:, j].astype(float)
        ok = np.isfinite(p[:, 0])
        p, r = p[ok], r[ok]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1) * 1000.0
        rstep = np.array([angle_deg(r[i], r[i + 1]) for i in range(len(r) - 1)])
        stay = step <= args.survival_mm
        if surv_deg is not None:
            stay &= rstep <= surv_deg
        dev = np.linalg.norm(p - p[0], axis=1) * 1000.0
        # anchored stability: share of frames in which the grasp is still the frame-0 grasp (position AND angle).
        # Unlike survival, oscillating between close neighbours does not count against it, jumping away and
        # staying there does.
        a_deg = args.anchor_deg if args.anchor_deg is not None else surv_deg
        rdev0 = np.array([angle_deg(r[0], r[i]) for i in range(len(r))])
        anchored = dev <= args.anchor_mm
        if a_deg is not None:
            anchored &= rdev0 <= a_deg
        c0, dc, _ = nearest_candidate(cands_c, 0, p[0], r[0], 1.0, None)       # exact with --select_nms
        if c0 is None:
            c0, dc, _ = nearest_candidate(cands_c, 0, p[0], r[0], match_mm, group_deg)
        cg = c0.get("group_rank") if c0 else None
        gc = next((g for g in groups_c if g["group_rank"] == cg), None)
        # A: the A group (zone) the B grasp STARTS in, by distance to the group's grasp point (not only frame 0:
        # an A group may have no member in frame 0 and still be a persistent zone)
        ga, da, aa = nearest_group(groups_a, p[0], r[0], match_mm, group_deg)
        ag = ga["group_rank"] if ga else None
        # how much of B's trajectory lies inside the top-N A zones (position only)
        if len(zone_pts):
            dz = np.linalg.norm(p[:, None, :] - zone_pts[None], axis=-1) * 1000.0   # (F, N)
            inside = dz.min(axis=1) <= zone_mm
            zone_frac = float(inside.mean())
            hit = dz.argmin(axis=1)[inside]
            zone_main = int(zones[np.bincount(hit).argmax()]["group_rank"]) if len(hit) else None
        else:
            zone_frac, zone_main = 0.0, None
        on_pipe0 = masks.hit(int(tr["frame"][0]), p[0]) if masks is not None else None
        tracks.append({
            "track": j,
            "on_pipe_frame0": on_pipe0,
            "score_frame0": round(float(tr["scores"][0, j]), 4),
            "survival": round(float(stay.mean()) if len(stay) else 1.0, 3),
            "anchored": round(float(anchored.mean()), 3),
            "jumps": int((step > args.survival_mm).sum()),
            "in_A_zones": round(zone_frac, 3), "A_zone_main": zone_main,
            "jitter_mm": round(float(np.sqrt((p.std(axis=0) ** 2).sum()) * 1000.0), 2),
            "drift_mm": round(float(dev.max()), 2),
            "max_step_mm": round(float(step.max()) if len(step) else 0.0, 2),
            "max_rot_step_deg": round(float(rstep.max()) if len(rstep) else 0.0, 1),
            "c_group": cg, "c_support": gc["support"] if gc else None,
            "c_spread_mm": round(gc["spread_mm"]["rms"], 2) if gc else None, "c_match_mm": dc,
            "a_group": ag, "a_support": ga["support"] if ga else None, "a_match_mm": da, "a_match_deg": aa,
            "traj": p, "rot_first": r[0],
            "width_first": float(tr["widths"][ok][0, j]), "depth_first": float(tr["depths"][ok][0, j]),
        })

    # ---- report
    print(f"groups: radius {group_mm:.0f} mm, angle {group_deg if group_deg is not None else 'off'}"
          f"{' (regrouped here, not run A settings)' if regroup else ' (as run A)'} | top_k {top_k}, rank_mode "
          f"{rank_mode} | B survival: steps <= {args.survival_mm:.0f} mm, {surv_deg} deg | "
          f"A zones: top {args.zone_top}, radius {zone_mm:.0f} mm | {n_frames} frames")
    print(f"\nA (detector + grouping): {len(groups_a)} groups | C (tracker grasps + grouping): {len(groups_c)} groups")
    print(f"{'#':>3} | {'A frames':>8} {'A spread':>8} {'A rot':>11} | {'C frames':>8} {'C spread':>8} {'C rot':>11}")
    print(f"{'':>3} | {'':>8} {'(rms)':>8} {'(mean/max)':>11} | {'':>8} {'(rms)':>8} {'(mean/max)':>11}")
    for i in range(min(10, max(len(groups_a), len(groups_c)))):
        ga = groups_a[i] if i < len(groups_a) else None
        gc = groups_c[i] if i < len(groups_c) else None
        fa = f"{ga['frames_hit']}/{n_a}" if ga else "-"
        fc = f"{gc['frames_hit']}/{n_frames}" if gc else "-"
        sa = f"{ga['spread_mm']['rms']:.1f}mm" if ga else "-"
        sc = f"{gc['spread_mm']['rms']:.1f}mm" if gc else "-"
        # orientation spread: angle of every member to the group's representative (best) grasp
        ra = f"{ga['orientation_spread_deg']['mean']:.0f}/{ga['orientation_spread_deg']['max']:.0f}deg" if ga else "-"
        rc = f"{gc['orientation_spread_deg']['mean']:.0f}/{gc['orientation_spread_deg']['max']:.0f}deg" if gc else "-"
        print(f"{i + 1:>3} | {fa:>8} {sa:>8} {ra:>11} | {fc:>8} {sc:>8} {rc:>11}")
    for tag, gs in (("A", groups_a), ("C", groups_c)):
        if gs:
            top = gs[:10]
            print(f"  {tag} top-10 mean: support {np.mean([g['support'] for g in top]):.2f}, "
                  f"spread {np.mean([g['spread_mm']['rms'] for g in top]):.1f} mm, "
                  f"orientation spread {np.mean([g['orientation_spread_deg']['mean'] for g in top]):.0f} deg mean / "
                  f"{np.mean([g['orientation_spread_deg']['max'] for g in top]):.0f} deg max")

    print(f"\nB (tracking) per tracked grasp, and the group it started in:")
    print(f"{'trk':>3} {'score':>6} {'surv':>5} {'anch':>5} {'jumps':>5} {'jitter':>7} {'drift':>7} | "
          f"{'C grp':>5} {'C supp':>6} | {'A grp':>5} {'A supp':>6} {'A dist':>6} | {'in A top':>8} {'main':>4}")
    for t in tracks:
        cs = f"{t['c_support']:.2f}" if t["c_support"] is not None else "-"
        asu = f"{t['a_support']:.2f}" if t["a_support"] is not None else "-"
        ad = f"{t['a_match_mm']:.1f}" if t["a_match_mm"] is not None else "-"
        print(f"{t['track']:>3} {t['score_frame0']:6.3f} {t['survival']:5.2f} {t['anchored']:5.2f} {t['jumps']:>5} "
              f"{t['jitter_mm']:6.1f}mm {t['drift_mm']:6.1f}mm | {t['c_group'] or '-':>5} {cs:>6} | "
              f"{t['a_group'] or '-':>5} {asu:>6} {ad:>6} | {t['in_A_zones'] * 100:7.0f}% {t['A_zone_main'] or '-':>4}"
              + ("" if t["on_pipe_frame0"] is None else ("  pipe" if t["on_pipe_frame0"] else "  OFF pipe")))

    baseline = None
    corr = {
        "C_support_vs_B_survival": spearman([t["c_support"] or np.nan for t in tracks], [t["survival"] for t in tracks]),
        "C_support_vs_B_jitter": spearman([t["c_support"] or np.nan for t in tracks], [t["jitter_mm"] for t in tracks]),
        "A_support_vs_B_survival": spearman([t["a_support"] or np.nan for t in tracks], [t["survival"] for t in tracks]),
        "C_support_vs_B_anchored": spearman([t["c_support"] or np.nan for t in tracks], [t["anchored"] for t in tracks]),
        "A_support_vs_B_anchored": spearman([t["a_support"] or np.nan for t in tracks], [t["anchored"] for t in tracks]),
    }
    n_a_matched = sum(t["a_group"] is not None for t in tracks)
    print(f"\nSpearman (n = {len(tracks)}, only indicative with so few grasps): "
          + " | ".join(f"{k} {v:+.2f}" if v is not None else f"{k} n/a" for k, v in corr.items()))
    print(f"B grasps that START within {match_mm:.0f} mm of an A group's grasp point: {n_a_matched}/{len(tracks)} "
          f"(low = the detector and the tracker propose different grasps)")
    if tracks and len(zone_pts):
        allf = np.mean([t["in_A_zones"] for t in tracks])
        stable = [t["in_A_zones"] for t in tracks if t["survival"] >= 0.6]
        unstable = [t["in_A_zones"] for t in tracks if t["survival"] < 0.6]
        covered = {t["A_zone_main"] for t in tracks if t["A_zone_main"] is not None}
        print(f"B trajectory frames inside the top-{len(zones)} A zones (r {zone_mm:.0f} mm): {allf * 100:.0f}% overall | "
              f"stable tracks (surv >= 0.6, n={len(stable)}): "
              f"{(np.mean(stable) * 100 if stable else float('nan')):.0f}% | "
              f"unstable (n={len(unstable)}): {(np.mean(unstable) * 100 if unstable else float('nan')):.0f}% | "
              f"A zones visited by some B track: {len(covered)}/{len(zones)}")

        # baseline: share of the pipe's visible surface inside the same zones = what a grasp placed at random on
        # the pipe would score. Only the excess over it says that B and A point at the same places.
        try:
            surf = pipe_surface(seq, int(tr["frame"][0]))
        except FileNotFoundError:
            surf = np.zeros((0, 3))
        if len(surf):
            sub = surf[:: max(1, len(surf) // 20000)]             # plenty of points for a percentage
            dz = np.linalg.norm(sub[:, None, :] - zone_pts[None], axis=-1).min(axis=1) * 1000.0
            base = float((dz <= zone_mm).mean())

            def lift(x):
                return f"{x / base:.2f}x" if base > 0 and not np.isnan(x) else "n/a"

            def mean_or_nan(v):
                return float(np.mean(v)) if v else float("nan")
            stable_a = [t["in_A_zones"] for t in tracks if t["anchored"] >= args.anchored_min]
            unstable_a = [t["in_A_zones"] for t in tracks if t["anchored"] < args.anchored_min]
            s_m, u_m = mean_or_nan(stable), mean_or_nan(unstable)
            sa_m, ua_m = mean_or_nan(stable_a), mean_or_nan(unstable_a)
            print(f"BASELINE: {base * 100:.0f}% of the pipe surface lies inside those zones (a random point on the pipe)"
                  f" -> B overall {allf * 100:.0f}% ({lift(allf)})")
            print(f"  by survival >= 0.6        : stable {s_m * 100:.0f}% ({lift(s_m)}, n={len(stable)}) | "
                  f"unstable {u_m * 100:.0f}% ({lift(u_m)}, n={len(unstable)})")
            print(f"  by anchored >= {args.anchored_min:.1f} ({args.anchor_mm:.0f} mm): stable {sa_m * 100:.0f}% "
                  f"({lift(sa_m)}, n={len(stable_a)}) | unstable {ua_m * 100:.0f}% ({lift(ua_m)}, n={len(unstable_a)})")
            print("  lift > 1: B sits in A's persistent zones more than chance; stable lift > unstable lift: "
                  "the two measures agree")

            def r3(x):
                return None if np.isnan(x) else round(float(x), 3)
            baseline = {"pipe_surface_in_zones": round(base, 3), "B_overall": round(float(allf), 3),
                        "B_stable": r3(s_m), "B_unstable": r3(u_m),
                        "n_stable": len(stable), "n_unstable": len(unstable),
                        "B_stable_anchored": r3(sa_m), "B_unstable_anchored": r3(ua_m),
                        "n_stable_anchored": len(stable_a), "n_unstable_anchored": len(unstable_a),
                        "lift_overall": r3(allf / base) if base > 0 else None,
                        "lift_stable": r3(s_m / base) if base > 0 else None,
                        "lift_unstable": r3(u_m / base) if base > 0 else None,
                        "lift_stable_anchored": r3(sa_m / base) if base > 0 else None,
                        "lift_unstable_anchored": r3(ua_m / base) if base > 0 else None}
        else:
            print("BASELINE: no mask_XXX.png / depth in the sequence: baseline skipped")

    out = {
        "burst_root": str(root), "track": str(track_dir), "frames": n_frames,
        "settings": {"top_k": top_k, "group_mm": group_mm, "group_deg": group_deg, "rank_mode": rank_mode,
                     "regrouped": regroup, "match_mm": match_mm, "survival_mm": args.survival_mm,
                     "anchor_mm": args.anchor_mm, "anchor_deg": args.anchor_deg, "anchored_min": args.anchored_min,
                     "survival_deg": surv_deg, "zone_top": args.zone_top, "zone_mm": zone_mm},
        "groups_A": [{k: g[k] for k in ("group_rank", "frames_hit", "support", "avg_rank", "avg_score", "spread_mm",
                                        "orientation_spread_deg", "grasp")} for g in groups_a],
        "groups_C": [{k: g[k] for k in ("group_rank", "frames_hit", "support", "avg_rank", "avg_score", "spread_mm",
                                        "orientation_spread_deg", "grasp")} for g in groups_c],
        "mask": {"on": args.mask, "dilate_px": args.dilate},
        "tracks_B": [{k: v for k, v in t.items() if k not in ("traj", "rot_first")} for t in tracks],
        "spearman": corr,
        "zone_baseline": baseline,
    }
    (track_dir / "compare.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[INFO] saved -> {track_dir / 'compare.json'}")

    if args.vis:
        visualise(root / "burst" / "cloud_000.npy", args.view.upper(), groups_a, groups_c, cands_a, cands_c,
                  tracks, db.PALETTE, args.show, 0.005, args.gripper_mm / 1000.0, args.point_mm / 1000.0,
                  not args.no_members)


if __name__ == "__main__":
    main()
