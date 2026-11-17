"""AnyGrasp on the 3D MODEL of the scene vs on the Isaac Sim POINT CLOUD, and the quality of both candidate sets.

One frame of a sequence recorded with scripts/capture_sequence_pipefit.py (it saves meshes.npz: the bin and tube
meshes, and poses.npz: their poses + the camera pose). The detector runs on up to four clouds of the SAME scene,
all in the camera frame:

  sensor         the camera's depth of that frame (with the Gemini 2 noise)          -> what the robot would get
  sensor_clean   the same view without noise (frames/depth_clean_000.npy, frame 0 only)
  model_visible  the meshes rendered from the camera pose: perfect depth, only the visible side
  model_full     points sampled on the WHOLE surface of the tubes (all sides, also the hidden ones)

Every candidate of every cloud is then checked against the TRUE geometry (the meshes, tubes + bin), with the
model-free check of graspnetAPI (same gripper model as the GraspNet benchmark):
  collision   the fingers, the palm or the approach corridor go through an object
  empty       nothing between the fingers
  no_fc       something between the fingers, but a contact normal is outside the friction cone (mu): it would slip
  ok          none of the above (antipodal grasp that fits)
and "tube" = the object between the fingers is a tube (not the bin).

Report per cloud: candidates, ok / collision / empty / no_fc rates, ok-on-a-tube rate of the top-10 / top-50 by
score (what a robot picking by score would get), and how many of its top-50 the sensor cloud also proposes
(within MATCH mm / deg).

Environment: anygrasp (+ open3d, scipy). Needs group_and_trace.py next to it (compare_burst_tracking.py for --vis).
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/model_vs_cloud.py --seq runs/pipefit/seq_0000
    ... --frame 5 --clouds sensor model_visible --vis
Output: <seq>/model_vs_cloud/ summary.csv + grasps_<cloud>.csv (one row per candidate, with its label)
"""

# =====================================================================================================
# CONFIG: edit here
# =====================================================================================================
CONFIG = {
    "FRAME": 0,
    "CLOUDS": ["sensor", "sensor_clean", "model_visible", "model_full"],
    "MODEL": {
        "visible_objects": "all",          # model_visible: "all" (tubes + bin, a perfect sensor) | "targets" (tubes)
        "full_objects": "targets",         # model_full:    "targets" (tubes only) | "all"
        "sample_mm": 2.0,                  # model_full: surface sampling spacing
    },
    "DETECTOR": {                          # same settings as seq_group_track.py
        "workspace": [(-0.4, 0.4), (-0.4, 0.4), (0.05, 1.0)],
        "voxel": 0.005,
        "max_gripper_width": 0.10,
        "gripper_height": 0.03,
        "collision_detection": True,
        "nms": True,
        "top_k": 0,
    },
    "EVAL": {                              # graspnetAPI ModelFreeCollisionDetector + an antipodal check
        "gt_spacing_mm": 2.0,              # density of the true-geometry points
        "finger_width": 0.01,
        "finger_length": 0.06,
        "height": 0.02,
        "approach_dist": 0.05,
        "collision_points": 3,             # more true points than this inside fingers / palm / corridor = collision
        "empty_points": 5,                 # fewer true points than this between the fingers = empty
        "contact_band_mm": 2.0,            # points this close to the outermost one on each side form a contact
        "mu": 0.4,                         # friction: contact normal within atan(mu) = 21.8 deg of the closing axis
    },
    "MATCH": {"mm": 20.0, "deg": 30.0},    # "same grasp" between two clouds (flip-aware angle)
    "TOP": [10, 50],
    "VIS": {"top": 10, "labels": 5, "gripper_mm": 3.0, "gt_points": 60000},
}
# =====================================================================================================

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import group_and_trace as gat  # noqa: E402

TAGS = {"sensor": "S", "sensor_clean": "C", "model_visible": "V", "model_full": "F"}
COLOURS = {"sensor": (0.12, 0.47, 0.95), "sensor_clean": (0.05, 0.80, 0.85), "model_visible": (1.0, 0.5, 0.05),
           "model_full": (0.10, 0.75, 0.20)}


# --------------------------------------------------------------- geometry helpers
def quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pose(p, q):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = quat_wxyz_to_mat(q), p
    return t


def sample_surface(v, f, spacing, rng):
    """Area-uniform surface points with their face normals: ~1 point per spacing^2."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    area = np.linalg.norm(cr, axis=1) / 2
    n = max(1, int(np.ceil(area.sum() / spacing ** 2)))
    face = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * a[face] + (s * (1 - r2))[:, None] * b[face] + (s * r2)[:, None] * c[face]
    nrm = cr[face] / np.maximum(np.linalg.norm(cr[face], axis=1, keepdims=True), 1e-12)
    return pts, nrm


def depth_to_points(depth_m, k):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ok = np.isfinite(depth_m) & (depth_m > 0)
    z = depth_m[ok]
    return np.stack([(u[ok] - k[0, 2]) * z / k[0, 0], (v[ok] - k[1, 2]) * z / k[1, 1], z], 1), ok


def render_visible(meshes, k, w, h):
    """Perfect depth of the meshes seen from the camera (camera frame = OpenCV, as the meshes)."""
    import open3d as o3d
    try:
        scene = o3d.t.geometry.RaycastingScene()
        for v, f in meshes:
            scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32)))
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(k.astype(np.float64)), extrinsic_matrix=o3d.core.Tensor(np.eye(4)),
            width_px=int(w), height_px=int(h))
        hit = scene.cast_rays(rays)["t_hit"].numpy()
        r = rays.numpy()
        ok = np.isfinite(hit)
        return r[..., :3][ok] + r[..., 3:][ok] * hit[ok][:, None]
    except AttributeError:                               # old open3d: hidden point removal on dense samples
        rng = np.random.default_rng(0)
        pts = np.concatenate([sample_surface(v, f, 0.002, rng)[0] for v, f in meshes])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        _, idx = pcd.hidden_point_removal([0.0, 0.0, 0.0], 100.0 * np.linalg.norm(pts, axis=1).max())
        return pts[idx]


def angle_ok(ra, rb, cos_min):
    """Flip-aware 'same orientation' test for many pairs: (n, 3, 3) x (n, 3, 3) -> (n,) bool."""
    col = np.einsum("pkj,pkj->pj", ra, rb)
    tr = np.maximum(col.sum(1), col[:, 0] - col[:, 1] - col[:, 2])
    return (tr - 1.0) / 2.0 >= cos_min - 1e-9


def matched(pa, ra, pb, rb, cfg):
    """For every grasp of set A: is there a grasp of set B within MATCH mm and deg?"""
    out = np.zeros(len(pa), bool)
    if not len(pa) or not len(pb):
        return out
    d = np.linalg.norm(pa[:, None] - pb[None], axis=2) * 1000.0
    i, j = np.nonzero(d <= cfg["mm"])
    if len(i):
        ok = angle_ok(ra[i], rb[j], np.cos(np.radians(cfg["deg"])))
        out[np.unique(i[ok])] = True
    return out


# --------------------------------------------------------------- scene
def load_scene(seq: Path, frame: int):
    meta = json.loads((seq / "meta.json").read_text())
    poses = np.load(seq / "poses.npz")
    if not (seq / "meshes.npz").exists():
        raise SystemExit(f"{seq / 'meshes.npz'} not found: record the sequence with scripts/capture_sequence_pipefit.py "
                         "(the version that exports the meshes)")
    mz = np.load(seq / "meshes.npz")
    k = np.asarray(meta["K"], float)
    t_cam_w = np.linalg.inv(pose(poses["cam_pos_w"][frame], poses["cam_quat_w_ros"][frame]))
    targets = set(meta.get("targets") or [n for n in mz["names"] if str(n).startswith("pipe")])
    objects = []
    for name in [str(n) for n in mz["names"]]:
        t = t_cam_w @ pose(poses[f"{name}_pos_w"][frame], poses[f"{name}_quat_w"][frame])
        v = mz[f"{name}_verts"].astype(np.float64) @ t[:3, :3].T + t[:3, 3]
        objects.append({"name": name, "verts": v, "faces": mz[f"{name}_faces"], "target": name in targets})
    return meta, k, objects


def in_workspace(v, ws):
    return np.all([(v[:, a] > lo) & (v[:, a] < hi) for a, (lo, hi) in enumerate(ws)], axis=0).any()


def build_clouds(seq, meta, k, objects, cfg, frame):
    from PIL import Image
    w, h = meta["resolution"]
    ws = cfg["DETECTOR"]["workspace"]
    scale = meta.get("depth_scale", 1000.0)
    rgb = np.asarray(Image.open(seq / "frames" / f"color_{frame:03d}.png"))[..., :3] / 255.0
    clouds = {}
    for name in cfg["CLOUDS"]:
        if name == "sensor":
            d = np.load(seq / "frames" / f"depth_{frame:03d}.npy").astype(np.float64) / scale
            pts, ok = depth_to_points(d, k)
            clouds[name] = (pts, rgb[ok])
        elif name == "sensor_clean":
            p = seq / "frames" / "depth_clean_000.npy"
            if frame != 0 or not p.exists():
                print(f"[skip] sensor_clean: only frame 0 has a clean depth ({p.name})")
                continue
            pts, ok = depth_to_points(np.load(p).astype(np.float64) / scale, k)
            clouds[name] = (pts, rgb[ok])
        elif name in ("model_visible", "model_full"):
            which = cfg["MODEL"]["visible_objects" if name == "model_visible" else "full_objects"]
            sel = [o for o in objects if (which == "all" or o["target"]) and in_workspace(o["verts"], ws)]
            if not sel:
                print(f"[skip] {name}: no object inside the workspace")
                continue
            if name == "model_visible":
                pts = render_visible([(o["verts"], o["faces"]) for o in sel], k, w, h)
            else:
                rng = np.random.default_rng(0)
                pts = np.concatenate([sample_surface(o["verts"], o["faces"], cfg["MODEL"]["sample_mm"] / 1000.0, rng)[0]
                                      for o in sel])
            clouds[name] = (pts, np.full((len(pts), 3), 0.7))
        else:
            raise SystemExit(f"unknown cloud {name!r}")
    return clouds


# --------------------------------------------------------------- detection (+ cache)
def detect(name, pts, rgb, cfg, cache_dir):
    key = hashlib.sha1(json.dumps({"n": len(pts), "s": float(np.round(pts.sum(), 4)), "det": cfg["DETECTOR"]},
                                  default=str, sort_keys=True).encode()).hexdigest()[:16]
    path = cache_dir / f"{name}.npz"
    if path.exists():
        z = dict(np.load(path))
        if str(z["key"]) == key:
            print(f"[detect] {name}: cached ({len(z['score'])} grasps)")
            return z
    print(f"[detect] {name}: {len(pts)} points")
    cands = gat.run_detector([(0, pts, rgb)], cfg["DETECTOR"])
    z = {"key": key, "rank": np.array([c["rank"] for c in cands], int),
         "score": np.array([c["score"] for c in cands], float), "width": np.array([c["width"] for c in cands], float),
         "depth": np.array([c["depth"] for c in cands], float),
         "pos": np.array([c["pos"] for c in cands], float).reshape(-1, 3),
         "R": np.array([c["R"] for c in cands], float).reshape(-1, 3, 3)}
    np.savez_compressed(path, **z)
    return z


# --------------------------------------------------------------- evaluation against the true geometry
class TrueGeometry:
    def __init__(self, objects, cfg):
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(1)
        pts, nrm, oid = [], [], []
        for i, o in enumerate(objects):
            p, n = sample_surface(o["verts"], o["faces"], cfg["gt_spacing_mm"] / 1000.0, rng)
            pts.append(p), nrm.append(n), oid.append(np.full(len(p), i))
        self.pts, self.nrm, self.oid = np.concatenate(pts), np.concatenate(nrm), np.concatenate(oid)
        self.objects, self.cfg, self.tree = objects, cfg, cKDTree(self.pts)
        print(f"[eval] true geometry: {len(self.pts)} points on {len(objects)} objects "
              f"({', '.join(o['name'] for o in objects)})")

    def check(self, t, r, width, depth):
        c = self.cfg
        fl, fw, hh, ad = c["finger_length"], c["finger_width"], c["height"] / 2, c["approach_dist"]
        reach = np.sqrt((fl + fw + ad + depth) ** 2 + (width / 2 + fw) ** 2 + hh ** 2)
        idx = np.asarray(self.tree.query_ball_point(t, reach), int)
        if not len(idx):
            return {"label": "empty", "object": "", "tube": False, "inner": 0, "coll_pts": 0,
                    "angle_a": None, "angle_b": None, "thickness_mm": None}
        loc = (self.pts[idx] - t) @ r                       # grasp frame: x approach, y closing, z height
        x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
        m1 = np.abs(z) < hh
        m2 = (x > depth - fl) & (x < depth)
        m3, m4 = y > -(width / 2 + fw), y < -width / 2
        m5, m6 = y < (width / 2 + fw), y > width / 2
        m7 = (x <= depth - fl) & (x > depth - fl - fw)
        m8 = (x <= depth - fl - fw) & (x > depth - fl - fw - ad)
        coll = ((m1 & m2 & m3 & m4) | (m1 & m2 & m5 & m6) | (m1 & m3 & m5 & m7) | (m1 & m3 & m5 & m8)).sum()
        inner = m1 & m2 & ~m4 & ~m6
        n_in = int(inner.sum())
        res = {"inner": n_in, "coll_pts": int(coll), "angle_a": None, "angle_b": None, "thickness_mm": None,
               "object": "", "tube": False}
        if n_in:
            ids = self.oid[idx[inner]]
            main = np.bincount(ids).argmax()
            res["object"], res["tube"] = self.objects[main]["name"], bool(self.objects[main]["target"])
            sel = inner.copy()
            sel[inner] = ids == main
            ys, nn = y[sel], self.nrm[idx[sel]]
            band = c["contact_band_mm"] / 1000.0
            res["thickness_mm"] = float((ys.max() - ys.min()) * 1000.0)
            closing = r[:, 1]
            for key, side in (("angle_a", ys <= ys.min() + band), ("angle_b", ys >= ys.max() - band)):
                d = nn[side] @ closing
                n_mean = (nn[side] * np.sign(d + 1e-12)[:, None]).mean(0)
                res[key] = float(np.degrees(np.arccos(np.clip(abs(n_mean @ closing) / max(np.linalg.norm(n_mean), 1e-12),
                                                              -1, 1))))
        cone = np.degrees(np.arctan(c["mu"]))
        if coll > c["collision_points"]:
            res["label"] = "collision"
        elif n_in < c["empty_points"]:
            res["label"] = "empty"
        elif max(res["angle_a"], res["angle_b"]) > cone:
            res["label"] = "no_fc"
        else:
            res["label"] = "ok"
        return res


# --------------------------------------------------------------- main
def pct(x):
    return f"{100 * x:5.1f}%" if x is not None and np.isfinite(x) else "    -"


def main():
    ap = argparse.ArgumentParser(description="AnyGrasp on the 3D model vs on the sim point cloud")
    ap.add_argument("--seq", required=True, help="sequence from scripts/capture_sequence_pipefit.py")
    ap.add_argument("--frame", type=int, help="override CONFIG FRAME")
    ap.add_argument("--clouds", nargs="+", help="override CONFIG CLOUDS: sensor sensor_clean model_visible model_full")
    ap.add_argument("--out", help="output folder (default <seq>/model_vs_cloud)")
    ap.add_argument("--save_clouds", action="store_true",
                    help="also save every cloud as <out>/clouds/<cloud>/cloud_XXX.npy (N, 6)")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()
    cfg = CONFIG
    if args.frame is not None:
        cfg["FRAME"] = args.frame
    if args.clouds:
        cfg["CLOUDS"] = args.clouds
    seq = Path(args.seq).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else seq / "model_vs_cloud"
    (out / ".cache").mkdir(parents=True, exist_ok=True)
    frame = cfg["FRAME"]

    meta, k, objects = load_scene(seq, frame)
    clouds = build_clouds(seq, meta, k, objects, cfg, frame)
    if args.save_clouds:                                   # one folder per cloud, seq_group_track.py can read it
        for name, (pts, rgb) in clouds.items():
            d = out / "clouds" / name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"cloud_{frame:03d}.npy",
                    np.c_[pts, np.clip(rgb * 255.0, 0, 255)].astype(np.float32))
        print(f"[clouds] saved to {out / 'clouds'}/<cloud>/cloud_{frame:03d}.npy  (N, 6): xyz m, camera frame + rgb")
    geo = TrueGeometry(objects, cfg["EVAL"])

    sets = {}
    for name, (pts, rgb) in clouds.items():
        z = detect(name, pts, rgb, cfg, out / ".cache")
        order = np.argsort(-z["score"])
        z = {key: (v[order] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(order) else v)
             for key, v in z.items()}
        z["eval"] = [geo.check(z["pos"][i], z["R"][i], z["width"][i], z["depth"][i]) for i in range(len(z["score"]))]
        z["n_points"] = len(pts)
        sets[name] = z

    # agreement with the sensor cloud
    ref = sets.get("sensor")
    for name, z in sets.items():
        z["in_sensor"] = matched(z["pos"], z["R"], ref["pos"], ref["R"], cfg["MATCH"]) if ref is not None else None

    # tables
    rows = []
    for name, z in sets.items():
        lab = np.array([e["label"] for e in z["eval"]])
        tube = np.array([e["tube"] for e in z["eval"]])
        good = (lab == "ok") & tube
        n = len(lab)
        row = {"cloud": name, "points": z["n_points"], "grasps": n,
               "score_mean": round(float(z["score"].mean()), 4) if n else None,
               "ok": (lab == "ok").mean() if n else None, "collision": (lab == "collision").mean() if n else None,
               "empty": (lab == "empty").mean() if n else None, "no_fc": (lab == "no_fc").mean() if n else None,
               "on_tube": tube.mean() if n else None, "ok_on_tube": good.mean() if n else None}
        for t in cfg["TOP"]:
            row[f"top{t}_ok_on_tube"] = good[:t].mean() if n else None
        row["top50_also_in_sensor"] = (z["in_sensor"][:50].mean() if n else None) \
            if z["in_sensor"] is not None and name != "sensor" else None
        rows.append(row)
        with open(out / f"grasps_{name}.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["rank", "score", "label", "object", "tube", "angle_a_deg", "angle_b_deg", "thickness_mm",
                         "inner_pts", "collision_pts", "in_sensor", "width_mm", "depth_mm", "tilt_deg", "x", "y", "z"]
                        + [f"r{a}{b}" for a in range(3) for b in range(3)])
            for i, e in enumerate(z["eval"]):
                rr = z["R"][i]
                wr.writerow([i + 1, round(float(z["score"][i]), 4), e["label"], e["object"], e["tube"],
                             None if e["angle_a"] is None else round(e["angle_a"], 1),
                             None if e["angle_b"] is None else round(e["angle_b"], 1),
                             None if e["thickness_mm"] is None else round(e["thickness_mm"], 1), e["inner"],
                             e["coll_pts"], None if z["in_sensor"] is None else bool(z["in_sensor"][i]),
                             round(float(z["width"][i]) * 1000, 1), round(float(z["depth"][i]) * 1000, 1),
                             round(gat.tilt_deg(rr), 1)] + [round(float(v), 4) for v in z["pos"][i]]
                            + [round(float(v), 5) for v in rr.ravel()])
    gat.write_csv(out / "summary.csv", [{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}
                                        for r in rows])

    cone = np.degrees(np.arctan(cfg["EVAL"]["mu"]))
    print(f"\nFRAME {frame} of {seq.name}: every candidate checked against the true meshes "
          f"(collision / empty / friction cone {cone:.0f} deg)")
    print(f"  {'cloud':<14} {'points':>7} {'grasps':>6} {'score':>6} | {'ok':>6} {'collis':>6} {'empty':>6} "
          f"{'no_fc':>6} | {'tube':>6} {'ok+tube':>7} | " + " ".join(f"{'top' + str(t):>6}" for t in cfg["TOP"])
          + f" | {'in sens':>7}")
    for r in rows:
        print(f"  {r['cloud']:<14} {r['points']:>7} {r['grasps']:>6} {r['score_mean'] or 0:6.3f} | {pct(r['ok'])} "
              f"{pct(r['collision'])} {pct(r['empty'])} {pct(r['no_fc'])} | {pct(r['on_tube'])} {pct(r['ok_on_tube'])} | "
              + " ".join(pct(r[f"top{t}_ok_on_tube"]) for t in cfg["TOP"]) + f" | {pct(r.get('top50_also_in_sensor'))}")
    print("  ok = no collision, something between the fingers, both contacts inside the friction cone | tube = a "
          "tube between the fingers | topN = 'ok + tube' among the N best by score | in sens = share of its top 50 "
          f"the sensor cloud also proposes (<= {cfg['MATCH']['mm']:.0f} mm, {cfg['MATCH']['deg']:.0f} deg)")
    print(f"[out] {out}: summary.csv, grasps_<cloud>.csv")

    if args.vis:
        visualise(clouds, sets, geo, cfg, f"{seq.name} frame {frame}: AnyGrasp on the model vs on the cloud")


def visualise(clouds, sets, geo, cfg, title):
    import open3d as o3d
    import compare_burst_tracking as cbt
    v = cfg["VIS"]
    geoms, labels = [], []
    if "sensor" in clouds:
        pts, rgb = clouds["sensor"]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        geoms.append(("sensor cloud", pcd.voxel_down_sample(0.003)))
    sel = np.random.default_rng(0).choice(len(geo.pts), min(len(geo.pts), v["gt_points"]), replace=False)
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(geo.pts[sel]))
    gt.paint_uniform_color((0.6, 0.6, 0.6))
    geoms.append(("true meshes", gt))
    for name, z in sets.items():
        tag, col = TAGS.get(name, "?"), COLOURS.get(name, (1, 0, 1))
        good, bad = o3d.geometry.TriangleMesh(), o3d.geometry.TriangleMesh()
        for i in range(min(v["top"], len(z["score"]))):
            e = z["eval"][i]
            ok = e["label"] == "ok" and e["tube"]
            m = cbt.gripper_mesh(z["pos"][i], z["R"][i], z["width"][i], z["depth"][i],
                                 col if ok else tuple(0.35 * c for c in col), v["gripper_mm"] / 1000.0)
            if ok:
                good += m
            else:
                bad += m
            if i < v["labels"]:
                labels.append((z["pos"][i] + [0, 0, -0.02], f"{tag}{i + 1} {e['label']}{'' if e['tube'] else ' (bin)'}"))
        if len(good.vertices):
            geoms.append((f"{name} top{v['top']} ok", good))
        if len(bad.vertices):
            geoms.append((f"{name} top{v['top']} failed (dark)", bad))
    print("[vis] colours: sensor blue, sensor_clean cyan, model_visible orange, model_full green; failed = dark; "
          "tick the geometries on the right to compare")
    try:
        import open3d.visualization.gui as gui
        app = gui.Application.instance
        app.initialize()
        w = o3d.visualization.O3DVisualizer(title, 1400, 900)
        for name, g in geoms:
            w.add_geometry(name, g)
        for p, t in labels:
            w.add_3d_label(p, t)
        w.point_size = 2
        w.show_settings = True
        w.reset_camera_to_default()
        app.add_window(w)
        app.run()
    except Exception as e:
        print(f"[vis] labelled viewer unavailable ({e}); colours only")
        o3d.visualization.draw_geometries([g for _, g in geoms], window_name=title)


if __name__ == "__main__":
    main()
