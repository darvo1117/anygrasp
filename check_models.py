"""Detector (gsnet.so) vs tracker (tracker.so) on ONE point cloud: same network? same weights? collision check?

Input: one point cloud, either
  --cloud file.npy   (N, 3) or (N, 6) XYZ [+ RGB], camera frame, e.g. latest_cloud.npy / burst/cloud_000.npy
                     (metres; values above 20 are taken as millimetres; RGB 0..255 or 0..1)
  --seq runs/...     one frame (--frame) of a sequence from capture_sequence.py
Both libraries get exactly the same points. Each runs in its own process (they are never loaded together).

  1. weights    open both checkpoints with torch: size, layers, parameters. For the layers they share: identical?
                Layers only in the tracking checkpoint = the temporal association module.
                (If a checkpoint cannot be opened, e.g. encrypted, it is reported and skipped.)
  2. speed      same cloud, timed (2 warm-up calls, then --repeats, median):
                  detector: voxel 5 mm + collision check (= detect_burst.py), voxel without, full cloud without
                  tracker:  full cloud (= track_replay.py), voxel 5 mm
  3. overlap    SAME voxel cloud, no collision check on either side: share of tracker grasps with a detector grasp
                within 10 mm / 30 deg and vice versa. High = same geometry network; low = different networks.
  4. collisions OUR OWN check, applied identically to every output: gripper as two fingers + base + approach
                corridor (graspnet-baseline's model-free check), a grasp collides if >= --coll_points cloud points
                fall inside. The detector WITH its collision check is the control (should be ~0 %), the detector
                WITHOUT it shows what the raw network proposes. If the tracker looks like the raw detector, the
                tracker skips the collision check. (This says nothing about the weights: that is check 1 and 3.)

Environment: anygrasp. From the project root:
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/check_models.py --cloud ~/<folder>/latest_cloud.npy --vis
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/check_models.py --seq runs/multi/seed_100/seq_0000
    ... --max_gripper_width 0.1 --stage report    (redo only the report / views from the saved outputs)
Output: anygrasp/check_models_out/ (json + npz) and the printed report. --vis opens one window per output:
grasps in green (free) / red (collides).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SDK = Path.home() / "anygrasp_sdk"
DET_DIR, TRK_DIR = SDK / "grasp_detection", SDK / "grasp_tracking"
WORKSPACE = [(-0.4, 0.4), (-0.4, 0.4), (0.05, 1.0)]   # as batch_detect.py / detect_grasps.py / track_replay.py
DET_RUNS = ("voxel_collision", "voxel_nocollision", "full_nocollision")
TRK_RUNS = ("full", "voxel")


# --------------------------------------------------------------- input
def load_input(args):
    """Full cloud and its 5 mm voxel version (camera frame, metres, workspace-cropped): ((p, c), (pv, cv))."""
    if args.cloud:
        data = np.load(Path(args.cloud).expanduser()).astype(np.float64)
        p = data[:, :3]
        if np.nanmax(np.abs(p)) > 20:            # millimetres (raw Orbbec output)
            p = p / 1000.0
        if data.shape[1] >= 6:
            c = data[:, 3:6]
            c = c / 255.0 if np.nanmax(c) > 1.0 else c
        else:
            c = np.full((len(p), 3), 0.6)
    else:
        from PIL import Image
        seq = Path(args.seq).expanduser()
        meta = json.loads((seq / "meta.json").read_text())
        k = np.asarray(meta["K"], dtype=np.float64)
        rgb = np.asarray(Image.open(seq / "frames" / f"color_{args.frame:03d}.png"), dtype=np.float64)[..., :3] / 255.0
        z = np.load(seq / "frames" / f"depth_{args.frame:03d}.npy").astype(np.float64) / meta.get("depth_scale", 1000.0)
        h, w = z.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        pts = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)
        valid = z > 0
        p, c = pts[valid], rgb[valid]
    keep = np.isfinite(p).all(axis=1) & ~np.all(p == 0, axis=1)
    for a, (lo, hi) in enumerate(WORKSPACE):
        keep &= (p[:, a] > lo) & (p[:, a] < hi)
    p, c = p[keep].astype(np.float32), c[keep].astype(np.float32)
    if not len(p):
        raise SystemExit("the workspace crop left no points: is the cloud in the camera frame, in metres?")
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    pcd = pcd.voxel_down_sample(0.005)
    return (p, c), (np.asarray(pcd.points, np.float32), np.asarray(pcd.colors, np.float32))


def timed(fn, repeats):
    for _ in range(2):
        out = fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return out, float(np.median(ts))


def arrays(gg):
    if gg is None or len(gg) == 0:
        return {k: np.zeros(s) for k, s in (("t", (0, 3)), ("R", (0, 3, 3)), ("s", (0,)), ("w", (0,)), ("d", (0,)))}
    return {"t": np.asarray(gg.translations, np.float64), "R": np.asarray(gg.rotation_matrices, np.float64),
            "s": np.asarray(gg.scores, np.float64), "w": np.asarray(gg.widths, np.float64),
            "d": np.asarray(gg.depths, np.float64)}


def save_npz(path, runs: dict):
    np.savez(path, **{f"{k}__{a}": v[a] for k, v in runs.items() for a in ("t", "R", "s", "w", "d")})


def load_npz(path):
    if not path.exists():
        return {}
    z = np.load(path)
    names = sorted({key.split("__")[0] for key in z.files})
    return {n: {a: z[f"{n}__{a}"] for a in ("t", "R", "s", "w", "d")} for n in names}


# --------------------------------------------------------------- stage 1: weights
def stage_weights(out: Path):
    try:
        import torch
    except ImportError as e:
        (out / "weights.json").write_text(json.dumps({"detection": {"error": str(e)}, "tracking": {"error": str(e)}}))
        return
    res = {}
    for name, path in (("detection", DET_DIR / "log" / "checkpoint_detection.tar"),
                       ("tracking", TRK_DIR / "log" / "checkpoint_tracking.tar")):
        info = {"path": str(path), "size_mb": round(path.stat().st_size / 1e6, 2) if path.exists() else None}
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            sd = ck
            if isinstance(ck, dict):
                info["top_keys"] = [str(k) for k in list(ck.keys())[:20]]
                for key in ("model_state_dict", "state_dict", "model", "net"):
                    if key in ck and isinstance(ck[key], dict):
                        sd = ck[key]
                        break
            tensors = {k: v for k, v in sd.items() if hasattr(v, "shape")} if isinstance(sd, dict) else {}
            info["tensors"] = len(tensors)
            info["parameters"] = int(sum(v.numel() for v in tensors.values()))
            info["layer_prefixes"] = sorted({k.split(".")[0] for k in tensors})
            res[name], res[name + "_sd"] = info, tensors
        except Exception as e:  # encrypted / custom format
            info["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            res[name] = info
    report = {k: v for k, v in res.items() if not k.endswith("_sd")}
    if res.get("detection_sd") and res.get("tracking_sd"):
        a, b = res["detection_sd"], res["tracking_sd"]
        shared = [k for k in a if k in b and a[k].shape == b[k].shape]
        same = [k for k in shared if bool((a[k].float() - b[k].float()).abs().max() == 0)]
        report["shared_layers"] = len(shared)
        report["identical_shared_layers"] = len(same)
        report["shared_parameters"] = int(sum(a[k].numel() for k in shared))
        report["identical_shared_parameters"] = int(sum(a[k].numel() for k in same))
        report["only_in_tracking"] = sorted({k.split(".")[0] for k in b if k not in a})
        report["only_in_detection"] = sorted({k.split(".")[0] for k in a if k not in b})
        report["only_in_tracking_parameters"] = int(sum(b[k].numel() for k in b if k not in a))
    (out / "weights.json").write_text(json.dumps(report, indent=2))


# --------------------------------------------------------------- stage 2: detector
def stage_detect(args, out: Path):
    (full, _), (vox, _) = load_input(args)
    sys.path.insert(0, str(DET_DIR))
    os.chdir(DET_DIR)
    from gsnet import create_detector
    det = create_detector(argparse.Namespace(checkpoint_path=str(DET_DIR / "log" / "checkpoint_detection.tar"),
                                             max_gripper_width=args.max_gripper_width, gripper_height=0.03))
    if det is None:
        raise SystemExit("detector license failed")
    rep, save = {"points_full": len(full), "points_voxel": len(vox)}, {}
    for name, pts, coll in (("voxel_collision", vox, True), ("voxel_nocollision", vox, False),
                            ("full_nocollision", full, False)):
        try:
            gg, ms = timed(lambda: det.get_grasp(pts, {"dense_grasp": False, "collision_detection": coll}),
                           args.repeats)
            rep[name] = {"ms": round(ms, 1), "grasps": 0 if gg is None else len(gg),
                         "after_nms": 0 if gg is None or not len(gg) else len(gg.nms())}
            save[name] = arrays(gg)
        except Exception as e:
            rep[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    (out / "detect.json").write_text(json.dumps(rep, indent=2))
    save_npz(out / "detect.npz", save)


# --------------------------------------------------------------- stage 3: tracker
def stage_track(args, out: Path):
    (full, fc), (vox, vc) = load_input(args)
    sys.path.insert(0, str(TRK_DIR))
    os.chdir(TRK_DIR)
    from tracker import create_tracker
    rep, save = {"points_full": len(full), "points_voxel": len(vox)}, {}
    for name, pts, col in (("full", full, fc), ("voxel", vox, vc)):
        trk = create_tracker(argparse.Namespace(checkpoint_path=str(TRK_DIR / "log" / "checkpoint_tracking.tar"),
                                                filter="oneeuro", vis=False))
        try:
            res0, ms0 = timed(lambda: trk.update(pts, col, [0]), args.repeats)          # as the first frame
            _, ms = timed(lambda: trk.update(pts, col, list(range(5))), args.repeats)  # as a tracking frame
            curr = res0[1]                                                              # the detections
            rep[name] = {"ms_update": round(ms, 1), "ms_first_frame": round(ms0, 1),
                         "grasps": 0 if curr is None else len(curr),
                         "after_nms": 0 if curr is None or not len(curr) else len(curr.nms())}
            save[name] = arrays(curr)
        except Exception as e:
            rep[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        del trk
    (out / "track.json").write_text(json.dumps(rep, indent=2))
    save_npz(out / "track.npz", save)


# --------------------------------------------------------------- our collision check
def collides(g, cloud, coll_points, finger_width=0.01, finger_length=0.06, height=0.02, approach_dist=0.05):
    """(n,) bool: graspnet-baseline's model-free check. Grasp frame: x = approach, y = closing, origin = grasp
    centre; fingers reach depth along x. Points inside either finger, the base or the approach corridor behind it
    count; >= coll_points of them = collision. The space BETWEEN the fingers (the object) is not counted."""
    n = len(g["t"])
    out = np.zeros(n, dtype=bool)
    reach = 0.25
    for i in range(n):
        near = cloud[np.linalg.norm(cloud - g["t"][i], axis=1) < reach]
        if not len(near):
            continue
        q = (near - g["t"][i]) @ g["R"][i]                   # into the grasp frame
        w, d = g["w"][i], g["d"][i]
        z_in = np.abs(q[:, 2]) < height / 2
        x_fing = (q[:, 0] > d - finger_length) & (q[:, 0] < d)
        y_band = (q[:, 1] > -(w / 2 + finger_width)) & (q[:, 1] < (w / 2 + finger_width))
        left = z_in & x_fing & (q[:, 1] > -(w / 2 + finger_width)) & (q[:, 1] < -w / 2)
        right = z_in & x_fing & (q[:, 1] < (w / 2 + finger_width)) & (q[:, 1] > w / 2)
        base = z_in & y_band & (q[:, 0] <= d - finger_length) & (q[:, 0] > d - finger_length - finger_width)
        path = z_in & y_band & (q[:, 0] <= d - finger_length - finger_width) & \
            (q[:, 0] > d - finger_length - finger_width - approach_dist)
        out[i] = int((left | right | base | path).sum()) >= coll_points
    return out


def gripper_mesh(t, R, w, d, color, thick=0.004):
    import open3d as o3d
    tail, base = 0.04, 0.02
    boxes = [((d + base + thick, thick, thick), (-(base + thick), -(w / 2 + thick), -thick / 2)),
             ((d + base + thick, thick, thick), (-(base + thick), w / 2, -thick / 2)),
             ((thick, w, thick), (-(thick + base), -w / 2, -thick / 2)),
             ((tail, thick, thick), (-(tail + thick + base), -thick / 2, -thick / 2))]
    m = o3d.geometry.TriangleMesh()
    for (sx, sy, sz), off in boxes:
        b = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
        b.translate(off)
        m += b
    m.rotate(np.asarray(R, np.float64), center=(0, 0, 0))
    m.translate(np.asarray(t, np.float64))
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


# --------------------------------------------------------------- report
def overlap(a, b, mm=10.0, deg=30.0):
    """Share of grasps in a that have a grasp of b within mm and deg (flip-aware)."""
    if len(a["t"]) == 0 or len(b["t"]) == 0:
        return None
    flip = np.diag([1.0, -1.0, -1.0])
    hit = 0
    for t, r in zip(a["t"], a["R"]):
        for j in np.flatnonzero(np.linalg.norm(b["t"] - t, axis=1) * 1000.0 <= mm):
            tr = max(np.trace(r.T @ b["R"][j]), np.trace(r.T @ b["R"][j] @ flip))   # larger trace = smaller angle
            if np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))) <= deg:
                hit += 1
                break
    return hit / len(a["t"])


def nms_top(g, k):
    order = np.argsort(-g["s"])[:k]
    return {a: v[order] for a, v in g.items()}


def report(args, out: Path):
    rep = {}
    for name in ("weights", "detect", "track"):
        p = out / f"{name}.json"
        rep[name] = json.loads(p.read_text()) if p.exists() else None

    print("\n=== 1. WEIGHTS ===")
    w = rep["weights"] or {}
    for m in ("detection", "tracking"):
        i = w.get(m, {})
        if "error" in i:
            print(f"  {m:<9}: {i.get('size_mb')} MB, could not be opened ({i['error']})")
        else:
            print(f"  {m:<9}: {i.get('size_mb')} MB, {i.get('tensors')} tensors, {i.get('parameters')} parameters, "
                  f"modules {i.get('layer_prefixes')}")
    if "shared_layers" in w:
        print(f"  shared layers: {w['shared_layers']} ({w['shared_parameters']} parameters), IDENTICAL: "
              f"{w['identical_shared_layers']} ({w['identical_shared_parameters']} parameters)")
        print(f"  only in tracking: {w['only_in_tracking']} ({w['only_in_tracking_parameters']} parameters) | "
              f"only in detection: {w['only_in_detection']}")

    print("\n=== 2. SPEED (same cloud, median ms) ===")
    d, t = rep["detect"] or {}, rep["track"] or {}
    print(f"  cloud: {d.get('points_full') or t.get('points_full')} points full, "
          f"{d.get('points_voxel') or t.get('points_voxel')} after the 5 mm voxel")
    for k in DET_RUNS:
        x = d.get(k, {})
        print(f"  detector {k:<18}: " + (x.get("error", "-") if "ms" not in x else
                                          f"{x['ms']:7.1f} ms, {x['grasps']} grasps ({x['after_nms']} after NMS)"))
    for k in TRK_RUNS:
        x = t.get(k, {})
        print(f"  tracker  {k:<18}: " + (x.get("error", "-") if "ms_update" not in x else
                                          f"{x['ms_update']:7.1f} ms per update ({x['ms_first_frame']:.1f} ms first "
                                          f"frame), {x['grasps']} grasps ({x['after_nms']} after NMS)"))

    dets, trks = load_npz(out / "detect.npz"), load_npz(out / "track.npz")
    print("\n=== 3. OVERLAP: same 5 mm voxel cloud, no collision check on either side (10 mm, 30 deg) ===")
    dv, tv = dets.get("voxel_nocollision"), trks.get("voxel")
    if dv is not None and tv is not None and len(dv["t"]) and len(tv["t"]):
        o_all = (overlap(tv, dv), overlap(dv, tv))
        o_top = (overlap(nms_top(tv, 50), dv), overlap(nms_top(dv, 50), tv))
        rep["overlap"] = {"tracker_in_detector": o_all[0], "detector_in_tracker": o_all[1],
                          "tracker_top50_in_detector": o_top[0], "detector_top50_in_tracker": o_top[1]}
        print(f"  tracker grasps with a detector grasp nearby : {o_all[0] * 100:5.1f}%  (top-50 by score: "
              f"{o_top[0] * 100:.1f}%)")
        print(f"  detector grasps with a tracker grasp nearby : {o_all[1] * 100:5.1f}%  (top-50 by score: "
              f"{o_top[1] * 100:.1f}%)")
        print("  high on both sides = same geometry network on the same input; low = different networks")
    else:
        print("  n/a (missing outputs)")

    print(f"\n=== 4. COLLISIONS: our own check on every output (>= {args.coll_points} cloud points in fingers / "
          f"base / approach corridor) ===")
    (_, _), (vox, vcol) = load_input(args)
    rep["collisions"], marks = {}, {}
    for label, g in [(f"detector {k}", dets.get(k)) for k in DET_RUNS] + [(f"tracker  {k}", trks.get(k))
                                                                           for k in TRK_RUNS]:
        if g is None or not len(g["t"]):
            print(f"  {label:<27}: n/a")
            continue
        c = collides(g, vox, args.coll_points)
        top = np.argsort(-g["s"])[:50]
        rep["collisions"][label.strip()] = {"grasps": int(len(c)), "collide": float(c.mean()),
                                            "top50_collide": float(c[top].mean())}
        marks[label.strip()] = (g, c)
        print(f"  {label:<27}: {c.mean() * 100:5.1f}% of {len(c)} collide | top-50 by score: {c[top].mean() * 100:5.1f}%")
    print("  control: 'detector voxel_collision' should be close to 0 %; if not, our check is stricter than the SDK's")

    (out / "check_models.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"\n[INFO] {out / 'check_models.json'}")

    if args.vis and marks:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vox.astype(np.float64)))
        pcd.colors = o3d.utility.Vector3dVector(vcol.astype(np.float64))
        flip = np.diag([1.0, 1.0, -1.0, 1.0])
        for label, (g, c) in marks.items():
            top = np.argsort(-g["s"])[:args.vis_top]
            meshes = o3d.geometry.TriangleMesh()
            for i in top:
                meshes += gripper_mesh(g["t"][i], g["R"][i], g["w"][i], g["d"][i],
                                       [0.9, 0.1, 0.1] if c[i] else [0.1, 0.8, 0.1])
            cloud = o3d.geometry.PointCloud(pcd)
            cloud.transform(flip)
            meshes.transform(flip)
            print(f"[vis] {label}: top {len(top)} by score, red = collides (close the window for the next one)")
            o3d.visualization.draw_geometries([cloud, meshes], window_name=f"{label}: red = collides",
                                              width=1280, height=800)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cloud", help=".npy point cloud (N,3)/(N,6), camera frame, metres or mm")
    src.add_argument("--seq", help="a sequence folder from capture_sequence.py (use --frame)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max_gripper_width", type=float, default=0.1, help="detector; the SDK allows (0, 0.1]")
    ap.add_argument("--coll_points", type=int, default=3, help="cloud points inside the gripper = collision")
    ap.add_argument("--vis", action="store_true", help="one window per output, red = collides")
    ap.add_argument("--vis_top", type=int, default=40, help="grasps drawn per window (best score first)")
    ap.add_argument("--stage", choices=["all", "weights", "detect", "track", "report"], default="all")
    ap.add_argument("--out", default=str(HERE / "check_models_out"))
    args = ap.parse_args()
    if args.cloud:
        args.cloud = str(Path(args.cloud).expanduser().resolve())
    if args.seq:
        args.seq = str(Path(args.seq).expanduser().resolve())
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.stage == "weights":
        stage_weights(out)
    elif args.stage == "detect":
        stage_detect(args, out)
    elif args.stage == "track":
        stage_track(args, out)
    elif args.stage == "report":
        report(args, out)
    else:
        env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8")}
        common = (["--cloud", args.cloud] if args.cloud else ["--seq", args.seq, "--frame", str(args.frame)]) + \
            ["--repeats", str(args.repeats), "--max_gripper_width", str(args.max_gripper_width), "--out", str(out)]
        for st in ("weights", "detect", "track"):
            print(f"[{st}] ...", flush=True)
            r = subprocess.run([sys.executable, __file__, "--stage", st, *common], env=env)
            if r.returncode != 0:
                print(f"[{st}] failed (exit {r.returncode}); continuing")
        report(args, out)


if __name__ == "__main__":
    main()
