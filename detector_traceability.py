"""Geometric traceability of the DETECTOR's grasps: how often is each grasp detected again, at the same pose?

No tracker. Uses the detections already in every scene's burst_result.json (detect_burst.py: AnyGrasp detector
WITH its collision check, NMS, all grasps kept with --top_k 1024), so nothing has to be run again.

For every grasp of the start frame (--start, default 0), anchored to its START pose:
  trace       share of the later frames that contain a detection within --match_mm and --match_deg of it
              (flip-aware: 180 deg about the approach = same parallel-jaw grasp)
  life        share of the later frames until the first frame without such a detection (1.0 = never missed)
  match_mm    mean distance of those re-detections to the start pose: how precisely it comes back
  score       its score in the start frame, and the mean score of its re-detections
Because the detector checks collisions in every frame, a grasp that collides in some frames is missing there:
its trace drops, which is what we want (a grasp at the limit is less reliable).

Only grasps on the target (GT pipe mask) are kept unless --no_mask.

Environment: anygrasp. From the project root:
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/detector_traceability.py --runs runs/multi --vis
    ... --seeds 100 103 --top 10 --vis           only these scenes, one window each
    ... --vis_trails                             also the re-detections of every top grasp (one dot per frame)
Window: point cloud of the start frame, the top grasps coloured by trace (green = 1.0 -> red = 0), labels
"#rank trace" (Open3D's labelled viewer; the right panel switches every grasp on / off).
Output: <seq>/detector_traceability.csv, <seq>/detector_top.npz (top grasps, camera frame, batch_detect.py keys),
        <runs>/detector_traceability.csv (all scenes), <runs>/detector_traceability_summary.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLIP_X = np.diag([1.0, -1.0, -1.0])


def angle_deg(ra, rb):
    tr = max(np.trace(ra.T @ rb), np.trace(ra.T @ rb @ FLIP_X))      # larger trace = smaller angle
    return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None

    def rank(v):
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        return (start + (cnt - 1) / 2.0)[inv]
    rx, ry = rank(x[ok]), rank(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def colour(v):
    """0 -> red, 0.5 -> yellow, 1 -> green."""
    v = float(np.clip(v, 0, 1))
    return [min(1.0, 2 * (1 - v)), min(1.0, 2 * v), 0.1]


def start_cloud(seq: Path, frame: int, voxel=0.004):
    """Coloured point cloud of one frame (camera frame), for the view."""
    import open3d as o3d
    from PIL import Image
    meta = json.loads((seq / "meta.json").read_text())
    k = np.asarray(meta["K"], dtype=np.float64)
    rgb = np.asarray(Image.open(seq / "frames" / f"color_{frame:03d}.png"), dtype=np.float64)[..., :3] / 255.0
    z = np.load(seq / "frames" / f"depth_{frame:03d}.npy").astype(np.float64) / meta.get("depth_scale", 1000.0)
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)
    keep = (z > 0) & (z < 1.0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep]))
    pcd.colors = o3d.utility.Vector3dVector(rgb[keep])
    return pcd.voxel_down_sample(voxel)


def process(seq: Path, args, cbt):
    res = json.loads((seq / "burst_result.json").read_text())
    n_frames = len(res["frames"])
    masks = None if args.no_mask else cbt.Masks(seq, args.dilate)
    by_frame = {}
    for c in res["candidates"]:
        p = np.array([c["x"], c["y"], c["z"]])
        if masks is not None and not masks.hit(c["frame"], p):
            continue
        by_frame.setdefault(c["frame"], []).append({"pos": p, "R": np.asarray(c["rotation_matrix"]),
                                                    "score": c["score"], "width": c["width"], "depth": c["depth"],
                                                    "rank": c["rank"]})
    s = args.start
    starts = by_frame.get(s, [])
    later = [f for f in range(s + 1, n_frames)]
    if not starts or not later:
        print(f"[skip] {seq}: no grasp in frame {s}" + ("" if starts else " (on the pipe)"))
        return None
    pos_f = {f: np.array([g["pos"] for g in by_frame.get(f, [])]).reshape(-1, 3) for f in later}

    rows = []
    for i, g in enumerate(starts):
        hit, dists, scores, trail, matches = [], [], [], [], []
        for f in later:
            cand = by_frame.get(f, [])
            found = None
            if len(cand):
                d = np.linalg.norm(pos_f[f] - g["pos"], axis=1) * 1000.0
                for j in np.argsort(d):
                    if d[j] > args.match_mm:
                        break
                    if angle_deg(g["R"], cand[j]["R"]) <= args.match_deg:
                        found = (j, d[j])
                        break
            hit.append(found is not None)
            matches.append((f, cand[found[0]] if found is not None else None, found[1] if found is not None else None))
            if found is not None:
                dists.append(found[1])
                scores.append(cand[found[0]]["score"])
                trail.append(cand[found[0]]["pos"])
        hit = np.array(hit)
        miss = np.flatnonzero(~hit)
        rows.append({
            "grasp": i, "rank_in_frame": g["rank"], "score": round(g["score"], 4),
            "trace": round(float(hit.mean()), 3),
            "life": round((int(miss[0]) if len(miss) else len(hit)) / len(hit), 3),
            "match_mm": round(float(np.mean(dists)), 2) if dists else None,
            "redetected_score": round(float(np.mean(scores)), 4) if scores else None,
            "width_mm": round(g["width"] * 1000, 1), "tilt_deg": round(float(np.degrees(np.arccos(np.clip(g["R"][2, 0], -1, 1)))), 1),
            "x": round(float(g["pos"][0]), 4), "y": round(float(g["pos"][1]), 4), "z": round(float(g["pos"][2]), 4),
            "_g": g, "_trail": np.array(trail).reshape(-1, 3), "_matches": matches,
        })
    rows.sort(key=lambda r: (-r["trace"], -r["score"]))
    for r, row in enumerate(rows):
        row["top_rank"] = r + 1
    return rows, n_frames


def visualise(seq: Path, title: str, rows, args, cbt):
    import open3d as o3d
    pcd = start_cloud(seq, args.start)
    geoms, labels = [("cloud", pcd)], []
    for row in rows[:args.top]:
        g, col = row["_g"], colour(row["trace"])
        name = f"#{row['top_rank']} trace {row['trace']:.2f}"
        geoms.append((f"{name} gripper", cbt.gripper_mesh(g["pos"], g["R"], g["width"], g["depth"], col,
                                                         args.gripper_mm / 1000.0)))
        geoms.append((f"{name} point", cbt.spheres([g["pos"]], args.point_mm / 1000.0, col)))
        if args.vis_trails and len(row["_trail"]):
            geoms.append((f"{name} re-detections", cbt.spheres(row["_trail"], args.point_mm / 2000.0, col)))
        labels.append((g["pos"] + [0, 0, -0.03], f"#{row['top_rank']} {row['trace']:.2f}"))
    try:
        import open3d.visualization.gui as gui
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer(f"{title}: top {args.top} most traceable (green = 1.0, red = 0)",
                                              1400, 900)
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
        o3d.visualization.draw_geometries([g for _, g in geoms], window_name=f"{title}: top {args.top}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/multi")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--start", type=int, default=0, help="frame whose grasps are scored")
    ap.add_argument("--match_mm", type=float, default=20.0)
    ap.add_argument("--match_deg", type=float, default=30.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no_mask", action="store_true", help="every grasp of the scene, not only the pipe")
    ap.add_argument("--dilate", type=int, default=8)
    ap.add_argument("--vis", action="store_true", help="one window per scene (close it for the next one)")
    ap.add_argument("--vis_trails", action="store_true", help="also draw the re-detections of each top grasp")
    ap.add_argument("--gripper_mm", type=float, default=3.0)
    ap.add_argument("--point_mm", type=float, default=4.0)
    args = ap.parse_args()

    runs = Path(args.runs).expanduser().resolve()
    sys.path.insert(0, str(HERE))
    import compare_burst_tracking as cbt  # noqa: E402  (masks, gripper mesh, spheres)

    dirs = ([runs / f"seed_{s}" for s in args.seeds] if args.seeds
            else sorted(runs.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])))
    pooled, per_scene, shown = [], [], []
    for d in dirs:
        seqs = sorted(p for p in d.glob("seq_*") if (p / "burst_result.json").exists())
        if not seqs:
            print(f"[skip] {d.name}: no burst_result.json")
            continue
        seq = seqs[-1]
        got = process(seq, args, cbt)
        if got is None:
            continue
        rows, n_frames = got
        tr = np.array([r["trace"] for r in rows])
        rho = spearman([r["score"] for r in rows], tr)
        per_scene.append({"scene": d.name, "grasps": len(rows), "frames": n_frames, "trace_mean": float(tr.mean()),
                          "share_trace_ge_0.8": float((tr >= 0.8).mean()), "share_trace_ge_0.5": float((tr >= 0.5).mean()),
                          "spearman_score_trace": rho})
        print(f"\n=== {d.name}: {len(rows)} grasps in frame {args.start}"
              f"{'' if args.no_mask else ' on the pipe'}, matched in {n_frames - args.start - 1} later frames "
              f"(<= {args.match_mm:.0f} mm, {args.match_deg:.0f} deg) ===")
        print(f"  mean trace {tr.mean():.2f} | trace >= 0.8: {(tr >= 0.8).mean() * 100:.0f}% | >= 0.5: "
              f"{(tr >= 0.5).mean() * 100:.0f}% | Spearman score~trace {'n/a' if rho is None else f'{rho:+.2f}'}")
        print(f"  {'top':>3} {'trace':>5} {'life':>5} {'match':>7} {'score':>6} {'re-score':>8} {'rank':>4} "
              f"{'width':>6} {'tilt':>5}   x, y, z [m]")
        for r in rows[:args.top]:
            print(f"  {r['top_rank']:>3} {r['trace']:5.2f} {r['life']:5.2f} "
                  f"{(r['match_mm'] if r['match_mm'] is not None else float('nan')):5.1f}mm {r['score']:6.3f} "
                  f"{(r['redetected_score'] if r['redetected_score'] is not None else float('nan')):8.3f} "
                  f"{r['rank_in_frame']:>4} {r['width_mm']:5.1f}mm {r['tilt_deg']:4.0f}°   "
                  f"{r['x']:+.3f}, {r['y']:+.3f}, {r['z']:+.3f}")
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        with open(seq / "detector_traceability.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(clean[0].keys()))
            w.writeheader()
            w.writerows(clean)
        top = rows[:args.top]
        np.savez(seq / "detector_top.npz",  # for an execution test later (camera frame of the start frame)
                 translations=np.array([r["_g"]["pos"] for r in top], np.float32),
                 rotation_matrices=np.array([r["_g"]["R"] for r in top], np.float32),
                 widths=np.array([r["_g"]["width"] for r in top], np.float32),
                 depths=np.array([r["_g"]["depth"] for r in top], np.float32),
                 scores=np.array([r["score"] for r in top], np.float32),
                 trace=np.array([r["trace"] for r in top], np.float32), frame=args.start)
        pooled += [{"scene": d.name, **c} for c in clean]
        if args.vis:
            shown.append((seq, d.name, rows))

    if not pooled:
        raise SystemExit("no scene processed")
    tr = np.array([r["trace"] for r in pooled])
    sc = np.array([r["score"] for r in pooled])
    print(f"\nPOOLED over {len(per_scene)} scenes, {len(pooled)} grasps")
    print(f"  mean trace {tr.mean():.2f} | trace >= 0.8: {(tr >= 0.8).mean() * 100:.0f}% | >= 0.5: "
          f"{(tr >= 0.5).mean() * 100:.0f}% | < 0.2: {(tr < 0.2).mean() * 100:.0f}%")
    rho = spearman(sc, tr)
    pos = sum(1 for p in per_scene if (p["spearman_score_trace"] or 0) > 0)
    print(f"  Spearman score~trace {'n/a' if rho is None else f'{rho:+.2f}'} (positive in {pos}/{len(per_scene)} scenes)")
    q = np.quantile(sc, [1 / 3, 2 / 3])
    for lo, hi, name in ((-1, q[0], "low score"), (q[0], q[1], "mid score"), (q[1], 9, "high score")):
        m = (sc > lo) & (sc <= hi)
        if m.any():
            print(f"  {name:<10}: {m.sum():4d} grasps, mean trace {tr[m].mean():.2f}, trace >= 0.8 "
                  f"{(tr[m] >= 0.8).mean() * 100:.0f}%")
    with open(runs / "detector_traceability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pooled[0].keys()))
        w.writeheader()
        w.writerows(pooled)
    (runs / "detector_traceability_summary.json").write_text(json.dumps(
        {"settings": vars(args), "per_scene": per_scene,
         "pooled": {"grasps": len(pooled), "trace_mean": float(tr.mean()), "share_ge_0.8": float((tr >= 0.8).mean()),
                    "spearman_score_trace": rho}}, indent=2, default=float))
    print(f"[INFO] {runs / 'detector_traceability.csv'} and detector_traceability_summary.json; "
          f"per scene: <seq>/detector_traceability.csv, detector_top.npz")

    for seq, name, rows in shown:
        print(f"[vis] {name}: close the window for the next scene")
        visualise(seq, name, rows, args, cbt)


if __name__ == "__main__":
    main()
