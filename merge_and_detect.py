"""Merge one burst frame (camera depth) with the ground-truth points of the tube, run AnyGrasp once on the merged
cloud, keep the grasps on that tube (the same 3-D mask as gt_vs_grouping.py) and write the usual grasps.csv.

  sensor   <seq>/frames/depth_XXX.npy (+ color_XXX.png), back-projected with the K of <seq>/meta.json
  tube     <seq>/gt_vs_grouping/<pipe>/model_cloud_pipe.npy (written by gt_vs_grouping.py), same camera frame
  mask     grasp centre within TUBE_MM of the tube points (none: every grasp of the scene)

Environment: anygrasp. Needs group_and_trace.py and seq_group_track.py next to it.
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/merge_and_detect.py --seq runs/pipefit/seq_0007_b3 --pipe pipe_1
    ... --frame 5 --mask none --vis
Output: <seq>/merged/<pipe>_frame_XXX/
    cloud_000.npy   the merged cloud (N, 6): xyz m camera frame + rgb (sensor colours, tube points yellow)
    grasps.csv      seq_group_track.py grasps.csv columns (one frame: no group / trace) + tube_mm and the
                    true-geometry check of gt_vs_grouping.py: check (ok / collision / empty / no_fc),
                    between_fingers, angle_a_deg / angle_b_deg (contact normals vs the closing axis), bin_pts,
                    filters (why it would be dropped), pass
Needs also gt_vs_grouping.py and model_vs_cloud.py next to it, and <seq>/meshes.npz.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import group_and_trace as gat  # noqa: E402
import seq_group_track as sgt  # noqa: E402

TUBE_MM = 12.0


def main():
    ap = argparse.ArgumentParser(description="sensor frame + ground-truth tube points -> AnyGrasp -> grasps.csv")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--pipe", required=True, help="tube name, as in gt_vs_grouping/<pipe>/")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--tube_cloud", help="default <seq>/gt_vs_grouping/<pipe>/model_cloud_pipe.npy")
    ap.add_argument("--mask", choices=["pipe", "none"], default="pipe")
    ap.add_argument("--tube_mm", type=float, default=TUBE_MM)
    ap.add_argument("--voxel", type=float, help="detector voxel (m); default as seq_group_track.py")
    ap.add_argument("--out")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    seq = Path(args.seq).expanduser().resolve()
    meta = json.loads((seq / "meta.json").read_text())
    k = np.asarray(meta["K"], float)
    f = args.frame
    depth = np.load(seq / "frames" / f"depth_{f:03d}.npy")
    rgb = sgt.read_image(seq / "frames" / f"color_{f:03d}.png")
    fr = sgt.frame_from_depth(f, rgb, sgt.depth_to_m(depth, meta.get("depth_scale")), k)
    tube_path = Path(args.tube_cloud).expanduser() if args.tube_cloud else \
        seq / "gt_vs_grouping" / args.pipe / "model_cloud_pipe.npy"
    tube = np.load(tube_path).astype(np.float64)

    # sanity check: the tube points must sit on the sensor surface (same camera frame)
    from scipy.spatial import cKDTree
    d_tube = cKDTree(fr["xyz"]).query(tube[:, :3])[0] * 1000.0
    print(f"[merge] sensor frame {f}: {len(fr['xyz'])} points | tube {args.pipe}: {len(tube)} points; "
          f"tube point to nearest sensor point: median {np.median(d_tube):.1f} mm "
          f"(a few mm = aligned; the hidden side of the tube is farther)")

    xyz = np.concatenate([fr["xyz"], tube[:, :3]])
    col = np.concatenate([fr["rgb"], tube[:, 3:6] / 255.0])
    src = np.r_[np.zeros(len(fr["xyz"]), int), np.ones(len(tube), int)]
    out = Path(args.out).expanduser().resolve() if args.out else seq / "merged" / f"{args.pipe}_frame_{f:03d}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "cloud_000.npy", np.c_[xyz, col * 255.0].astype(np.float32))
    print(f"[merge] saved {out / 'cloud_000.npy'}  ({len(xyz)} points, (N, 6) xyz m + rgb 0-255)")

    det = dict(sgt.CONFIG["DETECTOR"])
    if args.voxel:
        det["voxel"] = args.voxel
    cands = gat.run_detector([(0, xyz, col)], det)
    n_raw = len(cands)
    tree = cKDTree(tube[:, :3])
    for c in cands:
        c["tube_mm"] = float(tree.query(c["pos"])[0] * 1000.0)
    if args.mask == "pipe":
        cands = [c for c in cands if c["tube_mm"] <= args.tube_mm]
        for r, c in enumerate(sorted(cands, key=lambda c: c["rank"])):
            c["rank"] = r + 1
    print(f"[detect] {n_raw} grasps on the merged cloud, {len(cands)} kept "
          f"({'within ' + str(args.tube_mm) + ' mm of ' + args.pipe if args.mask == 'pipe' else 'no mask'})")

    # the same true-geometry check and filters as gt_vs_grouping.py (meshes.npz of the sequence)
    import gt_vs_grouping as gvg
    import model_vs_cloud as mvc
    _, _, objects = mvc.load_scene(seq, 0)
    geo = mvc.TrueGeometry(objects, gvg.CONFIG["EVAL"])
    bin_ids = [i for i, o in enumerate(objects) if not o["target"]]
    for c in cands:
        c["pass"], c["eval"] = gvg.judge(geo, bin_ids, args.pipe, c["pos"], c["R"], c["width"], c["depth"], gvg.CONFIG)

    fmt = sgt.fmt
    rows = []
    for c in sorted(cands, key=lambda c: c["rank"]):
        rows.append({"grasp_id": len(rows), "frame": 0, "rank": c["rank"], "score": fmt(c["score"], 4),
                     "group": None, "group_best": False, "trace": None, "trace_hits": None,
                     "width_mm": fmt(c["width"] * 1000, 1), "depth_mm": fmt(c["depth"] * 1000, 1),
                     "tilt_deg": fmt(gat.tilt_deg(c["R"]), 1), "x": fmt(c["pos"][0], 4), "y": fmt(c["pos"][1], 4),
                     "z": fmt(c["pos"][2], 4), **sgt.rot_cols(c["R"]), "tube_mm": fmt(c["tube_mm"], 1),
                     "check": c["eval"]["label"], "between_fingers": c["eval"]["object"],
                     "angle_a_deg": fmt(c["eval"]["angle_a"], 1), "angle_b_deg": fmt(c["eval"]["angle_b"], 1),
                     "bin_pts": c["eval"]["bin_pts"], "filters": c["eval"]["why"], "pass": c["pass"]})
    gat.write_csv(out / "grasps.csv", rows)
    print(f"\n  {'rank':>4} {'score':>6} {'tilt':>5} {'width':>6} {'tube_mm':>7} {'check':>9} {'between':>8} "
          f"{'filters':>14}   {'x':>7} {'y':>7} {'z':>6}")
    for r in rows[:20]:
        print(f"  {r['rank']:>4} {r['score']:6.3f} {r['tilt_deg']:4.0f}° {r['width_mm']:4.0f}mm {r['tube_mm']:6.1f} "
              f"{r['check']:>9} {r['between_fingers'] or '-':>8} {r['filters']:>14}   "
              f"{r['x']:+7.3f} {r['y']:+7.3f} {r['z']:6.3f}")
    print(f"  pass (all filters, as gt_vs_grouping.py): {sum(r['pass'] for r in rows)}/{len(rows)}")
    print(f"[out] {out / 'grasps.csv'} ({len(rows)} grasps)")

    if args.vis:
        ext = [{"name": f"#{r['rank']}", "pos": c["pos"], "R": c["R"], "width": c["width"], "depth": c["depth"]}
               for r, c in zip(rows[:10], sorted(cands, key=lambda c: c["rank"]))]
        cfg = {"VIS": {**sgt.CONFIG["VIS"], "show_groups": 0}}
        gat.visualise(xyz, col, [], cands, [], ext, [{"group": None, "trace": 0.0} for _ in ext], cfg,
                      f"{seq.name} frame {f} + {args.pipe} ground truth: top 10 grasps")


if __name__ == "__main__":
    main()
