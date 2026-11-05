"""Which grasp should the tracker start from? Four initialisation strategies on the same frames of every scene.

Uses the scenes that run_multiseed.py already produced (runs/multi/seed_S/seq_0000 + its burst_result.json).
Every strategy gets the SAME split of the 1 s sequence:
  phase 1  frames 0 .. P-1   build the zones (only the strategies that use zones look at them)
  select   frame  P          each strategy picks K tracker grasps on the pipe in this frame
  phase 2  frames P+1 .. end the tracker follows them; we measure how well they hold

  (a) score   the K best tracker grasps by score in frame P (after NMS), as track_replay.py does on frame 0
  (b) C zones zones = groups of the TRACKER's own detections in phase 1 (detect_burst.py's grouping), most
              persistent first; per zone the frame-P tracker grasp closest to the zone centre
  (c) A zones zones = groups of the DETECTOR's detections (run A) in phase 1, most persistent first; per zone
              the frame-P tracker grasp with the highest score inside it  (your idea)
  (d) A and C A zones that also have a persistent C group (support >= --min_c_support) at the same place;
              per zone the frame-P tracker grasp closest to that C group
Zones are ranked by support, then mean score (persistence first). Only grasps on the pipe (GT mask) count.

Per selected grasp, over phase 2 (relative to its pose in frame P):
  anchored   share of frames within --anchor_mm and --anchor_deg of the start pose: "still the same grasp"
  drift      max distance from the start (mm); jitter = std of the position (mm)
  on_pipe    share of frames whose grasp centre is still on the pipe mask
  zone_stay  share of frames still inside its zone (b, c, d)
Per scene: top-1 (the grasp the robot would execute) and the mean of the K. Pooled: mean over scenes, and the
paired difference of each strategy against (a) with a sign test.

The tracker is re-run from frame 0 for every strategy (its state cannot be copied), ~3 s per strategy and scene.
Selected grasps are saved for a later execution test: <seq>/init_strategies/<strategy>.npz (camera OpenCV frame,
same keys as batch_detect.py: scores, widths, depths, translations, rotation_matrices).

Environment: anygrasp
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/compare_init_strategies.py --burst_root ~/Time_consistency_sim
    ... --seeds 100 101 --k 5 --phase1 10
Output: <runs>/init_strategies_summary.json, <seq>/init_strategies/result.json per scene
"""
import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STRATS = {"a": "score", "b": "C zones", "c": "A zones", "d": "A and C"}


def sign_test(diffs):
    d = [x for x in diffs if x is not None and np.isfinite(x) and x != 0]
    n = len(d)
    if n == 0:
        return None
    k = sum(x > 0 for x in d)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst_root", required=True, help="the _sim burst folder (for detect_burst.py's grouping code)")
    ap.add_argument("--runs", default="runs/multi", help="output folder of run_multiseed.py")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: every seed_* folder in --runs")
    ap.add_argument("--strategies", nargs="*", default=list(STRATS), choices=list(STRATS))
    ap.add_argument("--phase1", type=int, default=10, help="frames used to build the zones (selection on this frame)")
    ap.add_argument("--k", type=int, default=5, help="grasps selected per strategy")
    ap.add_argument("--group_mm", type=float, default=20.0)
    ap.add_argument("--group_deg", type=float, default=45.0)
    ap.add_argument("--min_c_support", type=float, default=0.6, help="(d): C group support needed at an A zone")
    ap.add_argument("--anchor_mm", type=float, default=20.0)
    ap.add_argument("--anchor_deg", type=float, default=30.0)
    ap.add_argument("--dilate", type=int, default=8)
    ap.add_argument("--filter", default="oneeuro")
    args = ap.parse_args()

    # resolve paths BEFORE importing track_replay: it changes the working directory
    runs = Path(args.runs).expanduser().resolve()
    root = Path(args.burst_root).expanduser().resolve()
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(root))
    import compare_burst_tracking as cbt  # noqa: E402  (masks, grouping helpers)
    import detect_burst as db  # noqa: E402  (the burst's own grouping)
    import track_replay as trp  # noqa: E402  (tracker, frames, NMS order)

    seed_dirs = ([runs / f"seed_{s}" for s in args.seeds] if args.seeds
                 else sorted(runs.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])))
    gmm, gdeg = args.group_mm, (None if args.group_deg <= 0 else args.group_deg)
    ckpt = str(trp.TRACK_DIR / "log" / "checkpoint_tracking.tar")
    per_seed = []

    for sd in seed_dirs:
        seqs = sorted(p for p in sd.glob("seq_*") if (p / "meta.json").exists())
        if not seqs or not (seqs[-1] / "burst_result.json").exists():
            print(f"[skip] {sd.name}: no sequence or burst_result.json (run run_multiseed.py first)")
            continue
        seq = seqs[-1]
        meta = json.loads((seq / "meta.json").read_text())
        k_mat = np.asarray(meta["K"], dtype=np.float32)
        n_frames, scale = meta["n_frames"], meta.get("depth_scale", 1000.0)
        s = args.phase1
        if s + 2 > n_frames:
            raise SystemExit(f"--phase1 {s} leaves no frames to track ({n_frames} frames)")
        masks = cbt.Masks(seq, args.dilate)
        frames = seq / "frames"
        print(f"\n=== {sd.name}: zones from frames 0-{s - 1}, select on {s}, track {s + 1}-{n_frames - 1} ===")

        # ---- A zones: detector candidates of phase 1, on the pipe, regrouped
        res_a = json.loads((seq / "burst_result.json").read_text())
        cands_a = [{"frame": c["frame"], "rank": c["rank"], "score": c["score"], "width": c["width"],
                    "depth": c["depth"], "pos": np.array([c["x"], c["y"], c["z"]]),
                    "R": np.asarray(c["rotation_matrix"])} for c in res_a["candidates"]
                   if c["frame"] < s and masks.hit(c["frame"], np.array([c["x"], c["y"], c["z"]]))]
        cands_a = cbt.rerank(cands_a)
        zones_a = db.summarise_groups(db.group_candidates(cands_a, gmm, gdeg), cands_a, s, 1024, "penalized") \
            if cands_a else []
        zones_a.sort(key=lambda g: (-g["support"], -g["avg_score"]))

        def pipe_order(gg, all_t, f):
            order, _ = trp.nms_order(gg, all_t)
            return np.array([i for i in order if masks.hit(f, all_t[i])], dtype=np.int64)

        def zone_center(g):
            gr = g["grasp"]
            return np.array([gr["x"], gr["y"], gr["z"]]), np.asarray(gr["rotation_matrix"])

        def in_zone(t, r, center, rot):
            d = np.linalg.norm(t - center, axis=1) * 1000.0
            ok = d <= gmm
            if gdeg is not None:
                ok &= np.array([cbt.angle_deg(rot, ri) <= gdeg for ri in r])
            return ok, d

        # ---- one tracker run for one strategy
        def run(strategy):
            tracker = trp.create_tracker(argparse.Namespace(checkpoint_path=ckpt, filter=args.filter, vis=False))
            if not tracker:
                raise SystemExit("could not create the tracker")
            det, ids, sel, traj_t, traj_r = [], [0], None, [], []
            for f in range(n_frames):
                colors, depth = trp.load_frame(frames, f, k_mat, scale)
                pts, cols = trp.build_cloud(colors, depth, k_mat)
                target_gg, curr_gg, tids, _ = tracker.update(pts, cols, ids)
                all_t = np.asarray(curr_gg.translations, dtype=np.float64)
                all_r = np.asarray(curr_gg.rotation_matrices, dtype=np.float64)
                sc = np.asarray(curr_gg.scores, dtype=np.float64)
                if f < s:                                   # phase 1: the tracker's detections, on the pipe
                    order = pipe_order(curr_gg, all_t, f)
                    for rnk, i in enumerate(order):
                        det.append({"frame": f, "rank": rnk + 1, "score": float(sc[i]),
                                    "width": float(curr_gg.widths[i]), "depth": float(curr_gg.depths[i]),
                                    "pos": all_t[i], "R": all_r[i]})
                elif f == s:                                # select
                    sel = select(strategy, curr_gg, all_t, all_r, sc, f, det)
                    if not sel:
                        break
                    ids = np.array([x["id"] for x in sel], dtype=np.int64)
                    start_t, start_r = all_t[ids], all_r[ids]
                else:                                       # phase 2: follow
                    ids = np.asarray(tids).reshape(-1).astype(np.int64)
                    n = len(sel)
                    t = np.full((n, 3), np.nan)
                    r = np.full((n, 3, 3), np.nan)
                    if target_gg is not None and len(target_gg):
                        tt = np.asarray(target_gg.translations, dtype=np.float64)[:n]
                        rr = np.asarray(target_gg.rotation_matrices, dtype=np.float64)[:n]
                        t[:len(tt)], r[:len(rr)] = tt, rr
                    traj_t.append(t)
                    traj_r.append(r)
            del tracker
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            if not sel:
                return None
            return evaluate(strategy, sel, start_t, start_r, np.stack(traj_t), np.stack(traj_r))

        # ---- the four selections (all pick tracker grasps of frame s on the pipe)
        def select(strategy, gg, all_t, all_r, sc, f, det):
            pipe = pipe_order(gg, all_t, f)
            if len(pipe) == 0:
                return []
            out, used = [], set()

            def take(i, zone=None, zsupp=None, zsupp_c=None):
                used.add(int(i))
                out.append({"id": int(i), "score": float(sc[i]), "zone": zone, "zone_support_A": zsupp,
                            "zone_support_C": zsupp_c})

            if strategy == "a":
                for i in pipe[:args.k]:
                    take(i)
                return out

            cands_c = cbt.rerank([dict(c) for c in det])
            zones_c = db.summarise_groups(db.group_candidates(cands_c, gmm, gdeg), cands_c, s, 1024, "penalized") \
                if cands_c else []
            zones_c.sort(key=lambda g: (-g["support"], -g["avg_score"]))
            free = lambda: np.array([i for i in pipe if int(i) not in used], dtype=np.int64)  # noqa: E731

            if strategy == "b":
                for g in zones_c:
                    if len(out) >= args.k:
                        break
                    fr = free()
                    if not len(fr):
                        break
                    c, rot = zone_center(g)
                    ok, d = in_zone(all_t[fr], all_r[fr], c, rot)
                    if ok.any():
                        take(fr[ok][np.argmin(d[ok])], zone=c.tolist(), zsupp_c=g["support"])
            elif strategy == "c":
                for g in zones_a:
                    if len(out) >= args.k:
                        break
                    fr = free()
                    if not len(fr):
                        break
                    c, rot = zone_center(g)
                    ok, _ = in_zone(all_t[fr], all_r[fr], c, rot)
                    if ok.any():
                        take(fr[ok][np.argmax(sc[fr[ok]])], zone=c.tolist(), zsupp=g["support"])
            elif strategy == "d":
                for g in zones_a:
                    if len(out) >= args.k:
                        break
                    ca, rota = zone_center(g)
                    gc_ = cbt.nearest_group([z for z in zones_c if z["support"] >= args.min_c_support],
                                            ca, rota, gmm, gdeg)[0]
                    if gc_ is None:
                        continue
                    fr = free()
                    if not len(fr):
                        break
                    cc, rotc = zone_center(gc_)
                    ok, d = in_zone(all_t[fr], all_r[fr], cc, rotc)
                    if ok.any():
                        take(fr[ok][np.argmin(d[ok])], zone=cc.tolist(), zsupp=g["support"], zsupp_c=gc_["support"])
            return out

        def evaluate(strategy, sel, t0, r0, tt, rr):
            grasps = []
            for j, x in enumerate(sel):
                p, r = tt[:, j], rr[:, j]
                ok = np.isfinite(p[:, 0])
                p, r = p[ok], r[ok]
                if not len(p):
                    continue
                dev = np.linalg.norm(p - t0[j], axis=1) * 1000.0
                ang = np.array([cbt.angle_deg(r0[j], ri) for ri in r])
                anch = (dev <= args.anchor_mm) & (ang <= args.anchor_deg)
                frames_idx = np.arange(s + 1, n_frames)[ok]
                on_pipe = np.mean([masks.hit(int(fi), q) for fi, q in zip(frames_idx, p)])
                zone_stay = None
                if x["zone"] is not None:
                    zone_stay = float((np.linalg.norm(p - np.asarray(x["zone"]), axis=1) * 1000.0 <= gmm).mean())
                grasps.append({**{k: v for k, v in x.items() if k != "zone"}, "rank": j + 1,
                               "anchored": round(float(anch.mean()), 3), "drift_mm": round(float(dev.max()), 1),
                               "jitter_mm": round(float(np.sqrt((p.std(axis=0) ** 2).sum()) * 1000.0), 1),
                               "on_pipe": round(float(on_pipe), 3),
                               "zone_stay": None if zone_stay is None else round(zone_stay, 3)})
            # selected grasps, for a later execution test (camera frame of frame s)
            out_dir = seq / "init_strategies"
            out_dir.mkdir(exist_ok=True)
            np.savez(out_dir / f"{strategy}_{STRATS[strategy].replace(' ', '_')}.npz",
                     translations=t0.astype(np.float32), rotation_matrices=r0.astype(np.float32),
                     scores=np.array([x["score"] for x in sel], np.float32), frame=s)
            return grasps

        # ---- run every strategy on this scene
        result = {"seed": sd.name, "seq": str(seq), "phase1": s, "k": args.k, "zones_A": len(zones_a), "strategies": {}}
        for st in args.strategies:
            g = run(st)
            if not g:
                print(f"  ({st}) {STRATS[st]:<8}: nothing selected")
                result["strategies"][st] = None
                continue
            top1 = g[0]
            summ = {"n": len(g), "top1_anchored": top1["anchored"], "top1_drift_mm": top1["drift_mm"],
                    "mean_anchored": round(float(np.mean([x["anchored"] for x in g])), 3),
                    "median_drift_mm": round(float(np.median([x["drift_mm"] for x in g])), 1),
                    "mean_on_pipe": round(float(np.mean([x["on_pipe"] for x in g])), 3),
                    "mean_score": round(float(np.mean([x["score"] for x in g])), 4),
                    "grasps": g}
            result["strategies"][st] = summ
            print(f"  ({st}) {STRATS[st]:<8}: n {summ['n']} | top-1 anchored {summ['top1_anchored']:.2f}, drift "
                  f"{summ['top1_drift_mm']:6.1f} mm | mean anchored {summ['mean_anchored']:.2f}, median drift "
                  f"{summ['median_drift_mm']:6.1f} mm | on pipe {summ['mean_on_pipe']:.2f} | score {summ['mean_score']:.3f}")
        (seq / "init_strategies" / "result.json").write_text(json.dumps(result, indent=2, default=float))
        per_seed.append(result)

    if not per_seed:
        raise SystemExit("no scene processed")

    # ---- pooled
    def col(st, key):
        return [r["strategies"][st][key] if r["strategies"].get(st) else None for r in per_seed]

    pooled = {}
    print(f"\nPOOLED over {len(per_seed)} scenes (k = {args.k}, zones from {args.phase1} frames)")
    print(f"{'strategy':<12} {'scenes':>6} {'top-1 anch':>10} {'mean anch':>9} {'med drift':>10} {'on pipe':>8} "
          f"{'score':>6} | vs (a): {'top-1':>14} {'mean':>14}")
    base_t1, base_m = col("a", "top1_anchored"), col("a", "mean_anchored")
    for st in args.strategies:
        t1, m = col(st, "top1_anchored"), col(st, "mean_anchored")
        dr, op, sc = col(st, "median_drift_mm"), col(st, "mean_on_pipe"), col(st, "mean_score")

        def mean(v):
            v = [x for x in v if x is not None]
            return float(np.mean(v)) if v else float("nan")
        row = {"scenes": sum(x is not None for x in m), "top1_anchored": mean(t1), "mean_anchored": mean(m),
               "median_drift_mm": mean(dr), "on_pipe": mean(op), "score": mean(sc)}
        vs = ""
        if st != "a" and "a" in args.strategies:
            d1 = [None if a is None or b is None else b - a for a, b in zip(base_t1, t1)]
            dm = [None if a is None or b is None else b - a for a, b in zip(base_m, m)]
            row["diff_top1_vs_a"] = {"mean": mean(d1), "positive": sum(1 for x in d1 if x is not None and x > 0),
                                     "n": sum(x is not None for x in d1), "sign_p": sign_test(d1)}
            row["diff_mean_vs_a"] = {"mean": mean(dm), "positive": sum(1 for x in dm if x is not None and x > 0),
                                     "n": sum(x is not None for x in dm), "sign_p": sign_test(dm)}

            def fmt(q):
                p = f"p={q['sign_p']:.2f}" if q["sign_p"] is not None else "p=n/a"
                return f"{q['mean']:+.2f} {q['positive']}/{q['n']} {p}"
            vs = f"{fmt(row['diff_top1_vs_a']):>14} {fmt(row['diff_mean_vs_a']):>14}"
        pooled[st] = row
        print(f"({st}) {STRATS[st]:<8} {row['scenes']:>6} {row['top1_anchored']:>10.2f} {row['mean_anchored']:>9.2f} "
              f"{row['median_drift_mm']:>9.1f}mm {row['on_pipe']:>8.2f} {row['score']:>6.3f} | {vs}")
    print("  vs (a): mean difference in anchored | scenes where it is better | sign test p (< 0.05 with 10 scenes "
          "needs 9/10)")
    (runs / "init_strategies_summary.json").write_text(json.dumps(
        {"settings": vars(args), "pooled": pooled, "per_scene": [
            {"seed": r["seed"], **{st: ({k: v for k, v in r["strategies"][st].items() if k != "grasps"}
                                         if r["strategies"].get(st) else None) for st in args.strategies}}
            for r in per_seed]}, indent=2, default=float))
    print(f"[INFO] {runs / 'init_strategies_summary.json'}")


if __name__ == "__main__":
    main()
