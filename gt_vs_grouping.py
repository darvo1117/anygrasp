"""Grouping over a 30-frame Isaac burst vs ONE AnyGrasp run on the 3D model of the scene (the reference).

Question: do the super candidates of the burst grouping look like the grasps AnyGrasp finds on the exact geometry?

Input: a sequence of scripts/capture_sequence_pipefit.py (frames/, meta.json, poses.npz, meshes.npz). Static camera.

  REFERENCE (model)  AnyGrasp ONCE on the WHOLE 3D model: points on every surface of the bin and of all tubes
                     (camera frame). Its collision check sees the complete geometry.
  GROUPING (burst)   AnyGrasp on every frame of the burst (camera depth with noise), then the rules of
                     seq_group_track.py: grouping, traceability, super candidates (one per group, by frequency).

Both are segmented to the SAME tube (--pipe, default: the most visible one) with the same 3-D rule: the grasp
centre within TUBE_MM of that tube's mesh and that tube between the fingers. Both are filtered the same way:
  - approach within MAX_TILT_DEG of the camera z axis (as SUPER max_tilt_deg in seq_group_track.py)
  - checked against the true meshes (model_vs_cloud.py): no collision, not empty, contacts inside the friction cone
    (EXCLUDE), and no bin point anywhere in the gripper (fingers, palm, approach corridor or between the fingers)
  - grouping only: trace >= MIN_TRACE
Then: model -> the N_MODEL (50) best by score; grouping -> N_GROUPING (10) super candidates.

Comparison: every super candidate is matched to the nearest model candidate (<= MATCH mm and deg, flip-aware):
precision = share of the 10 that the model also proposes; coverage = share of the model's top 10 / 50 that some
group (or some super candidate) reproduces.

Environment: anygrasp. Needs next to it: seq_group_track.py, group_and_trace.py, model_vs_cloud.py
(and compare_burst_tracking.py for --vis).
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/gt_vs_grouping.py --seq runs/pipefit/seq_0001
    ... --pipe pipe_1 --voxel 0.003 --vis
Output: <seq>/gt_vs_grouping/<pipe>/
  burst_grasps.csv   every burst grasp on the tube (all frames), seq_group_track.py grasps.csv columns
  burst_groups.csv   the burst groups on the tube, seq_group_track.py groups.csv columns
  model_grasps.csv   every model grasp on the tube, same grasp columns (frame 0, no group / trace)
    + at the end of each: tube_mm, check (ok / collision / empty / no_fc), between_fingers, bin_pts,
      filters (why it is dropped, "ok" = passes), pass, and super / model_top / by_super / by_group
  super.csv, model_top.csv   the two compared shortlists; summary.json
  model_cloud.npy    the cloud AnyGrasp received for the model (N, 6: xyz m camera frame + rgb by object);
  model_cloud_pipe.npy  only the selected tube's points of it
Re-running is cheap: both detections are cached in .cache/ (the detector does not run again).
"""

# =====================================================================================================
# CONFIG: edit here
# =====================================================================================================
CONFIG = {
    "PIPE": None,                          # tube to segment (meshes.npz name, e.g. "pipe_1"); None = most visible
    "TUBE_MM": 12.0,                       # grasp centre within this distance of the tube's surface (hose r ~8 mm)
    "MODEL": {
        "objects": "all",                  # reference cloud: "all" (bin + every tube) | "targets" (tubes only)
        "surface": "full",                 # "full" (every side) | "visible" (only what the camera sees)
        "sample_mm": 2.0,
    },
    "DETECTOR": {                          # the SAME settings for the model and for every frame of the burst
        "workspace": [(-0.4, 0.4), (-0.4, 0.4), (0.05, 1.0)],
        "voxel": 0.005,
        "max_gripper_width": 0.10,
        "gripper_height": 0.03,
        "collision_detection": True,
        "nms": True,
        "top_k": 0,
    },
    "GROUPING": {"group_mm": 20.0, "group_deg": 45.0, "rank_mode": "penalized", "min_support": 0.0},
    "TRACKING": {"match_mm": 20.0, "match_deg": 30.0},
    "FILTER": {
        "max_tilt_deg": 30.0,              # approach vs the camera z axis, both sets
        "min_trace": 0.5,                  # grouping super candidates only
        "exclude": ["collision", "empty", "no_fc"],   # labels of the true-geometry check that are dropped
        "no_bin_contact": True,            # drop grasps with any bin point in the gripper
    },
    "EVAL": {                              # true-geometry check (as model_vs_cloud.py)
        "gt_spacing_mm": 2.0, "finger_width": 0.01, "finger_length": 0.06, "height": 0.02,
        "approach_dist": 0.05, "collision_points": 3, "empty_points": 5, "contact_band_mm": 2.0, "mu": 0.4,
    },
    "N_MODEL": 50,
    "N_GROUPING": 10,
    "MATCH": {"mm": 20.0, "deg": 30.0},    # super candidate = model candidate
    "VIS": {"gripper_mm": 3.0, "model_labels": 10},
}
# =====================================================================================================

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import group_and_trace as gat  # noqa: E402
import model_vs_cloud as mvc   # noqa: E402
import seq_group_track as sgt  # noqa: E402


# --------------------------------------------------------------- helpers
def bin_points_in_gripper(geo, bin_ids, t, r, width, depth):
    """Number of bin surface points inside the whole gripper volume (fingers, palm, corridor, between the fingers)."""
    c = geo.cfg
    fl, fw, hh, ad = c["finger_length"], c["finger_width"], c["height"] / 2, c["approach_dist"]
    reach = np.sqrt((fl + fw + ad + depth) ** 2 + (width / 2 + fw) ** 2 + hh ** 2)
    idx = np.asarray(geo.tree.query_ball_point(t, reach), int)
    if not len(idx):
        return 0
    idx = idx[np.isin(geo.oid[idx], bin_ids)]
    if not len(idx):
        return 0
    loc = (geo.pts[idx] - t) @ r
    x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
    inside = (np.abs(z) < hh) & (np.abs(y) < width / 2 + fw) & (x < depth) & (x > depth - fl - fw - ad)
    return int(inside.sum())


def flip_angle(ra, rb):
    return gat.angle_deg(ra, rb)


def judge(geo, bin_ids, pipe, pos, R, width, depth, cfg):
    """True-geometry check + the filters shared by both sets. Returns (passes, info)."""
    e = geo.check(pos, R, width, depth)
    e["bin_pts"] = bin_points_in_gripper(geo, bin_ids, pos, R, width, depth)
    e["tilt"] = gat.tilt_deg(R)
    f = cfg["FILTER"]
    why = []
    if e["tilt"] > f["max_tilt_deg"]:
        why.append("tilt")
    if e["label"] in f["exclude"]:
        why.append(e["label"])
    if f["no_bin_contact"] and e["bin_pts"] > 0:
        why.append("bin")
    if e["object"] and e["object"] != pipe:
        why.append("other_object")
    e["why"] = "+".join(why) if why else "ok"
    return not why, e


def choose_pipe(objects, sensor_pts, rng):
    """Most visible tube: sensor points (frame 0) within 4 mm of its surface."""
    from scipy.spatial import cKDTree
    counts = {}
    for o in objects:
        if not o["target"]:
            continue
        s, _ = mvc.sample_surface(o["verts"], o["faces"], 0.003, rng)
        counts[o["name"]] = int((cKDTree(s).query(sensor_pts, distance_upper_bound=0.004)[0] < 0.004).sum())
    return max(counts, key=counts.get), counts


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="burst grouping vs one AnyGrasp run on the 3D model (reference)")
    ap.add_argument("--seq", required=True, help="sequence from scripts/capture_sequence_pipefit.py")
    ap.add_argument("--pipe", help="tube to compare (e.g. pipe_1); default: the most visible")
    ap.add_argument("--voxel", type=float, help="detector voxel for BOTH (m), e.g. 0.003")
    ap.add_argument("--n_model", type=int, help="model candidates kept (CONFIG N_MODEL)")
    ap.add_argument("--n_grouping", type=int, help="super candidates (CONFIG N_GROUPING)")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--out", help="output folder (default <seq>/gt_vs_grouping/<pipe>)")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()
    cfg = CONFIG
    if args.pipe:
        cfg["PIPE"] = args.pipe
    if args.voxel:
        cfg["DETECTOR"]["voxel"] = args.voxel
    if args.n_model:
        cfg["N_MODEL"] = args.n_model
    if args.n_grouping:
        cfg["N_GROUPING"] = args.n_grouping
    seq = Path(args.seq).expanduser().resolve()
    rng = np.random.default_rng(0)

    # ---- scene, frames
    meta, k, objects = mvc.load_scene(seq, 0)
    s_cfg = {**sgt.CONFIG, "DETECTOR": {**sgt.CONFIG["DETECTOR"], **cfg["DETECTOR"]}}
    frames, kind = sgt.load_frames(seq, s_cfg)
    n = len(frames)
    pipe, counts = (cfg["PIPE"], None) if cfg["PIPE"] else choose_pipe(objects, frames[0]["xyz"], rng)
    names = [o["name"] for o in objects]
    if pipe not in names:
        raise SystemExit(f"--pipe {pipe!r}: the scene has {names}")
    out = Path(args.out).expanduser().resolve() if args.out else seq / "gt_vs_grouping" / pipe
    (out / ".cache").mkdir(parents=True, exist_ok=True)
    print(f"[scene] {seq.name}: {n} frames ({kind}), objects {names}")
    if counts:
        print(f"[pipe] most visible: {pipe}  (sensor points on each tube: {counts})")
    geo = mvc.TrueGeometry(objects, cfg["EVAL"])
    bin_ids = [i for i, o in enumerate(objects) if not o["target"]]
    pipe_obj = objects[names.index(pipe)]
    from scipy.spatial import cKDTree
    pipe_tree = cKDTree(mvc.sample_surface(pipe_obj["verts"], pipe_obj["faces"], 0.002, rng)[0])

    def pipe_mm(p):
        return pipe_tree.query(np.atleast_2d(p))[0] * 1000.0

    def on_pipe(p):
        return pipe_mm(p) <= cfg["TUBE_MM"]

    # ---- 1. REFERENCE: AnyGrasp once on the whole model
    mcfg = cfg["MODEL"]
    ws = cfg["DETECTOR"]["workspace"]
    sel = [o for o in objects if (mcfg["objects"] == "all" or o["target"]) and mvc.in_workspace(o["verts"], ws)]
    if mcfg["surface"] == "visible":
        pts = mvc.render_visible([(o["verts"], o["faces"]) for o in sel], k, *meta["resolution"])
        from scipy.spatial import cKDTree as _T            # which object each rendered point belongs to
        d = np.stack([_T(mvc.sample_surface(o["verts"], o["faces"], 0.002, rng)[0]).query(pts)[0] for o in sel], 1)
        obj = d.argmin(1)
    else:
        srng = np.random.default_rng(0)                    # own seed: the same model cloud on every run
        parts = [mvc.sample_surface(o["verts"], o["faces"], mcfg["sample_mm"] / 1000.0, srng)[0] for o in sel]
        pts = np.concatenate(parts)
        obj = np.concatenate([np.full(len(p), j) for j, p in enumerate(parts)])
    print(f"[model] {mcfg['surface']} surface of {', '.join(o['name'] for o in sel)}: {len(pts)} points")
    # the cloud AnyGrasp received (camera frame, m): all objects, and only the selected tube
    col = np.array([(0.55, 0.55, 0.55) if not o["target"] else (0.9, 0.75, 0.1) if o["name"] == pipe
                    else (0.3, 0.5, 0.9) for o in sel])[obj] * 255.0
    np.save(out / "model_cloud.npy", np.c_[pts, col].astype(np.float32))
    is_pipe = np.array([o["name"] == pipe for o in sel])[obj]
    np.save(out / "model_cloud_pipe.npy", np.c_[pts[is_pipe], col[is_pipe]].astype(np.float32))
    print(f"[model] cloud saved: model_cloud.npy ({len(pts)} pts, all objects), model_cloud_pipe.npy "
          f"({int(is_pipe.sum())} pts, {pipe}) - (N, 6) xyz m camera frame + rgb (bin grey, {pipe} yellow, "
          "other tubes blue)")
    if args.no_cache and (out / ".cache" / "model.npz").exists():
        (out / ".cache" / "model.npz").unlink()
    z = mvc.detect("model", pts, np.full((len(pts), 3), 0.7), {"DETECTOR": cfg["DETECTOR"]}, out / ".cache")
    order = np.argsort(-z["score"])
    model_all, model = [], []
    for i in order:
        c = {"pos": z["pos"][i], "R": z["R"][i], "score": float(z["score"][i]), "width": float(z["width"][i]),
             "depth": float(z["depth"][i]), "rank_all": len(model_all) + 1}
        model_all.append(c)
        c["tube_mm"] = float(pipe_mm(c["pos"])[0])
        if c["tube_mm"] > cfg["TUBE_MM"]:
            continue
        ok, e = judge(geo, bin_ids, pipe, c["pos"], c["R"], c["width"], c["depth"], cfg)
        c["eval"], c["pass"] = e, ok
        if ok:
            model.append(c)
    n_on_pipe = sum(1 for c in model_all if "eval" in c)
    why = {}
    for c in model_all:
        if "eval" in c and c["eval"]["why"] != "ok":
            for w in c["eval"]["why"].split("+"):
                why[w] = why.get(w, 0) + 1
    print(f"[model] {len(model_all)} grasps, {n_on_pipe} on {pipe}, {len(model)} pass the filters "
          f"(dropped: {why or 'none'}) -> top {cfg['N_MODEL']}")
    model_top = model[:cfg["N_MODEL"]]

    # ---- 2. GROUPING: the burst, seq_group_track.py rules, same tube
    key = hashlib.sha1(json.dumps({"seq": str(seq), "n": n, "pts": [len(f["xyz"]) for f in frames],
                                   "det": cfg["DETECTOR"]}, default=str, sort_keys=True).encode()).hexdigest()[:16]
    cache = out / ".cache" / "burst.npz"
    cands = None if args.no_cache else sgt.load_cache(cache, key)
    if cands is None:
        print(f"[burst] AnyGrasp detector on {n} frames")
        cands = gat.run_detector([(f["frame"], f["xyz"], f["rgb"]) for f in frames], cfg["DETECTOR"])
        sgt.save_cache(cache, key, cands)
    else:
        print(f"[burst] cached detections ({len(cands)} grasps)")
    n_raw = len(cands)
    P = np.array([c["pos"] for c in cands]).reshape(-1, 3)
    dmm = pipe_mm(P) if len(P) else []
    for c, d in zip(cands, dmm):
        c["tube_mm"] = float(d)
    cands = [c for c in cands if c["tube_mm"] <= cfg["TUBE_MM"]]
    by_frame = {}
    for c in cands:
        by_frame.setdefault(c["frame"], []).append(c)
    for fl in by_frame.values():
        for r, c in enumerate(sorted(fl, key=lambda c: c["rank"])):
            c["rank"] = r + 1
    print(f"[burst] {n_raw} grasps over {n} frames, {len(cands)} on {pipe}")
    if not cands:
        raise SystemExit("no burst grasp on that tube")
    groups = gat.group(cands, n, cfg["GROUPING"])
    for g in groups:
        g["n_frames"] = n
    P = np.array([c["pos"] for c in cands])
    R = np.array([c["R"] for c in cands])
    F = np.array([c["frame"] for c in cands])
    fid = [fr["frame"] for fr in frames]
    hit = sgt.redetected(P, R, P, R, F, fid, cfg["TRACKING"])
    hit &= F[:, None] != np.array(fid)[None, :]
    for c, h in zip(cands, hit.sum(1)):
        c["trace_hits"], c["trace"] = int(h), h / max(n - 1, 1)
    if groups:
        rep_hit = sgt.redetected(np.array([g["centre"] for g in groups]), np.array([g["R"] for g in groups]),
                                 P, R, F, fid, cfg["TRACKING"])
        for g, h in zip(groups, rep_hit):
            tr = [cands[i]["trace"] for i in g["member_ids"]]
            g["trace_rep"], g["trace_mean"], g["trace_max"] = h.sum() / n, float(np.mean(tr)), float(np.max(tr))
    for c in cands:                                        # every burst grasp through the same check (for the table)
        c["pass"], c["eval"] = judge(geo, bin_ids, pipe, c["pos"], c["R"], c["width"], c["depth"], cfg)

    # super candidates: groups by frequency, per group the best member that passes the SAME filters + trace
    supers, skipped = [], []
    for g in sorted(groups, key=lambda g: (-g["frames_hit"], -g["members"], g["penalized_rank"])):
        if len(supers) == cfg["N_GROUPING"]:
            break
        for kk, i in enumerate(g["member_ids"]):
            c = cands[i]
            if gat.tilt_deg(c["R"]) > cfg["FILTER"]["max_tilt_deg"] or c["trace"] < cfg["FILTER"]["min_trace"]:
                continue
            if c["pass"]:
                c["super"] = len(supers) + 1
                supers.append({**c, "group": g["group"], "frames_hit": g["frames_hit"], "cand": kk + 1})
                g["super"] = len(supers)
                break
        else:
            skipped.append(g["group"])
    print(f"[grouping] {len(groups)} groups on {pipe}; {len(supers)} super candidates "
          f"(skipped groups with no member passing the filters: {len(skipped)})")

    # ---- 3. comparison
    mm, deg = cfg["MATCH"]["mm"], cfg["MATCH"]["deg"]

    def nearest(c, pool):
        best = None
        for j, m in enumerate(pool):
            d = np.linalg.norm(c["pos"] - m["pos"]) * 1000.0
            if d > mm:
                continue
            a = flip_angle(c["R"], m["R"])
            if a <= deg and (best is None or d < best[1]):
                best = (j, d, a)
        return best

    for s in supers:
        s["match"] = nearest(s, model_top)
    rep = [{"pos": g["centre"], "R": g["R"]} for g in groups]
    for m in model_top:
        m["by_super"] = nearest(m, supers) is not None
        m["by_group"] = nearest(m, rep) is not None        # some group's representative pose
    prec = np.mean([s["match"] is not None for s in supers]) if supers else np.nan
    cov10 = np.mean([m["by_super"] for m in model_top[:10]]) if model_top else np.nan
    cov_s = np.mean([m["by_super"] for m in model_top]) if model_top else np.nan
    cov_g = np.mean([m["by_group"] for m in model_top]) if model_top else np.nan
    dists = [s["match"][1] for s in supers if s["match"]]
    angs = [s["match"][2] for s in supers if s["match"]]

    # ---- 4. report
    print(f"\nSUPER CANDIDATES (grouping, {n} frames) vs the model's top {len(model_top)} on {pipe} "
          f"(match <= {mm:.0f} mm, {deg:.0f} deg)")
    print(f"  {'#':>2} {'grp':>4} {'frames':>6} {'cand':>4} {'score':>6} {'trace':>5} {'tilt':>4} {'width':>5} | "
          f"{'model rank':>10} {'dist':>6} {'angle':>5}")
    for k_, s in enumerate(supers, 1):
        m = s["match"]
        mt = (f"{m[0] + 1:>10} {m[1]:5.1f}mm {m[2]:4.0f}°" if m else f"{'none':>10} {'-':>6} {'-':>5}")
        print(f"  {k_:>2} {s['group']:>4} {s['frames_hit']:>3}/{n:<2} {s['cand']:>4} {s['score']:6.3f} {s['trace']:5.2f} "
              f"{s['eval']['tilt']:3.0f}° {s['width'] * 1000:4.0f}mm | {mt}")
    print(f"\n  precision : {pct(prec)} of the {len(supers)} super candidates are also proposed by the model"
          + (f" (mean {np.mean(dists):.1f} mm, {np.mean(angs):.0f} deg away)" if dists else ""))
    print(f"  coverage  : model top 10 -> {pct(cov10)} reproduced by a super candidate | top {len(model_top)} -> "
          f"{pct(cov_s)} by a super candidate, {pct(cov_g)} by some group of the burst")
    print(f"  model top-{len(model_top)} score {np.mean([m['score'] for m in model_top]):.3f} | super candidates "
          f"score {np.mean([s['score'] for s in supers]) if supers else float('nan'):.3f} (different clouds: compare "
          "poses, not scores)")

    row = lambda c: {"x": round(float(c["pos"][0]), 4), "y": round(float(c["pos"][1]), 4),  # noqa: E731
                     "z": round(float(c["pos"][2]), 4), **sgt.rot_cols(c["R"])}
    gat.write_csv(out / "super.csv", [{"n": k_, "group": s["group"], "frames_hit": s["frames_hit"], "cand": s["cand"],
                                       "frame": s["frame"], "score": round(s["score"], 4), "trace": round(s["trace"], 3),
                                       "tilt_deg": round(s["eval"]["tilt"], 1), "width_mm": round(s["width"] * 1000, 1),
                                       "check": s["eval"]["label"], "model_rank": s["match"][0] + 1 if s["match"] else None,
                                       "model_dist_mm": round(s["match"][1], 1) if s["match"] else None,
                                       "model_angle_deg": round(s["match"][2], 1) if s["match"] else None, **row(s)}
                                      for k_, s in enumerate(supers, 1)])
    gat.write_csv(out / "model_top.csv", [{"rank": k_, "rank_all": m["rank_all"], "score": round(m["score"], 4),
                                           "tilt_deg": round(m["eval"]["tilt"], 1), "width_mm": round(m["width"] * 1000, 1),
                                           "by_super": m["by_super"], "by_group": m["by_group"], **row(m)}
                                          for k_, m in enumerate(model_top, 1)])
    # ---- the full tables, in seq_group_track.py's format (+ the check columns at the end)
    fmt = sgt.fmt

    def grasp_row(gid, c, frame, rank, group, best, extra):
        e = c["eval"]
        r = {"grasp_id": gid, "frame": frame, "rank": rank, "score": fmt(c["score"], 4), "group": group,
             "group_best": best, "trace": fmt(c.get("trace")), "trace_hits": c.get("trace_hits"),
             "width_mm": fmt(c["width"] * 1000, 1), "depth_mm": fmt(c["depth"] * 1000, 1),
             "tilt_deg": fmt(e["tilt"], 1), "x": fmt(c["pos"][0], 4), "y": fmt(c["pos"][1], 4),
             "z": fmt(c["pos"][2], 4), **sgt.rot_cols(c["R"]),
             "tube_mm": fmt(c["tube_mm"], 1), "check": e["label"], "between_fingers": e["object"],
             "bin_pts": e["bin_pts"], "filters": e["why"], "pass": c.get("pass", e["why"] == "ok")}
        r.update(extra)
        return r

    member_of = {i: (g["group"], k_ == 0) for g in groups for k_, i in enumerate(g["member_ids"])}
    rows = []
    for i, c in sorted(enumerate(cands), key=lambda ic: (ic[1]["frame"], ic[1]["rank"])):
        grp, best = member_of.get(i, (None, False))
        rows.append(grasp_row(len(rows), c, c["frame"], c["rank"], grp, best, {"super": c.get("super")}))
    gat.write_csv(out / "burst_grasps.csv", rows)
    grows = []
    for g in groups:
        mem = [cands[i] for i in g["member_ids"]]
        grows.append({"group": g["group"], "n_grasps": g["members"], "frames_hit": g["frames_hit"],
                      "support": fmt(g["support"]), "avg_rank": fmt(g["avg_rank"], 2),
                      "penalized_rank": fmt(g["penalized_rank"], 2), "avg_score": fmt(g["avg_score"], 4),
                      "max_score": fmt(g["max_score"], 4), "spread_mm": fmt(g["spread_mm"], 1),
                      "rot_spread_deg": fmt(g["rot_spread_deg"], 1), "trace_rep": fmt(g["trace_rep"]),
                      "trace_mean": fmt(g["trace_mean"]), "trace_max": fmt(g["trace_max"]),
                      "width_mm": fmt(g["width"] * 1000, 1), "depth_mm": fmt(g["depth"] * 1000, 1),
                      "tilt_deg": fmt(g["tilt_deg"], 1), "x": fmt(g["centre"][0], 4), "y": fmt(g["centre"][1], 4),
                      "z": fmt(g["centre"][2], 4), **sgt.rot_cols(g["R"]),
                      "members_pass": sum(m["pass"] for m in mem), "super": g.get("super")})
    gat.write_csv(out / "burst_groups.csv", grows)
    top_rank = {id(m): k_ for k_, m in enumerate(model_top, 1)}
    mrows, rk = [], 0
    for c in model_all:
        if "eval" not in c:                                # not on the selected tube
            continue
        rk += 1
        mrows.append(grasp_row(len(mrows), c, 0, rk, None, False,
                               {"rank_all": c["rank_all"], "model_top": top_rank.get(id(c)),
                                "by_super": c.get("by_super"), "by_group": c.get("by_group")}))
    gat.write_csv(out / "model_grasps.csv", mrows)
    print(f"[out] tables: burst_grasps.csv ({len(rows)}), burst_groups.csv ({len(grows)}), "
          f"model_grasps.csv ({len(mrows)}) - seq_group_track.py columns + check columns")

    (out / "summary.json").write_text(json.dumps({
        "seq": str(seq), "pipe": pipe, "frames": n, "model_points": len(pts), "model_grasps": len(model_all),
        "model_on_pipe": n_on_pipe, "model_pass": len(model), "model_dropped": why, "burst_grasps": n_raw,
        "burst_on_pipe": len(cands), "groups": len(groups), "super": len(supers), "precision": prec,
        "coverage_top10_super": cov10, "coverage_super": cov_s, "coverage_groups": cov_g, "config": cfg},
        indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)))
    print(f"[out] {out}: super.csv, model_top.csv, summary.json")

    if args.vis:
        visualise(frames[0], geo, names.index(pipe), model_top, supers, cfg, f"{seq.name} {pipe}: grouping vs model")


def pct(x):
    return f"{100 * x:.0f}%" if x is not None and np.isfinite(x) else "-"


def visualise(fr, geo, pipe_idx, model_top, supers, cfg, title):
    import open3d as o3d
    import compare_burst_tracking as cbt
    t = cfg["VIS"]["gripper_mm"] / 1000.0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fr["xyz"]))
    pcd.colors = o3d.utility.Vector3dVector(fr["rgb"])
    geoms = [("sensor frame 0", pcd.voxel_down_sample(0.003))]
    pp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(geo.pts[geo.oid == pipe_idx]))
    pp.paint_uniform_color((1.0, 0.85, 0.1))
    geoms.append(("selected tube (true mesh)", pp))
    mg, sg, labels = o3d.geometry.TriangleMesh(), o3d.geometry.TriangleMesh(), []
    for k_, m in enumerate(model_top, 1):
        mg += cbt.gripper_mesh(m["pos"], m["R"], m["width"], m["depth"], (0.1, 0.75, 0.2), t * 0.7)
        if k_ <= cfg["VIS"]["model_labels"]:
            labels.append((m["pos"] + [0, 0, -0.025], f"M{k_}"))
    for k_, s in enumerate(supers, 1):
        sg += cbt.gripper_mesh(s["pos"], s["R"], s["width"], s["depth"], (1.0, 0.0, 1.0), t * 1.5)
        labels.append((s["pos"] + [0, 0, -0.045], f"S{k_}" + (f"=M{s['match'][0] + 1}" if s["match"] else "")))
    if len(mg.vertices):
        geoms.append((f"model top {len(model_top)} (green)", mg))
    if len(sg.vertices):
        geoms.append((f"super candidates (magenta)", sg))
    print("[vis] green = model reference (M#), magenta = grouping super candidates (S#, '=M3' = matches model #3), "
          "yellow = the selected tube")
    try:
        import open3d.visualization.gui as gui
        app = gui.Application.instance
        app.initialize()
        w = o3d.visualization.O3DVisualizer(title, 1400, 900)
        for name, g in geoms:
            w.add_geometry(name, g)
        for p, txt in labels:
            w.add_3d_label(p, txt)
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
