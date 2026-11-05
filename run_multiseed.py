"""Run the whole burst-vs-tracking comparison on several scenes (seeds) and pool the results.

Per seed, in order (each step is skipped if its output already exists, so an interrupted run resumes):
  1. capture   scripts/capture_sequence.py      (env_isaaclab)  new scene, 1 s at 30 fps  -> <out>/seed_S/seq_0000
  2. burst     seq_to_burst.py + detect_burst.py (anygrasp)     run A on those frames    -> seq/burst_result.json
  3. track     track_replay.py                  (anygrasp)      run B (+ C detections)   -> seq/tracking/<tag>/
  4. compare   compare_burst_tracking.py        (anygrasp)      A vs B vs C, mask on     -> .../compare.json
Then it pools every compare.json:
  per seed: baseline, lifts (overall, stable vs unstable by survival and by anchored stability), Spearman
  pooled:   mean +- std of the lifts, paired difference stable - unstable per seed + sign test, Spearman on all tracks
Outputs: <out>/summary.json, <out>/tracks.csv (one row per tracked grasp of every seed), logs in <out>/seed_S/logs/

detect_burst.py works in <burst_root>/burst and overwrites <burst_root>/burst_result.json, so the seeds run one
after the other; each seed's result is copied into its own sequence folder.

Run from the project root with the anygrasp python (needs numpy only):
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/run_multiseed.py --burst_root ~/Time_consistency_sim --n 10
    ... --seeds 7 11 23              explicit seeds instead of --n / --seed0
    ... --device cpu                 passed to Isaac (physics on CPU)
    ... --aggregate_only             only pool the compare.json files that exist
    ... --redo                       recompute every step
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # <project>/anygrasp
PROJECT = HERE.parent
HOME = Path.home()


def run(cmd, log: Path, cwd=None, env=None) -> bool:
    """Run a command, stdout + stderr to `log`. True if it exited with 0."""
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log, "w") as f:
        f.write("$ " + " ".join(str(c) for c in cmd) + "\n\n")
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=env).returncode
    print(f"      {'ok ' if rc == 0 else 'FAILED'} ({time.time() - t:5.0f} s)  log: {log}")
    return rc == 0


def latest_seq(seed_dir: Path):
    seqs = sorted(p for p in seed_dir.glob("seq_*") if (p / "meta.json").exists())
    return seqs[-1] if seqs else None


def sign_test(diffs) -> float | None:
    """Two-sided sign test p-value: are the per-seed differences positive more often than chance?"""
    d = [x for x in diffs if x is not None and not np.isnan(x) and x != 0]
    n = len(d)
    if n == 0:
        return None
    k = sum(x > 0 for x in d)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


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


def mean_std(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)


# --------------------------------------------------------------- one seed
def do_seed(seed: int, args, isaac_py, ag_py, root: Path, out: Path, tag: str) -> Path | None:
    seed_dir = out / f"seed_{seed}"
    logs = seed_dir / "logs"
    env_ag = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8")}
    print(f"\n=== seed {seed} ===")

    # 1. capture
    seq = latest_seq(seed_dir)
    if seq is None or args.redo:
        print("  [1/4] capture (Isaac)")
        cmd = [isaac_py, PROJECT / "scripts" / "capture_sequence.py", "--enable_cameras", "--headless",
               "--fps", args.fps, "--duration", args.duration, "--seed", seed, "--out", seed_dir]
        if args.device:
            cmd += ["--device", args.device]
        if not run(cmd, logs / "1_capture.log", cwd=PROJECT):
            return None
        seq = latest_seq(seed_dir)
    else:
        print(f"  [1/4] capture: exists ({seq.name})")
    meta = json.loads((seq / "meta.json").read_text())
    if not meta.get("scene_ok", True):
        print("  scene_ok = False (the pipe did not settle in the bin): seed skipped")
        return None

    # 2. burst (run A)
    result = seq / "burst_result.json"
    if not result.exists() or args.redo:
        print("  [2/4] burst (detector + grouping)")
        if not run([ag_py, HERE / "seq_to_burst.py", "--seq", seq, "--burst_root", root], logs / "2a_seq_to_burst.log",
                   cwd=PROJECT, env=env_ag):
            print("  (seq_to_burst refused or failed: is --burst_root the _sim copy? see the log)")
            return None
        if not run([ag_py, "detect_burst.py", "--top_k", 1024, "--group_mm", args.group_mm, "--group_deg",
                    args.group_deg, "--rank_mode", "penalized"], logs / "2b_detect_burst.log", cwd=root, env=env_ag):
            return None
        shutil.copy(root / "burst_result.json", result)
    else:
        print("  [2/4] burst: exists")

    # 3. tracking (run B, and the per-frame detections for C)
    track_dir = seq / "tracking" / tag
    if not (track_dir / "tracks.npz").exists() or args.redo:
        print("  [3/4] tracking")
        if not run([ag_py, HERE / "track_replay.py", "--seq", seq, "--no_cone", "--num_track", args.num_track,
                    "--select_nms", "--save_topk", 1024], logs / "3_track_replay.log", cwd=PROJECT, env=env_ag):
            return None
    else:
        print("  [3/4] tracking: exists")

    # 4. compare
    cmp_json = track_dir / "compare.json"
    if not cmp_json.exists() or args.redo:
        print("  [4/4] compare")
        if not run([ag_py, HERE / "compare_burst_tracking.py", "--burst_root", root, "--burst_result", result,
                    "--track", track_dir, "--mask", "--group_mm", args.group_mm, "--group_deg", args.group_deg,
                    "--zone_top", args.zone_top, "--anchor_mm", args.anchor_mm],
                   logs / "4_compare.log", cwd=PROJECT, env=env_ag):
            return None
    else:
        print("  [4/4] compare: exists")
    return cmp_json


# --------------------------------------------------------------- pooling
def aggregate(cmp_files: dict, out: Path):
    rows, per_seed = [], []
    for seed, path in sorted(cmp_files.items()):
        c = json.loads(Path(path).read_text())
        b = c.get("zone_baseline") or {}
        tracks = c["tracks_B"]
        top_a = c["groups_A"][:10]
        top_c = c["groups_C"][:10]
        per_seed.append({
            "seed": seed,
            "tracks": len(tracks),
            "A_groups": len(c["groups_A"]), "C_groups": len(c["groups_C"]),
            "A_top10_support": float(np.mean([g["support"] for g in top_a])) if top_a else None,
            "C_top10_support": float(np.mean([g["support"] for g in top_c])) if top_c else None,
            "baseline": b.get("pipe_surface_in_zones"),
            "lift_overall": b.get("lift_overall"),
            "lift_stable": b.get("lift_stable"), "lift_unstable": b.get("lift_unstable"),
            "lift_stable_anch": b.get("lift_stable_anchored"), "lift_unstable_anch": b.get("lift_unstable_anchored"),
            "n_stable_anch": b.get("n_stable_anchored"),
            "rho_Csupp_anch": c["spearman"].get("C_support_vs_B_anchored"),
            "rho_Csupp_surv": c["spearman"].get("C_support_vs_B_survival"),
        })
        for t in tracks:
            rows.append({"seed": seed, **{k: t.get(k) for k in (
                "track", "on_pipe_frame0", "score_frame0", "survival", "anchored", "jumps", "jitter_mm", "drift_mm",
                "c_group", "c_support", "a_group", "a_support", "a_match_mm", "in_A_zones", "A_zone_main")}})

    def diff(s, a, b):
        return None if s[a] is None or s[b] is None else s[a] - s[b]
    d_surv = [diff(s, "lift_stable", "lift_unstable") for s in per_seed]
    d_anch = [diff(s, "lift_stable_anch", "lift_unstable_anch") for s in per_seed]
    pooled = {
        "seeds": len(per_seed),
        "tracks": len(rows),
        "lift_overall": mean_std([s["lift_overall"] for s in per_seed]),
        "lift_stable_minus_unstable_survival": {"mean_std_n": mean_std(d_surv),
                                                "seeds_positive": sum(1 for d in d_surv if d is not None and d > 0),
                                                "sign_test_p": sign_test(d_surv)},
        "lift_stable_minus_unstable_anchored": {"mean_std_n": mean_std(d_anch),
                                                "seeds_positive": sum(1 for d in d_anch if d is not None and d > 0),
                                                "sign_test_p": sign_test(d_anch)},
        "spearman_all_tracks": {
            "C_support_vs_anchored": spearman([r["c_support"] if r["c_support"] is not None else np.nan for r in rows],
                                              [r["anchored"] for r in rows]),
            "C_support_vs_survival": spearman([r["c_support"] if r["c_support"] is not None else np.nan for r in rows],
                                              [r["survival"] for r in rows]),
            "A_support_vs_anchored": spearman([r["a_support"] if r["a_support"] is not None else np.nan for r in rows],
                                              [r["anchored"] for r in rows]),
            "anchored_vs_in_A_zones": spearman([r["anchored"] for r in rows], [r["in_A_zones"] for r in rows]),
        },
    }
    (out / "summary.json").write_text(json.dumps({"per_seed": per_seed, "pooled": pooled}, indent=2))
    if rows:
        with open(out / "tracks.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- print
    def f2(x):
        return f"{x:5.2f}" if isinstance(x, (int, float)) else "    -"
    print(f"\n{'seed':>6} {'trk':>4} {'A grp':>5} {'C grp':>5} {'A sup':>5} {'C sup':>5} {'base':>5} {'lift':>5} | "
          f"{'surv: st':>8} {'unst':>5} | {'anch: st':>8} {'unst':>5} {'n st':>4} | {'rho C/anch':>10}")
    for s in per_seed:
        print(f"{s['seed']:>6} {s['tracks']:>4} {s['A_groups']:>5} {s['C_groups']:>5} {f2(s['A_top10_support'])} "
              f"{f2(s['C_top10_support'])} {f2(s['baseline'])} {f2(s['lift_overall'])} | {f2(s['lift_stable']):>8} "
              f"{f2(s['lift_unstable'])} | {f2(s['lift_stable_anch']):>8} {f2(s['lift_unstable_anch'])} "
              f"{s['n_stable_anch'] if s['n_stable_anch'] is not None else '-':>4} | {f2(s['rho_Csupp_anch']):>10}")

    def ms(t):
        m, sd, n = t
        return f"{m:+.2f} +- {sd:.2f} (n={n})" if m is not None else "n/a"
    p = pooled
    print(f"\nPOOLED over {p['seeds']} seeds, {p['tracks']} tracked grasps")
    print(f"  lift overall                         : {ms(p['lift_overall'])}")
    for key, name in (("lift_stable_minus_unstable_survival", "stable - unstable lift (survival)"),
                      ("lift_stable_minus_unstable_anchored", "stable - unstable lift (anchored)")):
        q = p[key]
        pv = f"{q['sign_test_p']:.3f}" if q["sign_test_p"] is not None else "n/a"
        print(f"  {name:<37}: {ms(q['mean_std_n'])} | positive in {q['seeds_positive']} seeds | sign test p = {pv}")
    print("  Spearman on all tracks               : " + " | ".join(
        f"{k} {v:+.2f}" if v is not None else f"{k} n/a" for k, v in p["spearman_all_tracks"].items()))
    print(f"\n[INFO] {out / 'summary.json'} and {out / 'tracks.csv'}")


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst_root", required=True, help="the _sim copy of the burst folder")
    ap.add_argument("--n", type=int, default=10, help="number of seeds (seed0, seed0+1, ...)")
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit seeds (overrides --n/--seed0)")
    ap.add_argument("--out", default="runs/multi")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--device", default=None, help="Isaac device, e.g. cpu")
    ap.add_argument("--group_mm", type=float, default=20.0)
    ap.add_argument("--group_deg", type=float, default=45.0)
    ap.add_argument("--zone_top", type=int, default=10)
    ap.add_argument("--anchor_mm", type=float, default=20.0)
    ap.add_argument("--num_track", type=int, default=100)
    ap.add_argument("--isaac_python", default=str(HOME / "miniforge3" / "envs" / "env_isaaclab" / "bin" / "python"))
    ap.add_argument("--anygrasp_python", default=str(HOME / "miniforge3" / "envs" / "anygrasp" / "bin" / "python"))
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    root = Path(args.burst_root).expanduser().resolve()
    out = (PROJECT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds else list(range(args.seed0, args.seed0 + args.n))
    tag = f"oneeuro_nocone_n{args.num_track}_nms"
    for py in (args.isaac_python, args.anygrasp_python):
        if not Path(py).exists() and not args.aggregate_only:
            raise SystemExit(f"python not found: {py} (use --isaac_python / --anygrasp_python)")
    if not (root / "detect_burst.py").exists():
        raise SystemExit(f"{root} has no detect_burst.py")

    cmp_files, failed, t0 = {}, [], time.time()
    for seed in seeds:
        if args.aggregate_only:
            seq = latest_seq(out / f"seed_{seed}")
            path = seq / "tracking" / tag / "compare.json" if seq else None
        else:
            path = do_seed(seed, args, args.isaac_python, args.anygrasp_python, root, out, tag)
        if path is not None and Path(path).exists():
            cmp_files[seed] = path
        else:
            failed.append(seed)
    print(f"\n{len(cmp_files)}/{len(seeds)} seeds done in {(time.time() - t0) / 60:.1f} min"
          + (f" | failed or skipped: {failed}" if failed else ""))
    if cmp_files:
        aggregate(cmp_files, out)


if __name__ == "__main__":
    sys.exit(main())
