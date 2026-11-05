"""Play one scene frame by frame and watch the top-N most traceable detector grasps come back (or not).

Same scoring as detector_traceability.py (re-detection within --match_mm / --match_deg of the START pose).
In every frame you see that frame's point cloud and, for each top grasp (one colour each):
  - a small sphere at its START pose (frame 0): the anchor
  - its gripper where the detector re-detected it in this frame (solid), or, if it was NOT re-detected,
    a dark grey gripper at the start pose
The terminal prints, per frame, which grasps were found and how far from the anchor.

Controls (window focused):  Right / N  next frame   Left / P  previous   Space  play / pause
                            S  save a screenshot (also every frame while --save is on)   Q / Esc  quit

Environment: anygrasp. From the project root:
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/play_detector_top.py --runs runs/multi --seed 100
    ... --top 5 --fps 4 --save            screenshots to <seq>/play_top/ (make a video with ffmpeg later)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PALETTE = [(0.12, 0.47, 0.95), (1.00, 0.50, 0.05), (0.10, 0.75, 0.20), (0.75, 0.25, 0.85), (0.05, 0.80, 0.85),
           (0.95, 0.85, 0.10), (0.90, 0.20, 0.20), (0.55, 0.35, 0.20)]
NAMES = ["blue", "orange", "green", "purple", "cyan", "yellow", "red", "brown"]
GREY = (0.25, 0.25, 0.25)
FLIP = np.diag([1.0, 1.0, -1.0, 1.0])     # camera frame -> a view that is not upside down (as track_replay.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/multi")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--match_mm", type=float, default=20.0)
    ap.add_argument("--match_deg", type=float, default=30.0)
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--dilate", type=int, default=8)
    ap.add_argument("--fps", type=float, default=4.0, help="playback speed with Space")
    ap.add_argument("--gripper_mm", type=float, default=3.0)
    ap.add_argument("--point_mm", type=float, default=4.0)
    ap.add_argument("--save", action="store_true", help="save a screenshot of every frame shown")
    args = ap.parse_args()

    runs = Path(args.runs).expanduser().resolve()
    sys.path.insert(0, str(HERE))
    import compare_burst_tracking as cbt  # noqa: E402
    import detector_traceability as dt  # noqa: E402
    import open3d as o3d

    seqs = sorted(p for p in (runs / f"seed_{args.seed}").glob("seq_*") if (p / "burst_result.json").exists())
    if not seqs:
        raise SystemExit(f"no burst_result.json in {runs / f'seed_{args.seed}'}")
    seq = seqs[-1]
    got = dt.process(seq, args, cbt)
    if got is None:
        raise SystemExit("nothing to show")
    rows, n_frames = got
    top = rows[:min(args.top, len(PALETTE))]
    print(f"seed {args.seed}: top {len(top)} of {len(rows)} grasps, frames {args.start}-{n_frames - 1}")
    for k, r in enumerate(top):
        print(f"  {NAMES[k]:<7} #{r['top_rank']}  trace {r['trace']:.2f}  score {r['score']:.3f}  "
              f"width {r['width_mm']:.0f} mm  tilt {r['tilt_deg']:.0f} deg")
    frames = list(range(args.start, n_frames))
    clouds = {}
    save_dir = seq / "play_top"
    if args.save:
        save_dir.mkdir(exist_ok=True)

    def geometry(f):
        if f not in clouds:
            c = dt.start_cloud(seq, f)
            c.transform(FLIP)
            clouds[f] = c
        meshes = o3d.geometry.TriangleMesh()
        status = []
        for k, r in enumerate(top):
            g, col = r["_g"], PALETTE[k]
            anchor = cbt.spheres([g["pos"]], args.point_mm / 1000.0, col)
            meshes += anchor
            if f == args.start:
                found, dist = g, 0.0
            else:
                _, found, dist = r["_matches"][f - args.start - 1]
            if found is not None:
                meshes += cbt.gripper_mesh(found["pos"], found["R"], found["width"], found["depth"], col,
                                           args.gripper_mm / 1000.0)
                status.append(f"{NAMES[k]} ok {dist:4.1f}mm")
            else:
                meshes += cbt.gripper_mesh(g["pos"], g["R"], g["width"], g["depth"], GREY, args.gripper_mm / 2000.0)
                status.append(f"{NAMES[k]} --  missing")
        meshes.transform(FLIP)
        return clouds[f], meshes, status

    state = {"i": 0, "play": False, "last": 0.0, "geoms": []}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"seed {args.seed}: top {len(top)} frame by frame (Space play, arrows step)",
                      width=1400, height=900)

    def show(i, reset=False):
        f = frames[i]
        cloud, meshes, status = geometry(f)
        for gm in state["geoms"]:
            vis.remove_geometry(gm, reset_bounding_box=False)
        state["geoms"] = [cloud, meshes]
        for gm in state["geoms"]:
            vis.add_geometry(gm, reset_bounding_box=reset)
        print(f"frame {f:3d}/{n_frames - 1} | " + " | ".join(status))
        if args.save:
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(save_dir / f"frame_{f:03d}.png"), do_render=True)

    def step(d):
        def cb(_vis):
            state["i"] = (state["i"] + d) % len(frames)
            show(state["i"])
            return False
        return cb

    def toggle(_vis):
        state["play"] = not state["play"]
        print("play" if state["play"] else "pause")
        return False

    def shot(_vis):
        save_dir.mkdir(exist_ok=True)
        p = save_dir / f"shot_{frames[state['i']]:03d}.png"
        vis.capture_screen_image(str(p), do_render=True)
        print(f"saved {p}")
        return False

    def animate(_vis):
        if state["play"] and time.time() - state["last"] >= 1.0 / args.fps:
            state["last"] = time.time()
            step(1)(_vis)
        return False

    for key, fn in ((262, step(1)), (ord("N"), step(1)), (263, step(-1)), (ord("P"), step(-1)),
                    (32, toggle), (ord("S"), shot)):
        vis.register_key_callback(key, fn)
    vis.register_animation_callback(animate)
    vis.get_render_option().point_size = 2.0
    show(0, reset=True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
