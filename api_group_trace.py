"""Send the first frame of a burst to the pipeFit API, then evaluate the API's grasp candidates with our grouping
and traceability over the WHOLE burst (group_and_trace.py).

  1. request   frame START of the burst -> API scene (v01/color.png, depth.npy, camera.json), using api_scene.py
               sim sequence: native format (colour PNG + depth in metres + K), no reprojection
               real burst:   cloud_XXX.npy (N, 6) through api_scene.py's .npy path (needs --calib)
  2. API       sent, polled until done (2-6 min), answer saved as api_result.json
               (or --api_result to reuse an answer you already have, without sending anything)
  3. convert   every candidate's T_base_grasp (AnyGrasp grasp frame: col 0 approach, col 1 closing, origin =
               grasp centre) back to the camera frame with the SAME T_base_cam that was sent:
               T_cam_grasp = inv(T_base_cam) @ T_base_grasp
  4. evaluate  group_and_trace.py on all frames of the burst with these candidates as --external grasps:
               the group each candidate falls in (support, avg rank, group traceability) and its own trace
  5. merge     api_candidates_eval.csv: the API's own features (score, width, tilt, d_from_cog, d_wall, clearance,
               ...) next to ours (group, support, trace)

Settings: CONFIG block below (pose, which candidates); grouping / tracking / mask settings are the CONFIG of
group_and_trace.py (edit them there; --mask overrides the mask mode).

Needs: api_scene.py next to this script (or --api_script), the lab network for the API.
Environment: anygrasp. Examples (project root):
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/api_group_trace.py --seq runs/multi/seed_100/seq_0000 --mask sim --vis
    ... --detections runs/multi/seed_100/seq_0000/burst_result.json     reuse detections (no detector run)
    ... --api_result runs/.../api_group_trace/api_result.json            reuse an API answer (no request)
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/api_group_trace.py --clouds ~/Time_consistency/burst \
        --calib camera_calibration.json --mask box
Output: <input>/api_group_trace/: request.zip, T_base_cam.json, api_result.json, api_candidates.json,
        api_candidates_eval.csv, and group_and_trace's groups.csv / grasps.csv / external.csv
"""

# =====================================================================================================
# CONFIG: edit here
# =====================================================================================================
CONFIG = {
    "START_FRAME": 0,              # the frame sent to the API (= group_and_trace TRACKING start_frame: keep equal)
    "POSE": "virtual",             # T_base_cam sent with the scene:
                                   #   "virtual"  api_scene.py's camera looking straight down (its default)
                                   #   "sim"      the real wrist-camera pose from the sim (poses.npz + robot base below)
                                   #   a path     a json file with {"T_base_cam": 4x4}
    "SIM_ROBOT_BASE": {            # "sim" pose only: robot root in the world, as in tasks/ur10e_env_cfg.py
        "pos": (0.0, 0.0, 1.341),
        "yaw_deg": 180.0,
    },
    "CANDIDATES": {
        "per_tube": 0,             # candidates kept per tube, best first; 0 = all the API returns
        "pose_key": "T_base_grasp",  # grasp centre in AnyGrasp's convention (comparable with our detections);
                                     # "T_base_tip" is the fingertip centre the robot executes (depth already added)
    },
    "ANYGRASP_VIEW": "v01",
}
# =====================================================================================================

import argparse
import csv
import importlib.util
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_api(path: Path):
    if not path.exists():
        raise SystemExit(f"{path} not found: put api_scene.py next to this script or pass --api_script")
    spec = importlib.util.spec_from_file_location("api_scene", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def sim_T_base_cam(seq: Path, frame: int) -> np.ndarray:
    """Camera pose in the robot base frame from the sim: T_base_cam = inv(T_world_base) @ T_world_cam."""
    p = np.load(seq / "poses.npz")
    t_wc = np.eye(4)
    t_wc[:3, :3] = quat_to_mat(p["cam_quat_w_ros"][frame])    # OpenCV optical frame (as the API wants)
    t_wc[:3, 3] = p["cam_pos_w"][frame]
    b = CONFIG["SIM_ROBOT_BASE"]
    yaw = np.radians(b["yaw_deg"])
    t_wb = np.eye(4)
    t_wb[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    t_wb[:3, 3] = b["pos"]
    return np.linalg.inv(t_wb) @ t_wc


# --------------------------------------------------------------- 1. request
def request_from_seq(api, seq: Path, frame: int, t_base_cam: np.ndarray) -> bytes:
    """Native API scene from a sim frame: colour PNG + depth in METRES + K + pose. No reprojection."""
    from PIL import Image
    meta = json.loads((seq / "meta.json").read_text())
    rgb = np.asarray(Image.open(seq / "frames" / f"color_{frame:03d}.png"))[..., :3]
    clean = seq / "frames" / f"depth_clean_{frame:03d}.npy"
    if clean.exists():                       # the API adds the real D435's noise itself: send clean depth
        depth_file = clean
    else:
        depth_file = seq / "frames" / f"depth_{frame:03d}.npy"
        if meta.get("noise", False):
            print("[WARN] only NOISY depth in this sequence (captured before depth_clean_000.npy existed): the API "
                  "will add its own noise on top. Re-capture with the current capture_sequence.py for clean depth")
    print(f"[request] depth from {depth_file.name}")
    depth = np.load(depth_file).astype(np.float32) / meta.get("depth_scale", 1000.0)
    cam = {"K": np.asarray(meta["K"], float).tolist(), "T_base_cam": t_base_cam.tolist(), "convention": "opencv"}
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v01/color.png", api.png_bytes(rgb))
        zf.writestr("v01/depth.npy", api.npy_bytes(depth))
        zf.writestr("v01/camera.json", json.dumps(cam, indent=1))
    print(f"[request] sim frame {frame}: {rgb.shape[1]}x{rgb.shape[0]}, depth {depth[depth > 0].min():.3f}-"
          f"{depth.max():.3f} m")
    return b.getvalue()


def request_from_cloud(api, cloud: Path, calib: Path, pose_json, tcp_pose=None) -> tuple[bytes, np.ndarray]:
    """Real burst frame through api_scene.py's .npy path (reprojects the cloud with the calibration K).
    tcp_pose (UR TCP at capture, m + rotvec) -> T_base_cam = T_base_tcp @ inv(T_flange_tcp) @ T_flange_cam."""
    ns = argparse.Namespace(cloud=str(cloud), colors=None, png=None, calib=str(calib), tcp_pose=tcp_pose,
                            pose_json=pose_json)
    z = api.npy_to_zip(ns)
    cam = json.loads(zipfile.ZipFile(io.BytesIO(z)).read("v01/camera.json"))
    return z, np.asarray(cam["T_base_cam"], float)


def send(api, z: bytes, url, token, out: Path) -> dict:
    q = f"?anygrasp_view={CONFIG['ANYGRASP_VIEW']}"
    job = api.request(url, token, "POST", "/api/sim/scenes" + q, z, "application/zip")["job"]
    print(f"[api] job {job} sent, waiting (the server takes 2-6 min per scene; the real robot goes first)...")
    t0, last = time.time(), None
    while True:
        st = api.request(url, token, "GET", f"/api/sim/jobs/{job}", timeout=30)
        line = f"{st.get('state')} {st.get('step') or ''} {st.get('waiting_for') or ''}".strip()
        if line != last:
            print(f"  [{time.time() - t0:5.0f} s] {line}", flush=True)
            last = line
        if st.get("state") == "done":
            break
        if st.get("state") in ("failed", "cancelled"):
            raise SystemExit(f"job {st.get('state')}: {st.get('error')}")
        time.sleep(3.0)
    res = api.request(url, token, "GET", f"/api/sim/jobs/{job}/result", timeout=60)
    (out / "api_result.json").write_text(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------- 3. convert
def candidates_in_camera(res: dict, t_base_cam: np.ndarray):
    """[{name, tube, rank, x, y, z, rotation_matrix, width, depth, score, features...}] in the camera frame."""
    t_cam_base = np.linalg.inv(t_base_cam)
    key = CONFIG["CANDIDATES"]["pose_key"]
    order = {tid: i for i, tid in enumerate(res.get("pick_order", []) or [])}
    tubes = {str(t.get("id")): t for t in res.get("tubes", [])}
    out = []
    for gi, g in enumerate(res.get("grasps", [])):
        tube = g.get("tube_id", g.get("tube", g.get("id", gi)))
        cands = g.get("candidates", [])
        if CONFIG["CANDIDATES"]["per_tube"]:
            cands = cands[:CONFIG["CANDIDATES"]["per_tube"]]
        for r, c in enumerate(cands):
            if key not in c:
                print(f"[WARN] tube {tube} candidate {r}: no '{key}', skipped (keys: {sorted(c)[:12]})")
                continue
            t = t_cam_base @ np.asarray(c[key], float)
            feats = {k: v for k, v in c.items() if not isinstance(v, (list, dict))}
            tb = tubes.get(str(tube), {})
            out.append({"name": f"t{tube}_c{r + 1}", "tube": tube, "cand_rank": r + 1,
                        "pick_order": order.get(tube, order.get(str(tube))),
                        "tube_type": tb.get("type"), "tube_p_correct": tb.get("p_correct"),
                        "x": float(t[0, 3]), "y": float(t[1, 3]), "z": float(t[2, 3]),
                        "rotation_matrix": t[:3, :3].tolist(),
                        "width": c.get("width_m", c.get("width")), "depth": c.get("depth_m", c.get("depth")),
                        "score": c.get("score"), **{f"api_{k}": v for k, v in feats.items()}})
    return out


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="pipeFit API candidates -> our grouping + traceability")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", help="sim sequence (capture_sequence.py)")
    src.add_argument("--clouds", help="real burst folder with cloud_XXX.npy (capture_burst.py)")
    ap.add_argument("--calib", help="camera_calibration.json (real burst: K and image size for api_scene.py)")
    ap.add_argument("--tcp_pose", type=float, nargs=6, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                    help="real burst: UR TCP pose at capture (m, rotvec, as the teach pendant / getActualTCPPose) "
                         "-> real T_base_cam from the calibration (otherwise CONFIG POSE)")
    ap.add_argument("--detections", help="reuse a burst_result.json for the grouping / tracking (no detector run)")
    ap.add_argument("--api_result", help="reuse an API answer (api_result.json): nothing is sent")
    ap.add_argument("--api_script", default=str(HERE / "api_scene.py"))
    ap.add_argument("--url", default=None)
    ap.add_argument("--token", default=None, help="default: the one in api_scene.py")
    ap.add_argument("--mask", choices=["none", "sim", "image", "box"], help="passed to group_and_trace.py")
    ap.add_argument("--out", help="output folder (default <input>/api_group_trace)")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    api = load_api(Path(args.api_script).expanduser())
    url, token = args.url or api.DEF_URL, args.token or api.DEF_TOKEN
    s = CONFIG["START_FRAME"]
    if args.seq:
        seq = Path(args.seq).expanduser().resolve()
        root = seq
    else:
        clouds = Path(args.clouds).expanduser().resolve()
        root = clouds
        files = sorted(clouds.glob("cloud_*.npy"))
        if not files:
            raise SystemExit(f"no cloud_XXX.npy in {clouds}")
    out = Path(args.out).expanduser().resolve() if args.out else root / "api_group_trace"
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1 + 2: request and answer
    if args.api_result:
        res = json.loads(Path(args.api_result).expanduser().read_text())
        tbc = out / "T_base_cam.json"
        if not tbc.exists():
            raise SystemExit(f"--api_result needs the T_base_cam that was sent: {tbc} (written by the run that sent it)")
        t_base_cam = np.asarray(json.loads(tbc.read_text())["T_base_cam"], float)
        print(f"[api] reusing {args.api_result}")
    else:
        pose = CONFIG["POSE"]
        if args.seq:
            if pose == "virtual":
                t_base_cam = np.asarray(api.T_VIRTUAL, float)
            elif pose == "sim":
                t_base_cam = sim_T_base_cam(seq, s)
            else:
                t_base_cam = np.asarray(json.loads(Path(pose).expanduser().read_text())["T_base_cam"], float)
            z = request_from_seq(api, seq, s, t_base_cam)
        else:
            if not args.calib:
                raise SystemExit("--clouds needs --calib (camera_calibration.json)")
            pose_json = None if pose in ("virtual", "sim") else pose
            z, t_base_cam = request_from_cloud(api, files[s], Path(args.calib).expanduser(), pose_json, args.tcp_pose)
            if args.tcp_pose:
                pose = "tcp_pose"
        print(f"[pose] {pose}: camera at {np.round(t_base_cam[:3, 3], 3).tolist()} in the base frame")
        (out / "request.zip").write_bytes(z)
        (out / "T_base_cam.json").write_text(json.dumps({"T_base_cam": t_base_cam.tolist(), "pose": pose,
                                                         "frame": s}, indent=1))
        res = send(api, z, url, token, out)
    if isinstance(res.get("camera"), dict):
        print(f"[api] camera info from the API: {json.dumps(res['camera'])[:300]}")

    # ---- 3: candidates to the camera frame
    cands = candidates_in_camera(res, t_base_cam)
    if not cands:
        raise SystemExit("the API returned no grasp candidates")
    (out / "api_candidates.json").write_text(json.dumps(cands, indent=1))
    print(f"[api] {len(res.get('tubes', []))} tubes, pick_order {res.get('pick_order')}, {len(cands)} candidates "
          f"-> camera frame of frame {s}")

    # ---- 4: grouping + traceability over the whole burst, candidates as external grasps
    gat = HERE / "group_and_trace.py"
    cmd = [sys.executable, str(gat), "--external", str(out / "api_candidates.json"), "--out", str(out)]
    if args.detections:
        cmd += ["--detections", str(Path(args.detections).expanduser().resolve())]
    elif args.seq:
        cmd += ["--seq", str(seq)]
    else:
        cmd += ["--clouds", str(clouds)]
    if args.mask:
        cmd += ["--mask", args.mask]
    if args.vis:
        cmd += ["--vis"]
    print("[eval] " + " ".join(cmd[1:]))
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("group_and_trace.py failed")

    # ---- 5: merge
    ext = {r["name"]: r for r in csv.DictReader(open(out / "external.csv"))}
    rows = []
    for c in cands:
        e = ext.get(c["name"], {})
        rows.append({"name": c["name"], "tube": c["tube"], "pick_order": c["pick_order"], "cand_rank": c["cand_rank"],
                     "api_score": c["score"], "group": e.get("group") or None, "group_support": e.get("group_support"),
                     "group_avg_rank": e.get("group_avg_rank"), "group_trace_mean": e.get("group_trace_mean"),
                     "trace": e.get("trace"), "nearest_detection_mm": e.get("nearest_detection_mm"),
                     "in_mask": e.get("in_mask"), "width": c["width"], "depth": c["depth"],
                     **{k: v for k, v in c.items() if k.startswith("api_") and k != "api_score"}})
    with open(out / "api_candidates_eval.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nAPI CANDIDATES evaluated over the burst")
    print(f"  {'name':<10} {'pick':>4} {'API score':>9} | {'group':>5} {'support':>7} {'grp trace':>9} | "
          f"{'trace':>5} {'nearest':>8}")
    for r in rows:
        f = lambda v, fmt: (fmt % float(v)) if v not in (None, "", "None") else "-"  # noqa: E731
        print(f"  {r['name']:<10} {str(r['pick_order']) if r['pick_order'] is not None else '-':>4} "
              f"{f(r['api_score'], '%.3f'):>9} | {('G' + str(r['group'])) if r['group'] else 'none':>5} "
              f"{f(r['group_support'], '%.2f'):>7} {f(r['group_trace_mean'], '%.2f'):>9} | "
              f"{f(r['trace'], '%.2f'):>5} {f(r['nearest_detection_mm'], '%.1f'):>6}mm")
    print(f"[INFO] {out / 'api_candidates_eval.csv'}")


if __name__ == "__main__":
    main()
