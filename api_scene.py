#!/usr/bin/env python3
"""Send one scene to the pipeFit API and save its answer as JSON.

Input, one of:
  A) your capture (.npy):   --cloud CLOUD.npy [--colors COLORS.npy] [--png COLOR.png] --calib camera_calibration.json
       CLOUD: (N,3) XYZ, or (N,6) XYZ+RGB, camera optical frame (x right, y down, z forward), metres
       COLORS: (N,3), 0..1 or 0..255          PNG: colour image of the same camera (preferred for colour)
  B) the API's native format:  --scene FOLDER_OR_ZIP
       FOLDER: v01/color.png, v01/depth.npy, v01/camera.json (more views allowed: v06/, a1/ ...),
               or a single view directly: color.png, depth.npy, camera.json
       ZIP:    the same, zipped (sent as is)

Camera pose (only for A; B already has it in camera.json):
  default         virtual camera looking straight down (fine if the camera was ~vertical)
  --tcp-pose X Y Z RX RY RZ   real UR TCP pose at capture (m, rotvec)  -> T_base_cam from the calibration
  --pose-json F   file with {"T_base_cam": 4x4}

Output:
  --out result.json        the API answer, unchanged
  --save-request req.zip   (optional) the exact zip that was sent: reusable later with --scene req.zip

Examples:
  python3 api_scene.py --cloud scene.npy --colors scene_colors.npy --png scene.png --calib camera_calibration.json --out result.json
  python3 api_scene.py --scene my_scene/ --out result.json
  python3 api_scene.py --scene req.zip --out result_again.json
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DEF_URL = "http://172.26.0.172:8798"
DEF_TOKEN = (os.environ.get("PIPEFIT_TOKEN") or (Path.home() / ".pipefit_token").read_text().strip()) if (Path.home() / ".pipefit_token").exists() else os.environ.get("PIPEFIT_TOKEN", "")

# virtual camera: straight down, at the height of the pipeFit cell's v01 view
T_VIRTUAL = np.array([[0, -1, 0, 0.351171], [-1, 0, 0, 0.331158], [0, 0, -1, 0.637301], [0, 0, 0, 1.0]])


# ------------------------------------------------------------------ helpers
def rotvec_to_R(rv):
    rv = np.asarray(rv, float); th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def pose6_to_T(p):
    T = np.eye(4); T[:3, :3] = rotvec_to_R(p[3:6]); T[:3, 3] = p[:3]; return T


def png_bytes(rgb):
    rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8))
    try:
        from PIL import Image
        b = io.BytesIO(); Image.fromarray(rgb).save(b, format="PNG"); return b.getvalue()
    except ImportError:
        import cv2
        return cv2.imencode(".png", rgb[..., ::-1])[1].tobytes()


def read_png(path):
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    except ImportError:
        import cv2
        return cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1].copy()


def npy_bytes(a):
    b = io.BytesIO(); np.save(b, np.asarray(a, np.float32)); return b.getvalue()


# ------------------------------------------------------------------ A) .npy -> API format
def npy_to_zip(a):
    calib = json.load(open(a.calib))
    K = np.array(calib["camera"]["K"], float)
    W, H = calib["camera"]["resolution"]

    P = np.load(a.cloud).astype(np.float64)
    C = None
    if P.ndim == 2 and P.shape[1] >= 6:
        P, C = P[:, :3], P[:, 3:6]
    if a.colors:
        C = np.load(a.colors).astype(np.float64)

    # project the cloud with K -> depth image (the closest point wins per pixel)
    ok = np.isfinite(P).all(1) & (P[:, 2] > 0)
    u = np.rint(K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]).astype(int)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.where(ok)[0][np.argsort(-P[ok, 2])]
    depth = np.zeros((H, W), np.float32); depth[v[idx], u[idx]] = P[idx, 2]
    err = float(np.abs(K[0, 0] * P[ok, 0] / P[ok, 2] + K[0, 2] - u[ok]).max()) if ok.any() else 0
    print(f"[npy] {ok.sum()} points -> {(depth > 0).sum()} depth pixels "
          f"({depth[depth > 0].min():.3f}-{depth.max():.3f} m), reprojection off-grid {err:.2f} px")
    if err > 0.25:
        print("  ⚠ the points do not fall on the pixel grid: is the calibration K the one of this cloud?")

    if a.png:
        rgb = read_png(a.png)
        if rgb.shape[:2] != (H, W):
            sys.exit(f"PNG is {rgb.shape[1]}x{rgb.shape[0]}, the calibration is {W}x{H}")
    elif C is not None:
        C = C / 255.0 if C.max() > 1.5 else C
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[v[idx], u[idx]] = np.clip(C[idx] * 255 + 0.5, 0, 255).astype(np.uint8)
    else:
        sys.exit("no colour: pass --png, --colors or an (N,6) cloud")

    if a.tcp_pose:
        T_fl_tcp = pose6_to_T(np.array(calib["robot"]["tcp_offset_xyz_rotvec"], float))
        T = pose6_to_T(np.array(a.tcp_pose)) @ np.linalg.inv(T_fl_tcp) @ np.array(calib["T_flange_cam"], float)
        src = "real (TCP pose)"
    elif a.pose_json:
        T, src = np.array(json.load(open(a.pose_json))["T_base_cam"], float), a.pose_json
    else:
        T, src = T_VIRTUAL, "virtual (straight down)"
    print(f"[pose] {src}")

    cam = {"K": K.tolist(), "T_base_cam": T.tolist(), "convention": "opencv"}
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v01/color.png", png_bytes(rgb))
        zf.writestr("v01/depth.npy", npy_bytes(depth))
        zf.writestr("v01/camera.json", json.dumps(cam, indent=1))
    return b.getvalue()


# ------------------------------------------------------------------ B) native API format
def native_to_zip(path):
    p = Path(path)
    if p.is_file() and p.suffix == ".zip":
        names = zipfile.ZipFile(p).namelist()
        if not any(n.endswith("camera.json") for n in names):
            sys.exit(f"{p} has no camera.json: not an API scene zip")
        print(f"[native] zip as is: {sorted(names)}")
        return p.read_bytes()
    if not p.is_dir():
        sys.exit(f"{p} is neither a folder nor a .zip")
    files = {"color.png", "depth.npy", "camera.json"}
    views = [d for d in sorted(p.iterdir()) if d.is_dir() and files <= {f.name for f in d.iterdir()}]
    single = files <= {f.name for f in p.iterdir()}
    if not views and not single:
        sys.exit(f"{p}: expected <view>/color.png, depth.npy, camera.json (or those three files directly)")
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        for d, name in ([(p, "v01")] if single and not views else [(d, d.name) for d in views]):
            for f in files:
                zf.write(d / f, f"{name}/{f}")
            check_view(d, name)
    print(f"[native] views: {[n for n in (['v01'] if single and not views else [d.name for d in views])]}")
    return b.getvalue()


def check_view(d, name):
    cam = json.load(open(d / "camera.json"))
    dep = np.load(d / "depth.npy")
    for k in ("K", "T_base_cam"):
        if k not in cam:
            sys.exit(f"{name}/camera.json has no '{k}'")
    if dep.ndim != 2:
        sys.exit(f"{name}/depth.npy must be (H, W), it is {dep.shape}")
    z = dep[np.isfinite(dep) & (dep > 0)]
    if z.size and z.max() > 20:
        print(f"  ⚠ {name}: depth up to {z.max():.0f} -> looks like mm, the API wants METRES")


# ------------------------------------------------------------------ API
def request(url, token, method, path, data=None, ctype=None, timeout=300):
    r = urllib.request.Request(url.rstrip("/") + path, data=data, method=method)
    r.add_header("X-Token", token)
    if ctype:
        r.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {path}: {e.read().decode('utf-8', 'replace')[:500]}"
                 + (" (token wrong or changed?)" if e.code in (401, 403) else ""))
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {url}: {e.reason} (lab network? server down?)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud"); ap.add_argument("--colors"); ap.add_argument("--png"); ap.add_argument("--calib")
    ap.add_argument("--scene", help="native API scene: folder or .zip")
    ap.add_argument("--tcp-pose", nargs=6, type=float); ap.add_argument("--pose-json")
    ap.add_argument("--url", default=DEF_URL); ap.add_argument("--token", default=DEF_TOKEN)
    ap.add_argument("--view", default="v01", help="view whose image AnyGrasp uses")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--save-request")
    ap.add_argument("--poll", type=float, default=3.0); ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()

    if a.scene:
        z = native_to_zip(a.scene)
    elif a.cloud:
        if not a.calib:
            sys.exit("--cloud needs --calib (for K and the image size)")
        z = npy_to_zip(a)
    else:
        sys.exit("give --cloud ... or --scene ...  (see --help)")
    if a.save_request:
        Path(a.save_request).write_bytes(z); print(f"[request] saved {a.save_request}")

    q = f"?anygrasp_view={a.view}" + (f"&note={urllib.request.quote(a.note)}" if a.note else "")
    j = request(a.url, a.token, "POST", "/api/sim/scenes" + q, z, "application/zip")
    job = j["job"]; print(f"[api] job {job} sent, waiting...")
    t0, last = time.time(), None
    while True:
        st = request(a.url, a.token, "GET", f"/api/sim/jobs/{job}", timeout=30)
        line = f"{st.get('state')} {st.get('step') or ''} {st.get('waiting_for') or ''}".strip()
        if line != last:
            print(f"  [{time.time() - t0:5.0f} s] {line}", flush=True); last = line
        if st.get("state") == "done":
            break
        if st.get("state") in ("failed", "cancelled"):
            sys.exit(f"job {st.get('state')}: {st.get('error')}")
        if time.time() - t0 > a.timeout:
            sys.exit(f"timeout after {a.timeout:.0f} s (job {job} may still be queued)")
        time.sleep(a.poll)

    res = request(a.url, a.token, "GET", f"/api/sim/jobs/{job}/result", timeout=60)
    Path(a.out).write_text(json.dumps(res, indent=1))
    n = sum(len(g.get("candidates", [])) for g in res.get("grasps", []))
    print(f"[done] {len(res.get('tubes', []))} tubes, pick_order={res.get('pick_order')}, "
          f"{n} grasp candidates -> {a.out}")


if __name__ == "__main__":
    main()
