"""Find the image resolution a cloud_XXX.npy was captured at, and write a calibration that matches it.

The Orbbec cloud is depth aligned to colour at the colour stream's resolution. camera_calibration.json has the
1920x1080 K; if the stream ran at e.g. 1280x720, projecting with the 1920 K leaves the points off the pixel grid
and the image the API gets is full of holes (the bin is not found). Same sensor, so K scales with the resolution.

For each candidate resolution: K_scaled = K * s (cx, cy with the half-pixel shift), off-grid error (0 = exact) and
the fraction of image pixels the cloud fills. Writes <calib>_<W>x<H>.json for the best one and a preview PNG of
the colour image the API would get.

    ~/miniforge3/envs/anygrasp/bin/python anygrasp/fit_cloud_calib.py \
        --cloud ~/Time_consistency/burst/cloud_000.npy --calib ~/Time_consistency/camera_calibration.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

CANDIDATES = [(1920, 1080), (1280, 720), (960, 540), (640, 360), (2560, 1440), (3840, 2160), (1280, 800), (640, 480)]


def project(P, K, W, H):
    u = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]
    v = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
    ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    err = np.abs(np.r_[u[ok] - ui[ok], v[ok] - vi[ok]])
    pix = np.unique(vi[ok] * W + ui[ok]).size
    return float(np.percentile(err, 95)) if ok.any() else 1.0, pix / (W * H), int(ok.sum()), ui, vi, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--calib", required=True)
    args = ap.parse_args()

    calib_path = Path(args.calib).expanduser()
    calib = json.loads(calib_path.read_text())
    K0 = np.asarray(calib["camera"]["K"], float)
    W0, H0 = calib["camera"]["resolution"]
    D = np.load(Path(args.cloud).expanduser()).astype(np.float64)
    P = D[:, :3]
    P = P[np.isfinite(P).all(1) & (P[:, 2] > 0)]
    print(f"cloud: {len(P)} points, z {P[:, 2].min():.3f}-{P[:, 2].max():.3f} m"
          + ("   (looks like mm!)" if P[:, 2].max() > 20 else ""))

    rows = []
    for W, H in CANDIDATES:
        sx, sy = W / W0, H / H0
        K = K0.copy()
        K[0, 0] *= sx; K[1, 1] *= sy
        K[0, 2] = (K0[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (K0[1, 2] + 0.5) * sy - 0.5
        err, fill, n_in, *_ = project(P, K, W, H)
        rows.append((err, -fill, W, H, K, fill, n_in))
        print(f"  {W:4d}x{H:<4d}  off-grid p95 {err:.2f} px   fills {100 * fill:5.1f} % of the image   "
              f"{n_in} points inside")

    # the true resolution: points land on the grid (err ~0) AND at most one point per pixel
    good = [r for r in rows if r[6] <= r[2] * r[3]]
    err, _, W, H, K, fill, _ = min(good, key=lambda r: (round(r[0], 2), -r[5]))
    print(f"\nbest: {W}x{H} (off-grid {err:.2f} px, fills {100 * fill:.1f} %)")
    if err > 0.1:
        print("  [WARN] no candidate puts the points on the grid: the cloud may be undistorted / filtered, "
              "or captured at a resolution not in the list. The best one is still written.")
    if (W, H) == (W0, H0):
        print("  the calibration already matches; the problem is elsewhere (look at the preview PNG)")

    out = dict(calib)
    out["camera"] = dict(calib["camera"], K=K.tolist(), resolution=[W, H],
                         intrinsics_source=f"{calib['camera'].get('intrinsics_source', '')} scaled to {W}x{H}")
    out_path = calib_path.with_name(f"{calib_path.stem}_{W}x{H}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"written {out_path}")

    # preview of the colour image the API receives
    _, _, _, ui, vi, ok = project(P, K, W, H)
    if D.shape[1] >= 6:
        C = D[np.isfinite(D[:, :3]).all(1) & (D[:, 2] > 0), 3:6]
        C = C / 255.0 if C.max() > 1.5 else C
        img = np.zeros((H, W, 3), np.uint8)
        img[vi[ok], ui[ok]] = np.clip(C[ok] * 255 + 0.5, 0, 255).astype(np.uint8)
        prev = Path(args.cloud).expanduser().with_name(f"preview_{W}x{H}.png")
        try:
            import cv2
            cv2.imwrite(str(prev), img[:, :, ::-1])
        except ImportError:
            from PIL import Image
            Image.fromarray(img).save(prev)
        print(f"preview  {prev}  (open it: the bin should look solid and blue, no stripes)")


if __name__ == "__main__":
    main()
