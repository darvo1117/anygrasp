"""Turn a sequence from scripts/capture_sequence.py into the burst/ folder that detect_burst.py reads, so the
burst method runs on exactly the same frames as anygrasp/track_replay.py.

capture_burst.py (real Orbbec)                 this script (Isaac)
  depth aligned to colour, camera optical frame  same frame: x right, y down, z forward (OpenCV)
  PointCloudFilter -> mm, (N, 6) XYZ + RGB        depth_XXX.npy (mm) + K -> (N, 6) XYZ [m] + RGB [0..255]
  drop (0, 0, 0), mm -> m, SAVE_BOX crop          drop depth 0, same SAVE_BOX crop
Output: <burst_root>/burst/cloud_XXX.npy + burst_meta.json (same keys as capture_burst.py, plus "source": "isaac").

Refuses to overwrite a burst/ that came from the real camera (no "source": "isaac" in its meta), so pointing it at
the original folder by mistake cannot delete a real burst. Work in the copy (<folder>_sim).

Environment: anygrasp (numpy + PIL only)
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/seq_to_burst.py --seq runs/seq_0004 --burst_root ~/<folder>_sim
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SAVE_BOX = [(-0.5, 0.5), (-0.5, 0.5), (0.05, 1.2)]
"""Same loose crop as capture_burst.py (camera frame, m)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="runs/seq_XXXX from capture_sequence.py")
    ap.add_argument("--burst_root", required=True, help="the COPY of the burst folder (holds detect_burst.py)")
    ap.add_argument("--max_frames", type=int, default=60, help="same safety limit as capture_burst.py")
    ap.add_argument("--force", action="store_true", help="overwrite a burst/ even if it did not come from Isaac")
    args = ap.parse_args()

    seq = Path(args.seq).expanduser().resolve()
    root = Path(args.burst_root).expanduser().resolve()
    if not (root / "detect_burst.py").exists():
        raise SystemExit(f"{root} has no detect_burst.py: pass the copy of the burst folder")
    out = root / "burst"
    old_meta = out / "burst_meta.json"
    if old_meta.exists() and json.loads(old_meta.read_text()).get("source") != "isaac" and not args.force:
        raise SystemExit(f"{out} holds a REAL camera burst. Use the copy (<folder>_sim), or --force.")

    meta = json.loads((seq / "meta.json").read_text())
    k = np.asarray(meta["K"], dtype=np.float64)
    fps, scale = meta["fps"], meta.get("depth_scale", 1000.0)
    n = min(meta["n_frames"], args.max_frames)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    clouds = []
    for i in range(n):
        rgb = np.asarray(Image.open(seq / "frames" / f"color_{i:03d}.png"), dtype=np.float32)[..., :3]
        z = np.load(seq / "frames" / f"depth_{i:03d}.npy").astype(np.float64) / scale
        h, w = z.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        xyz = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)
        valid = z > 0
        data = np.concatenate([xyz[valid], rgb[valid]], axis=1)
        n_valid = len(data)
        keep = np.ones(n_valid, dtype=bool)
        for a, (lo, hi) in enumerate(SAVE_BOX):
            keep &= (data[:, a] > lo) & (data[:, a] < hi)
        data = data[keep].astype(np.float32)
        name = f"cloud_{i:03d}.npy"
        np.save(out / name, data)
        clouds.append({"file": name, "t_s": round(i / fps, 4), "valid_points": int(n_valid),
                       "saved_points": int(len(data))})

    burst_meta = {
        "source": "isaac",
        "seq": str(seq),
        "seq_level": meta.get("level"),
        "seed": meta.get("seed"),
        "seconds_requested": round(n / fps, 4),
        "span_s": round((n - 1) / fps, 4),
        "frames": n,
        "dropped_framesets": 0,
        "fps": fps,
        "has_color": True,
        "save_box_m": SAVE_BOX,
        "clouds": clouds,
    }
    (out / "burst_meta.json").write_text(json.dumps(burst_meta, indent=2))
    print(f"{n} frames of {seq.name} ({fps} fps) -> {out}  "
          f"(~{np.mean([c['saved_points'] for c in clouds]):.0f} points per cloud)")


if __name__ == "__main__":
    main()
