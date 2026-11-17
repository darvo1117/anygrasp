"""PNG mask of ONE tube (white = that tube), for seq_group_track.py --mask.

The sim masks (frames/mask_XXX.png) hold every tube (semantic class "pipe"). This renders the scene meshes
(meshes.npz + poses.npz of the sequence, from capture_sequence_pipefit.py) from the camera and keeps the pixels
where the chosen tube is the front surface. Static camera: one mask is valid for every frame of the burst.

Environment: anygrasp (open3d). Needs model_vs_cloud.py next to it.
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/make_pipe_mask.py --seq runs/pipefit/seq_0007_b3            # list
    ~/miniforge3/envs/anygrasp/bin/python anygrasp/make_pipe_mask.py --seq runs/pipefit/seq_0007_b3 --pipe pipe_1
Output: <seq>/mask_<pipe>.png  (same size as the frames)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model_vs_cloud as mvc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="PNG mask of one tube from the scene meshes")
    ap.add_argument("--seq", required=True, help="sequence folder with meta.json, poses.npz, meshes.npz")
    ap.add_argument("--pipe", help="tube name (e.g. pipe_1); without it: list the tubes and their visible pixels")
    ap.add_argument("--frame", type=int, default=0, help="camera pose of this frame (static camera: any)")
    ap.add_argument("--out", help="output PNG (default <seq>/mask_<pipe>.png)")
    args = ap.parse_args()

    import open3d as o3d
    from PIL import Image
    seq = Path(args.seq).expanduser().resolve()
    meta = json.loads((seq / "meta.json").read_text())
    _, k, objects = mvc.load_scene(seq, args.frame)
    w, h = meta["resolution"]
    scene = o3d.t.geometry.RaycastingScene()
    ids = {}
    for o in objects:
        gid = scene.add_triangles(o3d.core.Tensor(o["verts"].astype(np.float32)),
                                  o3d.core.Tensor(o["faces"].astype(np.uint32)))
        ids[int(gid)] = o["name"]
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(k), extrinsic_matrix=o3d.core.Tensor(np.eye(4)),
        width_px=int(w), height_px=int(h))
    gid = scene.cast_rays(rays)["geometry_ids"].numpy().astype(np.int64)     # (h, w), invalid id for no hit
    counts = {name: int((gid == i).sum()) for i, name in ids.items()}
    print(f"[mask] visible pixels per object ({w}x{h}): {counts}")
    if not args.pipe:
        tubes = {n: c for n, c in counts.items() if n.startswith("pipe")}
        print(f"       tubes: {tubes}  -> run again with --pipe <name>")
        return
    if args.pipe not in counts:
        raise SystemExit(f"--pipe {args.pipe!r}: objects are {list(counts)}")
    pid = next(i for i, n in ids.items() if n == args.pipe)
    mask = (gid == pid)
    out = Path(args.out).expanduser() if args.out else seq / f"mask_{args.pipe}.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(out)
    # how well it matches the sim's own mask (all tubes): the tube must lie inside it
    sim = seq / "frames" / f"mask_{args.frame:03d}.png"
    if sim.exists():
        m = np.asarray(Image.open(sim))
        m = (m[..., 0] if m.ndim == 3 else m) > 127
        print(f"[mask] {args.pipe}: {int(mask.sum())} px, {100 * (mask & m).sum() / max(mask.sum(), 1):.0f}% of them "
              f"inside the sim mask {sim.name} (~100% = aligned)")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
