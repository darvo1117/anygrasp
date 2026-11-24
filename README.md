# anygrasp/ — grasp detection, grouping and traceability tools

Analysis scripts for AnyGrasp on the pipeFit bin-picking task, for both Isaac Sim captures and real
Orbbec bursts. The main question they answer: **which AnyGrasp grasps hold up over time?** In other words,
which grasps are detected again and again across a burst of frames. They also compare those grasps with
grasps found on the true 3D geometry and with the pipeFit API's candidates.

All scripts run with the `anygrasp` conda env's Python from the project root (`isaacsim_anygrasp/`):

```bash
PY=~/miniforge3/envs/anygrasp/bin/python
$PY anygrasp/<script>.py --help
```

Every script starts with a detailed docstring covering its inputs, steps, outputs and examples. Read it
before running the script.

## Requirements

- The **AnyGrasp SDK** (detector `gsnet.so` / tracker `tracker.so`) with its license and weights.
  These are machine-bound and **not included** in this repo (see `.gitignore`).
- Python packages: `numpy`, `open3d`, `scipy`, `Pillow`, `torch`, `graspnetAPI`.
- Input data comes from the Isaac Lab capture scripts in `../scripts/` (`capture.py`,
  `capture_sequence.py`, `capture_sequence_pipefit.py`) or from real Orbbec bursts (`cloud_XXX.npy`).
  Outputs are written into `runs/`, which is git-ignored.
- pipeFit API scripts: these need the lab network. The token is read from `$PIPEFIT_TOKEN` or `~/.pipefit_token`.

## Data formats

| Input | Contents |
|---|---|
| sim sequence | `meta.json` (K) + `frames/color_XXX.png` + `frames/depth_XXX.npy` (+ `poses.npz`, `meshes.npz` for pipeFit) |
| real burst | `cloud_XXX.npy`, (N, 6) XYZ + RGB in the camera optical frame, metres |
| API scene | `v01/color.png`, `v01/depth.npy`, `v01/camera.json` (folder or zip) |

Grasps use the camera's OpenCV optical frame (x right, y down, z forward). The grasp frame follows AnyGrasp:
column 0 is the approach axis, column 1 is the closing axis, and the origin is the grasp centre.

## Scripts

### Core: grouping + traceability
| Script | What it does |
|---|---|
| `seq_group_track.py` | **Main tool.** Runs the detector on every frame of a sequence (real or sim), groups all grasps, computes traceability and writes `grasps.csv` / `groups.csv`. Optional PNG mask. |
| `group_and_trace.py` | Grouping + traceability on a burst, plus evaluation of external grasps (e.g. API candidates) against both. Library used by most scripts below. |
| `burst_group.py` | Earlier version of the above for bursts / sequences / API scenes. |
| `detector_traceability.py` | Measures how often each start-frame grasp is re-detected at the same pose, using `burst_result.json`. |
| `play_detector_top.py` | Interactive frame-by-frame player for the most traceable grasps. |

### Ground truth vs. sensor (pipeFit scenes)
| Script | What it does |
|---|---|
| `model_vs_cloud.py` | AnyGrasp on the 3D model vs. on the sensor cloud. Checks every candidate against the true meshes with `TrueGeometry.check()`, which reports collision / empty / no_fc / ok. |
| `gt_vs_grouping.py` | Burst grouping "super candidates" vs. one AnyGrasp run on the whole 3D model: precision and coverage. Reuses `TrueGeometry`. |
| `merge_and_detect.py` | Merges one sensor frame with ground-truth tube points, detects once and writes `grasps.csv` with the true-geometry check. Reuses `TrueGeometry`. |
| `make_pipe_mask.py` | Renders a PNG mask of a single tube from the scene meshes. |

### pipeFit API
| Script | What it does |
|---|---|
| `api_scene.py` | Sends one scene (a `.npy` cloud or the API's native format) to the pipeFit API and saves the JSON answer. |
| `api_group_trace.py` | Sends a burst's first frame to the API, then scores its candidates with grouping + traceability over the whole burst. |
| `fit_cloud_calib.py` | Finds the resolution a cloud was captured at and writes a matching calibration JSON. |

### Detection vs. tracking studies
| Script | What it does |
|---|---|
| `track_replay.py` | Runs the AnyGrasp tracker on a recorded sequence as if live (approach cone, realtime mode, jitter/drift metrics). |
| `compare_burst_tracking.py` | Burst grouping (A) vs. tracking (B) vs. grouping on the tracker's grasps (C), with a 3D view. |
| `compare_init_strategies.py` | Compares four ways of choosing which grasp the tracker should start from. |
| `run_multiseed.py` | Runs capture → burst → track → compare over many seeds and pools the results. |
| `seq_to_burst.py` | Converts a sim sequence into the `burst/` format used by `detect_burst.py`. |
| `check_models.py` | Detector vs. tracker on one cloud: weights, speed, overlap and collision rates. |

### Batch detection (single frame)
| Script | What it does |
|---|---|
| `batch_detect.py` | Runs AnyGrasp on every cloud of a capture batch and keeps the grasps on the target mask. |
| `view_grasps.py` | Viewer for one env after `batch_detect.py`, with the tilt filter applied. |

Files ending in `_old.py` / `_1.py` are earlier versions kept for reference.

## Typical workflow (pipeFit, sim)

```bash
# 1. capture a burst in Isaac Lab (env_isaaclab)
python scripts/capture_sequence_pipefit.py ...            # -> runs/pipefit/seq_XXXX
# 2. group + trace the detector grasps
$PY anygrasp/seq_group_track.py --input runs/pipefit/seq_XXXX
# 3. compare against the true geometry
$PY anygrasp/model_vs_cloud.py   --seq runs/pipefit/seq_XXXX
$PY anygrasp/gt_vs_grouping.py   --seq runs/pipefit/seq_XXXX
$PY anygrasp/merge_and_detect.py --seq runs/pipefit/seq_XXXX --pipe pipe_1
```
