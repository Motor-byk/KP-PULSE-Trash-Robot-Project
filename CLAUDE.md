# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working style

**Explain before acting, not after.** Before running a command, editing a file, or adding a new
file, say what it is and what it will do — then do it. Never fold this into an end-of-turn summary.

For each command, edit, or new file, cover:

- **What** it does — the concrete effect. Files written, paths touched, how long it runs, whether
  it overwrites anything.
- **Why** it's needed *here* — the specific problem in this project it solves.
- **The concept underneath**, briefly — what the flag, API, or pattern actually does, and why the
  obvious alternative is worse. This is a learning project; an explanation should transfer to the
  next decision, not just justify this one.

A line or two per item is usually right. Go longer when the concept is new, shorter when it's a
repeat of something already explained this session.

Read-only inspection (`cat`, `grep`, `ls`, reading a file) doesn't need a preamble unless what it
turns up changes the plan.

**Never hand back an action — "run this", "test that", "go move the camera" — while any command,
edit, or new file leading up to it is still unexplained.**

## Project

Vision stage for a trash-sorting robot arm. A YOLO detector finds objects on the work surface,
classifies them into concrete item types, and maps each to one of three disposal bins
(`recycle` / `compost` / `waste`), emitting a centroid for the arm to grasp.

The arm itself is **4DOF with no wrist roll or yaw**. A grasp angle is therefore not actionable,
which is why this uses axis-aligned detection rather than oriented boxes.

## Commands

Local venv at `.venv` (Python 3.12, ultralytics 8.4.x, torch+cu130, CUDA available).

```bash
source .venv/bin/activate

python main.py             # live detection + bin decisions + arm control
python calibrate_camera.py # one-time: solve the pixel -> millimetre mapping
python test_kinematics.py  # self-check the arm math, no hardware needed
python download_dataset.py # pull the labeled dataset from Roboflow
python train.py            # fine-tune yolo11s.pt -> runs/detect/trash-bins/weights/best.pt
```

`main.py` keys: SPACE pick the highlighted target, A toggle auto, H home, C clear the
picked list, Q quit.

`test_kinematics.py` is the only test. It is a plain script, not pytest. Run it after any
edit to `arm_config.py` — it round-trips the IK through the FK and verifies the joint
limits and servo ranges still agree. No linters or build steps are configured.

`download_dataset.py` needs `ROBOFLOW_API_KEY` in `.env`, and its `WORKSPACE`/`PROJECT` constants
must be pointed at the current Roboflow project.

`docs/vision-plan.md` holds the full rebuild plan and the evidence behind the decisions below —
read it before questioning why the OBB pipeline was abandoned.

## Architecture

Two pipelines. Training passes state through the filesystem:

`download_dataset.py` → `datasets/trash-bins/` → `train.py` → `runs/detect/trash-bins/weights/best.pt` → `main.py`

and at runtime a detection becomes a motion:

`main.py` → `workspace.py` (pixels → mm) → `kinematics.py` (mm → joint angles) → `pick_place.py` (sequence) → `arm_driver.py` (serial)

`docs/arm-setup.md` is the bring-up guide and holds the serial protocol the Arduino sketch
must implement. The sketch itself is not in this repo.

### Every physical constant lives in one place

`arm_config.py` holds every dimension, joint limit, servo trim, timing, and port.
`bins.py` holds where the bins physically are. `camera_calibration.json` holds the
pixel→millimetre mapping and is written by `calibrate_camera.py`. **Never write a
measurement inline anywhere else** — the whole arm layer was built so that finishing the
robot means editing one file, not hunting through five.

World frame for all of it: origin at the arm's base rotation axis at table level, +X
forward, +Y to the arm's left, +Z up, millimetres. Joint angles are geometric (shoulder
from vertical, others from the previous link); servo trim and direction are a separate
mapping, so a backwards servo is fixed by flipping a sign in `arm_config.SERVOS`, never by
negating something in `kinematics.py`.

### A servo's travel is a joint limit

`arm_config.EFFECTIVE_JOINT_LIMITS_DEG` intersects `JOINT_LIMITS_DEG` with the geometric
range each servo can actually produce from where it is mounted, and that intersection is
what `kinematics.check_limits` enforces. Keep it that way. When the two are allowed to
disagree, the driver is left choosing between silently clamping — which puts the arm
somewhere nobody solved for — and failing mid-sequence with an object in the gripper.
`joint_to_servo_deg` raises rather than clamping for the same reason.

### Plan the whole motion before moving

`pick_place.plan_pick` solves and limit-checks every waypoint before the first servo is
commanded, and refuses the whole sequence if any step is unreachable. A sequence that
fails halfway leaves the arm holding trash over the workspace with no plan; refusing up
front leaves it at home, which is always recoverable.

### The arm blocks the vision loop, deliberately

A pick takes seconds, during which the arm occludes the mat and the camera keeps
buffering. After any motion `main.py` calls `workspace.flush()` and `TrackVoter.reset()`,
so what resumes is the world as it is now rather than the scene from before the grasp.
Keep both calls on every path that moves the arm.

### Perception predicts objects; policy assigns bins

This split is the central design decision. The model predicts **visually grounded item types**
(`plastic_bottle`, `banana_peel`, `chip_bag`), and `bins.py` maps those to bins in a plain dict.

Bin rules are *regional policy* and depend on properties the pixels don't carry — whether a plastic
is rigid or film, whether paper is food-soiled. Training the network to output bins directly would
bake local rules into the weights, so a rule change would require recollecting and retraining. Keep
new classes concrete and separable from a single image; put every judgment call in `bins.py`.

Unmapped classes fall through to `UNKNOWN` by design. **Never let an unmapped or unstable detection
drive the arm.**

### Prototype mode

`main.py` has a `PROTOTYPE_MODE` flag. When `True` it runs stock COCO `yolo11s.pt` filtered to
`bins.COCO_CLASS_IDS` (bottle, cup, banana, apple, orange, pizza…), so the whole pipeline is
exercisable before any custom data exists. Set it to `False` to use `CUSTOM_WEIGHTS`.
`bins.bin_for()` takes a matching `prototype` argument selecting which table to read.

### Temporal stability is a safety property

`TrackVoter` in `main.py` accumulates per-track-ID class votes over a sliding window and only
reports a target once `VOTE_MIN_AGREE` frames agree. A single frame is never enough to move the arm.
`prune()` must keep being called each frame or the vote dicts grow without bound in long runs.

Detections with `boxes.id is None` (before a track is established) are drawn but never become
targets — keep that guard when editing the loop.

### Thresholds are set explicitly

`main.py` passes `conf=0.5`, `iou=0.5`, `max_det=20`. The Ultralytics default `conf=0.25` is far too
permissive here and readmits background false positives. Don't drop these back to defaults.

## Dataset guidance

- Export from Roboflow as **YOLOv11 detect** format with **augmentation disabled** — resize only.
  Ultralytics augments at train time; a second baked-in layer degraded the previous dataset.
- **Split by capture session, not randomly.** Random splits of near-duplicate frames leak between
  train and val and produce inflated metrics that collapse in the real world.
- Include ~10% hard negatives (empty mat, a hand, tools). Images with no boxes are valid data and
  directly suppress false positives.
- All training data must come from the final camera geometry. Re-mounting the camera invalidates it.

## Historical context

An earlier iteration trained a **YOLO-OBB** model on the Roboflow `Trash-Detection-13` dataset
(classes Glass/Metal/Paper/Plastic/Waste). It reached only mAP50 0.34 and was abandoned because that
dataset is *outdoor litter photography* — 53% of its objects are under 1% of image area — which does
not transfer to a close-range fixed camera. It predicted `Plastic` on background 56% of the time,
and its catch-all `Waste` class was missed 77% of the time. It also contains no compost/organics data
at all, so it cannot express the three-bin taxonomy.

`Trash-Detection-13/`, `runs/obb/`, and `yolo11s-obb.pt` remain on disk but are unused. Don't build
on them.

## Repo contents vs. working tree

`.gitignore` excludes `*.pt`, `datasets/`, `Trash-Detection-13/`, `runs/`, `.venv/`, and `.env` — the
committed repo is only the scripts. `camera_calibration.json` is committed on purpose: it is
small, it is specific to this camera mount, and losing it means redoing the calibration. Weights, datasets, and training output exist locally only and
cannot be recovered from git. Assume a fresh clone has none of them.
