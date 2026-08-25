# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

python main.py             # live webcam detection + bin decisions (press Q to quit)
python download_dataset.py # pull the labeled dataset from Roboflow
python train.py            # fine-tune yolo11s.pt -> runs/detect/trash-bins/weights/best.pt
```

No tests, linters, or build steps are configured.

`download_dataset.py` needs `ROBOFLOW_API_KEY` in `.env`, and its `WORKSPACE`/`PROJECT` constants
must be pointed at the current Roboflow project.

`docs/vision-plan.md` holds the full rebuild plan and the evidence behind the decisions below —
read it before questioning why the OBB pipeline was abandoned.

## Architecture

Three scripts plus a mapping module, run manually. State passes between them through the filesystem:

`download_dataset.py` → `datasets/trash-bins/` → `train.py` → `runs/detect/trash-bins/weights/best.pt` → `main.py`

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
committed repo is only the scripts. Weights, datasets, and training output exist locally only and
cannot be recovered from git. Assume a fresh clone has none of them.
