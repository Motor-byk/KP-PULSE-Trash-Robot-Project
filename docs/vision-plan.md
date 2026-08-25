# Rebuilding the Trash Sorter Vision Pipeline

> **Status (2026-08-25):** Steps 0, 4 and 5 are implemented — `bins.py`, `main.py`,
> `train.py` and `download_dataset.py` are written and the prototype path is verified.
> Steps 1-3 (freeze camera geometry, finalize class list, capture and label ~300-500
> images) are physical work and still outstanding.

## Context

The current model is inaccurate — it fires on background clutter, mislabels objects, and its
classes are too vague to act on. Investigation showed the root cause is not a tuning problem: the
model was trained on the wrong data for the wrong task. This plan replaces the vision stage with a
detector trained on the robot's own camera, outputting object types that map to the three bins
used locally (waste / recycle / compost).

## Why the current model fails

Final metrics (`runs/obb/train/results.csv`): **mAP50 0.339, mAP50-95 0.273, precision 0.52,
recall 0.38.** The model is genuinely unreliable, for four measurable reasons.

**1. Domain mismatch.** `val_batch0_pred.jpg` shows the dataset is *outdoor litter photography* —
trash on grass, gravel, sand, asphalt, shot from standing height at a distance. **53% of training
objects are <1% of image area; 27% are <0.1%** (~6px). It is tuned to find specks of litter in
outdoor texture, not to identify one close-up object on a tray.

**2. It learned "outdoor texture → Plastic."** From `confusion_matrix_normalized.png`, background is
predicted as Plastic **0.56** of the time (Paper 0.17, Waste 0.14, Metal 0.09, Glass 0.03). In the
val images it puts `Plastic 0.5` on bare dirt, `Plastic 0.6` on dried fronds, `Plastic 0.9` on
grass. Plastic is also a classification sink: true Metal → Plastic 0.20, true Paper → Plastic 0.30.
That is exactly the "picks up everything / mislabels" symptom.

**3. `Waste` is non-functional.** Per-class diagonal: Glass 0.40, Metal 0.31, Paper 0.30,
Plastic 0.58, **Waste 0.10** — missed 77% of the time. A catch-all class overlapping all others
cannot be learned. That is the "too vague" symptom.

**4. The dataset cannot express the target taxonomy.** It has **zero** compost/organics data.
Recycling-vs-landfill for plastic depends on rigidity; recycling-vs-compost for paper depends on
food contamination — neither is annotated. Also, 6147 train images are only **1269 unique sources**
(3 Roboflow copies each), with baked-in augmentation redundant with Ultralytics' own.

*Ruled out:* the variable-length polygon labels are handled correctly — Ultralytics converts them
via `cv2.minAreaRect` (`ultralytics/utils/ops.py:385`). Not a bug.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Box type | **Axis-aligned detection**, drop OBB | 4DOF arm has no wrist roll/yaw, so a grasp angle is not actionable. Also ~3x faster to label. |
| Output | **Object classes + bin lookup table** | Bin rules are regional policy; keeping them in a dict means rule changes need no retraining, and failures are debuggable. |
| Vocabulary | ~14 concrete item types | Enough to cover all 3 bins with items you physically own. |
| Starting weights | `yolo11s.pt` (COCO), already in repo | Already knows bottle, cup, banana, apple, orange, bowl, pizza — directly relevant. |
| `Trash-Detection-13` | Set aside | Its "texture → plastic" bias would carry in. |

## Step 0 — Zero-shot prototype (do this first, ~1 hour)

Before labeling anything, wire the full pipeline using COCO classes that `yolo11s.pt` already
detects. This validates the plumbing and unblocks arm work immediately.

Create `bins.py` with the mapping table, and a `main.py` variant using `YOLO("yolo11s.pt")` filtered
to `classes=[39,40,41,45,46,47,49,50,51,53,54,55]` (bottle, wine glass, cup, bowl, banana, apple,
orange, broccoli, carrot, pizza, donut, cake). You get real bin decisions and centroid coordinates
on day one, with a known-good detector — so any later problem is clearly in *your* model, not the
plumbing.

## Step 1 — Fix the camera rig (blocking)

**Nothing should be labeled until this is frozen.** Training data captured at a geometry you later
change is wasted. Pin down and write into `README`:

- Camera height, angle, and distance to the work surface — mark the mount position physically.
- **A plain, consistent background mat** (solid matte color, non-reflective) under the objects.
  This is the single cheapest accuracy win available: it nearly eliminates background false
  positives, and it is what production sorting rigs do.
- Consistent lighting — add a fixed lamp; avoid varying window light.
- Fixed capture resolution. Record it; `imgsz` should match its shorter side.

## Step 2 — Define the class list

Target ~14 visually-grounded classes. **The key fix over the old dataset: split plastics by form,
not material** — a rigid bottle and a film wrapper look completely different and are separately
learnable, whereas one merged "Plastic" class is what created the sink.

| Class | Bin |
|---|---|
| `plastic_bottle` | recycle |
| `metal_can` | recycle |
| `glass_bottle` | recycle |
| `cardboard` | recycle |
| `paper_clean` | recycle |
| `beverage_carton` | recycle |
| `food_scrap` | compost |
| `banana_peel` | compost |
| `apple_core` | compost |
| `soiled_paper` | compost |
| `plastic_wrapper` | waste |
| `styrofoam` | waste |
| `plastic_utensil` | waste |
| `ceramic` | waste |

Adjust to what you actually own. Two rules when editing: every class must be **separable from a
single image** (drop anything needing knowledge the pixels don't carry), and keep `soiled_paper`
examples *obviously* soiled — visible grease and residue — or it will collide with `paper_clean`.

## Step 3 — Capture and label (~300-500 images)

- **Multi-object scenes**, 3-5 items per image. Matches deployment and yields ~1500-2000 instances,
  roughly 120+ per class. Aim for **≥100 instances per class**; top up rare ones deliberately.
- Vary position across the whole workspace, rotation, lighting, partial occlusion, and overlap.
- Include **~10% hard negatives**: empty mat, a hand in frame, tools, a phone. Images with no boxes
  are valid training data and directly suppress the false-positive behavior seen in the old model.
- Label upright boxes in Roboflow (your `ROBOFLOW_API_KEY` is already set up), export **YOLOv11
  detect** format.
- **Turn Roboflow augmentation completely off** — resize only. Ultralytics augments at train time;
  doubling up is what degraded the previous dataset.
- **Split by capture session, not randomly.** Random splitting of near-duplicate frames leaks
  between train and val and produces inflated metrics that collapse in the real world. Hold out
  entire sessions for val/test.

## Step 4 — Training (`train.py`)

```python
from ultralytics import YOLO

model = YOLO('yolo11s.pt')          # COCO-pretrained detect, not -obb
model.train(
    data='./datasets/trash-bins/data.yaml',
    epochs=150, imgsz=640, batch=16, device=0,
    patience=30, cos_lr=True,
    degrees=180, flipud=0.5, fliplr=0.5,   # top-down view: any orientation is valid
    scale=0.5, translate=0.1,
    hsv_v=0.4, hsv_s=0.7,                  # lighting robustness
    mosaic=1.0, close_mosaic=15,
)
```

`degrees=180` is deliberate and is the direct fix for the old run's `degrees: 0.0` — objects land on
the tray at arbitrary angles. Set `flipud=0.0` if the camera ends up angled rather than top-down.
Move `train.py`'s hardcoded output path handling too: Ultralytics writes to `runs/detect/train2/`
on a second run, so pass an explicit `name=` and have `main.py` read the same constant.

## Step 5 — Rewrite `main.py`

The existing structure is sound and mostly survives; the changes are:

1. **`result.obb` → `result.boxes`**, `xyxyxyxy` polygons → `xyxy` rectangles, `cv2.polylines` →
   `cv2.rectangle`. Centroid becomes the box midpoint. Keep the existing `id is None` guard.
2. **Set thresholds explicitly**: `model.track(frame, persist=True, conf=0.5, iou=0.5)`. The default
   `conf=0.25` is what admits the weak background detections.
3. **Bin mapping** via the `bins.py` dict from Step 0; color the box by bin.
4. **Temporal stability voting** — the most important addition for a robot. Do not act on a single
   frame. Accumulate per-`track_id` class votes across N frames (~5) and only emit a grasp target
   once a track's class is stable. Track IDs are already available.
5. **Explicit `unknown` path** — if no class clears the threshold, emit `unknown` rather than
   guessing. Route to landfill or hold for review; never let a low-confidence guess drive the arm.

## Verification

1. **Held-out session metrics**: `model.val()` on sessions never seen in training. For a fixed
   camera and controlled scene, **mAP50 > 0.85** is a realistic bar — treat anything near the old
   0.34 as a sign something is still wrong.
2. **Bin-level confusion matrix**: aggregate the 14 object classes into the 3 bins and score that.
   This is the metric that actually matters — a `banana_peel`/`apple_core` mix-up is free, while a
   `plastic_wrapper`→`plastic_bottle` error puts film plastic in recycling.
3. **False-positive check**: run live on an **empty mat** for 60s. Expect zero detections. This is
   the specific failure the old model exhibited most.
4. **Live test**: place each of the 14 item types in the workspace, confirm correct bin and a stable
   track ID, and confirm the centroid lands on the object.
5. **Occlusion/overlap**: two overlapping items, confirm both are detected separately.

## Files

- `bins.py` — *new*: object-class → bin mapping table.
- `main.py` — rewrite per Step 5 (detect boxes, thresholds, bin mapping, vote stabilization).
- `train.py` — rewrite per Step 4 (`yolo11s.pt`, detect task, rotation augmentation).
- `download_dataset.py` — repoint to the new Roboflow project. Note the existing bug:
  `load_dotenv` is imported but never called, so `ROBOFLOW_API_KEY` is only read if already
  exported in the shell. Add the `load_dotenv()` call.
- `CLAUDE.md` — update; its OBB-centric guidance becomes wrong once the task switches to detect.
- `Trash-Detection-13/`, `runs/obb/`, `yolo11s-obb.pt` — leave in place, unused.
