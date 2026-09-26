"""Train the trash sorting detector.

Axis-aligned detection (not OBB) -- the 4DOF arm has no wrist roll/yaw, so a
grasp angle would not be actionable. Starts from COCO-pretrained yolo11s.pt,
which already knows bottle, cup, banana, apple, orange and bowl.
"""

from ultralytics import YOLO

# Explicit run name so the weights path is stable. Without it Ultralytics
# writes to train/, then train2/, train3/... on each rerun and main.py silently
# keeps loading the old model.
RUN_NAME = "trash-bins"
DATA = "./datasets/trash-bins/data.yaml"

model = YOLO("yolo11s.pt")

model.train(
    data=DATA,
    name=RUN_NAME,
    exist_ok=True,

    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    patience=30,
    cos_lr=True,

    # Objects land on the tray at arbitrary angles, so rotation matters.
    # The previous OBB run used degrees=0.0, which was a large missed signal.
    degrees=180,
    flipud=0.5,      # valid for a top-down camera; set 0.0 if angled
    fliplr=0.5,
    scale=0.5,
    translate=0.1,

    # Lighting robustness
    hsv_v=0.4,
    hsv_s=0.7,

    mosaic=1.0,
    close_mosaic=15,
)
