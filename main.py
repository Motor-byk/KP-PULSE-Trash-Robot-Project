"""Live vision stage for the trash sorting arm.

Detects objects, maps them to a disposal bin, and emits stable grasp targets
(centroid + bin) for the arm controller to consume.

Runs in two modes:
  PROTOTYPE_MODE = True   stock COCO yolo11s.pt, works before any data exists
  PROTOTYPE_MODE = False  the custom detector trained by train.py
"""

import cv2
from collections import Counter, deque
from ultralytics import YOLO

import bins


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

PROTOTYPE_MODE = True

# Must match `name=` in train.py once PROTOTYPE_MODE is False.
CUSTOM_WEIGHTS = "runs/detect/trash-bins/weights/best.pt"
MODEL_PATH = "yolo11s.pt" if PROTOTYPE_MODE else CUSTOM_WEIGHTS

# Detection thresholds. The Ultralytics default of conf=0.25 is far too
# permissive for a robot -- it admits weak background detections.
CONF = 0.5
IOU = 0.5
MAX_DET = 20

# Temporal stability. The arm must never act on a single frame, so a track's
# class must agree across several frames before it becomes a grasp target.
VOTE_WINDOW = 5        # frames of history kept per track
VOTE_MIN_AGREE = 3     # matching votes needed within that window
TRACK_TTL = 30         # frames a track survives unseen before being dropped

CAMERA_INDEX = 0


class TrackVoter:
    """Per-track class voting, so momentary misclassifications get filtered out."""

    def __init__(self, window=VOTE_WINDOW, min_agree=VOTE_MIN_AGREE, ttl=TRACK_TTL):
        self.window = window
        self.min_agree = min_agree
        self.ttl = ttl
        self.votes = {}       # track_id -> deque of class names
        self.last_seen = {}   # track_id -> frame index

    def update(self, track_id, class_name, frame_idx):
        """Record one observation and return (stable_class, is_stable)."""
        if track_id not in self.votes:
            self.votes[track_id] = deque(maxlen=self.window)

        self.votes[track_id].append(class_name)
        self.last_seen[track_id] = frame_idx

        winner, count = Counter(self.votes[track_id]).most_common(1)[0]
        return winner, count >= self.min_agree

    def prune(self, frame_idx):
        """Drop tracks that have not been seen recently, so the dicts stay bounded."""
        stale = [
            tid for tid, seen in self.last_seen.items()
            if frame_idx - seen > self.ttl
        ]
        for tid in stale:
            self.votes.pop(tid, None)
            self.last_seen.pop(tid, None)


def draw_detection(frame, xyxy, label, color, center):
    """Draw one box, its centroid, and a label with a filled background."""
    x1, y1, x2, y2 = xyxy.astype(int)
    cx, cy = center

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 5, color, -1)

    # Label sits above the box, on a filled strip so it stays readable
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y1, th + 6)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
    cv2.putText(
        frame, label, (x1 + 3, ty - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    cv2.putText(
        frame, f"({cx}, {cy})", (cx + 8, cy + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
    )


def main():
    model = YOLO(MODEL_PATH)
    voter = TrackVoter()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # In prototype mode, restrict to the COCO classes we have a bin rule for.
    track_kwargs = {
        "persist": True,
        "conf": CONF,
        "iou": IOU,
        "max_det": MAX_DET,
        "verbose": False,
    }
    if PROTOTYPE_MODE:
        track_kwargs["classes"] = bins.COCO_CLASS_IDS

    print(f"Model: {MODEL_PATH}  (prototype={PROTOTYPE_MODE})")
    print("Press Q to quit.\n")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_idx += 1
        results = model.track(frame, **track_kwargs)

        # Grasp targets for this frame, handed to the arm controller later
        targets = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy_list = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.int().cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            # Tracking IDs are absent on frames before a track is established
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().numpy()
            else:
                track_ids = [None] * len(xyxy_list)

            for xyxy, track_id, cls, conf in zip(xyxy_list, track_ids, class_ids, confs):
                x1, y1, x2, y2 = xyxy
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                class_name = model.names[int(cls)]

                # Untracked detections are drawn but never become grasp targets
                if track_id is None:
                    bin_name = bins.bin_for(class_name, PROTOTYPE_MODE)
                    draw_detection(
                        frame, xyxy,
                        f"{class_name} {conf:.2f} (no id)",
                        bins.BIN_COLORS[bins.UNKNOWN], center,
                    )
                    continue

                track_id = int(track_id)
                stable_class, is_stable = voter.update(track_id, class_name, frame_idx)
                bin_name = bins.bin_for(stable_class, PROTOTYPE_MODE)

                # Only a stable track with a known bin is safe to act on
                actionable = is_stable and bin_name != bins.UNKNOWN
                if actionable:
                    targets.append({
                        "track_id": track_id,
                        "class": stable_class,
                        "bin": bin_name,
                        "center": center,
                    })

                color = bins.BIN_COLORS[bin_name if actionable else bins.UNKNOWN]
                mark = "" if actionable else " ?"
                label = f"#{track_id} {stable_class} -> {bin_name}{mark} {conf:.2f}"
                draw_detection(frame, xyxy, label, color, center)

        voter.prune(frame_idx)

        cv2.putText(
            frame, f"targets: {len(targets)}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.imshow("Trash Sorter - Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
