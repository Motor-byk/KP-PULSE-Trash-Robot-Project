"""Live vision and pick-and-place for the trash sorting arm.

Detects objects, maps them to a disposal bin, converts the centroid to world
millimetres, and drives the arm to pick the item up and drop it in the right bin.

Runs in two detector modes:
  PROTOTYPE_MODE = True   stock COCO yolo11s.pt, works before any data exists
  PROTOTYPE_MODE = False  the custom detector trained by train.py

and degrades gracefully: with no camera calibration, or with the arm disabled, it
is exactly the vision-only program it used to be.

Keys:
  SPACE  pick the highlighted target
  A      toggle automatic picking (off by default)
  H      send the arm home
  C      forget which items have already been picked
  Q      quit
"""

import contextlib
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from collections import Counter, deque
from ultralytics import YOLO

import arm_config as cfg
import arm_driver
import bins
import kinematics
import pick_place
import workspace


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

# Arm. SIMULATE_ARM stays True until the hardware exists and has been checked --
# real motion is something you opt into, never the default you forget to change.
ARM_ENABLED = True     # False runs the original vision-only program
SIMULATE_ARM = True    # False talks to the Arduino on cfg.SERIAL_PORT

AUTO_MODE_DEFAULT = False
# In auto mode a target must have been actionable for this many consecutive
# frames before the arm commits. Class voting proves *what* the object is; this
# proves it has been sitting still long enough that the hand which put it there
# is out of the way.
AUTO_SETTLE_FRAMES = 20

# Set to True if you want to save decision and frame data for the current
# session
IS_RECORDING_DATA = True
RECORDING_INTERVAL = 0.5

# Label IS_RECORDING_DATA = True sessions using the start time as a naming 
# convention
CUR_DATETIME = datetime.now()
RECORD_DIR = "captures/" + CUR_DATETIME.strftime("%Y%m%d-%H%M%S")

class TrackVoter:
    """Per-track class voting, so momentary misclassifications get filtered out."""

    def __init__(self, window=VOTE_WINDOW, min_agree=VOTE_MIN_AGREE, ttl=TRACK_TTL):
        self.window = window
        self.min_agree = min_agree
        self.ttl = ttl
        self.votes = {}         # track_id -> deque of class names
        self.last_seen = {}     # track_id -> frame index
        self.stable_since = {}  # track_id -> frame it first became stable

    def update(self, track_id, class_name, frame_idx):
        """Record one observation and return (stable_class, is_stable)."""
        if track_id not in self.votes:
            self.votes[track_id] = deque(maxlen=self.window)

        self.votes[track_id].append(class_name)
        self.last_seen[track_id] = frame_idx

        winner, count = Counter(self.votes[track_id]).most_common(1)[0]
        is_stable = count >= self.min_agree

        if is_stable:
            self.stable_since.setdefault(track_id, frame_idx)
        else:
            self.stable_since.pop(track_id, None)

        return winner, is_stable

    def stable_frames(self, track_id, frame_idx):
        """How long this track has been continuously stable. 0 if it is not."""
        since = self.stable_since.get(track_id)
        return 0 if since is None else frame_idx - since

    def prune(self, frame_idx):
        """Drop tracks that have not been seen recently, so the dicts stay bounded."""
        stale = [
            tid for tid, seen in self.last_seen.items()
            if frame_idx - seen > self.ttl
        ]
        for tid in stale:
            self.votes.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.stable_since.pop(tid, None)

    def reset(self):
        """Forget everything.

        Called after the arm has moved. During a pick the arm occludes the mat,
        so any votes accumulated across that gap describe a scene that was partly
        a robot. Track IDs only ever increase, so cleared entries cannot be
        confused with the tracks that come back afterwards.
        """
        self.votes.clear()
        self.last_seen.clear()
        self.stable_since.clear()


class FrameRecorder:
    """Save raw frames and their detections as a YOLO dataset, at a fixed interval.

    The output uploads to Roboflow as pre-annotated images: every box the model
    found is already drawn, so labeling means correcting mistakes rather than
    drawing from scratch. One folder per run, so datasets can later be split by
    capture session instead of randomly.
    """

    def __init__(self, session_dir, interval_s, names):
        self.interval_s = interval_s
        self.last_save = None   # monotonic time of the last save; None = never saved
        self.count = 0          # number of the last file written

        self.session = Path(session_dir)
        self.images = self.session / "images"
        self.labels = self.session / "labels"
        # No exist_ok: a clash means two runs in the same second, and failing
        # beats one run silently overwriting the other's files.
        self.images.mkdir(parents=True)
        self.labels.mkdir()

        # Class id -> name table, so the ids in the label files mean something
        with open(self.session / "data.yaml", "w") as f:
            yaml.safe_dump({"names": dict(names)}, f)

    def maybe_save(self, frame, results):
        """Save this frame if the interval has passed.

        Must be called before anything is drawn on the frame -- training images
        with boxes painted on them teach the model to look for rectangles.
        """
        # monotonic, not time.time(): a wall-clock adjustment can't make it
        # skip saves or fire a burst of them
        now = time.monotonic()
        if self.last_save is not None and now - self.last_save < self.interval_s:
            return
        self.last_save = now
        self.count += 1

        stem = f"{self.count:06d}"
        # Image before label: a crash in between leaves an unlabeled image,
        # which is harmless, never a label with no image
        cv2.imwrite(str(self.images / f"{stem}.jpg"), frame)

        # Every box, tracked or not -- labels don't need track IDs.
        # xywhn is already YOLO format: center x, center y, width, height, 0-1.
        lines = []
        for result in results:
            if result.boxes is None:
                continue
            for cls, (cx, cy, w, h) in zip(result.boxes.cls.int().tolist(),
                                          result.boxes.xywhn.tolist()):
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Written even when empty: an empty label file is a hard negative
        with open(self.labels / f"{stem}.txt", "w") as f:
            f.writelines(lines)


def draw_detection(frame, xyxy, label, color, center, selected=False):
    """Draw one box, its centroid, and a label with a filled background."""
    x1, y1, x2, y2 = xyxy.astype(int)
    cx, cy = center

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if selected else 2)
    cv2.circle(frame, (cx, cy), 5, color, -1)

    # The one target SPACE would act on gets a second outline, so there is never
    # ambiguity about what is about to be picked.
    if selected:
        cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (255, 255, 255), 1)

    # Label sits above the box, on a filled strip so it stays readable
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y1, th + 6)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
    cv2.putText(
        frame, label, (x1 + 3, ty - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )


def annotate_position(frame, center, color, world, reachable):
    """Print the centroid's position by the dot -- pixels, and mm if calibrated."""
    cx, cy = center
    if world is None:
        text = f"({cx}, {cy})px"
    else:
        mark = "" if reachable else "  NO REACH"
        text = f"({world[0]:.0f}, {world[1]:.0f})mm{mark}"
    cv2.putText(
        frame, text, (cx + 8, cy + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
    )


def locate(space, center):
    """Pixel centroid -> ((x_mm, y_mm), reachable), or (None, False).

    None means the arm layer should ignore this detection entirely: either there
    is no calibration, or the object is outside the mat.
    """
    if space is None:
        return None, False

    x, y = space.pixel_to_world(*center)
    x, y = space.correct_for_height(x, y)
    if not space.in_bounds(x, y):
        return None, False

    return (x, y), kinematics.reachable(x, y, cfg.GRASP_Z_MM)


def select_target(targets, already_picked):
    """Choose the one target the arm would act on, or None.

    Nearest to the base among the reachable ones. Nearest rather than
    highest-confidence because a short reach is the fastest and most accurate
    move the arm has, and every pick clears the way for the ones behind it.
    """
    candidates = [
        t for t in targets
        if t["reachable"] and t["track_id"] not in already_picked
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda t: t["world"][0] ** 2 + t["world"][1] ** 2)


def draw_hud(frame, lines):
    """Status text down the top-left corner."""
    for index, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text, (10, 28 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
        )


def setup_arm():
    """Build the arm layer, or explain why it is unavailable.

    Returns (workspace_or_None, driver_or_None, message). Every failure here is
    non-fatal: vision still runs, the arm simply stays out of it. That keeps this
    program useful during the whole period where the camera is mounted but the
    robot is not.
    """
    if not ARM_ENABLED:
        return None, None, "arm disabled in config"

    try:
        space = workspace.Workspace.load(expected_resolution=cfg.CAMERA_RESOLUTION)
    except workspace.CalibrationError as exc:
        return None, None, f"no calibration ({exc})"

    problems = pick_place.check_bin_positions()
    if problems:
        print("Arm disabled -- configured bin positions are not reachable:")
        for problem in problems:
            print(f"  - {problem}")
        print("Fix BIN_DROP_XYZ in bins.py or the link lengths in arm_config.py.")
        return None, None, "bin positions unreachable"

    return space, arm_driver.make_driver(SIMULATE_ARM), None


def main():
    # Model instantiated
    model = YOLO(MODEL_PATH)
    voter = TrackVoter()

    try:
        cap, resolution = workspace.open_camera()
    except workspace.CalibrationError as exc:
        print(f"Error: {exc}")
        return

    space, driver, arm_note = setup_arm()

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

    recorder = None
    if IS_RECORDING_DATA:
        recorder = FrameRecorder(RECORD_DIR, RECORDING_INTERVAL, model.names)

    print(f"Model: {MODEL_PATH}  (prototype={PROTOTYPE_MODE})")
    print(f"Camera: {resolution[0]}x{resolution[1]}")
    if driver is None:
        print(f"Arm: OFF -- {arm_note}")
    else:
        print(f"Arm: {'SIMULATED' if SIMULATE_ARM else 'LIVE HARDWARE'}")
    if recorder is None:
        print("Recording: off")
    else:
        print(f"Recording: {RECORD_DIR} every {RECORDING_INTERVAL}s")
    print("Keys: SPACE pick | A auto | H home | C clear picked | Q quit\n")

    frame_idx = 0
    auto_mode = AUTO_MODE_DEFAULT
    already_picked = set()
    status = ""

    # ExitStack so the driver's context manager still runs -- homing the arm and
    # closing the port -- when there is no driver at all, and on any exception.
    with contextlib.ExitStack() as stack:
        picker = None
        if driver is not None:
            try:
                stack.enter_context(driver)
                picker = pick_place.PickAndPlace(driver)
            except arm_driver.ArmError as exc:
                print(f"Arm unavailable: {exc}")
                driver = None
                arm_note = str(exc)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame_idx += 1
            results = model.track(frame, **track_kwargs)

            # Before any drawing: from here on the frame gets painted in place
            if recorder is not None:
                recorder.maybe_save(frame, results)

            # Grasp targets for this frame, in the order they were detected
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
                    world, can_reach = (None, False)
                    if actionable:
                        world, can_reach = locate(space, center)
                        targets.append({
                            "track_id": track_id,
                            "class": stable_class,
                            "bin": bin_name,
                            "center": center,
                            "xyxy": xyxy,
                            "conf": float(conf),
                            "world": world,
                            "reachable": can_reach,
                        })

                    color = bins.BIN_COLORS[bin_name if actionable else bins.UNKNOWN]
                    mark = "" if actionable else " ?"
                    label = f"#{track_id} {stable_class} -> {bin_name}{mark} {conf:.2f}"
                    draw_detection(frame, xyxy, label, color, center)
                    annotate_position(frame, center, color, world, can_reach)

            voter.prune(frame_idx)

            selected = select_target(targets, already_picked)
            if selected is not None:
                draw_detection(
                    frame, selected["xyxy"],
                    f"PICK #{selected['track_id']} -> {selected['bin']}",
                    bins.BIN_COLORS[selected["bin"]], selected["center"],
                    selected=True,
                )

            # ------------------------------------------------------------------
            # Overlay, then show. imshow only actually paints during waitKey, so
            # the draw has to happen before the key is read, not after.
            # ------------------------------------------------------------------
            if picker is None:
                arm_line = (f"ARM OFF: {arm_note}", (0, 165, 255))
            elif auto_mode:
                arm_line = ("AUTO", (0, 255, 255))
            else:
                arm_line = ("MANUAL - SPACE to pick", (255, 255, 255))

            hud = [
                (f"targets: {len(targets)}   picked: {len(already_picked)}", (255, 255, 255)),
                arm_line,
            ]
            if recorder is not None:
                hud.append((f"REC  {recorder.count} saved", (0, 0, 255)))
            if status:
                hud.append((status, (200, 200, 200)))
            draw_hud(frame, hud)

            cv2.imshow("Trash Sorter", frame)

            # ------------------------------------------------------------------
            # Decide whether to move
            # ------------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            wants_pick = key == ord(" ")

            if auto_mode and selected is not None and picker is not None:
                settled = voter.stable_frames(selected["track_id"], frame_idx)
                wants_pick = wants_pick or settled >= AUTO_SETTLE_FRAMES

            if key == ord("q"):
                break
            if key == ord("a"):
                auto_mode = not auto_mode
                status = f"auto {'ON' if auto_mode else 'off'}"
            elif key == ord("c"):
                already_picked.clear()
                status = "cleared picked list"
            elif key == ord("h") and picker is not None:
                driver.home()
                workspace.flush(cap)
                voter.reset()
                status = "homed"
            elif wants_pick:
                if picker is None:
                    status = f"arm unavailable: {arm_note}"
                elif selected is None:
                    status = "nothing to pick"
                else:
                    try:
                        ok, reason = picker.execute(selected["world"], selected["bin"])
                    except arm_driver.ArmError as exc:
                        ok, reason = False, str(exc)
                    if ok:
                        already_picked.add(selected["track_id"])
                        status = f"picked #{selected['track_id']} -> {selected['bin']}"
                    else:
                        status = f"refused: {reason}"

                    # The arm occluded the mat for several seconds and the camera
                    # kept buffering. Without this the next frames show the scene
                    # as it was before the pick, complete with the item that is no
                    # longer there -- and in auto mode the arm would go for it again.
                    workspace.flush(cap)
                    voter.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
