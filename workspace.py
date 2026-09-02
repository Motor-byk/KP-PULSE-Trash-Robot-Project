"""Pixel coordinates -> world millimetres on the table.

Owns the camera_calibration.json format, so calibrate_camera.py (which writes it)
and main.py (which reads it) cannot drift apart.

Why a homography rather than a full camera calibration
-------------------------------------------------------
Every object we grasp sits on one flat surface. A 3x3 homography is an exact
description of the mapping between the image plane and another plane, which makes
it precisely the right tool for that case -- and it needs nothing but four points
whose real position you measured with a ruler. No checkerboard, no lens model, no
camera pose. Camera tilt is absorbed into the matrix, so an angled mount and a
top-down mount run identical code.

What it does NOT do is correct lens distortion. If calibrate_camera.py's live
check shows the error growing toward the edges of the frame, that is a barrel
distortion signature and the fix is a real cv2.calibrateCamera pass, not more
homography points.
"""

import json
import os

import cv2
import numpy as np

import arm_config as cfg


class CalibrationError(RuntimeError):
    """Raised when the calibration is missing, malformed, or does not apply."""


def save_calibration(path, homography, resolution, pixel_points, world_points):
    """Write the matrix plus everything needed to audit or re-solve it later."""
    payload = {
        "resolution": list(resolution),
        "homography": np.asarray(homography, dtype=float).tolist(),
        "pixel_points": [[float(a), float(b)] for a, b in pixel_points],
        "world_points_mm": [[float(a), float(b)] for a, b in world_points],
        "note": (
            "Pixel->world homography for the table plane. World origin is the "
            "arm's base rotation axis at table level, +X forward, +Y left. "
            "Only valid at the resolution recorded here, and only until the "
            "camera is moved."
        ),
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


class Workspace:
    """Maps image pixels to table millimetres, and says what is in bounds."""

    def __init__(self, homography, resolution):
        self.homography = np.asarray(homography, dtype=float)
        self.resolution = tuple(resolution)

    @classmethod
    def load(cls, path=None, expected_resolution=None):
        """Load a calibration, refusing one that does not match the live camera.

        The resolution check is not pedantry. The homography's inputs are raw
        pixels, so running it against a frame captured at a different size scales
        every world coordinate the arm receives -- quietly, with no error, and
        with the arm reaching to the wrong place.
        """
        path = path or cfg.CALIBRATION_FILE
        if not os.path.exists(path):
            raise CalibrationError(
                f"No calibration at {path!r}. Run: python calibrate_camera.py"
            )

        with open(path) as handle:
            payload = json.load(handle)

        try:
            homography = payload["homography"]
            resolution = tuple(payload["resolution"])
        except (KeyError, TypeError) as exc:
            raise CalibrationError(f"{path!r} is malformed: {exc}") from exc

        if expected_resolution is not None and resolution != tuple(expected_resolution):
            raise CalibrationError(
                f"Calibration was recorded at {resolution[0]}x{resolution[1]} but "
                f"the camera is running at {expected_resolution[0]}x{expected_resolution[1]}. "
                "The mapping is in pixel units and does not survive a resolution "
                "change -- re-run calibrate_camera.py."
            )

        return cls(homography, resolution)

    def pixel_to_world(self, cx, cy):
        """(px, py) -> (x_mm, y_mm) on the table.

        The third component of the result is the perspective divisor; dividing by
        it is what makes this a projective rather than an affine mapping, and it
        is the part that handles camera tilt.
        """
        point = np.array([float(cx), float(cy), 1.0])
        mapped = self.homography @ point
        if abs(mapped[2]) < 1e-9:
            raise CalibrationError(
                "Degenerate mapping for this pixel -- the calibration points were "
                "probably collinear. Re-run calibrate_camera.py with four points "
                "spread across the mat."
            )
        return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])

    def correct_for_height(self, x, y, object_height_mm=None):
        """Undo the parallax of grasping a solid object, not a flat sticker.

        The bounding box centroid is the object's *visual* middle, roughly half
        its height up. For a tilted camera, the ray through that point meets the
        table further from the camera than the object actually sits, so the arm
        would reach past it. Walking back along that same ray:

            true = nadir + (observed - nadir) * (1 - h / H)

        where nadir is the point on the table directly under the lens and H is
        the lens height. For a genuinely top-down camera the object is at its own
        nadir and this is a no-op, which is why it stays disabled (CAMERA_HEIGHT_MM
        = None) until you have measured the mount.
        """
        if cfg.CAMERA_HEIGHT_MM is None:
            return x, y

        height = cfg.DEFAULT_OBJECT_HEIGHT_MM if object_height_mm is None else object_height_mm
        if height <= 0 or height >= cfg.CAMERA_HEIGHT_MM:
            return x, y

        nadir_x, nadir_y = cfg.CAMERA_GROUND_XY_MM
        shrink = 1.0 - (height / cfg.CAMERA_HEIGHT_MM)
        return (
            nadir_x + (x - nadir_x) * shrink,
            nadir_y + (y - nadir_y) * shrink,
        )

    @staticmethod
    def in_bounds(x, y):
        """Is this world point somewhere we are willing to send the arm?

        A first filter before the IK: cheap, and it rejects things the arm could
        physically reach but should not, like an object that has rolled off the
        mat.
        """
        return (
            cfg.WORKSPACE_X_MM[0] <= x <= cfg.WORKSPACE_X_MM[1]
            and cfg.WORKSPACE_Y_MM[0] <= y <= cfg.WORKSPACE_Y_MM[1]
        )


def open_camera(index=None, resolution=None):
    """Open the webcam at the configured resolution, verifying it took effect.

    Lives here, next to the calibration, because the resolution requirement comes
    from the calibration: cap.set() is a *request*, and a webcam that does not
    support the mode silently gives you the nearest one it does. Unchecked, that
    is how a valid-looking calibration ends up applied to differently-sized frames.

    Also sets a buffer depth of 1. OpenCV otherwise queues frames, and after the
    arm has spent three seconds blocking the loop, the next read() returns the
    stale frame from before the arm moved.
    """
    index = cfg.CAMERA_INDEX if index is None else index
    resolution = cfg.CAMERA_RESOLUTION if resolution is None else resolution

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise CalibrationError(f"Could not open camera {index}.")

    width, height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if actual != tuple(resolution):
        cap.release()
        raise CalibrationError(
            f"Camera {index} gave {actual[0]}x{actual[1]}, not the configured "
            f"{width}x{height}. Set CAMERA_RESOLUTION in arm_config.py to a mode "
            "this camera actually supports, then re-run the calibration."
        )
    return cap, actual


def flush(cap, frames=5):
    """Throw away buffered frames, so what we read next is the world as it is now.

    Called after the arm has been moving: during a pick the camera kept running
    and the driver kept a queue, so without this the first frame after a grasp
    shows the scene before it, complete with the object that is no longer there.
    """
    for _ in range(frames):
        cap.grab()


def solve_homography(pixel_points, world_points):
    """Fit the mapping from four or more measured correspondences.

    findHomography rather than getPerspectiveTransform so that more than four
    points can be supplied -- extra points are least-squares averaged, which is
    the cheapest way to dilute the error in any one ruler measurement.
    """
    src = np.asarray(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(world_points, dtype=np.float32).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src, dst)
    if homography is None:
        raise CalibrationError(
            "Could not fit a homography. The four points are probably collinear "
            "or nearly so -- spread them out over the mat, ideally near its corners."
        )
    return homography
