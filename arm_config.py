"""Physical constants for the 4DOF arm and its camera.

THIS IS THE ONLY FILE YOU EDIT WHEN THE ROBOT IS BUILT. Everything downstream
(kinematics, driver, pick sequence) reads its numbers from here, so no dimension,
servo trim value, or port name should ever be written inline anywhere else.

Every value below is a placeholder. Each one carries a comment saying how to
measure it. See docs/arm-setup.md for the order to do it in.

Conventions used everywhere in this project
-------------------------------------------
Units: millimetres, and degrees in this file only. The math in kinematics.py
works in radians; the conversion happens when these are read.

World frame: origin on the table surface, on the arm's base rotation axis.
    +X  forward, the direction the arm points at base angle 0
    +Y  to the arm's left
    +Z  up

Joint angles are GEOMETRIC, not servo positions:
    base      rotation about +Z. 0 = pointing along +X, positive = toward +Y.
    shoulder  measured from straight up. 0 = upper arm vertical, positive =
              leaning forward (away from the base).
    elbow     measured from the previous link. 0 = forearm continues straight.
    wrist     measured from the previous link. 0 = gripper continues straight.

Servo trim, direction, and range are a separate mapping (SERVOS below), so a
servo mounted backwards is fixed by flipping a sign here -- never by putting a
minus sign in the kinematics.
"""

import math


# --------------------------------------------------------------------------
# Link lengths
#
# Measure between rotation axes, not between the outsides of the brackets.
# With the arm laid out straight, a caliper or steel rule on the pivot bolts.
# --------------------------------------------------------------------------

BASE_HEIGHT_MM = 60.0        # table surface up to the shoulder pivot axis
BASE_X_OFFSET_MM = 0.0       # horizontal offset from the base rotation axis to
                             # the shoulder pivot. Usually 0; nonzero if the
                             # shoulder bracket is mounted off-centre.

L_SHOULDER_ELBOW_MM = 105.0  # upper arm: shoulder axis -> elbow axis
L_ELBOW_WRIST_MM = 95.0      # forearm:   elbow axis -> wrist axis
L_WRIST_TIP_MM = 75.0        # wrist axis -> the point between the closed
                             # fingertips. Close the gripper and measure to
                             # where it actually pinches, not to the finger tips.


# --------------------------------------------------------------------------
# Joint limits, in degrees, in the geometric convention above.
#
# These are the primary guard against the arm driving itself into the table or
# folding back onto its own base. Start conservative and open them up only after
# you have watched the arm move through the range by hand.
# --------------------------------------------------------------------------

JOINT_LIMITS_DEG = {
    "base": (-90.0, 90.0),        # how far the base can slew each way
    "shoulder": (-20.0, 135.0),   # -20 = leaning slightly back, 135 = past horizontal
    "elbow": (0.0, 150.0),        # this arm is assumed to bend one way only; if
                                  # yours folds the other way, use (-150, 0)
    "wrist": (-20.0, 150.0),
}

JOINT_NAMES = ("base", "shoulder", "elbow", "wrist")


# --------------------------------------------------------------------------
# Servo mapping: geometric angle -> servo command in degrees.
#
#   servo_deg = clamp(zero_deg + sign * geometric_deg, min_deg, max_deg)
#
# `channel` is the index in the serial command, and must match the order the
# Arduino sketch attaches its Servo objects.
#
# To find `zero_deg`: command the servo to its midpoint, then physically set the
# joint so it matches the geometric zero described at the top of this file, and
# read off what number produced that. To find `sign`: increase the servo command
# a little; if the joint moves the way the convention calls negative, use -1.
#
# min_deg/max_deg is the servo's real mechanical travel. A servo cannot go past
# it, so it is a joint limit just as much as JOINT_LIMITS_DEG is -- see
# EFFECTIVE_JOINT_LIMITS_RAD at the bottom of this file, which intersects the two.
#
# Note how `zero_deg` differs per joint below. A servo has only 180 degrees to
# give, so it has to be mounted with that range positioned over the angles the
# joint actually needs. An elbow that only ever bends forwards wants its whole
# travel on that side, not centred with half of it wasted.
# --------------------------------------------------------------------------

SERVOS = {
    "base":     {"channel": 0, "zero_deg": 90.0, "sign": 1, "min_deg": 0.0, "max_deg": 180.0},
    "shoulder": {"channel": 1, "zero_deg": 20.0, "sign": 1, "min_deg": 0.0, "max_deg": 180.0},
    "elbow":    {"channel": 2, "zero_deg": 0.0,  "sign": 1, "min_deg": 0.0, "max_deg": 180.0},
    "wrist":    {"channel": 3, "zero_deg": 20.0, "sign": 1, "min_deg": 0.0, "max_deg": 180.0},
}

# The gripper is not a kinematic joint -- it has no effect on where the tip is,
# so it is commanded directly in servo degrees rather than through the IK.
GRIPPER_CHANNEL = 4
GRIPPER_OPEN_DEG = 60.0      # wide enough to clear the largest item you sort
GRIPPER_CLOSED_DEG = 20.0    # NOT 0. Stop where the fingers grip, not where the
                             # servo stalls -- a stalled hobby servo burns out.


# --------------------------------------------------------------------------
# Serial link to the Arduino
# --------------------------------------------------------------------------

SERIAL_PORT = "/dev/ttyACM0"   # `ls /dev/ttyACM* /dev/ttyUSB*` with the board plugged in
BAUD = 115200
CONNECT_TIMEOUT_S = 10.0       # how long to wait for READY after opening the port
MOVE_TIMEOUT_S = 15.0          # how long to wait for OK before giving up on a move


# --------------------------------------------------------------------------
# Motion profile
# --------------------------------------------------------------------------

TRAVEL_HEIGHT_MM = 100.0    # fingertip height while moving across the workspace.
                            # Must clear the tallest item you expect to sort --
                            # but note that raising it shrinks the reachable area,
                            # since height and distance trade off against each other.
GRASP_Z_MM = 15.0           # fingertip height at the moment of closing. Small and
                            # positive: 0 would drive the fingers into the table.
LIFT_HEIGHT_MM = 100.0      # height to raise to immediately after gripping

MOVE_DURATION_MS = 900      # how long the arm takes per waypoint. Slower is safer
                            # and less likely to fling the item out of the gripper.
GRIPPER_SETTLE_MS = 500     # pause after open/close, so the fingers finish moving
                            # before the arm starts travelling

# Resting pose, in geometric degrees (base, shoulder, elbow, wrist). Leaned back
# and high, so the arm is behind its own base rather than hanging over the mat --
# an arm parked above the work surface blocks the camera's view of exactly the
# area it needs to see.
#
# Check this with test_kinematics.py after changing it: that prints where the
# fingertip actually ends up, which is the quickest way to catch a pose that reads
# fine as four angles but puts the arm somewhere useless.
HOME_POSE_DEG = (0.0, -18.0, 2.0, 2.0)


# --------------------------------------------------------------------------
# Camera and workspace
# --------------------------------------------------------------------------

# Pinned explicitly: the pixel->world homography is measured in pixels, so it is
# only valid at the resolution it was calibrated at. Leaving this to the webcam
# default would let a driver update silently rescale every coordinate the arm
# uses. calibrate_camera.py records this value and workspace.py refuses to load a
# calibration that does not match.
CAMERA_RESOLUTION = (1280, 720)

CAMERA_INDEX = 0             # /dev/videoN. Shared by main.py and calibrate_camera.py
                             # so they cannot end up calibrating one camera and
                             # running another.

CALIBRATION_FILE = "camera_calibration.json"

# Height of the camera lens above the table, needed for the parallax correction
# in workspace.py. Set to None to disable that correction (correct for a truly
# top-down camera, and the safe choice until you have measured it).
CAMERA_HEIGHT_MM = None
CAMERA_GROUND_XY_MM = (0.0, 0.0)   # where the lens is, projected straight down
                                   # onto the table, in world mm

DEFAULT_OBJECT_HEIGHT_MM = 40.0    # assumed item height, used only by the
                                   # parallax correction

# Bounds on where a detection is allowed to be, in world mm. Anything outside is
# ignored, so a bottle on a shelf behind the mat never becomes a grasp target.
#
# This is a cheap rectangular pre-filter, not a reachability test -- the corners
# of this box are further away than its middle and will still be refused by the
# IK. Set it to the mat, and let kinematics have the final say.
WORKSPACE_X_MM = (60.0, 150.0)     # (min, max) forward distance from the base
WORKSPACE_Y_MM = (-120.0, 120.0)   # (min, max) left/right


# --------------------------------------------------------------------------
# Derived helpers -- radians for the math layer
# --------------------------------------------------------------------------

def _servo_reachable_range_deg(name):
    """The geometric angles a servo can physically produce, given its mounting.

    Inverting  servo = zero + sign * geometric  over the servo's travel. A
    negative sign flips which end is which, hence the sort.
    """
    spec = SERVOS[name]
    ends = sorted(
        (
            (spec["min_deg"] - spec["zero_deg"]) / spec["sign"],
            (spec["max_deg"] - spec["zero_deg"]) / spec["sign"],
        )
    )
    return ends[0], ends[1]


# What the arm can ACTUALLY do: your mechanical limits intersected with what the
# servos can reach from where they are mounted.
#
# Keeping these separate and intersecting them is the point. A servo has 180
# degrees of travel; if the kinematics is allowed to ask for 200, the driver has
# no good options left -- clamping silently puts the arm somewhere the solver
# never intended, and refusing mid-sequence strands it holding an object. Folding
# the servo range into the limit check means such a pose is rejected while it is
# still just a number, before anything moves.
EFFECTIVE_JOINT_LIMITS_DEG = {}
for _name in JOINT_NAMES:
    _mech_lo, _mech_hi = JOINT_LIMITS_DEG[_name]
    _servo_lo, _servo_hi = _servo_reachable_range_deg(_name)
    EFFECTIVE_JOINT_LIMITS_DEG[_name] = (
        max(_mech_lo, _servo_lo),
        min(_mech_hi, _servo_hi),
    )
    if EFFECTIVE_JOINT_LIMITS_DEG[_name][0] >= EFFECTIVE_JOINT_LIMITS_DEG[_name][1]:
        raise ValueError(
            f"The {_name} joint has no usable range: JOINT_LIMITS_DEG says "
            f"{JOINT_LIMITS_DEG[_name]} but the servo mounting only reaches "
            f"({_servo_lo:.0f}, {_servo_hi:.0f}). Check its zero_deg and sign."
        )

JOINT_LIMITS_RAD = {
    name: (math.radians(lo), math.radians(hi))
    for name, (lo, hi) in EFFECTIVE_JOINT_LIMITS_DEG.items()
}

HOME_POSE_RAD = tuple(math.radians(d) for d in HOME_POSE_DEG)

MAX_REACH_MM = L_SHOULDER_ELBOW_MM + L_ELBOW_WRIST_MM
