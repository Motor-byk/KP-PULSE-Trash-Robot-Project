"""Forward and inverse kinematics for the 4DOF arm.

Pure math over numbers -- no serial, no OpenCV, no global state. Everything here
can be exercised without the robot existing, which is the point: test_kinematics.py
verifies the whole module by round-tripping IK through FK.

Angle convention is defined at the top of arm_config.py. In short, all angles are
measured from the previous link, with the shoulder measured from straight up, and
positive means leaning forward.

The wrist is spent
-------------------
A 4DOF arm has one fewer joint than it needs to reach an arbitrary position AND
orientation. This solver spends the wrist on holding the gripper pointing straight
down, which is what a top-down grasp needs. Two consequences:

  * The IK becomes exact and closed-form. With the gripper vertical, the wrist
    pivot must sit directly above the target, so what remains is the textbook
    two-link problem.
  * The gripper's approach direction is not a free parameter. Combined with the
    missing wrist roll, the jaws always close perpendicular to the direction the
    arm is reaching. See docs/arm-setup.md.
"""

import math
from collections import namedtuple

import arm_config as cfg


# Geometric joint angles in radians, in the order the arm is assembled.
JointAngles = namedtuple("JointAngles", cfg.JOINT_NAMES)


def _normalize(angle):
    """Wrap an angle into (-pi, pi], so limit checks compare like with like."""
    return math.atan2(math.sin(angle), math.cos(angle))


def forward_kinematics(angles):
    """Joint angles -> world position of the point between the fingertips, in mm.

    Used to verify the IK by round-trip, and to report where a pose actually puts
    the arm when debugging against the simulator.
    """
    angles = JointAngles(*angles)

    # Each link's direction, measured from vertical, is the running sum of the
    # pitch joints before it.
    a1 = angles.shoulder
    a2 = a1 + angles.elbow
    a3 = a2 + angles.wrist

    r = (
        cfg.BASE_X_OFFSET_MM
        + cfg.L_SHOULDER_ELBOW_MM * math.sin(a1)
        + cfg.L_ELBOW_WRIST_MM * math.sin(a2)
        + cfg.L_WRIST_TIP_MM * math.sin(a3)
    )
    z = (
        cfg.BASE_HEIGHT_MM
        + cfg.L_SHOULDER_ELBOW_MM * math.cos(a1)
        + cfg.L_ELBOW_WRIST_MM * math.cos(a2)
        + cfg.L_WRIST_TIP_MM * math.cos(a3)
    )

    return (r * math.cos(angles.base), r * math.sin(angles.base), z)


def check_limits(angles):
    """Return (ok, reason). `reason` names the first joint out of range."""
    angles = JointAngles(*angles)
    for name in cfg.JOINT_NAMES:
        lo, hi = cfg.JOINT_LIMITS_RAD[name]
        value = getattr(angles, name)
        if not (lo <= value <= hi):
            return False, (
                f"{name} would be {math.degrees(value):.1f} deg, "
                f"outside its limit of "
                f"{math.degrees(lo):.1f}..{math.degrees(hi):.1f}"
            )
    return True, None


def inverse_kinematics(x, y, z):
    """World position in mm -> JointAngles, or None if the arm cannot get there.

    Returns None for anything unreachable rather than a clamped approximation:
    an arm that silently goes "as close as it can" to a point outside its
    workspace is an arm that drives into the table.
    """
    solution, _ = solve(x, y, z)
    return solution


def solve(x, y, z):
    """inverse_kinematics, but also returning why it failed.

    Returns (JointAngles, None) on success or (None, reason) on failure, so the
    caller can put something useful on screen instead of just refusing.
    """
    base = math.atan2(y, x)

    # Collapse to the vertical plane the arm reaches in. Because the gripper is
    # held vertical, its last link contributes nothing horizontally and exactly
    # -L_WRIST_TIP vertically, so the wrist pivot sits directly above the target.
    u = math.hypot(x, y) - cfg.BASE_X_OFFSET_MM
    v = (z + cfg.L_WRIST_TIP_MM) - cfg.BASE_HEIGHT_MM

    l1 = cfg.L_SHOULDER_ELBOW_MM
    l2 = cfg.L_ELBOW_WRIST_MM
    d = math.hypot(u, v)

    if d > l1 + l2:
        return None, f"too far: needs {d:.0f}mm of reach, arm has {l1 + l2:.0f}mm"
    if d < abs(l1 - l2):
        return None, f"too close to the base: {d:.0f}mm is inside the arm's dead zone"

    # Law of cosines on the shoulder-elbow-wrist triangle. Clamped because
    # floating point can push this a hair past 1 exactly at the workspace edge.
    cos_elbow = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_magnitude = math.acos(cos_elbow)

    # Direction from the shoulder pivot to the wrist pivot, from vertical.
    phi = math.atan2(u, v)

    # Both elbow bends reach the same point. Rather than assume which one your
    # arm can physically do, try both, drop the ones that break a joint limit,
    # and prefer the one holding the elbow higher -- that keeps the joint away
    # from the table and from the object being approached.
    candidates = []
    rejections = []
    for branch in (1.0, -1.0):
        elbow = branch * elbow_magnitude
        # Angle between the upper arm and the straight line to the wrist pivot.
        alpha = math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
        shoulder = phi - alpha
        # Whatever pitch the first two joints accumulated, the wrist cancels, so
        # the gripper ends up pointing straight down (pi from vertical).
        wrist = _normalize(math.pi - (shoulder + elbow))

        angles = JointAngles(
            base=_normalize(base),
            shoulder=_normalize(shoulder),
            elbow=_normalize(elbow),
            wrist=wrist,
        )
        ok, reason = check_limits(angles)
        if ok:
            elbow_z = cfg.BASE_HEIGHT_MM + l1 * math.cos(angles.shoulder)
            candidates.append((elbow_z, angles))
        else:
            rejections.append(reason)

    if not candidates:
        return None, "; ".join(dict.fromkeys(rejections))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1], None


def reachable(x, y, z):
    """True if the arm can put its fingertips exactly there, within its limits."""
    return inverse_kinematics(x, y, z) is not None


def describe(angles):
    """One-line human-readable pose, for logging and the simulated driver."""
    angles = JointAngles(*angles)
    return "  ".join(
        f"{name}={math.degrees(getattr(angles, name)):7.2f}"
        for name in cfg.JOINT_NAMES
    )
