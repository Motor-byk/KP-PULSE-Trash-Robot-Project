"""Self-check for the kinematics, runnable with no hardware.

    python test_kinematics.py

The main test round-trips positions through inverse_kinematics and back through
forward_kinematics and checks they agree. That is worth doing because sign and
convention mistakes are the likeliest bug in kinematics code and they do not look
like bugs -- they produce perfectly plausible angles that put the arm in the wrong
place. Two independently written directions of the same math agreeing is real
evidence; eyeballing the angles is not.

Run this again after editing any dimension in arm_config.py.
"""

import math

import arm_config as cfg
import arm_driver
import kinematics
import pick_place


TOLERANCE_MM = 1.0


def test_round_trip():
    """Every solvable point in a grid must come back where it started."""
    tested = 0
    worst = 0.0

    for x in range(40, 320, 20):
        for y in range(-200, 220, 20):
            for z in (cfg.GRASP_Z_MM, 60.0, cfg.TRAVEL_HEIGHT_MM):
                angles = kinematics.inverse_kinematics(x, y, z)
                if angles is None:
                    continue          # out of reach is a valid answer, not a failure

                bx, by, bz = kinematics.forward_kinematics(angles)
                error = math.dist((x, y, z), (bx, by, bz))
                worst = max(worst, error)
                tested += 1

                assert error < TOLERANCE_MM, (
                    f"round trip failed at ({x}, {y}, {z}): came back at "
                    f"({bx:.1f}, {by:.1f}, {bz:.1f}), off by {error:.2f}mm"
                )

    assert tested > 0, (
        "No point in the test grid was reachable. The link lengths in "
        "arm_config.py are probably far too short for WORKSPACE_X_MM."
    )
    print(f"  round trip: {tested} poses, worst error {worst:.4f}mm")


def test_gripper_is_vertical():
    """The solver must always leave the last link pointing straight down.

    This is the constraint that makes a 4DOF arm solvable at all, so if it ever
    stops holding, the IK is silently solving a different problem than the grasp
    sequence assumes.
    """
    checked = 0
    for x in (80, 150, 220):
        for y in (-100, 0, 100):
            angles = kinematics.inverse_kinematics(x, y, cfg.GRASP_Z_MM)
            if angles is None:
                continue
            total = angles.shoulder + angles.elbow + angles.wrist
            assert abs(math.cos(total) + 1.0) < 1e-6, (
                f"gripper is not vertical at ({x}, {y}): the links sum to "
                f"{math.degrees(total):.1f} deg from vertical, expected 180"
            )
            checked += 1
    print(f"  vertical gripper: {checked} poses")


def test_unreachable_returns_none():
    """Out of range must be None, never a clamped guess."""
    far = cfg.MAX_REACH_MM + cfg.L_WRIST_TIP_MM + 500
    cases = {
        "far beyond reach": (far, 0.0, cfg.GRASP_Z_MM),
        "below the table": (150.0, 0.0, -400.0),
        "far above reach": (150.0, 0.0, far),
    }
    for label, xyz in cases.items():
        result = kinematics.inverse_kinematics(*xyz)
        assert result is None, f"{label} {xyz} should be unreachable, got {result}"

        _, reason = kinematics.solve(*xyz)
        assert reason, f"{label} produced no explanation"
    print(f"  unreachable: {len(cases)} cases correctly refused")


def test_base_angle_tracks_target():
    """An object to the left must swing the base left, and vice versa.

    A sign error in the base joint is the single most dangerous mistake here --
    it mirrors the whole workspace, and the arm would confidently reach exactly
    the wrong way.
    """
    left = kinematics.inverse_kinematics(150.0, 100.0, cfg.GRASP_Z_MM)
    right = kinematics.inverse_kinematics(150.0, -100.0, cfg.GRASP_Z_MM)
    assert left is not None and right is not None, (
        "the test points are unreachable -- widen the arm's limits or lengthen "
        "the links in arm_config.py"
    )
    assert left.base > 0 > right.base, (
        f"base angle sign is inverted: +Y gave {math.degrees(left.base):.1f} deg, "
        f"-Y gave {math.degrees(right.base):.1f} deg"
    )
    print("  base angle: sign correct")


def test_servo_mapping_agrees_with_limits():
    """Anything the limit check accepts, the servos must be able to produce.

    This is the check that catches JOINT_LIMITS_DEG and the servo mounting
    drifting apart -- the failure that otherwise shows up as the arm quietly
    stopping 20 degrees short of where the solver sent it.
    """
    for name in cfg.JOINT_NAMES:
        lo, hi = cfg.JOINT_LIMITS_RAD[name]
        for radians in (lo, (lo + hi) / 2, hi):
            arm_driver.joint_to_servo_deg(name, radians)   # raises if unreachable

        # And an angle well outside the limits must be refused, not clamped.
        try:
            arm_driver.joint_to_servo_deg(name, hi + math.radians(45))
        except arm_driver.ArmError:
            pass
        else:
            raise AssertionError(f"{name} silently accepted an out-of-range angle")

    ranges = "  ".join(
        f"{name} {lo:.0f}..{hi:.0f}"
        for name, (lo, hi) in cfg.EFFECTIVE_JOINT_LIMITS_DEG.items()
    )
    print(f"  servo mapping: consistent ({ranges})")


def test_home_pose_is_legal():
    ok, reason = kinematics.check_limits(cfg.HOME_POSE_RAD)
    assert ok, f"HOME_POSE_DEG violates the joint limits: {reason}"
    tip = kinematics.forward_kinematics(cfg.HOME_POSE_RAD)
    print(f"  home pose: legal, tip at ({tip[0]:.0f}, {tip[1]:.0f}, {tip[2]:.0f})mm")


def report_bin_reachability():
    """Not an assertion -- the placeholder bin positions are expected to be wrong.

    Reported rather than asserted so this script still passes on a fresh clone,
    while telling you exactly what to fix once you have real numbers.
    """
    problems = pick_place.check_bin_positions()
    if problems:
        print("\n  NOTE: bin positions in bins.py are not reachable yet:")
        for problem in problems:
            print(f"    - {problem}")
        print("  Expected until you measure the real bin positions and arm links.")
    else:
        print("  bin positions: all reachable")


def main():
    print("kinematics self-check\n")
    for test in (
        test_round_trip,
        test_gripper_is_vertical,
        test_unreachable_returns_none,
        test_base_angle_tracks_target,
        test_servo_mapping_agrees_with_limits,
        test_home_pose_is_legal,
    ):
        test()
    report_bin_reachability()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
