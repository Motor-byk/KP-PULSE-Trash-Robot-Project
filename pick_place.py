"""The pick-and-place sequence: object on the mat -> item in a bin.

Plan first, then move
----------------------
Every waypoint is solved and limit-checked before the first servo is commanded.
If any step in the sequence is unreachable, nothing moves at all and the caller
gets a reason it can put on screen.

The alternative -- solving each waypoint as you arrive at it -- fails halfway
through, leaving the arm holding an object somewhere over the workspace with no
plan for putting it down. Refusing up front leaves it parked at home, which is
always a recoverable state.
"""

import arm_config as cfg
import bins
import kinematics


class Plan:
    """A fully solved sequence, ready to execute or reject."""

    def __init__(self, steps, description):
        self.steps = steps            # list of ("move", label, JointAngles)
                                      #      or ("grip", label, servo_degrees)
        self.description = description


def _drop_repeats(steps, tolerance_rad=1e-4):
    """Remove a move to a pose the arm is already in.

    Waypoints legitimately coincide: if a bin's drop height equals the travel
    height, "above bin" and "at bin" are the same pose. Sending both wastes a
    full move duration doing nothing and makes the log read as though a step
    was skipped.
    """
    trimmed = []
    previous = None
    for step in steps:
        kind, _, payload = step
        if kind == "move":
            if previous is not None and all(
                abs(a - b) < tolerance_rad for a, b in zip(payload, previous)
            ):
                continue
            previous = payload
        elif kind == "home":
            previous = None
        trimmed.append(step)
    return trimmed


def plan_pick(world_xy, bin_name):
    """Build the full sequence for one item. Returns (Plan, None) or (None, reason)."""
    x, y = world_xy

    drop = bins.BIN_DROP_XYZ.get(bin_name)
    if drop is None:
        return None, f"no drop position configured for bin {bin_name!r}"

    # Positions, in the order the arm visits them. Approach and retreat are both
    # vertical: the gripper comes straight down onto the object and straight back
    # up. Moving in at grasp height would sweep the fingers sideways through the
    # object and knock it over before closing.
    moves = [
        ("above object", (x, y, cfg.TRAVEL_HEIGHT_MM)),
        ("descend", (x, y, cfg.GRASP_Z_MM)),
        ("lift", (x, y, cfg.LIFT_HEIGHT_MM)),
        ("above bin", (drop[0], drop[1], cfg.TRAVEL_HEIGHT_MM)),
        ("at bin", drop),
    ]

    solved = {}
    for label, xyz in moves:
        angles, reason = kinematics.solve(*xyz)
        if angles is None:
            return None, f"{label} at ({xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f})mm: {reason}"
        solved[label] = angles

    steps = _drop_repeats([
        ("grip", "open", cfg.GRIPPER_OPEN_DEG),
        ("move", "above object", solved["above object"]),
        ("move", "descend", solved["descend"]),
        ("grip", "close", cfg.GRIPPER_CLOSED_DEG),
        ("move", "lift", solved["lift"]),
        ("move", "above bin", solved["above bin"]),
        ("move", "at bin", solved["at bin"]),
        ("grip", "release", cfg.GRIPPER_OPEN_DEG),
        ("home", "home", None),
    ])

    description = (
        f"({x:.0f}, {y:.0f})mm -> {bin_name} bin at "
        f"({drop[0]:.0f}, {drop[1]:.0f}, {drop[2]:.0f})mm"
    )
    return Plan(steps, description), None


class PickAndPlace:
    """Runs plans on a driver, and reports what happened."""

    def __init__(self, driver, verbose=True):
        self.driver = driver
        self.verbose = verbose

    def execute(self, world_xy, bin_name):
        """Do one full pick. Returns (succeeded, reason_if_not)."""
        plan, reason = plan_pick(world_xy, bin_name)
        if plan is None:
            if self.verbose:
                print(f"[pick] refused: {reason}")
            return False, reason

        if self.verbose:
            print(f"[pick] {plan.description}")

        for kind, label, payload in plan.steps:
            if self.verbose:
                print(f"[pick]   {label}")
            if kind == "move":
                self.driver.move_to(payload)
            elif kind == "grip":
                self.driver.set_gripper(payload)
            elif kind == "home":
                self.driver.home()

        if self.verbose:
            print("[pick] done")
        return True, None


def check_bin_positions():
    """Verify every configured bin is somewhere the arm can actually reach.

    Run at startup, before any object is picked. Discovering that the compost bin
    is 40mm out of reach is survivable while the gripper is empty and unbearable
    once it is holding a banana peel.

    Returns a list of human-readable problems; empty means all good.
    """
    problems = []
    for bin_name, drop in sorted(bins.BIN_DROP_XYZ.items()):
        for label, xyz in (
            ("approach", (drop[0], drop[1], cfg.TRAVEL_HEIGHT_MM)),
            ("drop", drop),
        ):
            angles, reason = kinematics.solve(*xyz)
            if angles is None:
                problems.append(f"{bin_name} bin, {label} height: {reason}")
    return problems
