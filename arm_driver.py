"""Talking to the arm: geometric joint angles in, servo commands out.

Two backends behind one interface. SimulatedArmDriver is the default and needs no
hardware -- it prints what would be sent and sleeps the real move duration, so the
whole pipeline is exercisable before the robot is built. SerialArmDriver is the
same interface over pyserial to an Arduino.

Serial protocol (what the Arduino sketch must implement)
---------------------------------------------------------
Host -> board:
    M <base> <shoulder> <elbow> <wrist> <gripper> <ms>\\n
        Move all five servos to these positions, in servo degrees, interpolating
        over <ms> milliseconds. Reply OK when the motion has finished.
    H\\n     Go to the home pose. Reply OK when finished.
    P\\n     Ping. Reply OK immediately.

Board -> host:
    READY\\n        Sent once at boot, after the servos reach home.
    OK\\n           The previous command has completed.
    ERR <text>\\n   The previous command was rejected.

Integers, space separated, newline terminated: parseable with Serial.parseInt()
and readable by a human in the Arduino serial monitor while debugging.
"""

import math
import time

import arm_config as cfg
import kinematics


class ArmError(RuntimeError):
    """The arm could not be commanded, or did not confirm that it moved."""


def joint_to_servo_deg(joint_name, radians):
    """Geometric joint angle -> the number the servo library wants.

    The trim, direction, and range all come from arm_config.SERVOS, which is why
    a servo mounted backwards is fixed by flipping a sign in the config rather
    than by putting a minus sign in the kinematics.

    Out of range raises rather than clamping. Clamping is the tempting choice --
    it protects the servo and the program keeps running -- but it moves the arm
    to a pose nobody solved for, silently, with the error growing the further out
    of range the request was. Since arm_config folds each servo's travel into the
    joint limits, and every waypoint is limit-checked before the sequence starts,
    getting here at all means a bug rather than an ambitious target.
    """
    spec = cfg.SERVOS[joint_name]
    value = spec["zero_deg"] + spec["sign"] * math.degrees(radians)
    if not (spec["min_deg"] <= value <= spec["max_deg"]):
        raise ArmError(
            f"{joint_name} at {math.degrees(radians):.1f} deg needs servo position "
            f"{value:.1f}, outside its travel of {spec['min_deg']:.0f}..{spec['max_deg']:.0f}. "
            "This should have been caught by the limit check -- the servo mapping "
            "and JOINT_LIMITS_DEG in arm_config.py disagree."
        )
    return value


def pose_to_servo_degrees(angles):
    """JointAngles -> servo degrees, ordered by each servo's channel index."""
    angles = kinematics.JointAngles(*angles)
    by_channel = sorted(cfg.JOINT_NAMES, key=lambda name: cfg.SERVOS[name]["channel"])
    return [joint_to_servo_deg(name, getattr(angles, name)) for name in by_channel]


class ArmDriver:
    """Common behaviour: pose bookkeeping, gripper state, context management."""

    def __init__(self):
        self.angles = kinematics.JointAngles(*cfg.HOME_POSE_RAD)
        self.gripper_deg = cfg.GRIPPER_OPEN_DEG
        self.connected = False

    # -- context manager -----------------------------------------------------
    # Homing on exit matters more than it looks: an arm left holding a pose is an
    # arm holding torque, and one left extended over the mat blocks the camera.
    # __exit__ runs on exceptions and Ctrl-C too, which is exactly when a crashed
    # program would otherwise abandon the arm mid-reach.

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.connected:
                self.home()
        finally:
            self.close()
        return False

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False

    # -- commands ------------------------------------------------------------

    def move_to(self, angles, duration_ms=None):
        """Move the four joints, holding the gripper where it is."""
        angles = kinematics.JointAngles(*angles)
        ok, reason = kinematics.check_limits(angles)
        if not ok:
            raise ArmError(f"refusing to move: {reason}")

        duration = cfg.MOVE_DURATION_MS if duration_ms is None else duration_ms
        self._send_pose(angles, self.gripper_deg, duration)
        self.angles = angles

    def set_gripper(self, servo_deg, duration_ms=None):
        """Open or close the gripper, holding the arm where it is."""
        duration = cfg.GRIPPER_SETTLE_MS if duration_ms is None else duration_ms
        self._send_pose(self.angles, servo_deg, duration)
        self.gripper_deg = servo_deg

    def open_gripper(self):
        self.set_gripper(cfg.GRIPPER_OPEN_DEG)

    def close_gripper(self):
        self.set_gripper(cfg.GRIPPER_CLOSED_DEG)

    def home(self):
        """Return to the resting pose defined in arm_config."""
        self._send_home()
        self.angles = kinematics.JointAngles(*cfg.HOME_POSE_RAD)

    # -- backend hooks -------------------------------------------------------

    def _send_pose(self, angles, gripper_deg, duration_ms):
        raise NotImplementedError

    def _send_home(self):
        raise NotImplementedError


class SimulatedArmDriver(ArmDriver):
    """Prints commands and waits the real duration. No hardware required.

    This is not a stub -- it runs the same timing as the real arm, so the vision
    loop's stalls, buffer flushing, and track resets all behave the way they will
    once the servos are attached.
    """

    def __init__(self, realtime=True):
        super().__init__()
        self.realtime = realtime

    def connect(self):
        super().connect()
        print("[arm] simulated driver -- no hardware will move")

    def _send_pose(self, angles, gripper_deg, duration_ms):
        servos = pose_to_servo_degrees(angles)
        tip = kinematics.forward_kinematics(angles)
        print(
            f"[arm] {kinematics.describe(angles)}  grip={gripper_deg:5.1f}"
            f"  -> servos {[round(v) for v in servos]}"
            f"  tip=({tip[0]:6.1f}, {tip[1]:6.1f}, {tip[2]:6.1f})mm"
        )
        if self.realtime:
            time.sleep(duration_ms / 1000.0)

    def _send_home(self):
        print("[arm] home")
        if self.realtime:
            time.sleep(cfg.MOVE_DURATION_MS / 1000.0)


class SerialArmDriver(ArmDriver):
    """The real arm, over USB serial to an Arduino."""

    def __init__(self, port=None, baud=None):
        super().__init__()
        self.port = port or cfg.SERIAL_PORT
        self.baud = baud or cfg.BAUD
        self.serial = None

    def connect(self):
        try:
            import serial
        except ImportError as exc:
            raise ArmError(
                "pyserial is not installed. `pip install pyserial`, or run with "
                "the simulated driver."
            ) from exc

        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=1.0)
        except serial.SerialException as exc:
            raise ArmError(f"could not open {self.port}: {exc}") from exc

        # Opening the port pulls DTR, which resets most Arduino boards. Anything
        # written before the sketch finishes booting is lost, so wait for the
        # board to announce itself rather than guessing at a sleep duration.
        self._await("READY", cfg.CONNECT_TIMEOUT_S)
        super().connect()
        print(f"[arm] connected on {self.port} at {self.baud} baud")

    def close(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None
        super().close()

    def _write(self, line):
        self.serial.write((line + "\n").encode("ascii"))
        self.serial.flush()

    def _await(self, expected, timeout_s):
        """Block until the board says `expected`, or give up loudly.

        Waiting for an acknowledgement rather than sleeping the move duration is
        what keeps Python's idea of where the arm is in step with reality. A
        sleep that is even slightly short means the next command arrives while
        the arm is still moving, and the error compounds over a sequence.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            raw = self.serial.readline().decode("ascii", errors="replace").strip()
            if not raw:
                continue
            if raw == expected:
                return
            if raw.startswith("ERR"):
                raise ArmError(f"arm rejected the command: {raw}")
            # Anything else is the sketch's own debug chatter; show it, keep waiting.
            print(f"[arm] {raw}")
        raise ArmError(
            f"timed out after {timeout_s:.0f}s waiting for {expected!r}. The board "
            "may be unplugged, running a sketch that does not speak this protocol, "
            "or stuck mid-move."
        )

    def _send_pose(self, angles, gripper_deg, duration_ms):
        servos = pose_to_servo_degrees(angles)
        parts = [str(round(v)) for v in servos]
        parts.append(str(round(gripper_deg)))
        parts.append(str(int(duration_ms)))
        self._write("M " + " ".join(parts))
        self._await("OK", cfg.MOVE_TIMEOUT_S)

    def _send_home(self):
        self._write("H")
        self._await("OK", cfg.MOVE_TIMEOUT_S)


def make_driver(simulated=True):
    """Pick a backend. Simulated by default -- real motion has to be asked for."""
    return SimulatedArmDriver() if simulated else SerialArmDriver()
