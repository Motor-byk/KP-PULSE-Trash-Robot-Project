# Arm bring-up

Everything from "the code exists" to "the arm picks up a bottle". The code is
written and self-checks pass; what remains is measuring your robot and typing the
numbers into `arm_config.py`.

Do these in order. Each step depends on the one before it, and steps 2 and 3 are
invalidated by moving the camera, so do not start them until the mount is final.

---

## 0. Where the numbers live

| File | Holds |
|---|---|
| `arm_config.py` | every dimension, joint limit, servo trim, timing, port |
| `bins.py` → `BIN_DROP_XYZ` | where the three bins physically sit |
| `camera_calibration.json` | the pixel→millimetre mapping, written by `calibrate_camera.py` |

Nothing else in the project contains a physical constant. If you find yourself
wanting to put one somewhere else, put it in `arm_config.py` instead.

## The world frame

Everything is measured from **the arm's base rotation axis, at table level**:

```
        +Y (arm's left)
         ^
         |
         |
  base   O ------> +X (forward, where the arm points at base angle 0)
         
  +Z is up out of the table.
```

Millimetres throughout. Use the same origin for the calibration points, the bin
positions, and the workspace bounds — mixing origins is the single easiest way to
send the arm confidently to the wrong place.

---

## 1. Measure the arm

Fill in the link lengths in `arm_config.py`. Measure **between rotation axes**
(the pivot bolts), not between the outsides of the brackets, with the arm laid out
straight:

- `BASE_HEIGHT_MM` — table surface up to the shoulder pivot
- `L_SHOULDER_ELBOW_MM` — shoulder axis to elbow axis
- `L_ELBOW_WRIST_MM` — elbow axis to wrist axis
- `L_WRIST_TIP_MM` — wrist axis to where the closed fingers actually pinch

Then the servos. For each joint:

1. Command the servo to its midpoint.
2. Physically set the joint to the geometric zero described at the top of
   `arm_config.py` (shoulder straight up, elbow and wrist straight).
3. Whatever servo number produced that is `zero_deg`.
4. Nudge the command upward. If the joint moves the way the convention calls
   negative, set `sign` to `-1`.

`min_deg`/`max_deg` are the servo's real travel, usually 0–180.

Set `GRIPPER_CLOSED_DEG` to where the fingers grip, **not** where the servo
stalls. A stalled hobby servo draws full current until it burns out.

Then:

```bash
python test_kinematics.py
```

This round-trips positions through the IK and back, checks the base angle's sign,
checks that your joint limits and servo ranges agree, and prints where the home
pose actually puts the fingertip. Run it after every edit to `arm_config.py`.

## 2. Mount the camera, then freeze it

Fix the camera so it cannot move, and mark the mount position. Overhead or angled
both work — the homography handles either. From here on, bumping the camera
invalidates both the calibration and any training data captured before it.

Set `CAMERA_RESOLUTION` in `arm_config.py` to a mode the camera really supports.
The code verifies this rather than trusting it: `cap.set()` is a request, and a
webcam given a mode it lacks silently substitutes another.

## 3. Calibrate

```bash
python calibrate_camera.py
```

Press SPACE to freeze, click four points on the mat spread as widely as the frame
allows, then type each one's real X/Y in millimetres from the base. Tape crosses
or a printed sheet are much easier to click accurately than a blank mat.

The script prints how well the fit reproduces your own measurements. A worst
residual above ~10mm means a mistyped number or a sloppy click — redo it, because
the arm inherits that error directly.

Then hover the mouse over known points and check the readout against a ruler. If
the error is small in the middle and grows toward the frame edges, that is lens
distortion, and the fix is a real `cv2.calibrateCamera` pass rather than more
homography points.

## 4. Place the bins

Put the three bins down, measure each one's position from the base, and write them
into `BIN_DROP_XYZ` in `bins.py`. Mark their footprints on the table so they can be
put back. `Z` is the height the gripper opens at — a little above the rim, so items
drop in rather than being placed on the edge.

`python test_kinematics.py` reports whether all three are actually reachable.
`main.py` refuses to enable the arm if any is not.

## 5. Dry run, then live

```bash
python main.py           # SIMULATE_ARM = True
```

The arm layer runs fully — IK, sequencing, timing — and prints every command
instead of sending it. Put an object on the mat, wait for a stable track, press
SPACE, and read the waypoints. Check that the base angle follows the object as you
move it left and right, and that the printed tip position matches where the object
actually is.

Then, with the **servo power supply off** but the Arduino connected, set
`SIMULATE_ARM = False` and repeat. Confirm the board acknowledges every command.

Finally, power the servos with the arm clear of everything, and pick something.

---

## Serial protocol

The Arduino sketch is not in this repo — this is the contract it has to implement.
`arm_driver.py` is the other half.

**Host → board**

| Message | Meaning |
|---|---|
| `M <base> <shoulder> <elbow> <wrist> <gripper> <ms>` | move all five servos to these positions in servo degrees, interpolating over `<ms>` milliseconds; reply `OK` when the motion has finished |
| `H` | go to the home pose; reply `OK` when finished |
| `P` | ping; reply `OK` immediately |

**Board → host**

| Message | Meaning |
|---|---|
| `READY` | sent once at boot, after the servos reach home |
| `OK` | the previous command has completed |
| `ERR <text>` | the previous command was rejected |

Integers, space separated, newline terminated. Parseable with `Serial.parseInt()`
and readable by a human in the serial monitor while debugging.

Two things the sketch must get right:

- **Send `READY` after booting.** Opening the serial port resets most Arduino
  boards, so anything written before the sketch is up is lost. The driver waits
  for this line rather than guessing at a delay.
- **Send `OK` when the move is genuinely finished**, not when it is accepted. The
  driver blocks on it, which is what keeps Python's idea of where the arm is in
  step with reality. Acknowledging early means the next command arrives mid-move
  and the error compounds across the sequence.

The sketch should interpolate between poses rather than calling `servo.write()`
once — a servo commanded to jump 90 degrees goes as fast as it physically can,
which throws the item out of the gripper and browns out the supply.

---

## Known limitations, by design

These are expected behaviour, not bugs to report:

**The jaws cannot rotate.** With no wrist roll, the fingers always close
perpendicular to the direction the arm is reaching. An elongated object lying
along that same line will not be gripped. The arm attempts every pick regardless,
so you will find out empirically which shapes work; there is no orientation
handling to fix, only a different wrist.

**Grasp height is fixed.** One camera on a plane gives no depth, so the fingers
always descend to `GRASP_Z_MM`. Tall objects get gripped low on the body and very
flat ones may be missed. It is one constant to tune, and the mat is assumed flat.

**Objects are assumed to sit on the table.** The homography maps the *table plane*.
An object on top of another object maps to the wrong place — the parallax
correction in `workspace.py` compensates only for an item's own height, and only
once `CAMERA_HEIGHT_MM` is measured.

**The vision pauses while the arm moves.** This is deliberate: the arm occludes
the mat mid-pick, so tracking through it would produce garbage. Afterwards the
frame buffer is flushed and the vote history cleared, so what resumes is the world
as it is now.
