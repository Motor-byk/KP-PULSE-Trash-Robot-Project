"""One-time camera calibration: teach the program where the table is.

Run this once, after the camera is physically fixed in place and after the arm's
base position is decided. Re-run it any time the camera moves, even slightly --
a bumped mount invalidates the mapping, and nothing downstream can detect that.

    python calibrate_camera.py

Procedure
---------
 1. Aim the camera, then press SPACE to freeze a frame.
 2. Click four points on the mat, ideally near its corners, spread as widely as
    the frame allows. Something with a visible corner (tape crosses, a printed
    sheet) is easier to click accurately than a blank mat.
 3. For each point in the order you clicked it, type its real position in
    millimetres, measured from the arm's base rotation axis: +X forward from the
    base, +Y to the arm's left.
 4. Check the live readout against a ruler before trusting it.

Four points is the minimum. More is better -- the fit least-squares averages them,
which dilutes the error in any single ruler measurement.
"""

import cv2

import arm_config as cfg
import workspace


WINDOW = "Calibrate - SPACE freeze | click 4+ points | ENTER solve | R reset | Q quit"


def collect_points(cap):
    """Freeze a frame and gather clicked pixel positions. Returns a list or None."""
    clicks = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    frozen = None
    while True:
        if frozen is None:
            ret, frame = cap.read()
            if not ret:
                raise SystemExit("Lost the camera feed.")
            display = frame.copy()
            cv2.putText(
                display, "SPACE to freeze", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )
        else:
            display = frozen.copy()
            for index, (x, y) in enumerate(clicks, start=1):
                cv2.drawMarker(display, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(
                    display, str(index), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                )
            status = f"{len(clicks)} points - ENTER to solve (need 4+), R to reset"
            cv2.putText(
                display, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return None
        if key == ord(" ") and frozen is None:
            frozen = frame.copy()
        elif key == ord("r"):
            clicks.clear()
            frozen = None
        elif key in (13, 10) and frozen is not None:
            if len(clicks) < 4:
                print(f"Need at least 4 points, have {len(clicks)}.")
                continue
            return clicks


def ask_world_positions(clicks):
    """Prompt for the real-world mm of each clicked point, in click order."""
    print("\nFor each point, type its position in mm from the ARM BASE.")
    print("  X = forward from the base, Y = to the arm's left. Example:  180 -75\n")

    world = []
    for index, (px, py) in enumerate(clicks, start=1):
        while True:
            raw = input(f"  point {index} (pixel {px},{py})  X Y in mm: ").strip()
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                print("    Type two numbers separated by a space.")
                continue
            try:
                world.append((float(parts[0]), float(parts[1])))
                break
            except ValueError:
                print("    Those were not numbers.")
    return world


def report_fit_error(space, clicks, world):
    """Map the calibration points back through the fit and show the residuals.

    A good fit reproduces its own input almost exactly. A large residual here
    means a mistyped measurement or a sloppy click, and it is far cheaper to
    catch that now than to watch the arm miss by 3cm later.
    """
    print("\nFit check -- how well the solved mapping reproduces your measurements:")
    worst = 0.0
    for (px, py), (wx, wy) in zip(clicks, world):
        gx, gy = space.pixel_to_world(px, py)
        error = ((gx - wx) ** 2 + (gy - wy) ** 2) ** 0.5
        worst = max(worst, error)
        print(f"  measured ({wx:7.1f},{wy:7.1f})  ->  fitted ({gx:7.1f},{gy:7.1f})   off by {error:5.1f} mm")

    print(f"\n  worst residual: {worst:.1f} mm")
    if worst > 10.0:
        print("  That is high. Check for a mistyped measurement or a misplaced click,")
        print("  then press R and redo it -- the arm inherits this error directly.")


def live_check(cap, space):
    """Live feed with the world position under the cursor, for ruler checking."""
    cursor = [0, 0]

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            cursor[0], cursor[1] = x, y

    window = "Verify - hover over a known point and compare | Q to finish"
    cv2.destroyAllWindows()
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    print("\nHover the mouse over points you can measure and compare the readout")
    print("to a ruler. Press Q when satisfied.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x_mm, y_mm = space.pixel_to_world(*cursor)
        inside = space.in_bounds(x_mm, y_mm)
        color = (0, 255, 0) if inside else (0, 0, 255)

        cv2.drawMarker(frame, tuple(cursor), color, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            frame,
            f"X {x_mm:7.1f} mm   Y {y_mm:7.1f} mm   {'in workspace' if inside else 'OUT OF BOUNDS'}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    try:
        cap, _ = workspace.open_camera()
    except workspace.CalibrationError as exc:
        raise SystemExit(str(exc))

    try:
        clicks = collect_points(cap)
        if clicks is None:
            print("Cancelled -- nothing was written.")
            return

        world = ask_world_positions(clicks)
        homography = workspace.solve_homography(clicks, world)
        space = workspace.Workspace(homography, cfg.CAMERA_RESOLUTION)

        report_fit_error(space, clicks, world)

        workspace.save_calibration(
            cfg.CALIBRATION_FILE, homography, cfg.CAMERA_RESOLUTION, clicks, world
        )
        print(f"\nWrote {cfg.CALIBRATION_FILE}")

        live_check(cap, space)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
