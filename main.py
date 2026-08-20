import cv2
import numpy as np
from ultralytics import YOLO


# Load your trained OBB model
model = YOLO("runs/obb/train/weights/best.pt")


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO tracking
    results = model.track(
        frame,
        persist=True
    )

    # Process results
    for result in results:

        # Make sure OBB detections exist
        if result.obb is None:
            continue

        # Get OBB corner coordinates
        corners_list = result.obb.xyxyxyxy.cpu().numpy()

        # Get class IDs
        class_ids = result.obb.cls.int().cpu().numpy()

        # Tracking IDs may be None
        if result.obb.id is not None:
            track_ids = result.obb.id.int().cpu().numpy()
        else:
            track_ids = [None] * len(corners_list)

        # Process every detected object
        for corners, track_id, cls in zip(
            corners_list,
            track_ids,
            class_ids
        ):

            # --------------------------------
            # Draw oriented bounding box
            # --------------------------------

            pts = corners.astype(np.int32)
            pts = pts.reshape((-1, 1, 2))

            cv2.polylines(
                frame,
                [pts],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )


            # --------------------------------
            # Calculate center of object
            # --------------------------------

            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


            # --------------------------------
            # Get class name
            # --------------------------------

            class_name = model.names[int(cls)]


            # --------------------------------
            # Create label
            # --------------------------------

            if track_id is not None:
                label = f"ID: {int(track_id)} | {class_name}"
            else:
                label = f"{class_name}"


            # --------------------------------
            # Draw label
            # --------------------------------

            text_x = int(corners[0][0])
            text_y = int(corners[0][1])

            cv2.putText(
                frame,
                label,
                (text_x, text_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


            # --------------------------------
            # Draw center coordinates
            # --------------------------------

            coordinate_text = f"({center_x}, {center_y})"

            cv2.putText(
                frame,
                coordinate_text,
                (center_x + 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )


    # Display frame
    cv2.imshow(
        "Object Detection",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Clean up
cap.release()
cv2.destroyAllWindows()