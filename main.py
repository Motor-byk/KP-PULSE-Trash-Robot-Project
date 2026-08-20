import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/obb/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, persist=True)

    for result in results:
        
        if result.obb is not None and result.obb.id is not None:

            corners_list = result.obb.xyxyxyxy.cpu().numpy()

            track_ids = result.obb.id.int().cpu().numpy()

            class_ids = result.obb.cls.int().cpu().numpy()

            for corners, track_id, cls in zip(
                corners_list,
                track_ids,
                class_ids
            ):

                pts = corners.reshape(-1, 2).astype(np.int32)
                pts = pts.reshape((-1, 1, 2))

                cv2.polylines(
                    frame,
                    [pts],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2
                )

                text_x = int(corners[0][0])
                text_y = int(corners[0][1])

                label = f"ID: {track_id}, Class: {model.names[int(cls)]}"

                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()