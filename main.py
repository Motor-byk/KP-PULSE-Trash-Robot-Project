import cv2
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

def detect_objects(frame):
    results = model(frame)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            detected_objects.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(confidence),
                "class_id": class_id,
                "class_name": model.names[class_id]
            })
    return results, detected_objects

cap = cv2.VideoCapture(0);
while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        resized_frame = cv2.resize(frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    results, detected_objects = detect_objects(resized_frame)

    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        confidence = obj["confidence"]
        class_id = obj["class_id"]
        class_name = obj["class_name"]
        label = f"{class_name}: {confidence:.2f}"
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
