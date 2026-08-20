from ultralytics import YOLO

model = YOLO('yolo11s-obb.pt')
model.train(
    data='./Trash-Detection-13/data.yaml',  
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,  
)