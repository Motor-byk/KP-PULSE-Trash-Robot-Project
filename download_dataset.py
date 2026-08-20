import os
from dotenv import load_dotenv
from ultralytics import YOLO
from roboflow import Roboflow

Roboflow_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=Roboflow_key)
project = rf.workspace("trash-dataset-for-oriented-bounded-box").project("trash-detection-1fjjc")
version = project.version(13)
dataset = version.download("yolov11")
