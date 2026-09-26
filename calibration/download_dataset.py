"""Download the labeled dataset from Roboflow.

Export the project in YOLOv11 *detect* format with augmentation turned off --
Ultralytics augments at train time, and baking a second layer in on top of that
is what degraded the previous dataset.
"""

import os

from dotenv import load_dotenv
from roboflow import Roboflow

# Must be called before os.getenv, or ROBOFLOW_API_KEY is only picked up when
# it already happens to be exported in the shell.
load_dotenv()

WORKSPACE = "your-workspace"       # set to your Roboflow workspace slug
PROJECT = "trash-bins"             # set to your new project slug
VERSION = 1
LOCATION = "datasets/trash-bins"   # keep in sync with DATA in train.py

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY not set -- add it to .env")

rf = Roboflow(api_key=api_key)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)
dataset = version.download("yolov11", location=LOCATION)

print(f"Downloaded to {dataset.location}")
