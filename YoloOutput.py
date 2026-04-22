#YOLOnewTest.py

from ultralytics import YOLO
import os
import sys
import glob
import json
import subprocess
import cv2

#Configuration
model_tier = "m"
scores_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "epoch_scores.json")

#Returns the best model weights by mAP score if available, otherwise most recent
def find_best_existing_model(base_path=None):
    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            scores = json.load(f)
        if scores:
            best_key = max(scores, key=lambda k: scores[k]["map"])
            entry = scores[best_key]
            if os.path.exists(entry["weights"]):
                return entry["weights"]
    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")
    matches = glob.glob(os.path.join(base_path, "train_epochs*", "weights", "best.pt"))
    if matches:
        return max(matches, key=os.path.getmtime)
    return None

#Watches webcam and fires on_detected(frame) when a package is seen for 5 consecutive frames
def watch_for_box(on_detected=None, is_running=None, cam_index=0):
    weights_path = find_best_existing_model()
    if weights_path is None:
        weights_path = f"yolo11{model_tier}.pt"
    yolo_model = YOLO(weights_path)
    cap = cv2.VideoCapture(cam_index)
    consecutive = 0
    notified = False
    while True:
        if is_running and not is_running():
            break
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame, verbose=False, conf=0.60, iou=0.60)
        box_found = any(
            yolo_model.names[int(b.cls[0])].lower() in {"package", "box", "parcel"}
            for b in results[0].boxes
        )
        if box_found:
            consecutive += 1
        else:
            consecutive = 0
            notified = False
        if consecutive >= 5 and not notified:
            notified = True
            consecutive = 0
            if on_detected:
                snapshot = results[0].plot()
                on_detected(snapshot)
    cap.release()

#Launches YOLOTraining.py as a separate process
def launch_training():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YOLOTraining.py")
    subprocess.run([sys.executable, script], check=True)

#Launches YoloTkinter.py as a separate process
def launch_gui():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YoloTkinter.py")
    subprocess.run([sys.executable, script], check=True)

if __name__ == "__main__":
    launch_training()
    launch_gui()