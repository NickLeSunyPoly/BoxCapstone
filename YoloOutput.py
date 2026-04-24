#YoloOutput.py

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
            entry    = scores[best_key]
            if os.path.exists(entry["weights"]):
                return entry["weights"]
    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")
    matches = glob.glob(os.path.join(base_path, "train_epochs*", "weights", "best.pt"))
    if matches:
        return max(matches, key=os.path.getmtime)
    return None

#Launch YOLOTraining.py
def launch_training():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YOLOTraining.py")
    subprocess.run([sys.executable, script], check=True)

#Launch YoloTkinter.py
def launch_gui():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YoloTkinter.py")
    subprocess.run([sys.executable, script], check=True)

if __name__ == "__main__":
    launch_training()
    launch_gui()