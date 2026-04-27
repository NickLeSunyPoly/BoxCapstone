#YoloOutput.py

import os
import sys
import glob
import json
import subprocess
import time
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


#Returns the (cx, cy) center of the first detected package box, or None
def get_box_center(results, names):
    best = None
    best_conf = 0
    for b in results[0].boxes:
        if names[int(b.cls[0])].lower() in {"package", "box", "parcel"}:
            conf = float(b.conf[0])
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                best = ((x1 + x2) / 2, (y1 + y2) / 2)
    return best


#Estimates package size from bounding-box area relative to frame size
def estimate_size(results, names, frame_h, frame_w):
    for b in results[0].boxes:
        if names[int(b.cls[0])].lower() in {"package", "box", "parcel"}:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)
            if area_ratio < 0.10:
                return "Small"
            elif area_ratio < 0.25:
                return "Medium"
            else:
                return "Large"
    return "Medium"


#Runs the webcam detection loop in a background thread.
#
#on_detected(frame, predicted_size_short) is called on the calling thread
#via root.after — pass a wrapper that does root.after(0, ...) if needed,
#or pass a plain callable; PackageWatcher does not touch Tkinter directly.
#
#is_running_fn() should return True while the loop should keep running.
class PackageWatcher:
    def __init__(self, on_detected, is_running_fn,
                 move_threshold=60, required_seconds=2, conf=0.60, iou=0.45):
        self.on_detected      = on_detected
        self.is_running       = is_running_fn
        self.move_threshold   = move_threshold
        self.required_seconds = required_seconds
        self.conf             = conf
        self.iou              = iou

    def run(self):
        from ultralytics import YOLO as _YOLO

        weights_path = find_best_existing_model()
        if weights_path is None:
            weights_path = f"yolo11{model_tier}.pt"

        yolo_model         = _YOLO(weights_path)
        cap                = cv2.VideoCapture(0)
        stable_since       = None
        last_center        = None
        notified           = False
        last_notified_size = None

        while self.is_running():
            ret, frame = cap.read()
            if not ret:
                break
            h, w      = frame.shape[:2]
            results   = yolo_model(frame, verbose=False, conf=self.conf, iou=self.iou)
            annotated = results[0].plot()
            cv2.imshow("Package Detector — Live Feed", annotated)
            cv2.waitKey(1)
            center = get_box_center(results, yolo_model.names)

            if center is None:
                stable_since = None
                last_center  = None
                notified     = False
            elif last_center is None:
                last_center  = center
                stable_since = time.time()
            else:
                dx = abs(center[0] - last_center[0])
                dy = abs(center[1] - last_center[1])
                if dx > self.move_threshold or dy > self.move_threshold:
                    last_center  = center
                    stable_since = time.time()
                    notified     = False
                elif not notified and (time.time() - stable_since) >= self.required_seconds:
                    pred_size = estimate_size(results, yolo_model.names, h, w)
                    if pred_size != last_notified_size:
                        notified           = True
                        last_notified_size = pred_size
                        self.on_detected(annotated, pred_size)

        cap.release()
        cv2.destroyAllWindows()


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