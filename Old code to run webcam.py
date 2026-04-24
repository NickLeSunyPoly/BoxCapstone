#YOLOnewTest.py
from ultralytics import YOLO
import os
import glob
import cv2
import json
import urllib.request

#Configuration
data_path  = "C:/Users/Nick/Downloads/packages.v2i.yolov11/data.yaml"
test_image = "C:/Users/Nick/Downloads/packages.v2i.yolov11/test/images/img--69-_jpg.rf.f6e45cd71fd9e176d194a3dedfa6704a.jpg"

model_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

#tier of model: n=nano, s=small, m=medium, l=large, x=xlarge
model_tier = "m"

#Epoch counts
epoch_number = 75

scores_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "epoch_scores.json")


#Downloads the latest YOLO.pt model from GitHub, uses cache if already downloaded
def fetch_best_yolo_from_github(tier=model_tier, cache_dir=model_cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    api_url = "https://api.github.com/repos/ultralytics/assets/releases/latest"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "yolo-trainer-script"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            release_data = json.loads(resp.read().decode())

        assets = {
            a["name"]: a["browser_download_url"]
            for a in release_data.get("assets", [])
            if a["name"].endswith(".pt")
        }

        target_name = f"yolo11{tier}.pt"
        if target_name in assets:
            local_path = os.path.join(cache_dir, target_name)
            if os.path.exists(local_path):
                return local_path  # Already cached
            urllib.request.urlretrieve(assets[target_name], local_path)
            return local_path

    except Exception as e:
        print(f"[ERROR] GitHub model fetch failed: {e}")

    return f"yolo11{tier}.pt"


#Load saved mAP scores from disk
def load_scores():
    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            return json.load(f)
    return {}


#Save a run's mAP score so we don't re-validate it next time
def save_score(epochs, weights_path, map_score):
    scores = load_scores()
    scores[str(epochs)] = {"weights": weights_path, "map": map_score}
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)


#Returns the path to best.pt if a completed run exists for this epoch count
def find_existing_run(epochs, base_path=None):
    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")
    matches = glob.glob(os.path.join(base_path, f"train_epochs{epochs}*", "weights", "best.pt"))
    if matches:
        return max(matches, key=os.path.getmtime)
    return None


#Returns the best model weights — by mAP score if available, otherwise most recent
def find_best_existing_model(base_path=None):
    scores = load_scores()
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


#Live webcam detection loop — press Q or Esc to quit
def run_webcam_demo(weights_path):
    yolo_model = YOLO(weights_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        results   = yolo_model(frame, verbose=False, conf=0.60, iou=0.60)
        annotated = results[0].plot()

        cv2.putText(annotated, "LIVE PACKAGE DETECTION  |  Press Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("YOLO Package Detector", annotated)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


#Trains for the given epoch count, validates, and returns the result
def train_model(epochs, yolo_model_path):
    try:
        existing_weights = find_existing_run(epochs)

        if existing_weights:
            #Completed run found — skip straight to validation
            weights_path = existing_weights
        else:
            yolo_model = YOLO(yolo_model_path)
            yolo_model.train(
                data=data_path,
                epochs=epochs,
                imgsz=640,
                device=0,
                batch=8,       #Small batch to avoid VRAM overflow on 6GB GPU
                cache="disk",  #Cache images to disk instead of RAM
                amp=True,      #Mixed precision — faster with no accuracy loss
                workers=2,
                mosaic=1.0,    #Combines 4 images for varied scale detection
                flipud=0.5,
                fliplr=0.5,
                degrees=15.0,
                scale=0.5,
                plots=True,
                name=f"train_epochs{epochs}"
            )

            weights_path = os.path.join("runs/detect", f"train_epochs{epochs}", "weights", "best.pt")
            if not os.path.exists(weights_path):
                print(f"[ERROR] epochs={epochs}: weights not found after training.")
                return epochs, None, -1

        #Validate and record the mAP score
        metrics     = YOLO(weights_path).val()
        current_map = metrics.box.map

        save_score(epochs, weights_path, current_map)
        return epochs, weights_path, current_map

    except Exception as e:
        print(f"[ERROR] epochs={epochs}: {e}")
        return epochs, None, -1


if __name__ == "__main__":
    yolo_model_path = fetch_best_yolo_from_github()
    scores = load_scores()

    #Use cached result if already scored, otherwise train
    entry = scores.get(str(epoch_number))
    if entry and os.path.exists(entry["weights"]):
        print(f"[INFO] Using cached result for {epoch_number} epochs.")
        best_epoch, best_weights, best_map = epoch_number, entry["weights"], entry["map"]
    else:
        best_epoch, best_weights, best_map = train_model(epoch_number, yolo_model_path)

    if best_map < 0:
        print("[ERROR] Training did not complete successfully.")
    else:
        print(f"\nBest model — Epochs: {best_epoch} | mAP50-95: {best_map:.4f} | Weights: {best_weights}")

        # Run inference on the test image
        YOLO(best_weights)(test_image)[0].show()

        if input("\nRun live webcam demo? (y/n): ").strip().lower() == "y":
            run_webcam_demo(best_weights)


def watch_for_box(on_detected=None, is_running=None, log_callback=None, cam_index=0):
    weights_path = find_best_existing_model()
    if weights_path is None:
        weights_path = f"yolo11{model_tier}.pt"
    
    yolo_model = YOLO(weights_path)  # defined locally now — fixes the error
    
    cap = cv2.VideoCapture(cam_index)
    consecutive = 0
    notified    = False

    while True:
        if is_running and not is_running():
            break
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame, verbose=False, conf=0.60, iou=0.60)
        # check if any detection matches your box class
        box_found = any(
            yolo_model.names[int(b.cls[0])].lower() in {"package", "box", "parcel"}
            for b in results[0].boxes
        )

        if box_found:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= 5 and not notified:
            notified    = True
            consecutive = 0
            if on_detected:
                on_detected()

    cap.release()
    cv2.destroyAllWindows()