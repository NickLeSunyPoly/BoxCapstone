from ultralytics import YOLO
import os
import glob
import json
import urllib.request

#Configuration
data_path = "C:/Users/Nick/Downloads/packages.v2i.yolov11/data.yaml"
model_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
model_tier = "m"
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
                return local_path
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

#Trains the model for the given epoch count, validates, and returns the result
def train_model(epochs, yolo_model_path):
    try:
        existing_weights = find_existing_run(epochs)
        if existing_weights:
            weights_path = existing_weights
        else:
            yolo_model = YOLO(yolo_model_path)
            yolo_model.train(
            data=data_path,
            epochs=epochs,
            imgsz=640,
            device=0,      #notes for my self
            batch=8,       #Small batch size so we don't run out of VRAM on the 6GB GPU
            cache="disk",  #Caches images to disk instead of RAM — avoids memory warning
            amp=True,      #Mixed precision training — nearly 2x faster with no accuracy loss
            workers=2,     #Reduced to 2 so CPU doesn't push too much into VRAM at once
            mosaic=1.0,    #Combines 4 images — helps detect objects at varied sizes
            flipud=0.5,    #Random vertical flip for more angle variety
            fliplr=0.5,    #Random horizontal flip
            degrees=15.0,  #Random rotation up to ±15 degrees
            scale=0.5,     #Random zoom augmentation
            plots=True,    #Plot useful data
            name=f"train_epochs{epochs}"
            )
            weights_path = os.path.join("runs/detect", f"train_epochs{epochs}", "weights", "best.pt")
            if not os.path.exists(weights_path):
                print(f"[ERROR] epochs={epochs}: weights not found after training.")
                return epochs, None, -1
        metrics = YOLO(weights_path).val()
        current_map = metrics.box.map
        save_score(epochs, weights_path, current_map)
        return epochs, weights_path, current_map
    except Exception as e:
        print(f"[ERROR] epochs={epochs}: {e}")
        return epochs, None, -1

if __name__ == "__main__":
    yolo_model_path = fetch_best_yolo_from_github()
    scores = load_scores()
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