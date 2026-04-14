from ultralytics import YOLO
import os
import glob
import multiprocessing
import cv2
import json
import urllib.request
import urllib.error

#Configuration
data_path  = "C:/Users/Nick/Downloads/packages.v2i.yolov11/data.yaml"
test_image = "C:/Users/Nick/Downloads/packages.v2i.yolov11/test/images/img--69-_jpg.rf.f6e45cd71fd9e176d194a3dedfa6704a.jpg"

#Folder downloaded model weights are cached so don't re-download them
model_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

#Model
model = ["yolo11"]

#Size model to download: "n"=nano, "s"=small, "m"=medium, "l"=large, "x"=xlarge
model_tier = "m"

#Epoch counts
epoch_number = 75

#Set True to skip training and go straight to the webcam demo
webcam_demo_mode = False

#File that saves mAP scores so we never re-validate a run we already scored
scores_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "epoch_scores.json")


#GitHub Model Downloader
#Checks latest Ultralytics GitHub downloads the best matching model
def fetch_best_yolo_from_github(tier=model_tier, yolo=model, cache_dir=model_cache_dir):

    os.makedirs(cache_dir, exist_ok=True)
    print("[model] Checking GitHub for the latest Ultralytics model...")

    api_url = "https://api.github.com/repos/ultralytics/assets/releases/latest"

    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "yolo-trainer-script"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            release_data = json.loads(resp.read().decode())

        #Build a dict of {filename: download_url} for all .pt files in the release
        assets = {
            a["name"]: a["browser_download_url"]
            for a in release_data.get("assets", [])
            if a["name"].endswith(".pt")
        }

        print(f"[model] Found {len(assets)} .pt files in release: {release_data.get('tag_name', 'unknown')}")

        #Try each preferred version in order and grab the first match
        for version in yolo:
            target_name = f"{version}{tier}.pt"
            if target_name in assets:
                local_path = os.path.join(cache_dir, target_name)

                #Use the cached file if it already exists
                if os.path.exists(local_path):
                    print(f"[model] Using cached model: {local_path}")
                    return local_path, target_name

                #Download from GitHub and save locally
                print(f"[model] Downloading {target_name}...")
                urllib.request.urlretrieve(assets[target_name], local_path)
                print(f"[model] Saved to: {local_path}")
                return local_path, target_name

        print(f"[model] No match found for tier '{tier}'. Available: {list(assets.keys())}")

    except urllib.error.URLError as e:
        print(f"[model] GitHub unreachable: {e}")
    except Exception as e:
        print(f"[model] Download failed: {e}")

    #Fallback — let Ultralytics handle the download automatically
    fallback = "yolo11m.pt"
    print(f"[model] Falling back to: {fallback}")
    return fallback, fallback

    
def train_model_from_scratch(epochs, yolo_model_path, data_path):
    try:
        # Start COMPLETELY from scratch    
        base_model = yolo_model_path

        print(f"[epochs={epochs}] Starting fresh training with: {base_model}")
        yolo_model = YOLO(base_model)
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
            print(f"[epochs={epochs}] Training done but weights not found — skipping.")            
            return
        
        #Validate the trained model and get its mAP score
        print(f"[epochs={epochs}] Validating: {weights_path}")
        metrics = YOLO(weights_path).val()
        current_map = metrics.box.map
        print(f"[epochs={epochs}] mAP50-95: {current_map:.4f}")

        save_score(epochs, weights_path, current_map)
            
    except Exception as e:
        print(f"[epochs={epochs}] Error: {e}")
        
if __name__ == "__main__":

    #Download the best available YOLO base model from GitHub
    yolo_model_path, yolo_model_name = fetch_best_yolo_from_github()
    print(f"\n[model] Using: {yolo_model_name} → {yolo_model_path}\n")    
    train_model_from_scratch(epoch_number, yolo_model_path, data_path)
