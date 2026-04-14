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
epoch_number = [100]

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


#Checkpoint Finder
#Scans every folder in runs/detect
#Run the best epoch model that is downloaded and complete
def find_best_checkpoint_under(target_epochs, base_path=None):

    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")

    best_path = None
    best_epochs_found = 0

    #Check every subfolder that has a best.pt weights file
    for weights_path in glob.glob(os.path.join(base_path, "*", "weights", "best.pt")):
        run_folder = os.path.dirname(os.path.dirname(weights_path))
        folder_name = os.path.basename(run_folder)

        #First try to read the epoch count from results.csv — works for ALL folder names
        found_epochs = get_completed_epochs(run_folder)

        #If results.csv didn't give us anything, fall back to reading the folder name
        if found_epochs == 0:
            for part in folder_name.replace("train_epochs", "").replace("train_e", "").split("_"):
                try:
                    found_epochs = int(part)
                    break
                except ValueError:
                    continue

        #Keep it only if it's under our target and better than what we already have
        if found_epochs > 0 and found_epochs < target_epochs and found_epochs > best_epochs_found:
            best_epochs_found = found_epochs
            best_path = weights_path

    if best_path:
        print(f"[epochs={target_epochs}] Best checkpoint found: {best_epochs_found} epochs at {best_path}")
        print(f"[epochs={target_epochs}] Using this as starting point — skipping ~{best_epochs_found} epochs of work")

    return best_path


#Resume Helper
#Reads results.csv from a training run to count how many epochs finished
#Each data row = one completed epoch Returns 0 if the file doesn't exist
def get_completed_epochs(run_folder):

    results_csv = os.path.join(run_folder, "results.csv")
    if not os.path.exists(results_csv):
        return 0
    with open(results_csv, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    return max(len(lines) - 1, 0)  #Subtract 1 for the header row


#Looks for a partially-finished training run that we can continue
#Needs a last.pt checkpoint and must have fewer epochs done than the target
#Returns (run_folder, completed_epochs, last_pt_path) or None
def find_resumable_run(target_epochs, base_path=None):

    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")

    for run_folder in sorted(glob.glob(os.path.join(base_path, f"train_epochs{target_epochs}*")), key=os.path.getmtime, reverse=True):
        last_pt = os.path.join(run_folder, "weights", "last.pt")
        if not os.path.exists(last_pt):
            continue

        completed = get_completed_epochs(run_folder)

        #Already finished, skip
        if completed >= target_epochs:
            continue
        
        #Checkpoint exists but nothing was saved — likely corrupted
        if completed == 0:
            continue

        print(f"[epochs={target_epochs}] Partial run found: {completed}/{target_epochs} epochs done")
        return run_folder, completed, last_pt

    return None


#Score Cache Helpers
#Load saved mAP scores from disk. Returns empty dict if none exist
def load_scores():

    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            return json.load(f)
    return {}

#Save a run's mAP score so we never re-validate it
def save_score(epochs, weights_path, map_score):

    scores = load_scores()
    scores[str(epochs)] = {"weights": weights_path, "map": map_score}
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[epochs={epochs}] Score saved.")


#Existing Run Check
#Looks for a completed training run (best.pt) for the given epoch count
#Returns the path to best.pt if found, or None if not
def find_existing_run(epochs, base_path=None):

    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")

    matches = glob.glob(os.path.join(base_path, f"train_epochs{epochs}*", "weights", "best.pt"))
    if matches:
        best = max(matches, key=os.path.getmtime)
        print(f"[epochs={epochs}] Found existing run: {best}")
        return best
    return None


#Finds the best model by checking saved mAP scores
#Falls back to the most recently modified run if no scores are saved
def find_best_existing_model(base_path=None):

    scores = load_scores()

    if scores:
        best_key = max(scores, key=lambda k: scores[k]["map"])
        entry = scores[best_key]
        if os.path.exists(entry["weights"]):
            print(f"[webcam] Best model: epochs={best_key}, mAP={entry['map']:.4f}")
            return entry["weights"]
        print("[webcam] Saved best weights missing, scanning for fallback...")

    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")

    matches = glob.glob(os.path.join(base_path, "train_epochs*", "weights", "best.pt"))
    if not matches:
        return None

    best = max(matches, key=os.path.getmtime)
    print(f"[webcam] Using most recent model: {best}")
    return best


#Webcam Live Demo
def run_webcam_demo(weights_path):

    print(f"\n{'='*30}\nWEBCAM DEMO\n{'='*30}")
    print(f"Model: {weights_path}\nPress Q to quit.\n")

    yolo_model = YOLO(weights_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Webcam running...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame — exiting.")
            break
        
        #rev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rev_frame = frame

        #Run detection — raise conf to reduce false positives
        min_conf = 0.60
        # min_conf = 0.75
        results = yolo_model(rev_frame, verbose=False, conf=min_conf, iou=0.60)
        
        
        
        annotated = results[0].plot()

        cv2.putText(annotated, "LIVE PACKAGE DETECTION  |  Press Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("YOLO Package Detector — Live Demo", annotated)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam demo ended.")


#Training Worker
def train_model(epochs, yolo_model_path, result_queue):

    try:
        scores = load_scores()

        #Already done — skip training and validation entirely
        if str(epochs) in scores:
            entry = scores[str(epochs)]
            if os.path.exists(entry["weights"]):
                print(f"[epochs={epochs}] Already complete (mAP={entry['map']:.4f}), skipping.")
                result_queue.put((epochs, entry["weights"], entry["map"]))
                return
            else:
                print(f"[epochs={epochs}] Score cached but weights missing — retraining...")

        #Resume a partial run using its last.pt checkpoint
        resume_info = find_resumable_run(epochs)
        if resume_info:
            run_folder, completed, last_pt = resume_info
            print(f"\n[epochs={epochs}] Resuming: {completed}/{epochs} epochs done, {epochs - completed} remaining...")

            #resume=True loads the checkpoint and continues from where it stopped
            yolo_model = YOLO(last_pt)
            yolo_model.train(resume=True)
            weights_path = os.path.join(run_folder, "weights", "best.pt")

        else:
            #Check for a finished run that just hasn't been scored yet
            existing_weights = find_existing_run(epochs)
    
            #Run is complete — skip training, just validate
            if existing_weights:
                weights_path = existing_weights
                print(f"[epochs={epochs}] Run complete, skipping to validation...")

            else:
                #No full run found — check if we have a partial checkpoint from a
                #lower epoch run we can build on
                checkpoint = find_best_checkpoint_under(epochs)
                base_model = checkpoint if checkpoint else yolo_model_path

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
                    result_queue.put((epochs, None, -1))
                    return

        #Validate the trained model and get its mAP score
        print(f"[epochs={epochs}] Validating: {weights_path}")
        metrics = YOLO(weights_path).val()
        current_map = metrics.box.map
        print(f"[epochs={epochs}] mAP50-95: {current_map:.4f}")

        save_score(epochs, weights_path, current_map)
        result_queue.put((epochs, weights_path, current_map))

    except Exception as e:
        print(f"[epochs={epochs}] Error: {e}")
        result_queue.put((epochs, None, -1))


if __name__ == "__main__":

    #Webcam demo mode — load best existing model and stream live
    if webcam_demo_mode:
        best_weights = find_best_existing_model()
        if best_weights is None:
            print("No trained model found. Set webcam_demo_mode = False and train first.")
        else:
            run_webcam_demo(best_weights)
        exit(0)

    #Download the best available YOLO base model from GitHub
    yolo_model_path, yolo_model_name = fetch_best_yolo_from_github()
    print(f"\n[model] Using: {yolo_model_name} → {yolo_model_path}\n")

    result_queue = multiprocessing.Queue()
    processes = []
    scores = load_scores()

    #Skip launching processes for runs that are already fully complete
    epochs_to_run = []
    for epochs in epoch_number:
        entry = scores.get(str(epochs))
        if entry and os.path.exists(entry["weights"]):
            print(f"[epochs={epochs}] Already complete (mAP={entry['map']:.4f}), skipping launch.")
            result_queue.put((epochs, entry["weights"], entry["map"]))
        else:
            epochs_to_run.append(epochs)

    print(f"Launching {len(epochs_to_run)} training run(s): {epochs_to_run}")
    print(f"Skipping {len(epoch_number) - len(epochs_to_run)} already-complete run(s).\n")

    for epochs in epochs_to_run:
        p = multiprocessing.Process(target=train_model, args=(epochs, yolo_model_path, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    #Collect all results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    #Print summary table
    print(f"\n{'='*30}\nALL RESULTS\n{'='*30}")
    for epochs, weights, map_score in sorted(results, key=lambda x: x[0]):
        status = f"mAP50-95: {map_score:.4f}" if map_score >= 0 else "FAILED"
        print(f"Epochs: {epochs:>3} → {status}")

    valid_results = [(e, w, m) for e, w, m in results if m >= 0]

    if not valid_results:
        print("No valid runs completed.")
    else:
        #Pick the run with the highest mAP score
        best_epoch, best_weights, best_map = max(valid_results, key=lambda x: x[2])

        print(f"\n{'='*50}\nBEST MODEL\n{'='*50}")
        print(f"Epochs  : {best_epoch}")
        print(f"mAP50-95: {best_map:.4f}")
        print(f"Weights : {best_weights}")

        #Run inference on the test image with the best model
        print(f"\nRunning inference on test image...")
        YOLO(best_weights)(test_image)[0].show()

        #Optionally launch the webcam demo
        if input("\nRun live webcam demo? (y/n): ").strip().lower() == "y":
            run_webcam_demo(best_weights)