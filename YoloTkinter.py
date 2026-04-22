#YoloTkinter.py

import tkinter as tk
from tkinter import messagebox
import datetime
import threading
import cv2
from PIL import Image, ImageTk

try:
    import YOLOnewTest as yolo_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

root = tk.Tk()
root.title("Package Delivery Tracker")
root.geometry("480x500")
root.resizable(False, False)

#Top: add package controls
tk.Label(root, text="Box Size:").pack(pady=(14, 2))
size_var = tk.StringVar(value="Medium")
size_frame = tk.Frame(root)
size_frame.pack()
for sz in ("Small", "Medium", "Large"):
    tk.Radiobutton(size_frame, text=sz, variable=size_var, value=sz).pack(side="left", padx=8)

tk.Label(root, text="Expected Delivery Date (DD-MM-YYYY):").pack(pady=(8, 2))
date_entry = tk.Entry(root, width=20)
date_entry.insert(0, datetime.date.today().strftime("%d-%m-%Y"))
date_entry.pack()
tk.Button(root, text="Add Package", command=lambda: add_package()).pack(pady=(8, 2))

#Package list
list_frame = tk.Frame(root)
list_frame.pack(fill="both", expand=True, padx=10, pady=4)
scrollbar = tk.Scrollbar(list_frame)
scrollbar.pack(side="right", fill="y")
pkg_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8, selectmode="single")
pkg_listbox.pack(fill="both", expand=True)
scrollbar.config(command=pkg_listbox.yview)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=2)
tk.Button(btn_frame, text="Remove Package", command=lambda: remove_package()).pack(side="left", padx=6)
tk.Button(btn_frame, text="Mark as Arrived", command=lambda: mark_arrived()).pack(side="left", padx=6)

#Webcam toggle
cam_btn = tk.Button(root, text="Start Webcam Watch", command=lambda: toggle_webcam())
cam_btn.pack(pady=(4, 2))

status_label = tk.Label(root, text="Status: Idle", fg="gray")
status_label.pack(pady=(2, 8))

#Internal state
packages = []
cam_running = False
cam_thread = None
dialog_open = False

def add_package():
    date_str = date_entry.get().strip()
    size = size_var.get()
    try:
        datetime.datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        messagebox.showwarning("Bad Date", "Date must be DD-MM-YYYY.")
        return
    packages.append({"size": size, "date": date_str, "arrived": False})
    refresh_listbox()

def remove_package():
    sel = pkg_listbox.curselection()
    if not sel:
        messagebox.showwarning("None Selected", "Select a package to remove.")
        return
    packages.pop(sel[0])
    refresh_listbox()

def mark_arrived():
    sel = pkg_listbox.curselection()
    if not sel:
        messagebox.showwarning("None Selected", "Select a package to mark as arrived.")
        return
    packages[sel[0]]["arrived"] = True
    refresh_listbox()

def refresh_listbox():
    pkg_listbox.delete(0, tk.END)
    for pkg in packages:
        check = "[✔]" if pkg["arrived"] else "[ ]"
        pkg_listbox.insert(tk.END, f"{check}  {pkg['size']}  |  {pkg['date']}")
    total = len(packages)
    arrived = sum(1 for p in packages if p["arrived"])
    status_label.config(
        text=f"Total: {total}  |  ✔ Arrived: {arrived}  |  ⏳ Pending: {total - arrived}",
        fg="gray"
    )

#Called from background thread when a package is detected — marshalled to main thread
def on_package_detected(snapshot):
    global dialog_open
    if dialog_open:
        return
    dialog_open = True
    root.after(0, lambda: show_confirm_dialog(snapshot))

#Shows snapshot and asks user to confirm and match to a package
def show_confirm_dialog(snapshot):
    pending = [i for i, p in enumerate(packages) if not p["arrived"]]
    if not pending:
        return

    win = tk.Toplevel(root)
    win.title("Package Detected")
    win.grab_set()

    #Show snapshot
    img = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.thumbnail((360, 270))
    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(win, image=photo)
    img_label.image = photo
    img_label.pack(padx=10, pady=(10, 4))

    tk.Label(win, text="Is this one of your packages?").pack(pady=(4, 2))

    def on_yes():
        yes_btn.config(state="disabled")
        no_btn.config(state="disabled")
        match_frame.pack(pady=6)

    def on_no():
        global dialog_open
        dialog_open = False
        win.destroy()

    btn_row = tk.Frame(win)
    btn_row.pack()
    yes_btn = tk.Button(btn_row, text="Yes", width=8, command=on_yes)
    yes_btn.pack(side="left", padx=8)
    no_btn = tk.Button(btn_row, text="No", width=8, command=on_no)
    no_btn.pack(side="left", padx=8)

    #Match to list item
    match_frame = tk.Frame(win)
    tk.Label(match_frame, text="Which package is this?").pack(pady=(2, 2))
    match_var = tk.IntVar(value=pending[0])
    for i in pending:
        p = packages[i]
        tk.Radiobutton(match_frame, text=f"{p['size']}  |  {p['date']}", variable=match_var, value=i).pack(anchor="w", padx=16)

    def confirm_match():
        global dialog_open
        packages[match_var.get()]["arrived"] = True
        refresh_listbox()
        dialog_open = False
        win.destroy()

    tk.Button(match_frame, text="Confirm", command=confirm_match).pack(pady=6)

#Returns center (cx, cy) of the first matching box, or None
def get_box_center(results, names):
    for b in results[0].boxes:
        if names[int(b.cls[0])].lower() in {"package", "box", "parcel"}:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            return ((x1 + x2) / 2, (y1 + y2) / 2)
    return None

#Background loop — runs YOLO detection and shows live annotated feed in a cv2 window
def webcam_loop():
    import time
    from ultralytics import YOLO as _YOLO
    weights_path = yolo_detector.find_best_existing_model()
    if weights_path is None:
        weights_path = f"yolo11{yolo_detector.model_tier}.pt"
    yolo_model = _YOLO(weights_path)
    cap = cv2.VideoCapture(0)
    stable_since = None
    last_center = None
    notified = False
    #Box must stay within this many pixels of its original position to count as stable
    move_threshold = 60
    #Seconds the box must remain stable before triggering
    required_seconds = 5
    while cam_running:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame, verbose=False, conf=0.60, iou=0.60)
        annotated = results[0].plot()
        cv2.imshow("Package Detector — Live Feed", annotated)
        cv2.waitKey(1)
        center = get_box_center(results, yolo_model.names)
        if center is None:
            #No box — reset
            stable_since = None
            last_center = None
            notified = False
        elif last_center is None:
            #First detection — start tracking
            last_center = center
            stable_since = time.time()
        else:
            dx = abs(center[0] - last_center[0])
            dy = abs(center[1] - last_center[1])
            if dx > move_threshold or dy > move_threshold:
                #Box moved too much — reset timer
                last_center = center
                stable_since = time.time()
                notified = False
            elif not notified and (time.time() - stable_since) >= required_seconds:
                notified = True
                on_package_detected(annotated)
    cap.release()
    cv2.destroyAllWindows()
    root.after(0, lambda: status_label.config(text="Status: Webcam stopped", fg="gray"))

def toggle_webcam():
    global cam_running, cam_thread
    if not YOLO_AVAILABLE:
        messagebox.showerror("Missing Module", "YOLOnewTest.py could not be imported.")
        return
    if not cam_running:
        cam_running = True
        cam_btn.config(text="Stop Webcam Watch")
        status_label.config(text="Status: Watching…", fg="green")
        cam_thread = threading.Thread(target=webcam_loop, daemon=True)
        cam_thread.start()
    else:
        cam_running = False
        cam_btn.config(text="Start Webcam Watch")

root.mainloop()