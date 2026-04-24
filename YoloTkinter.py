#YoloTkinter.py

import tkinter as tk
from tkinter import ttk, messagebox
import datetime
import threading
import csv
import os
import cv2
from PIL import Image, ImageTk

try:
    import YOLOnewTest as yolo_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

#paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(BASE_DIR, "package_photos")
CSV_PATH   = os.path.join(BASE_DIR, "package_history.csv")
os.makedirs(PHOTOS_DIR, exist_ok=True)

#size labels mapped to short names used in filenames
SIZE_OPTIONS = {
    "Small  (4×4×4 – 8×8×8)":        "Small",
    "Medium (10×8×6 – 12×12×12)":     "Medium",
    "Large  (14×14×14 – 18×12×12)":   "Large",
}
SIZE_KEYS = list(SIZE_OPTIONS.keys())

#csv columns
CSV_COLS = ["id", "size", "expected_date", "arrived", "photo_path"]

def _load_csv():
    if not os.path.exists(CSV_PATH):
        return []
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["arrived"] = row["arrived"].lower() == "true"
            rows.append(row)
    return rows

def _save_csv(packages):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        for p in packages:
            writer.writerow({**p, "arrived": str(p["arrived"])})

def _next_id(packages):
    if not packages:
        return "1"
    return str(max(int(p["id"]) for p in packages) + 1)

def _save_photo(frame, size_short, date_str):
    safe_date = date_str.replace("-", "")
    filename  = f"{safe_date}_{size_short}.jpg"
    full_path = os.path.join(PHOTOS_DIR, filename)
    cv2.imwrite(full_path, frame)
    return full_path


#window
root = tk.Tk()
root.title("Package Delivery Tracker")
root.geometry("520x580")
root.resizable(False, False)

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)


#tab 1 — tracker
tab_tracker = tk.Frame(notebook)
notebook.add(tab_tracker, text="Tracker")

tk.Label(tab_tracker, text="Box Size:").pack(pady=(14, 2))
size_var = tk.StringVar(value=SIZE_KEYS[1])
size_frame = tk.Frame(tab_tracker)
size_frame.pack()
for label in SIZE_KEYS:
    tk.Radiobutton(size_frame, text=label, variable=size_var, value=label).pack(anchor="w", padx=8)

tk.Label(tab_tracker, text="Expected Delivery Date (DD-MM-YYYY):").pack(pady=(8, 2))
date_entry = tk.Entry(tab_tracker, width=20)
date_entry.insert(0, datetime.date.today().strftime("%d-%m-%Y"))
date_entry.pack()
tk.Button(tab_tracker, text="Add Package", command=lambda: add_package()).pack(pady=(8, 2))

list_frame = tk.Frame(tab_tracker)
list_frame.pack(fill="both", expand=True, padx=10, pady=4)
scrollbar = tk.Scrollbar(list_frame)
scrollbar.pack(side="right", fill="y")
pkg_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=7, selectmode="single")
pkg_listbox.pack(fill="both", expand=True)
scrollbar.config(command=pkg_listbox.yview)

btn_frame = tk.Frame(tab_tracker)
btn_frame.pack(pady=2)
tk.Button(btn_frame, text="Remove Package",  command=lambda: remove_package()).pack(side="left", padx=6)
tk.Button(btn_frame, text="Mark as Arrived", command=lambda: mark_arrived()).pack(side="left", padx=6)

cam_btn = tk.Button(tab_tracker, text="Start Webcam Watch", command=lambda: toggle_webcam())
cam_btn.pack(pady=(4, 2))

status_label = tk.Label(tab_tracker, text="Status: Idle", fg="gray")
status_label.pack(pady=(2, 8))


#tab 2 — history
tab_history = tk.Frame(notebook)
notebook.add(tab_history, text="History")

hist_frame = tk.Frame(tab_history)
hist_frame.pack(fill="both", expand=True, padx=8, pady=6)

hist_scroll = tk.Scrollbar(hist_frame)
hist_scroll.pack(side="right", fill="y")

hist_tree = ttk.Treeview(
    hist_frame,
    columns=("status", "size", "date", "photo"),
    show="headings",
    yscrollcommand=hist_scroll.set,
    height=16,
)
hist_tree.heading("status", text="Status")
hist_tree.heading("size",   text="Size")
hist_tree.heading("date",   text="Expected Date")
hist_tree.heading("photo",  text="Photo Saved")
hist_tree.column("status", width=80,  anchor="center")
hist_tree.column("size",   width=120, anchor="center")
hist_tree.column("date",   width=110, anchor="center")
hist_tree.column("photo",  width=160, anchor="w")
hist_tree.pack(fill="both", expand=True)
hist_scroll.config(command=hist_tree.yview)

hist_tree.tag_configure("arrived", foreground="green")
hist_tree.tag_configure("pending", foreground="red")
hist_tree.tag_configure("overdue", foreground="red")

preview_label = tk.Label(tab_history, text="Select a row to preview its photo.", fg="gray")
preview_label.pack(pady=(4, 2))
photo_preview = tk.Label(tab_history)
photo_preview.pack(pady=(0, 6))

def _show_photo(event):
    sel = hist_tree.selection()
    if not sel:
        return
    path = hist_tree.item(sel[0])["values"][3]
    if path and os.path.exists(str(path)):
        img = Image.open(path)
        img.thumbnail((240, 180))
        photo = ImageTk.PhotoImage(img)
        photo_preview.config(image=photo)
        photo_preview.image = photo
        preview_label.config(text=os.path.basename(str(path)))
    else:
        photo_preview.config(image="")
        photo_preview.image = None
        preview_label.config(text="No photo available.")

hist_tree.bind("<<TreeviewSelect>>", _show_photo)


#state
packages       = _load_csv()
cam_running    = False
cam_thread     = None
dialog_open    = False
_listbox_order = []


#helpers
def _short_size(label):
    return SIZE_OPTIONS.get(label, label)

def _today():
    return datetime.date.today()

def _parse_date(s):
    try:
        return datetime.datetime.strptime(s, "%d-%m-%Y").date()
    except ValueError:
        return None


#tracker actions
def add_package():
    date_str = date_entry.get().strip()
    if not _parse_date(date_str):
        messagebox.showwarning("Bad Date", "Date must be DD-MM-YYYY.")
        return
    pkg = {
        "id":            _next_id(packages),
        "size":          _short_size(size_var.get()),
        "expected_date": date_str,
        "arrived":       False,
        "photo_path":    "",
    }
    packages.append(pkg)
    _save_csv(packages)
    refresh_listbox()
    refresh_history()

def remove_package():
    sel = pkg_listbox.curselection()
    if not sel:
        messagebox.showwarning("None Selected", "Select a package to remove.")
        return
    packages.pop(_listbox_order[sel[0]])
    _save_csv(packages)
    refresh_listbox()
    refresh_history()

def mark_arrived():
    sel = pkg_listbox.curselection()
    if not sel:
        messagebox.showwarning("None Selected", "Select a package to mark as arrived.")
        return
    packages[_listbox_order[sel[0]]]["arrived"] = True
    _save_csv(packages)
    refresh_listbox()
    refresh_history()

def refresh_listbox():
    global _listbox_order
    pkg_listbox.delete(0, tk.END)
    today = _today()

    def sort_key(p):
        d    = _parse_date(p["expected_date"]) or datetime.date.max
        late = d < today and not p["arrived"]
        return (0 if late else 1 if not p["arrived"] else 2, d)

    ordered        = sorted(range(len(packages)), key=lambda i: sort_key(packages[i]))
    _listbox_order = ordered

    for i, idx in enumerate(ordered):
        pkg     = packages[idx]
        check   = "[✔]" if pkg["arrived"] else "[ ]"
        d       = _parse_date(pkg["expected_date"]) or datetime.date.max
        overdue = d < today and not pkg["arrived"]
        tag     = " ⚠ LATE" if overdue else ""
        pkg_listbox.insert(tk.END, f"{check}  {pkg['size']}  |  {pkg['expected_date']}{tag}")
        if pkg["arrived"]:
            pkg_listbox.itemconfig(i, fg="green")
        elif overdue:
            pkg_listbox.itemconfig(i, fg="red")

    total   = len(packages)
    arrived = sum(1 for p in packages if p["arrived"])
    status_label.config(
        text=f"Total: {total}  |  ✔ Arrived: {arrived}  |  ⏳ Pending: {total - arrived}",
        fg="gray"
    )

def refresh_history():
    for row in hist_tree.get_children():
        hist_tree.delete(row)
    today = _today()

    #pending first, then arrived — each sorted latest date first
    pending = sorted([p for p in packages if not p["arrived"]], key=lambda p: _parse_date(p["expected_date"]) or datetime.date.min, reverse=True)
    arrived = sorted([p for p in packages if     p["arrived"]], key=lambda p: _parse_date(p["expected_date"]) or datetime.date.min, reverse=True)

    for p in pending:
        d      = _parse_date(p["expected_date"]) or datetime.date.max
        late   = d < today
        status = "⚠ LATE" if late else "Pending"
        tag    = "overdue" if late else "pending"
        hist_tree.insert("", "end", values=(status, p["size"], p["expected_date"], p["photo_path"]), tags=(tag,))

    for p in arrived:
        hist_tree.insert("", "end", values=("✔ Arrived", p["size"], p["expected_date"], p["photo_path"]), tags=("arrived",))


#detection dialog
def on_package_detected(snapshot, predicted_size_short):
    global dialog_open
    if dialog_open:
        return
    dialog_open = True
    root.after(0, lambda: show_confirm_dialog(snapshot, predicted_size_short))

def show_confirm_dialog(snapshot, predicted_size_short):
    global dialog_open
    pending = [i for i, p in enumerate(packages) if not p["arrived"]]
    if not pending:
        dialog_open = False
        return

    win = tk.Toplevel(root)
    win.title("Package Detected")
    win.grab_set()

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
    no_btn  = tk.Button(btn_row, text="No",  width=8, command=on_no)
    no_btn.pack(side="left", padx=8)

    #predicted size listed first
    match_frame = tk.Frame(win)
    tk.Label(match_frame, text="Which package is this?").pack(pady=(2, 2))

    sorted_pending = sorted(pending, key=lambda i: 0 if packages[i]["size"] == predicted_size_short else 1)
    match_var      = tk.IntVar(value=sorted_pending[0])
    for i in sorted_pending:
        p    = packages[i]
        hint = "  ← predicted" if p["size"] == predicted_size_short else ""
        tk.Radiobutton(match_frame, text=f"{p['size']}  |  {p['expected_date']}{hint}",
                       variable=match_var, value=i).pack(anchor="w", padx=16)

    def confirm_match():
        global dialog_open
        idx  = match_var.get()
        pkg  = packages[idx]
        path = _save_photo(snapshot, pkg["size"], pkg["expected_date"])
        pkg["arrived"]    = True
        pkg["photo_path"] = path
        _save_csv(packages)
        refresh_listbox()
        refresh_history()
        dialog_open = False
        win.destroy()

    tk.Button(match_frame, text="Confirm", command=confirm_match).pack(pady=6)


#webcam helpers
def get_box_center(results, names):
    for b in results[0].boxes:
        if names[int(b.cls[0])].lower() in {"package", "box", "parcel"}:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            return ((x1 + x2) / 2, (y1 + y2) / 2)
    return None

def _estimate_size(results, names, frame_h, frame_w):
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


#webcam loop
def webcam_loop():
    import time
    from ultralytics import YOLO as _YOLO

    weights_path = yolo_detector.find_best_existing_model()
    if weights_path is None:
        weights_path = f"yolo11{yolo_detector.model_tier}.pt"

    yolo_model         = _YOLO(weights_path)
    cap                = cv2.VideoCapture(0)
    stable_since       = None
    last_center        = None
    notified           = False
    last_notified_size = None
    move_threshold     = 60
    required_seconds   = 2

    while cam_running:
        ret, frame = cap.read()
        if not ret:
            break
        h, w      = frame.shape[:2]
        results   = yolo_model(frame, verbose=False, conf=0.60, iou=0.60)
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
            if dx > move_threshold or dy > move_threshold:
                last_center  = center
                stable_since = time.time()
                notified     = False
            elif not notified and (time.time() - stable_since) >= required_seconds:
                pred_size = _estimate_size(results, yolo_model.names, h, w)
                if pred_size != last_notified_size:
                    notified           = True
                    last_notified_size = pred_size
                    on_package_detected(annotated, pred_size)

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


#boot
refresh_listbox()
refresh_history()
root.mainloop()