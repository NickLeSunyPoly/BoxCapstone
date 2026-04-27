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
    import YoloOutput as yolo_detector
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

#csv columns — "status" replaces the old "arrived" bool
#status values: "pending" | "arrived" | "removed"
CSV_COLS = ["id", "size", "expected_date", "status", "photo_path"]

def _load_csv():
    if not os.path.exists(CSV_PATH):
        return []
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            #back-compat: old files used "arrived" bool column
            if "arrived" in row and "status" not in row:
                row["status"] = "arrived" if row["arrived"].lower() == "true" else "pending"
            rows.append(row)
    return rows

def _save_csv(packages):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        for p in packages:
            writer.writerow({k: p[k] for k in CSV_COLS})

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
root.geometry("520x600")
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

#filter bar — All / Late / Arrived / Removed
filter_var = tk.StringVar(value="All")
filter_bar = tk.Frame(tab_history)
filter_bar.pack(fill="x", padx=8, pady=(6, 2))
for label in ("All", "Late", "Arrived", "Removed"):
    tk.Radiobutton(
        filter_bar, text=label, variable=filter_var, value=label,
        indicatoron=False, width=8, relief="groove",
        command=lambda: refresh_history()
    ).pack(side="left", padx=2)

hist_frame = tk.Frame(tab_history)
hist_frame.pack(fill="both", expand=True, padx=8, pady=4)

hist_scroll = tk.Scrollbar(hist_frame)
hist_scroll.pack(side="right", fill="y")

hist_tree = ttk.Treeview(
    hist_frame,
    columns=("status", "size", "date", "photo"),
    show="headings",
    yscrollcommand=hist_scroll.set,
    height=14,
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
hist_tree.tag_configure("pending", foreground="orange")
hist_tree.tag_configure("overdue", foreground="red")
hist_tree.tag_configure("removed", foreground="gray")

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

def _is_active(p):
    return p["status"] != "removed"


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
        "status":        "pending",
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
    #mark as removed rather than deleting — keeps it in history
    packages[_listbox_order[sel[0]]]["status"] = "removed"
    _save_csv(packages)
    refresh_listbox()
    refresh_history()

def mark_arrived():
    sel = pkg_listbox.curselection()
    if not sel:
        messagebox.showwarning("None Selected", "Select a package to mark as arrived.")
        return
    packages[_listbox_order[sel[0]]]["status"] = "arrived"
    _save_csv(packages)
    refresh_listbox()
    refresh_history()

def refresh_listbox():
    global _listbox_order
    pkg_listbox.delete(0, tk.END)
    today = _today()

    #only show active (non-removed) packages in the tracker list
    active = [i for i, p in enumerate(packages) if _is_active(p)]

    def sort_key(i):
        p    = packages[i]
        d    = _parse_date(p["expected_date"]) or datetime.date.max
        late = d < today and p["status"] == "pending"
        return (0 if late else 1 if p["status"] == "pending" else 2, d)

    ordered        = sorted(active, key=sort_key)
    _listbox_order = ordered

    for i, idx in enumerate(ordered):
        pkg     = packages[idx]
        check   = "[✔]" if pkg["status"] == "arrived" else "[ ]"
        d       = _parse_date(pkg["expected_date"]) or datetime.date.max
        overdue = d < today and pkg["status"] == "pending"
        tag     = " ⚠ LATE" if overdue else ""
        pkg_listbox.insert(tk.END, f"{check}  {pkg['size']}  |  {pkg['expected_date']}{tag}")
        if pkg["status"] == "arrived":
            pkg_listbox.itemconfig(i, fg="green")
        elif overdue:
            pkg_listbox.itemconfig(i, fg="red")

    total   = len(active)
    arrived = sum(1 for i in active if packages[i]["status"] == "arrived")
    status_label.config(
        text=f"Total: {total}  |  ✔ Arrived: {arrived}  |  ⏳ Pending: {total - arrived}",
        fg="gray"
    )

def refresh_history():
    for row in hist_tree.get_children():
        hist_tree.delete(row)
    today = _today()
    filt  = filter_var.get()

    def _include(p):
        s = p["status"]
        d = _parse_date(p["expected_date"]) or datetime.date.max
        if filt == "All":     return True
        if filt == "Arrived": return s == "arrived"
        if filt == "Removed": return s == "removed"
        if filt == "Late":    return s == "pending" and d < today
        return True

    #order: late → pending → arrived → removed, each by date desc
    def _order_key(p):
        d    = _parse_date(p["expected_date"]) or datetime.date.min
        s    = p["status"]
        late = s == "pending" and d < today
        rank = 0 if late else 1 if s == "pending" else 2 if s == "arrived" else 3
        return (rank, d)

    filtered = sorted([p for p in packages if _include(p)], key=_order_key)

    for p in filtered:
        s = p["status"]
        d = _parse_date(p["expected_date"]) or datetime.date.max
        if s == "arrived":
            label, tag = "✔ Arrived", "arrived"
        elif s == "removed":
            label, tag = "✖ Removed", "removed"
        elif d < today:
            label, tag = "⚠ LATE",    "overdue"
        else:
            label, tag = "Pending",   "pending"
        hist_tree.insert("", "end", values=(label, p["size"], p["expected_date"], p["photo_path"]), tags=(tag,))


#detection dialog — called via root.after from the webcam thread
def on_package_detected(snapshot, predicted_size_short):
    global dialog_open
    if dialog_open:
        return
    dialog_open = True
    root.after(0, lambda: show_confirm_dialog(snapshot, predicted_size_short))

def show_confirm_dialog(snapshot, predicted_size_short):
    global dialog_open
    pending = [i for i, p in enumerate(packages) if p["status"] == "pending"]
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
        pkg["status"]     = "arrived"
        pkg["photo_path"] = path
        _save_csv(packages)
        refresh_listbox()
        refresh_history()
        dialog_open = False
        win.destroy()

    tk.Button(match_frame, text="Confirm", command=confirm_match).pack(pady=6)


#webcam toggle — spins up PackageWatcher from YoloOutput in a background thread
def toggle_webcam():
    global cam_running, cam_thread
    if not YOLO_AVAILABLE:
        messagebox.showerror("Missing Module", "YoloOutput.py could not be imported.")
        return
    if not cam_running:
        cam_running = True
        cam_btn.config(text="Stop Webcam Watch")
        status_label.config(text="Status: Watching…", fg="green")

        def _thread_target():
            watcher = yolo_detector.PackageWatcher(
                on_detected   = lambda frame, size: root.after(0, lambda: on_package_detected(frame, size)),
                is_running_fn = lambda: cam_running,
            )
            watcher.run()
            root.after(0, lambda: status_label.config(text="Status: Webcam stopped", fg="gray"))

        cam_thread = threading.Thread(target=_thread_target, daemon=True)
        cam_thread.start()
    else:
        cam_running = False
        cam_btn.config(text="Start Webcam Watch")


#boot
refresh_listbox()
refresh_history()
root.mainloop()