import tkinter as tk
from tkinter import messagebox
import threading
import datetime

#Try to import existing YOLO detector
try:
    import YOLOnewTest as yolo_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

root = tk.Tk()
root.title("Package Delivery Tracker")
root.geometry("400x380")

#Box Size
tk.Label(root, text="Expected Box Size:").pack(pady=(20, 4))

size_var = tk.StringVar(value="Medium")
size_frame = tk.Frame(root)
size_frame.pack()
for sz in ("Small", "Medium", "Large"):
    tk.Radiobutton(size_frame, text=sz, variable=size_var, value=sz).pack(side="left", padx=8)

#Delivery Date
tk.Label(root, text="Expected Delivery Date (YYYY-MM-DD):").pack(pady=(16, 4))

date_entry = tk.Entry(root, width=20)
date_entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
date_entry.pack()

# ── Status label ──────────────────────────────────────────────────────────────
status_label = tk.Label(root, text="Status: Idle", fg="gray")
status_label.pack(pady=(16, 4))

# ── Detection state ────────────────────────────────────────────────────────────
detecting = False

def on_box_detected():
    """Called from the YOLO thread when a box is seen — shows popup on main thread."""
    def show_popup():
        answer = messagebox.askyesno(
            "📦 Package Detected!",
            f"YOLO thinks your {size_var.get()} package is here!\n\nIs your delivery here?"
        )
        if answer:
            status_label.config(text="Status: ✅ Delivery confirmed!", fg="green")
            stop_detection()
        else:
            status_label.config(text="Status: 🔍 Still watching...", fg="blue")
    root.after(0, show_popup)

def run_yolo():
    """Runs in a background thread so the GUI doesn't freeze."""
    if YOLO_AVAILABLE:
        yolo_detector.watch_for_box(
            on_detected  = on_box_detected,
            is_running   = lambda: detecting,
            log_callback = lambda msg: root.after(0, lambda: status_label.config(text=f"Status: {msg}"))
        )
    else:
        # Demo mode — simulates a detection after 4 seconds
        import time
        time.sleep(4)
        if detecting:
            on_box_detected()

def start_detection():
    global detecting
    # Validate date
    try:
        datetime.datetime.strptime(date_entry.get().strip(), "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Invalid Date", "Please enter date as YYYY-MM-DD")
        return

    detecting = True
    status_label.config(text="Status: 🔍 Watching for your package...", fg="blue")
    start_btn.config(state="disabled")
    stop_btn.config(state="normal")
    threading.Thread(target=run_yolo, daemon=True).start()

def stop_detection():
    global detecting
    detecting = False
    status_label.config(text="Status: Stopped", fg="gray")
    start_btn.config(state="normal")
    stop_btn.config(state="disabled")

#Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=20)

start_btn = tk.Button(btn_frame, text="Start Watching", command=start_detection)
start_btn.pack(side="left", padx=8)

stop_btn = tk.Button(btn_frame, text="Stop", command=stop_detection, state="disabled")
stop_btn.pack(side="left", padx=8)

root.mainloop()
