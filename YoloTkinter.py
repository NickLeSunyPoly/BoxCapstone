import tkinter as tk
from tkinter import messagebox
import datetime

try:
    import YOLOnewTest as yolo_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

root = tk.Tk()
root.title("Package Delivery Tracker")
root.geometry("400x380")

# Box Size
tk.Label(root, text="Expected Box Size:").pack(pady=(20, 4))

size_var = tk.StringVar(value="Medium")
size_frame = tk.Frame(root)
size_frame.pack()
for sz in ("Small", "Medium", "Large"):
    tk.Radiobutton(size_frame, text=sz, variable=size_var, value=sz).pack(side="left", padx=8)

# Delivery Date
tk.Label(root, text="Expected Delivery Date (DD-MM-YYYY):").pack(pady=(4, 16))

date_entry = tk.Entry(root, width=20)
date_entry.insert(0, datetime.date.today().strftime("%d-%m-%Y"))  # FIX: 4-digit year
date_entry.pack()

# Buttons
tk.Button(root, text="Add Package",    command=lambda: add_package()).pack(pady=(6, 2))

# Package list with scrollbar
list_frame = tk.Frame(root)
list_frame.pack(fill="both", expand=True, padx=10, pady=(4, 4))

scrollbar = tk.Scrollbar(list_frame)
scrollbar.pack(side="right", fill="y")

pkg_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8, selectmode="single")
pkg_listbox.pack(fill="both", expand=True)
scrollbar.config(command=pkg_listbox.yview)

tk.Button(root, text="Remove Package", command=lambda: remove_package()).pack(pady=(2, 2))
tk.Button(root, text="Mark as Arrived", command=lambda: mark_arrived()).pack(pady=(2, 4))

# Single status label at the bottom  ← FIX: removed the duplicate above
status_label = tk.Label(root, text="Status: Idle", fg="gray")
status_label.pack(pady=(4, 8))

# Internal package data
packages = []

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
        messagebox.showwarning("None Selected", "Please select a package to remove.")
        return
    packages.pop(sel[0])
    refresh_listbox()

def mark_arrived():
    sel = pkg_listbox.curselection()
    if not sel:
        messagebox.showwarning("None Selected", "Please select a package to mark as arrived.")
        return
    packages[sel[0]]["arrived"] = True
    refresh_listbox()

def refresh_listbox():
    pkg_listbox.delete(0, tk.END)
    for pkg in packages:
        check = "[✔]" if pkg["arrived"] else "[ ]"
        pkg_listbox.insert(tk.END, f"{check}  {pkg['size']}  |  {pkg['date']}")
    update_status()

def update_status():
    total = len(packages)
    arrived = sum(1 for p in packages if p["arrived"])
    pending = total - arrived
    status_label.config(
        text=f"Total: {total}  |  ✔ Arrived: {arrived}  |  ⏳ Pending: {pending}",
        fg="gray"
    )

root.mainloop()