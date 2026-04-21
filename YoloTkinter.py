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
tk.Label(root, text="Expected Delivery Date (DD-MM-YYYY):").pack(pady=(4, 16))

date_entry = tk.Entry(root, width=20)
date_entry.insert(0, datetime.date.today().strftime("%d-%m-%y"))
date_entry.pack()

#Status label
status_label = tk.Label(root, text="Status: Idle", fg="gray")
status_label.pack(pady=(16, 4))

root.mainloop()