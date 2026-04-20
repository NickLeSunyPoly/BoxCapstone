import tkinter as tk

root = tk.Tk()
root.title("My App")
root.geometry("400x300")

label = tk.Label(root, text="Enter something:")
label.pack(pady=10)

# Entry widget — single-line text input
entry = tk.Entry(root, width=30)
entry.pack(pady=5)

def get_value():
    print(entry.get())  # Read the text

button = tk.Button(root, text="Click Me", command=get_value)
button.pack(pady=10)

root.mainloop()