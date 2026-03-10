from ultralytics import YOLO

# Load a pre-trained YOLOv11 nano model
model = YOLO("yolo11n.pt") 

# Train the model on your dataset
# Point the 'data' argument to your downloaded data.yaml file
results = model.train(
    data="C:/Users/Nick/Downloads/packages.v2i.yolov11/data.yaml", 
    epochs=50,       # Number of training loops
    imgsz=640,       # Resize images to 640x640
    device=0,        # Use GPU 0 (Highly recommended for local training)
    mosaic=0.0       # Optional: Set to 0.0 to turn off mosaic augmentation if desired
)

#yolo train model=yolo11n.pt data=path/to/your/dataset/data.yaml epochs=50 imgsz=640 device=0 mosaic=0.0

# Load the best weights generated from your training run
model = YOLO("runs/detect/train/weights/best.pt")

# Run validation. It automatically uses the 'val' split defined in your data.yaml
metrics = model.val()

print(f"Mean Average Precision (mAP50-95): {metrics.box.map}")


# Pass a new image or video path to the trained model
results = model("C:/Users/Nick/Downloads/packages.v2i.yolov11/test/images/img--69-_jpg.rf.f6e45cd71fd9e176d194a3dedfa6704a.jpg")

# Display the image with the predicted bounding boxes
results[0].show()
