from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data/video_dataset/data.yaml",
    epochs=50,
    imgsz=640
)