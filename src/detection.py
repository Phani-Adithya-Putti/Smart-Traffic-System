import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load the pre-trained YOLOv8 model

video_path = "data/videos/cctv052x2004080516x01638.avi"

cap = cv.VideoCapture(video_path)

# Loads the video
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv.namedWindow("Traffic Detection (Normal)", cv.WINDOW_NORMAL)
    cv.resizeWindow("Traffic Detection (Normal)", 1200, 800)

    cv.imshow("Traffic Detection (Normal)", frame)

    cv.namedWindow("Traffic Detection Annotated", cv.WINDOW_NORMAL)
    cv.resizeWindow("Traffic Detection Annotated", 1200, 800)

    cv.imshow("Traffic Detection Annotated", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    #If not opened
    if not cap.isOpened():
        print("Error opening video stream or file")
        break

cap.release()
cv.destroyAllWindows()  


