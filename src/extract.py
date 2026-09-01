import cv2
from pathlib import Path


def extract_frames(video_dir, output_dir, frame_interval=10):

    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list(video_dir.glob("*.avi"))

    for video in videos:

        cv = cv2.VideoCapture(str(video))

        frame_count = 0
        saved_count = 0

        while True:

            ret, frame = cv.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:

                frame_name = f"{video.stem}_{saved_count}.jpg"

                cv2.imwrite(
                    str(output_dir / frame_name),
                    frame
                )

                saved_count += 1

            frame_count += 1

        cv.release()

        print(f"{video.name}: {saved_count} frames saved")


extract_frames(
    "data/video_dataset/train",
    "data/yolo_dataset/train",
    frame_interval=10
)

extract_frames(
    "data/video_dataset/val",
    "data/yolo_dataset/val",
    frame_interval=10
)