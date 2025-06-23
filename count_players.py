import argparse
import cv2
from ultralytics import YOLO


def count_players(video_path, model_path="yolov8n.pt"):
    """Print the number of detected players for every frame in the video.

    Parameters
    ----------
    video_path : str
        Path to the gameplay video file.
    model_path : str, optional
        Path to a YOLOv8 model weight file. ``yolov8n.pt`` is used by default.
    """

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)[0]
        person_idx = 0  # COCO class index for 'person'
        num_players = int((results.boxes.cls == person_idx).sum())
        print(f"Frame {frame_idx}: {num_players} players detected")
        frame_idx += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Count players in each frame of gameplay video using YOLO"
    )
    parser.add_argument("video", help="Path to EA25 gameplay video file")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to YOLOv8 weights (default: yolov8n.pt)",
    )
    args = parser.parse_args()
    count_players(args.video, args.model)


if __name__ == "__main__":
    main()
