import cv2
from ultralytics import YOLO


def count_players(video_path, model_path="yolov8n.pt", show=False, output=None):
    """Print the number of detected players for every frame in the video.

    Parameters
    ----------
    video_path : str
        Path to the gameplay video file.
    model_path : str, optional
        Path to a YOLOv8 model weight file. ``yolov8n.pt`` is used by default.
    show : bool, optional
        If ``True``, display the annotated video frames.
    output : str, optional
        If provided, save the annotated video to this path.
    """

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)[0]
        person_idx = 0  # COCO class index for 'person'
        mask = results.boxes.cls == person_idx
        num_players = int(mask.sum())
        boxes = results.boxes.xyxy[mask].cpu().numpy().astype(int)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Players: {num_players}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        print(f"Frame {frame_idx}: {num_players} players detected")
        if show:
            cv2.imshow("FIFA Analyzer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if writer:
            writer.write(frame)
        frame_idx += 1
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Hardcoded configuration
# ---------------------------------------------------------------------------
# Adjust these variables to use your own video, YOLO weights, and display
# preferences. ``output`` can be set to a file path to save the annotated video
# or ``None`` to disable saving.
VIDEO_PATH = "path/to/ea25_gameplay.mp4"
MODEL_PATH = "yolov8n.pt"
SHOW = False
OUTPUT = None


def main():
    """Run player counting using the hardcoded configuration."""
    count_players(VIDEO_PATH, MODEL_PATH, SHOW, OUTPUT)


if __name__ == "__main__":
    main()
