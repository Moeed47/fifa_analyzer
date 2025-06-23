# FIFA Analyzer

This repository demonstrates counting the number of players in each frame of an EA25 gameplay video using OpenCV and a pretrained YOLOv8 model.

## Setup

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the `count_players.py` script with the path to your gameplay video. You can
optionally specify a custom YOLOv8 weight file using `--model`:

```bash
python count_players.py /path/to/ea25_gameplay.mp4 --model yolov8n.pt
```

The script prints the detected number of players for every frame.
