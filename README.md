# FIFA Analyzer

This repository demonstrates counting the number of players in each frame of an EA25 gameplay video using OpenCV and a pretrained YOLOv8 model.

## Setup

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Usage

Edit the configuration variables at the bottom of `count_players.py` to set the
video path, model weights, and display options. Then run the script directly:

```bash
python count_players.py
```

The script prints the detected number of players for every frame and, if
enabled, shows or saves the annotated video.
