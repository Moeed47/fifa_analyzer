# FIFA Analyzer

This repository demonstrates counting the number of players in each frame of an EA25 gameplay video using OpenCV and a pretrained YOLOv8 model.

## Setup

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the `count_players.py` script and follow the prompts to provide the video
path and optional settings. You can specify a custom YOLOv8 weight file, choose
to display the annotated frames, and optionally save the result:

```bash
python count_players.py
```

After running the command, enter the requested parameters when prompted. The
script prints the detected number of players for every frame.
