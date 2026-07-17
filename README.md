# Low-FPS Multi-Object Tracking

Multi-object tracking (MOT) detects objects in video and attempts to preserve a consistent identity for each object across frames.

Tracking at 1 FPS is difficult because objects can move a large distance, change appearance, overlap, or leave the scene between consecutive frames.

This repository contains a tracking pipeline designed to operate directly on 1 FPS traffic-camera video.

## Installation

The project uses Poetry for dependency management and requires Python 3.11 or Python 3.12.

Poetry installation documentation:

* [Poetry installation guide](https://python-poetry.org/docs/#installation)
* [Poetry documentation](https://python-poetry.org/docs/)

Install Poetry using the official installer:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Make Poetry available in the current terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Confirm that it is installed:

```bash
poetry --version
```

Clone the repository and enter it:

```bash
git clone https://github.com/SLocs123/lowfps_commented.git
cd lowfps_commented
```

Configure Poetry to create the virtual environment inside the repository, if you want a .venv style:

```bash
poetry config virtualenvs.in-project true --local
```

Select Python 3.11:

```bash
poetry env use python3.11
```

Install the dependencies from `pyproject.toml`:
(see PyTorch and CUDA before running)
```bash
poetry install
```
Activate or just run through poetry:
```bash
source .venv/bin/activate
poetry run python main.py
```

## PyTorch and CUDA

PyTorch must be installed from a package source that provides a build compatible with the required CUDA version.
Check the installed NVIDIA driver and supported CUDA version:

```bash
nvidia-smi
```

The CUDA version shown by `nvidia-smi` represents the newest CUDA runtime supported by the installed NVIDIA driver. It does not mean that the full CUDA toolkit is installed.

The official PyTorch installation selector can be used to determine the appropriate PyTorch build:

[PyTorch installation selector](https://pytorch.org/get-started/locally/)

Select:

* Linux
* Pip
* Python
* A CUDA version supported by the NVIDIA driver

The current `pyproject.toml` uses the CUDA 12.8 PyTorch package source:

```toml
[[tool.poetry.source]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu128"
priority = "explicit"
```

If a different CUDA build is required, change the source URL to a supported PyTorch wheel repository and ensure the specified PyTorch versions are available from it. For example:

```toml
url = "https://download.pytorch.org/whl/cu126"
```

Do not choose a CUDA build newer than the installed NVIDIA driver supports. The available builds and installation commands should be checked using the official PyTorch selector.

Verify the installation after running `poetry install`:

```bash
poetry run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

## Code structure

```text
lowfps_commented/
├── main.py
├── detector.py
├── ByteTrack/
│   └── feature_extractor/
│   └── yolox/tracker/
│     └── byte_tracker.py
│     └── lowfps_metrics/
├── pyproject.toml
└── ...
```

### `main.py`

`main.py` is the main pipeline wrapper.

It reads the 1 FPS video frame by frame, sends each frame to the detector, passes the resulting detections to the tracker, and manages the output tracks.

The video is processed as provided. Frames are not skipped or resampled.

### `detector.py`

`detector.py` is a wrapper around the Ultralytics YOLO detector.

It loads the YOLO model, performs inference on each frame and converts the detections into the format required by the tracking code.

### `ByteTrack/`

This directory contains the ByteTrack-based tracking code.

The tracker associates detections between consecutive 1 FPS frames and attempts to assign a consistent track ID to each object.

FastReID is used to extract appearance features. These features provide additional information when position and motion alone are insufficient because of the large time gap between frames.

The FastReID integration is located under:

```text
ByteTrack/feature_extractor/
```

The required `.pth` model weights are not stored in the repository and must be added separately.

### `lowfps_metrics.py`

This file contains custom metrics designed for the 1 FPS application.

These metrics are intended to expose tracking behaviour that is particularly relevant when there are large spatial and temporal gaps between observations.

## Code sources

The repository contains:

* adapted ByteTrack tracking code;
* adapted FastReID appearance-extraction code;
* custom pipeline code;
* a custom YOLO detector wrapper;
* custom low-FPS tracking metrics.

## Notes

This repository is research code rather than a general-purpose tracking library. It may not be easy to use.
All outputs are saved into .txt and .txt processing code is not included here
I have not included the instructions for applying this code to a custom video, and therefore it will not work propoerly, if you intended to do this you will need to change the line and area codes for both detection and track output see:
byte_tracker.py, lines 436, 464
direction_lines.py, lines 8,9
detector.py, lines 135

