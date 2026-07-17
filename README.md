# Low-FPS Multi-Object Tracking

Multi-object tracking (MOT) detects objects in video and attempts to preserve a consistent identity for each object across frames.

Tracking at 1 FPS is difficult because objects can travel a large distance, change appearance, overlap, or leave the scene between consecutive frames.

This repository tests a tracking pipeline designed to operate directly on 1 FPS traffic-camera video.

## Installation

The project requires Python 3.11 or 3.12.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the project dependencies from `pyproject.toml`:

```bash
pip install .
```

PyTorch should be installed using a build compatible with your NVIDIA driver.

First check the installed driver:

```bash
nvidia-smi
```

Then use the official PyTorch installation selector and choose:

* Linux
* Pip
* Python
* A CUDA version supported by your NVIDIA driver

Run the generated command inside the activated virtual environment.

For example, a CUDA 12.8 installation may use:

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

Verify that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

The first value should normally be:

```text
True
```

## Code structure

```text
lowfps_commented/
├── main.py
├── detector.py
├── lowfps_metrics/
├── ByteTrack/
│   └── feature_extractor/
├── pyproject.toml
└── ...
```

### `main.py`

`main.py` is the main pipeline wrapper.

It reads the video frame by frame, passes each frame through the detector, sends the resulting detections to ByteTrack, and manages the resulting tracked objects and outputs.

Because the source video is already 1 FPS, the pipeline does not skip or resample frames.

### `detector.py`

`detector.py` wraps the Ultralytics YOLO detector.

It loads the detection model, runs inference on each frame, and converts YOLO predictions into the format expected by the tracker.

### `ByteTrack/`

This directory contains the tracking code.

ByteTrack associates detections between frames and assigns a persistent track ID to each object. The implementation also uses appearance information from FastReID to help compare objects across the large time gaps found in 1 FPS video.

The FastReID model and supporting files are located under:

```text
ByteTrack/feature_extractor/
```

The required `.pth` model weights are not stored in GitHub and must be added separately.

### `lowfps_metrics/`

This directory contains metrics designed specifically for evaluating the 1 FPS tracking application.

These metrics provide additional information beyond standard MOT scores, with emphasis on failures that become important when there is a large time and movement gap between frames.

## Code sources

The repository contains:

* ByteTrack tracking code
* FastReID appearance-extraction code
* Custom pipeline, detector-wrapper and low-FPS evaluation code
