"""Run the low-FPS vehicle counting pipeline on one video.

This script is intentionally small. Its job is mostly orchestration:
1. decide which video and output names to use,
2. create the detector and tracker,
3. step through the video frame by frame,
4. save the short counting-oriented track outputs to text files.

The tracker does create integer track IDs, but in this project they are only
temporary labels for short track fragments. They are useful for counting and
debugging, not as strong long-term object identities.

(Every important part of the tracking, detection, tracking, counting, lane assignment, is saved to txt here. This allowed me to process everything differently with controlled tracking or detection results(even though they should be fairly deterministic). I havent included to txt file processing code here but i can if you want it)
"""

import sys
import time

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from detector import YoloDetector
from test_paths import build_test_paths, ensure_parent_dirs, get_path

TEST_NAME = "demo_count"
VIDEO_PATH = "/path/to/your/video.mp4" 


def format_track_row(frame_number: int, track) -> str:
    """Convert one track object into the plain-text row format used by this project."""
    x, y, w, h = track.tlwh
    return (
        f"{frame_number},{int(track.track_id)},{x:.2f},{y:.2f},"
        f"{w:.2f},{h:.2f},{float(track.score):.4f},{int(track.cls)}\n"
    )

# Build every output path for this run from one test name.
paths = build_test_paths(TEST_NAME)
# Make sure the output folders exist before any files are written.
ensure_parent_dirs(list(paths.values()))

# Refuse to silently overwrite previous experiment outputs.
existing_outputs = [
    path for path in paths.values()
    if path.exists()
]
if existing_outputs:
    print(f"Test name '{TEST_NAME}' already exists. Existing files:")
    for path in existing_outputs:
        print(f"- {path}")
    answer = input("Do you want to continue and overwrite data? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Aborted by user.")
        sys.exit(1)

# The detector reads frames and produces vehicle detections.
det = YoloDetector("yolo26x.pt", output_txt_path=str(get_path(paths, "detections")))
det.open(VIDEO_PATH)

# The tracker links detections into short-lived track fragments that are later
# filtered for counting. The numeric IDs are fragile and only local to this run.
tracker = BYTETracker(output_paths=paths)

frame_idx = 0

start_time = time.time()

# The count-ready rows are written directly to `out_f` inside the loop.
# `all_track_rows` stores every active short track fragment before that filter.
all_track_rows = []
with open(get_path(paths, "tracks"), "w", encoding="utf-8") as out_f:
    while True:
        # Read the next frame and run YOLO on it.
        frame, detections = det.step()
        if frame is None:
            # `None` means the video has ended.
            break
        frame_idx += 1
        
        # The tracker expects detections in the form:
        # [[x1, y1, x2, y2, score, cls], ...]
        # tracker.update takes a detection list and the current frame, associates detections with existing tracks, and returns two lists:
        # 1. counting_tracks: tracks that are considered for counting (e.g., those that have been confirmed and are moving in the right direction).
        # 2. active_tracks: tracks that are currently active, regardless of whether they are counted or not. These are kept for debugging and offline inspection.
        # i used active tracks for debugging behaviour, and counting tracks for actual counting logic, both saved and run after this loop.
        counting_tracks, active_tracks = tracker.update(detections, frame)
        
        for track in counting_tracks:
            # Write one line per track fragment that currently looks useful for
            # the counting logic.
            out_f.write(format_track_row(frame_idx, track))
            
        for track in active_tracks:
            # Keep every active short track fragment in memory and flush them
            # once at the end for debugging and offline inspection.
            all_track_rows.append(format_track_row(frame_idx, track))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing completed in {elapsed_time:.2f} seconds.")

# Save the unfiltered short-track log after the loop finishes.
with open(get_path(paths, "all_tracks"), "w", encoding="utf-8") as all_out_f:
    all_out_f.writelines(all_track_rows)
    
    
print(f"Processed Test: {TEST_NAME}")