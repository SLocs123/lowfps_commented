from pathlib import Path
from config import LABELS_OUTPUT_ROOT, VISUAL_OUTPUT_ROOT

# this code just allows me to define an experiment name and it will build the necessary paths/directories, easier for version control and testing

def _validate_test_name(test_name: str) -> str:
    """Normalize and validate the experiment name used in output filenames."""
    name = test_name.strip()
    if not name:
        raise ValueError("test_name must be a non-empty string")
    return name


def build_test_paths(test_name: str) -> dict[str, Path]:
    """Return every output file used by one pipeline run."""
    name = _validate_test_name(test_name)
    return {
        "tracks": LABELS_OUTPUT_ROOT / "track_labels" / f"{name}.txt",
        "detections": LABELS_OUTPUT_ROOT / "det_labels" / f"{name}.txt",
        "kalman": LABELS_OUTPUT_ROOT / "kf_labels" / f"{name}.txt",
        "direction": LABELS_OUTPUT_ROOT / "direction_labels" / f"{name}.txt",
        "all_tracks": LABELS_OUTPUT_ROOT / "all_track_labels" / f"{name}.txt",
    }


def get_path(paths: dict[str, Path], key: str) -> Path:
    """Look up a named path and raise a helpful error if the key is missing."""
    if key not in paths:
        available = ", ".join(sorted(paths.keys()))
        raise KeyError(f"Missing path key '{key}'. Available keys: {available}")
    return paths[key]


def ensure_parent_dirs(paths: list[Path]) -> None:
    """Create every parent directory needed by the given output files."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def build_visual_output_path(test_name: str, kind: str, suffix: str) -> Path:
    """Build a visualisation filename such as `demo_tracks.png`."""
    name = _validate_test_name(test_name)
    clean_kind = kind.strip()
    clean_suffix = suffix.strip().lstrip(".")
    if not clean_kind:
        raise ValueError("kind must be a non-empty string")
    if not clean_suffix:
        raise ValueError("suffix must be a non-empty string")

    return VISUAL_OUTPUT_ROOT / f"{name}_{clean_kind}.{clean_suffix}"
