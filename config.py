from pathlib import Path

# A config file to easily define where i want my tests and processing to save to, you need to change this or you will get an error, or it might builg the dorectories itself which would be an error
# These roots decide where labels and visualisation outputs are written.

SET_TO_FALSE_TO_ACTIVATE = True # Change this to false once you get the error, i have done this so you dont accidently build unwanted directories
LABEL_OUTPUT = '/path/to/your/labels' # e.g., '/data/sam/lowfps/labels'
VISUAL_OUTPUT = '/path/to/your/visualisation_outputs' # e.g., '/data/sam/lowfps/visualisation_outputs' # visualisation code not included so you can ignore this


PROJECT_ROOT = Path(__file__).resolve().parent

# Set this to `Path(".")` if you want outputs inside this repository instead.
LABELS_ROOT = Path(LABEL_OUTPUT+"/labels")

# Root directory for visualisation image and video outputs.
VISUALISATION_OUTPUT_ROOT = Path(VISUAL_OUTPUT+"/visualisation_outputs")


def _resolve_root(root: Path) -> Path:
	"""Resolve relative paths from the repository root, keep absolute paths unchanged."""
	if root.is_absolute():
		return root
	return (PROJECT_ROOT / root).resolve()

if SET_TO_FALSE_TO_ACTIVATE:
	raise RuntimeError(
		"Please edit config.py to set LABEL_OUTPUT and VISUAL_OUTPUT to your desired output folders, "
		"then set SET_TO_FALSE_TO_ACTIVATE = False. This is a safety measure to avoid accidentally "
		"creating unwanted directories."
	)

LABELS_OUTPUT_ROOT = _resolve_root(LABELS_ROOT)
VISUAL_OUTPUT_ROOT = _resolve_root(VISUALISATION_OUTPUT_ROOT)

