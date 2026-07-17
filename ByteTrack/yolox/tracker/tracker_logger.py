from pathlib import Path
# This just helps me output some values so they are easier to save, instead of passing them all the way back to the initial call, or building a bunch of txt loggers within the tracker class


def _format_float(value, precision=2):
    """Format numeric values consistently for plain-text logs."""
    return f"{float(value):.{precision}f}"


class LabelLogger:
    """Tiny helper that writes CSV-like tracking logs with an optional header."""

    def __init__(self, output_path, header):
        self.output_path = output_path
        self.header = header
        self._initialized = False

    def write(self, line):
        """Append one line, creating the parent folder and header if needed."""
        if self.output_path is None:
            return

        out_path = Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a"
        if not self._initialized:
            mode = "w"

        with out_path.open(mode, encoding="utf-8") as f:
            if not self._initialized and self.header:
                f.write(self.header + "\n")
            f.write(line + "\n")

        self._initialized = True


def write_kf_prediction(logger, frame_id, track):
    """Log the tracker state immediately after Kalman prediction."""
    tlwh = track.tlwh
    line = (
        f"{int(frame_id)},{int(track.track_id)},"
        f"{_format_float(tlwh[0], precision=2)},{_format_float(tlwh[1], precision=2)},"
        f"{_format_float(tlwh[2], precision=2)},{_format_float(tlwh[3], precision=2)},"
        f"{int(track.state)},{float(track.score):.4f},{int(track.cls)}"
    )
    logger.write(line)


def write_direction_assignment(logger, frame_id, track):
    """Log the expected movement direction assigned to a track."""
    tlwh = track.tlwh
    bottom_middle_x = int(round(float(tlwh[0] + tlwh[2] / 2.0)))
    bottom_middle_y = int(round(float(tlwh[1] + tlwh[3])))
    direction_x = float(track.direction[0]) if track.direction is not None else 0.0
    direction_y = float(track.direction[1]) if track.direction is not None else 0.0
    ltr_dist = float(track.ltr_dist) if track.ltr_dist is not None else -1.0
    rtl_dist = float(track.rtl_dist) if track.rtl_dist is not None else -1.0

    line = (
        f"{int(frame_id)},{int(track.track_id)},{bottom_middle_x},{bottom_middle_y},"
        f"{track.direction_line},{direction_x:.4f},{direction_y:.4f},{ltr_dist:.4f},{rtl_dist:.4f}"
    )
    logger.write(line)
