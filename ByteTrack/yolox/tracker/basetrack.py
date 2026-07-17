import numpy as np
from collections import OrderedDict


class TrackState(object):
    """Small enum-like container for the lifecycle of a track."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    """Minimal base class shared by concrete tracker objects."""

    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        """Return the latest frame index in which this track was updated."""
        return self.frame_id

    @staticmethod
    def next_id():
        """Allocate the next unique integer track ID."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """Start a brand-new track."""
        raise NotImplementedError

    def predict(self):
        """Predict the next state before seeing a new detection."""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Update the track with a newly matched detection."""
        raise NotImplementedError

    def mark_lost(self):
        """Mark the track as temporarily unmatched."""
        self.state = TrackState.Lost

    def mark_removed(self):
        """Mark the track as permanently removed."""
        self.state = TrackState.Removed
