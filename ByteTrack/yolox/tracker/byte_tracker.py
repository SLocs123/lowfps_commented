"""Custom ByteTrack variant for low-frame-rate traffic counting.
(follows ByteTrack structure with custom tracking metrics, in low_fps_metrics.py, and more forgiving tolerances for low-fps application, the kalman filter is also initialised with initial movement based on the lane assignment logic)
(I have also added a feature embedding model with fastreid, 1-fps basically becomes a re-id problem so tracking is built around feature description)
(for re-id see https://www.ultralytics.com/glossary/object-re-identification-re-id as a baseline, basically tracking without motion prediction)

This file contains two main pieces:
1. `STrack`, which stores one short-lived track fragment.
2. `BYTETracker`, which links detections just long enough to support counting.

The design goal here is not robust long-term identity tracking. The IDs are
fragile and are treated as temporary labels that help the counter decide whether
recent detections belong to the same vehicle over a small number of frames.
"""

import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import shapely
import torch
import torch.nn.functional as F
from ultralytics import settings

from .kalman_filter import KalmanFilter
from .matching import *
from .basetrack import BaseTrack, TrackState
from .low_fps_metrics import low_fps_distance
from .direction_lines import assign_expected_direction
from .tracker_logger import LabelLogger, write_kf_prediction, write_direction_assignment
from ...feature_extractor.config import FeatureExtrator, Reid_config
from ...feature_extractor.extractor import Extractor
from kf_config import KALMAN_KWARGS


class STrack(BaseTrack):
    """State container for one short counting-oriented track fragment."""

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls):
        # Store the first detection as top-left x/y plus width/height.
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        # Extra metadata carried along with the geometric state.
        self.cls = cls
        self.score = score
        self.tracklet_len = 0
        # `history` is intentionally short-lived in practice because the whole
        # system is tuned for low-FPS counting rather than stable identity.
        self.history = [self.get_centre(self._tlwh)]
        self.embed_history = deque(maxlen=10)
        self.get_delta()

    def predict(self):
        """Run one Kalman prediction step for this track."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            # Lost tracks do not keep their height velocity.
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)  # type: ignore

    @staticmethod
    def multi_predict(stracks):
        """Predict many tracks together for efficiency."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            # Recompute expected lane direction before predicting motion.
            multi_deltas = STrack.get_multi_deltas(stracks)
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance, deltas=multi_deltas)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def get_multi_deltas(stracks):
        """Collect the expected lane-direction vector for each track."""
        deltas = []
        for st in stracks:
            st.get_delta()
            deltas.append(st.direction)
        return np.array(deltas)

    def get_delta(self):
        """Estimate expected motion from the track's bottom-middle point."""
        if self.is_activated:
            bottom_middle = (int(self._tlwh[0] + self._tlwh[2] / 2), int(self._tlwh[1] + self._tlwh[3]))
        else:
            tlwh = self.tlwh
            bottom_middle = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3]))
        delta, direction_line, direction_index, ltr_dist, rtl_dist = assign_expected_direction(bottom_middle)
        self.ltr_dist = ltr_dist
        self.rtl_dist = rtl_dist
        self.direction_line = direction_line
        self.direction_index = direction_index
        self.direction = delta
    
    def activate(self, kalman_filter, frame_id):
        """Start a new short track fragment from one unmatched detection."""

        self.get_delta()
        # Seed the initial velocity using the expected lane motion. This helps
        # more in low-FPS video than a zero-velocity start would.
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh), delta=self.direction)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # if frame_id == 1:
        #     self.is_activated = True
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reconnect a recently lost fragment when a plausible match appears."""
        self.mean, self.covariance = self.kalman_filter.update(  # type: ignore
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.history.append(self.get_centre(new_track.tlwh))
        self.embed_history.append(new_track.embed_history[-1])

    def update(self, new_track, frame_id):
        """Extend the current short track fragment with one more detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(  # type: ignore
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        if new_track.score > self.score:
            self.score = new_track.score
            self.cls = new_track.cls
        
        self.history.append(self.get_centre(new_tlwh))
        self.embed_history.append(new_track.embed_history[-1])

    @property
    def tlwh(self):
        """Return the current box as `(top_left_x, top_left_y, width, height)`."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Return the current box as `(x1, y1, x2, y2)`."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def get_centre(tlwh):
        """Return the centre point of a `tlwh` box."""
        centre = (
            float(tlwh[0] + tlwh[2] / 2.0),
            float(tlwh[1] + tlwh[3] / 2.0)
        )
        return centre

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert `tlwh` into `(center_x, center_y, aspect_ratio, height)`."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        """Return the track state in the measurement format used by the Kalman filter."""
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert `(x1, y1, x2, y2)` into `(x, y, w, h)`."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert `(x, y, w, h)` into `(x1, y1, x2, y2)`."""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        """Readable debug string showing ID and lifespan."""
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    """Build short low-FPS track fragments for counting using motion and appearance cues."""

    def __init__(self, track_thres=0.7, max_missed_frames=1, match_thresh=0.4, unconfirmed_match_thresh=0.7, second_math_thresh=0.5, kf_log_path=None, direction_log_path=None, output_paths=None, kalman_kwargs=None):
        # These three lists hold tracks in different lifecycle states.
        # Tracks are treated differently depending on whether they are currently active, recently lost, or permanently removed.
        # tracked_stracks are actively matched to new detections
        # lost_stracks are recently lost trackes that have to met the threshold to be removed, they are still compared against detections for potential re-activation
        # removed_stracks are permanently removed tracks that are no longer considered for matching or counting, basically used for debugging and offline inspection
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        if output_paths is not None:
            if "kalman" not in output_paths or "direction" not in output_paths:
                raise KeyError("output_paths must include 'kalman' and 'direction' keys")
            kf_log_path = str(output_paths["kalman"])
            direction_log_path = str(output_paths["direction"])

        # basic ByteTrack parameters
        # ByteTrack uses a 2 stage association process explaining the first and second thresholds
        # det_thres is how confident a detection must be to be considered for tracking, but i have manually removed this with 0.3 see line 382
        self.frame_id = 0
        self.track_thres = track_thres
        self.det_thresh = track_thres + 0.1
        self.match_thresh = match_thresh
        self.unconfirmed_match_thresh = unconfirmed_match_thresh
        self.second_math_thresh = second_math_thresh
        
        # In 1 FPS video we remove tracks quickly once they stop matching. So maxed missed frames is 1, accurate re-activation beyond 1 step for 1 fps is almost highly unlikely, even with a good feature embedding model
        self.max_time_lost = max_missed_frames
        if kalman_kwargs is None:
            kalman_kwargs = dict(KALMAN_KWARGS)

        # Use one Kalman filter per tracker plus a shared one for batched prediction.
        STrack.shared_kalman = KalmanFilter(**kalman_kwargs)
        self.kalman_filter = KalmanFilter(**kalman_kwargs)

        # The appearance extractor provides ReID embeddings for matching.
        extractor_settings = FeatureExtrator(reid_config=Reid_config())
        self.extractor = Extractor(extractor_settings)

        # Optional CSV-like logs for debugging prediction and direction assignment.
        self.kf_logger = LabelLogger(
            kf_log_path,
            "frame,track_id,x,y,w,h,state,score,cls",
        )
        self.direction_logger = LabelLogger(
            direction_log_path,
            "frame,track_id,bottom_middle_x,bottom_middle_y,direction_line,dx,dy,ltr_dist,rtl_dist",
        )

    def update(self, output_results, frame, img_info=[1,1], img_size=[1,1]):
        """Process one frame and return count-ready fragments plus all active fragments."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        classes = []

        if isinstance(output_results, list):
            if len(output_results) == 0:
                output_results = np.zeros((0, 6), dtype=np.float32)
            else:
                # Convert `(box, score, class)` tuples into a dense numeric matrix.
                output_results = np.stack(
                    [np.concatenate([np.asarray(bbox, dtype=np.float32).reshape(4),
                                    [np.float32(score)], [np.float32(_cls)]])
                    for (bbox, score, _cls) in output_results],
                    axis=0,
                )

        # Accept either torch tensors or NumPy arrays from upstream code.
        if torch.is_tensor(output_results):
            output_results = output_results.detach().cpu().numpy()
        else:
            output_results = np.asarray(output_results)

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            # The common case is `[x1, y1, x2, y2, confidence, class]`.
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = output_results[:, 5]

        # Split detections into a high-confidence pass and a lower-confidence pass.
        remain_inds = scores > self.track_thres
        inds_low = scores > 0.1
        inds_high = scores < self.track_thres

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            # Wrap raw detections in STrack objects so they carry embeddings,
            # history, and lane-direction metadata from the start.
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cls) for
                          (tlbr, s, cls) in zip(dets, scores_keep, classes[remain_inds])]
        else:
            detections = []

        # Add appearance vectors to the fresh detections before matching.
        self.extractor.extract_embeddings(frame, detections)

        # Keep only tracks that were truly activated in previous frames.
        tracked_stracks = [track for track in self.tracked_stracks if track.is_activated]  # type: list[STrack]

        # Step 1: try to match high-confidence detections first.
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict each old track forward to the current frame.
        STrack.multi_predict(strack_pool)

        for track in strack_pool:
            write_kf_prediction(self.kf_logger, self.frame_id, track)
        
        # Build a matching cost matrix and solve the assignment problem. 
        # Linear assignment is an application of the Hungarian algorithm, minimising global cost instead of just taking smallest distances greedily. 
        # typically matching is either done by hungarian or greedy matching, googling both should come up
        dists = low_fps_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        
        # linear assignment will return matched pairs, so this code just places the tracks in the right places and updates their states when needed
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 2: give unmatched fragments a second chance with lower-score detections.
        # ultimately the same process as above, but with a different threshold and a different set of detections
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cls) for
                          (tlbr, s, cls) in zip(dets_second, scores_second, classes[inds_second])]
        else:
            detections_second = []
        self.extractor.extract_embeddings(frame, detections_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        dists = low_fps_distance(r_tracked_stracks, detections_second)
        
        matches, u_track, u_detection_second = linear_assignment(dists, self.second_math_thresh)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 3: start new fragments from any remaining unmatched detections.
        # If detections are confident and not matched to existing tracks, they are considered new tracks and activated.
        # I have a low confidence threshold for activation (0.3) because in low-fps we do not typically have the ability to wait for the best detections as objects move quickly through the frame
        high_unmatched = [detections[i] for i in u_detection]
        low_unmatched = [detections_second[i] for i in u_detection_second]
        init_candidates = high_unmatched + low_unmatched

        for track in init_candidates:
            if track.score < 0.3:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 4: permanently remove tracks that have been lost for too long.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame >= self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Step 5: rebuild the tracked/lost/removed lists for the new frame.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # Step 6: keep a full copy of active fragments, then apply the final
        # counting filters.
        # these are basically positional filters, if track in/passed through area, add to count
        
        # normal tracking typically has similar processes on what tracks they want to show the user/testing algorithm, basically to optimise the tracking performance when benchmarking (see SMILE Track for example)
        active_stracks = [track for track in self.tracked_stracks if track.is_activated]
        all_out_tracks = active_stracks.copy()
        line_tracks = filter_track_history_last(active_stracks)
        counting_stracks = filter_tracks_count_box(active_stracks)
        counting_stracks = filter_tracks_edge(counting_stracks)
        
        # Re-add tracks whose recent path crossed the count line even if the
        # current box is outside the region-of-interest filter.
        for track in line_tracks:
            if track not in counting_stracks:
                counting_stracks.append(track)

        for track in counting_stracks:
            write_direction_assignment(self.direction_logger, self.frame_id, track)

        return counting_stracks, all_out_tracks


def filter_track_history_last(tracks):
    """Keep tracks whose last motion segment crosses the counting line."""
    points = [(193, 174), (591, 397)]
    line = shapely.geometry.LineString(points)
    filtered_tracks = []
    for track in tracks:
        track_points = track.history[-2:]
        if len(track_points) > 1:
            track_line = shapely.geometry.LineString(track_points)
            if line.intersects(track_line):
                filtered_tracks.append(track)
    return filtered_tracks


def filter_track_history_all(tracks):
    """Alternative count-line filter that checks the whole stored trajectory."""
    points = [(193, 174), (591, 397)]
    line = shapely.geometry.LineString(points)
    filtered_tracks = []
    for track in tracks:
        track_points = track.history
        if len(track_points) > 1:
            track_line = shapely.geometry.LineString(track_points)
            if line.intersects(track_line):
                filtered_tracks.append(track)
    return filtered_tracks


def filter_tracks_count_box(tracks):
    """Keep only tracks whose boxes intersect the counting region polygon."""
    points = [(280, 224), (453, 289), (293, 414), (107, 257)]
    detection_area = shapely.geometry.Polygon(points)
    if not detection_area.is_valid:
        raise ValueError("Invalid detection area polygon: check point order / self-intersections")
    
    filtered_tracks = []
    for track in tracks:
        x, y, w, h = track.tlwh
        box = [x, y, x + w, y + h]
        det_box = shapely.geometry.box(box[0], box[1], box[2], box[3])
        if detection_area.intersects(det_box):
            filtered_tracks.append(track)
    return filtered_tracks


def filter_tracks_edge(tracks, img_shape=[640, 480], size=5):
    """Discard boxes that touch the image edge, where crops are often unreliable."""
    filtered_tracks = []
    frame_box = shapely.geometry.box(0, 0, img_shape[0], img_shape[1])
    remove_box = shapely.geometry.box(size, size, img_shape[0] - size, img_shape[1] - size)
    edge_box = frame_box.difference(remove_box)
    for track in tracks:
        xyah = track.mean[:4]
        width = xyah[2] * xyah[3]
        height = xyah[3]
        box = [xyah[0] - width / 2, xyah[1] - height / 2, xyah[0] + width / 2, xyah[1] + height / 2]
        det_box = shapely.geometry.box(box[0], box[1], box[2], box[3])
        if edge_box.intersects(det_box):
            continue
        filtered_tracks.append(track)
    return filtered_tracks


def joint_stracks(tlista, tlistb):
    """Union two track lists while avoiding duplicate track IDs."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Return the tracks in `tlista` whose IDs do not appear in `tlistb`."""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Drop near-duplicate tracks, keeping the longer-lived one."""
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
