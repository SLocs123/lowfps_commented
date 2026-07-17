import os
from typing import Iterable

import cv2
import shapely
from ultralytics import YOLO


class YoloDetector:
    """Wrap a YOLO model so the rest of the pipeline can ask for one frame at a time. 
    Basically the same normal YOLO API but slightly tailored for my work, 
    with added NMS function because the built in YOLO NMS wasnt behaving as expected when run with agnostic_nms=True (I am still not sure why)."""

    def __init__(
        self,
        model_path="yolo26m.pt", # see ultralytics website for available models
        output_txt_path=None,
        iou=0.7,
        agnostic_nms=True,
        post_nms_iou=0.9,
    ):
        # Load the trained YOLO model once during startup.
        self.model = YOLO(model_path)
        self.cap = None
        self.frame_idx = 0
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.post_nms_iou = post_nms_iou
        
        # Start each run with a fresh detection output file.
        if output_txt_path and os.path.exists(output_txt_path):
            os.remove(output_txt_path)
        self.output_txt_path = output_txt_path

    def open(self, video_path: str):
        """Open the input video before calling `step`."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")

    def step(self, output_txt_path=None):
        """Read one frame, run detection, and return filtered detections.

        Returns:
            `(frame, detections)` while frames remain.
            `(None, None)` when the video reaches the end.
        """
        if self.cap is None:
            raise RuntimeError("Video not opened.")

        self.frame_idx += 1
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        y = self.model(
            frame,
            classes=[2, 3, 5, 7],
            verbose=False,
            iou=self.iou,
            agnostic_nms=self.agnostic_nms,
        )
        dets = self._to_dets(y)
        # Keep only detections in the road region of interest.
        filtered_dets = self.filter_dets_road(dets)
        # Remove near-duplicate boxes that survived YOLO's own NMS.
        filtered_dets = self.class_agnostic_nms(filtered_dets, self.post_nms_iou)
        
        if self.output_txt_path is not None:
            self._save_det_txt(filtered_dets, self.frame_idx)
        return frame, filtered_dets

    def _to_dets(self, yolo_results):
        """Convert Ultralytics results into `(box, confidence, class)` tuples."""
        r = yolo_results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy()
        return list(zip(xyxy, conf, cls))
    
    def _save_det_txt(self, dets, frame_num):
        """Append detections for this frame to the plain-text log file."""
        if self.output_txt_path is None:
            return
        with open(self.output_txt_path, "a", encoding="utf-8") as f:
            for (x1, y1, x2, y2), conf, cls in dets:
                f.write(f"{frame_num} {int(cls)} {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

    def _iou_xyxy(self, a, b):
        """Compute intersection-over-union for two `[x1, y1, x2, y2]` boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def class_agnostic_nms(self, dets, iou_thresh=0.9):
        """Keep the strongest box and suppress heavy overlaps regardless of class."""
        if not dets:
            return dets

        # Sort from highest confidence to lowest so the first kept box wins.
        order = sorted(range(len(dets)), key=lambda i: float(dets[i][1]), reverse=True)
        kept = []
        for idx in order:
            cand = dets[idx]
            cand_box = cand[0]
            should_keep = True
            for kept_det in kept:
                if self._iou_xyxy(cand_box, kept_det[0]) >= iou_thresh:
                    should_keep = False
                    break
            if should_keep:
                kept.append(cand)
        return kept

    def filter_dets_road(self, dets):
        """Discard detections that do not touch the road area of interest."""
        road_points = [(5, 290), (407, 206), (518, 230), (106, 459)]
        road = shapely.geometry.Polygon(road_points)
        filtered_dets = []
        for det in dets:
            x1, y1, x2, y2 = det[0]
            det_box = shapely.geometry.box(x1, y1, x2, y2)
            if det_box.intersects(road):
                filtered_dets.append(det)
        return filtered_dets