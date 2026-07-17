import cv2
import numpy as np
import torch

from .fastreid.config.config import get_cfg
from .fastreid.engine import DefaultPredictor
from .config import FeatureExtrator
# This is code from fast reid and yuqiang i am not 100% sure how much is original fastreid code, but i did simplify it a bit and give some diff function names

class Extractor:
    """Crop detections from a frame and turn them into appearance embeddings."""

    def __init__(self, config: FeatureExtrator) -> None:
        self.config = config
        self._setup()
  
    def _setup(self):
        """Load the FastReID model once so every frame can reuse it."""
        cfg = get_cfg()
        cfg.merge_from_file(self.config.reid_config.reid_config_path)
        cfg.MODEL.WEIGHTS = self.config.reid_config.reid_weight
        cfg.MODEL.DEVICE = self.config.reid_config.reid_device
        cfg.freeze()
        self.model = DefaultPredictor(cfg)

    def crop_img(self, img, box):
        """Crop one image using an `[x1, y1, x2, y2]` box."""

        if img is None or box is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = img.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = img[y1:y2, x1:x2]
        return crop

    def extract_embeddings(self, frame, detections):
        """Add one appearance vector to each detection object.

        The tracker later uses these vectors to decide whether a new detection
        looks like a previously seen object.
        """
        cfg = self.model.cfg
        h, w = cfg.INPUT.SIZE_TEST
        device = torch.device(self.config.reid_config.reid_device)
        is_cuda = device.type == "cuda"
        
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be (H, W, 3), got {frame.shape}")
        
        batch_tensors = []
        batch_indices = []
        
        # Convert each detection crop into the tensor format expected by FastReID.
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.tlbr
            
            crop = self.crop_img(frame, (x1, y1, x2, y2))
            if crop is None or crop.size == 0:
                raise ValueError(
                    f"Invalid crop for detection at {(x1, y1, x2, y2)}"
                )
            
            crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(crop).permute(2, 0, 1).float()
            batch_tensors.append(tensor)
            batch_indices.append(idx)
        
        if not batch_tensors:
            return
        
        # Stack all crops into one batch so the network can process them together.
        batch = torch.stack(batch_tensors, dim=0).to(device, non_blocking=is_cuda)
        with torch.no_grad():
            feats = self.model(batch)
        
        # Store the embedding back on the detection object for later matching.
        feats_np = feats.detach().cpu().numpy()
        for idx, feat in zip(batch_indices, feats_np):
            det = detections[idx]
            if not hasattr(det, "embed_history"):
                raise AttributeError("Detection object is missing embed_history")
            det.embed_history.append(feat)
            # Keep the latest embedding on a dedicated attribute for compatibility
            # with code that expects a single current embedding.
            det.embedding = feat