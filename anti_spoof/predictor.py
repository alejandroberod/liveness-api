"""Inference service based on Silent-Face-Anti-Spoofing (src/anti_spoof_predict.py + test.py)."""

import math
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from anti_spoof.model_lib.MiniFASNet import (
    MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE,
)
from anti_spoof.utils import CropImage, get_kernel, parse_model_name

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
MODELS_DIR = RESOURCES_DIR / "anti_spoof_models"
DETECTION_DIR = RESOURCES_DIR / "detection_model"

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE,
}

REAL_LABEL = 1


class FaceDetector:
    def __init__(self, confidence=0.6):
        self.net = cv2.dnn.readNetFromCaffe(
            str(DETECTION_DIR / "deploy.prototxt"),
            str(DETECTION_DIR / "Widerface-RetinaFace.caffemodel"),
        )
        self.confidence = confidence

    def get_bbox(self, img):
        """Returns [x, y, w, h] of the most confident face, or None if no face is found."""
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if width * height >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))),
                             interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.net.setInput(blob, 'data')
        out = self.net.forward('detection_out').squeeze()
        if out.ndim == 1:
            out = out[np.newaxis, :]
        if out.size == 0:
            return None

        idx = np.argmax(out[:, 2])
        if out[idx, 2] < self.confidence:
            return None
        left, top = out[idx, 3] * width, out[idx, 4] * height
        right, bottom = out[idx, 5] * width, out[idx, 6] * height
        return [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]


class AntiSpoofService:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.detector = FaceDetector()
        self.cropper = CropImage()
        self.models = [self._load_model(p) for p in sorted(MODELS_DIR.glob("*.pth"))]
        if not self.models:
            raise RuntimeError(f"No .pth models found in {MODELS_DIR}")
        self._lock = threading.Lock()

    def _load_model(self, model_path):
        h_input, w_input, model_type, scale = parse_model_name(model_path.name)
        model = MODEL_MAPPING[model_type](conv6_kernel=get_kernel(h_input, w_input)).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        if next(iter(state_dict)).startswith('module.'):
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        model.load_state_dict(state_dict)
        model.eval()
        return model, h_input, w_input, scale

    def _to_tensor(self, img):
        # Same as the original repo's custom ToTensor: BGR, HWC -> CHW, NOT divided by 255.
        return torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)

    def predict(self, image_bgr):
        """Returns {"label", "score", "bbox"} or None if no face is detected."""
        with self._lock:
            bbox = self.detector.get_bbox(image_bgr)
            if bbox is None:
                return None

            prediction = np.zeros((1, 3))
            with torch.no_grad():
                for model, h_input, w_input, scale in self.models:
                    patch = self.cropper.crop(
                        org_img=image_bgr, bbox=bbox, scale=scale,
                        out_w=w_input, out_h=h_input, crop=scale is not None,
                    )
                    output = model(self._to_tensor(patch))
                    prediction += F.softmax(output, dim=1).cpu().numpy()

        label = int(np.argmax(prediction))
        score = float(prediction[0][label] / len(self.models))
        return {
            "label": "real" if label == REAL_LABEL else "fake",
            "score": round(score, 4),
            "bbox": bbox,
        }
