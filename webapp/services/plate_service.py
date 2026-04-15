from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
import torch


class PlateRecognitionService:
    def __init__(
        self,
        repo_dir: str | Path,
        model_path: str | Path,
        results_dir: str | Path,
        conf_threshold: float = 0.25,
        blur_threshold: float = 80.0,
    ) -> None:
        self.repo_dir = str(Path(repo_dir).resolve())
        self.model_path = str(model_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.conf_threshold = conf_threshold
        self.blur_threshold = blur_threshold

        self.model = torch.hub.load(self.repo_dir, "custom", path=self.model_path, source="local")
        # Keep OCR on CPU for stable demo across machines.
        self.reader = easyocr.Reader(["en"], gpu=False)

    @staticmethod
    def blur_score(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def enhance_plate_for_ocr(plate: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(plate, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        y = clahe.apply(y)
        merged = cv2.merge((y, cr, cb))
        contrast = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        gaussian = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(contrast, 1.7, gaussian, -0.7, 0)
        return sharpened

    @staticmethod
    def _to_int_box(det: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(float(v))) for v in det[:4]]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        return x1, y1, x2, y2

    def _new_run_dir(self) -> Path:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "crops").mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _url_from_path(path: Path) -> str:
        idx = path.parts.index("static")
        relative = "/".join(path.parts[idx + 1 :])
        return f"/static/{relative}"

    def recognize(self, image: np.ndarray) -> dict[str, Any]:
        if image is None or image.size == 0:
            raise ValueError("Invalid image data")

        run_dir = self._new_run_dir()
        original_path = run_dir / "original.jpg"
        processed_path = run_dir / "processed.jpg"

        cv2.imwrite(str(original_path), image)

        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()

        drawn = image.copy()
        h, w = image.shape[:2]
        crops: list[dict[str, Any]] = []

        for idx, det in enumerate(detections, start=1):
            conf = float(det[4])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = self._to_int_box(det, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            plate = image[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            blur = self.blur_score(plate)
            enhanced = blur < self.blur_threshold
            ocr_input = self.enhance_plate_for_ocr(plate) if enhanced else plate

            texts = self.reader.readtext(
                ocr_input,
                detail=0,
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-",
            )
            plate_text = "".join(texts).strip() or "(khong doc duoc)"

            crop_path = run_dir / "crops" / f"plate_{idx:02d}.jpg"
            cv2.imwrite(str(crop_path), plate)

            label = f"{plate_text} {conf:.2f}"
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(drawn, (x1, max(0, y1 - 26)), (min(w, x1 + 380), y1), (0, 255, 0), -1)
            cv2.putText(drawn, label, (x1 + 4, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            crops.append(
                {
                    "text": plate_text,
                    "confidence": round(conf, 4),
                    "blur_score": round(blur, 2),
                    "enhanced": enhanced,
                    "crop_url": self._url_from_path(crop_path),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        cv2.imwrite(str(processed_path), drawn)

        return {
            "original_url": self._url_from_path(original_path),
            "processed_url": self._url_from_path(processed_path),
            "count": len(crops),
            "crops": crops,
        }
