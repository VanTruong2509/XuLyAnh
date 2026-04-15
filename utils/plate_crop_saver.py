from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np


def _safe_int(v: float | int) -> int:
    return int(round(float(v)))


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_plate_crop(
    image: np.ndarray,
    box: Sequence[float],
    output_dir: str | Path,
    source_name: str,
    index: int,
    confidence: Optional[float] = None,
    frame_idx: Optional[int] = None,
) -> Optional[Path]:
    """Save one cropped plate image and return output path, or None if crop is invalid."""
    if image is None or image.size == 0 or len(box) < 4:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = (_safe_int(box[0]), _safe_int(box[1]), _safe_int(box[2]), _safe_int(box[3]))
    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    out_dir = _ensure_dir(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    conf_part = f"_c{confidence:.2f}" if confidence is not None else ""
    frame_part = f"_f{frame_idx:06d}" if frame_idx is not None else ""
    filename = f"{source_name}{frame_part}_p{index:02d}{conf_part}_{stamp}.jpg"
    file_path = out_dir / filename

    ok = cv2.imwrite(str(file_path), crop)
    return file_path if ok else None


def save_plate_crops_from_detections(
    image: np.ndarray,
    detections: Iterable[Sequence[float]],
    output_dir: str | Path,
    source_name: str,
    conf_threshold: float,
    frame_idx: Optional[int] = None,
    max_crops: Optional[int] = None,
) -> list[Path]:
    """Save all valid crops from detections in xyxy conf cls format."""
    saved: list[Path] = []
    for idx, det in enumerate(detections, start=1):
        if len(det) < 5:
            continue
        conf = float(det[4])
        if conf < conf_threshold:
            continue

        out = save_plate_crop(
            image=image,
            box=det[:4],
            output_dir=output_dir,
            source_name=source_name,
            index=idx,
            confidence=conf,
            frame_idx=frame_idx,
        )
        if out is not None:
            saved.append(out)

        if max_crops is not None and len(saved) >= max_crops:
            break

    return saved
