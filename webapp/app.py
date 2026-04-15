from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from services.plate_service import PlateRecognitionService

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"
MODEL_PATH = Path(os.getenv("PLATE_MODEL_PATH", str(REPO_DIR / "runs" / "train" / "ket_qua_3202" / "weights" / "best.pt"))).resolve()

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(STATIC_DIR))
service = PlateRecognitionService(
    repo_dir=REPO_DIR,
    model_path=MODEL_PATH,
    results_dir=RESULTS_DIR,
    conf_threshold=float(os.getenv("PLATE_CONF", "0.25")),
    blur_threshold=float(os.getenv("PLATE_BLUR_THRESHOLD", "80")),
)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/recognize")
def recognize():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "Khong tim thay file image"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"ok": False, "error": "Ban chua chon anh"}), 400

    data = file.read()
    npbuf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"ok": False, "error": "File upload khong phai anh hop le"}), 400

    try:
        result = service.recognize(image)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Loi xu ly: {exc}"}), 500

    return jsonify({"ok": True, **result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
