from flask import Flask, render_template, request, redirect, Response, send_from_directory
import cv2, os, time
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import Counter

app = Flask(__name__)

# ================== CẤU HÌNH HỆ THỐNG ==================
BASE_VIOLATION = "violations"
VIOLATION_DIR = os.path.join(BASE_VIOLATION, "images")
PLATE_DIR = os.path.join(BASE_VIOLATION, "plates")
TXT_FILE = os.path.join(BASE_VIOLATION, "violations.txt")

for path in [VIOLATION_DIR, PLATE_DIR]:
    os.makedirs(path, exist_ok=True)

# ================== KHỞI TẠO MODEL ==================
vehicle_model = YOLO("yolov8n.pt")
ocr = easyocr.Reader(['en'], gpu=False)

cap = None
violated_ids = set()
violated_plates = {}

WIDTH, HEIGHT = 640, 360
STOP_LINE_Y = 320
STREAMING = True

# ================== NHẬN DIỆN ĐÈN GIAO THÔNG ==================
def detect_traffic_light(frame):
    h, w, _ = frame.shape
    roi = frame[0:int(h * 0.4), int(w * 0.5):w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    masks = {
        "red": cv2.inRange(hsv, (0, 150, 100), (10, 255, 255)) |
               cv2.inRange(hsv, (160, 150, 100), (180, 255, 255)),
        "yellow": cv2.inRange(hsv, (15, 150, 150), (35, 255, 255)),
        "green": cv2.inRange(hsv, (40, 100, 100), (90, 255, 255))
    }

    for color, mask in masks.items():
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 150:
                return color
    return "unknown"

# ================== OCR BIỂN SỐ ==================
def get_license_plate(vehicle_crop, obj_id):
    try:
        h, w = vehicle_crop.shape[:2]
        plate_region = vehicle_crop[int(h * 0.5):h, :]
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        plate_name = f"plate_{obj_id}_{int(time.time_ns())}.jpg"
        cv2.imwrite(os.path.join(PLATE_DIR, plate_name), thresh)

        results = ocr.readtext(thresh, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
        text = "".join([r[1] for r in results]).replace(" ", "").upper()

        return text if text else "UNKNOWN", plate_name
    except:
        return "UNKNOWN", "error.jpg"

# ================== ROUTES ==================
@app.route("/", methods=["GET", "POST"])
def index():
    global cap, violated_ids, violated_plates, STREAMING

    if request.method == "POST":
        video = request.files.get("video")
        if video:
            video.save("video.mp4")
            cap = cv2.VideoCapture("video.mp4")

            violated_ids.clear()
            violated_plates.clear()
            STREAMING = True

            if not os.path.exists(TXT_FILE):
                with open(TXT_FILE, "w", encoding="utf-8") as f:
                    f.write("DANH SÁCH XE VI PHẠM\n" + "=" * 60 + "\n")

            return redirect("/stream")

    return render_template("index.html")

@app.route("/stream")
def stream():
    return render_template("stream.html")

@app.route("/pause_stream")
def pause_stream():
    global STREAMING
    STREAMING = False
    return "paused"

@app.route("/resume_stream")
def resume_stream():
    global STREAMING
    STREAMING = True
    return "resumed"

# ================== STREAM VIDEO ==================
def generate_frames():
    global cap, STREAMING, violated_ids, violated_plates

    while cap and cap.isOpened():
        if not STREAMING:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        light = detect_traffic_light(frame)

        # ĐÈN CẤM ĐI = ĐỎ + VÀNG
        is_violation_light = light in ["red", "yellow"]

        # Màu hiển thị
        if light == "red":
            color_light = (0, 0, 255)
        elif light == "yellow":
            color_light = (0, 255, 255)
        else:
            color_light = (0, 255, 0)

        cv2.line(frame, (0, STOP_LINE_Y), (WIDTH, STOP_LINE_Y), color_light, 3)
        cv2.putText(frame, f"DEN: {light.upper()}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_light, 3)

        results = vehicle_model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls in zip(boxes, ids, clss):
                if cls not in [2, 3, 5, 7]:
                    continue

                x1, y1, x2, y2 = map(int, box)

                vehicle_type = {2: "O to", 3: "Xe may", 5: "Xe buyt", 7: "Xe tai"}.get(cls, "Khac")
                fine = "4 - 6 trieu" if vehicle_type == "Xe may" else "18 - 20 trieu"

                # ===== LOGIC VI PHẠM =====
                if is_violation_light and y2 > STOP_LINE_Y:
                    if obj_id not in violated_ids:
                        violated_ids.add(obj_id)

                        plate_text, plate_img = get_license_plate(frame[y1:y2, x1:x2], obj_id)
                        violated_plates[obj_id] = plate_text

                        car_img = f"car_{obj_id}_{int(time.time_ns())}.jpg"
                        cv2.imwrite(os.path.join(VIOLATION_DIR, car_img), frame[y1:y2, x1:x2])

                        with open(TXT_FILE, "a", encoding="utf-8") as f:
                            f.write(
                                f"Thoi gian: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                                f"Bien so: {plate_text} | "
                                f"Loai xe: {vehicle_type} | "
                                f"Muc phat: {fine} | "
                                f"Anh xe: {car_img} | "
                                f"Anh bien so: {plate_img}\n"
                            )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"VI PHAM - {violated_plates.get(obj_id, '')}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ================== QUẢN LÝ VI PHẠM ==================
@app.route("/violations")
def violations():
    data = []
    if os.path.exists(TXT_FILE):
        with open(TXT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    row = {}
                    for p in line.strip().split("|"):
                        if ":" in p:
                            k, v = p.split(":", 1)
                            row[k.strip()] = v.strip()
                    if row:
                        data.append(row)
    return render_template("violations.html", violations=data)

@app.route("/statistics")
def statistics():
    dates, plates = [], []
    if os.path.exists(TXT_FILE):
        with open(TXT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    parts = line.split("|")
                    dates.append(parts[0].split(":")[1].strip().split(" ")[0])
                    plates.append(parts[1].split(":")[1].strip())

    date_count = Counter(dates)
    plate_count = Counter(plates)

    return render_template(
        "statistics.html",
        total=len(plates),
        date_labels=list(date_count.keys()),
        date_values=list(date_count.values()),
        plate_labels=list(plate_count.keys()),
        plate_values=list(plate_count.values())
    )

@app.route("/violation_images/<filename>")
def violation_images(filename):
    return send_from_directory(VIOLATION_DIR, filename)

@app.route("/violation_plates/<filename>")
def violation_plates(filename):
    return send_from_directory(PLATE_DIR, filename)
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/locations")
def locations():
    return render_template("locations.html")


# ================== RUN ==================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
