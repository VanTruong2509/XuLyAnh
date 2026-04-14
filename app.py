from flask import Flask, render_template, request, redirect, Response, send_from_directory
import cv2, os, time
import shutil
from ultralytics import YOLO
import easyocr
from collections import Counter

app = Flask(__name__)

# ================== CẤU HÌNH HỆ THỐNG ==================
DETECTIONS_DIR = "detections"
VEHICLE_DIR = os.path.join(DETECTIONS_DIR, "images")
PLATE_DIR = os.path.join(DETECTIONS_DIR, "plates")
TXT_FILE = os.path.join(DETECTIONS_DIR, "detections.txt")
LEGACY_DIR = "violations"
LEGACY_VEHICLE_DIR = os.path.join(LEGACY_DIR, "images")
LEGACY_PLATE_DIR = os.path.join(LEGACY_DIR, "plates")
LEGACY_TXT_FILE = os.path.join(LEGACY_DIR, "violations.txt")

for path in [DETECTIONS_DIR, VEHICLE_DIR, PLATE_DIR]:
    os.makedirs(path, exist_ok=True)

if not os.path.exists(TXT_FILE) and os.path.exists(LEGACY_TXT_FILE):
    with open(LEGACY_TXT_FILE, "r", encoding="utf-8") as src, open(TXT_FILE, "w", encoding="utf-8") as dst:
        dst.write(src.read())

if os.path.isdir(LEGACY_VEHICLE_DIR):
    for name in os.listdir(LEGACY_VEHICLE_DIR):
        src = os.path.join(LEGACY_VEHICLE_DIR, name)
        dst = os.path.join(VEHICLE_DIR, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

if os.path.isdir(LEGACY_PLATE_DIR):
    for name in os.listdir(LEGACY_PLATE_DIR):
        src = os.path.join(LEGACY_PLATE_DIR, name)
        dst = os.path.join(PLATE_DIR, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

# ================== KHỞI TẠO MODEL ==================
vehicle_model = YOLO("yolov8n.pt")
ocr = easyocr.Reader(['en'], gpu=False)

cap = None
detected_ids = set()
detected_plates = {}
seen_plate_texts = set()
DETECT_ROI = None  # Normalized ROI: {x1, y1, x2, y2} in range [0, 1]
FIXED_CENTER_ROI = {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}

STREAMING = True

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
    global cap, detected_ids, detected_plates, seen_plate_texts, STREAMING

    if request.method == "POST":
        video = request.files.get("video")
        if video:
            video.save("video.mp4")
            cap = cv2.VideoCapture("video.mp4")

            detected_ids.clear()
            detected_plates.clear()
            seen_plate_texts.clear()
            STREAMING = True

            if not os.path.exists(TXT_FILE):
                with open(TXT_FILE, "w", encoding="utf-8") as f:
                    f.write("DANH SACH XE DA NHAN DIEN\n" + "=" * 60 + "\n")

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


@app.route("/set_roi", methods=["POST"])
def set_roi():
    return {
        "status": "ok",
        "message": "ROI dang duoc set cung o giua video",
        "roi": FIXED_CENTER_ROI,
    }


@app.route("/clear_roi", methods=["POST"])
def clear_roi():
    return {
        "status": "ok",
        "message": "ROI co dinh dang bat, khong can xoa",
        "roi": FIXED_CENTER_ROI,
    }

# ================== STREAM VIDEO ==================
# ================== STREAM VIDEO ==================
def generate_frames():
    global cap, STREAMING, detected_ids, detected_plates, seen_plate_texts

    while cap and cap.isOpened():
        if not STREAMING:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        # --- CẤU HÌNH ĐIỀU KIỆN CHỤP CHUẨN ---
        # 1. Viền an toàn (Margin): Khoảng cách (pixel) từ mép video. Xe lọt qua viền này mới tính là "Vào hẳn"
        MARGIN = 20 
        # 2. Kích thước tối thiểu: Lọc bỏ các xe ở tít đằng xa, phải đủ to mới chụp
        MIN_WIDTH = 120
        MIN_HEIGHT = 120

        # (Tùy chọn) Vẽ một đường viền mỏng màu vàng để bạn dễ hình dung khu vực "Vào hẳn"
        cv2.rectangle(frame, (MARGIN, MARGIN), (frame_w - MARGIN, frame_h - MARGIN), (0, 255, 255), 1)

        results = vehicle_model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls in zip(boxes, ids, clss):
                if cls not in [2, 3, 5, 7]:
                    continue

                x1, y1, x2, y2 = map(int, box)
                box_w = x2 - x1
                box_h = y2 - y1

                # -----------------------------------------------------
                # LOGIC CHỐNG NHIỄU & CHỐNG TRÙNG LẶP
                # 1. Nếu xe chạm hoặc nằm ngoài viền an toàn -> Bỏ qua chưa chụp vội
                if x1 <= MARGIN or y1 <= MARGIN or x2 >= (frame_w - MARGIN) or y2 >= (frame_h - MARGIN):
                    continue

                # 2. Nếu xe quá nhỏ (ở xa) -> Bỏ qua
                if box_w < MIN_WIDTH or box_h < MIN_HEIGHT:
                    continue
                # -----------------------------------------------------

                vehicle_type = {2: "O to", 3: "Xe may", 5: "Xe buyt", 7: "Xe tai"}.get(cls, "Khac")

                if obj_id not in detected_ids:
                    detected_ids.add(obj_id)

                    plate_text, plate_img = get_license_plate(frame[y1:y2, x1:x2], obj_id)
                    detected_plates[obj_id] = plate_text

                    # Avoid duplicated records when tracker re-IDs the same vehicle.
                    should_write = (plate_text == "UNKNOWN") or (plate_text not in seen_plate_texts)
                    if should_write:
                        if plate_text != "UNKNOWN":
                            seen_plate_texts.add(plate_text)

                        car_img = f"car_{obj_id}_{int(time.time_ns())}.jpg"
                        cv2.imwrite(os.path.join(VEHICLE_DIR, car_img), frame[y1:y2, x1:x2])

                        with open(TXT_FILE, "a", encoding="utf-8") as f:
                            f.write(
                                f"Thoi gian: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                                f"Track ID: {obj_id} | "
                                f"Bien so: {plate_text} | "
                                f"Loai xe: {vehicle_type} | "
                                f"Anh xe: {car_img} | "
                                f"Anh bien so: {plate_img}\n"
                            )

                plate_text = detected_plates.get(obj_id, "UNKNOWN")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"{vehicle_type} | {plate_text}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ================== DANH SACH XE DA NHAN DIEN ==================
@app.route("/detections")
def detections():
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
    return render_template("detections.html", detections=data)


@app.route("/violations")
def violations_redirect():
    return redirect("/detections")

@app.route("/statistics")
def statistics():
    dates, plates, hours, vehicles = [], [], [], []
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
                        ts = row.get("Thoi gian", "")
                        if " " in ts:
                            d, t = ts.split(" ", 1)
                            dates.append(d)
                            hours.append(t.split(":")[0])
                        plates.append(row.get("Bien so", "UNKNOWN"))
                        vehicles.append(row.get("Loai xe", "Khac"))

    date_count = Counter(dates)
    plate_count = Counter(plates)
    hour_count = Counter(hours)
    vehicle_count = Counter(vehicles)

    today = time.strftime("%Y-%m-%d")
    today_total = date_count.get(today, 0)
    peak_hour = f"{hour_count.most_common(1)[0][0]}h" if hour_count else "--"

    return render_template(
        "statistics.html",
        total=len(plates),
        today_total=today_total,
        peak_hour=peak_hour,
        motor_count=vehicle_count.get("Xe may", 0),
        car_count=vehicle_count.get("O to", 0),
        truck_count=vehicle_count.get("Xe tai", 0) + vehicle_count.get("Xe buyt", 0),
        date_labels=list(date_count.keys()),
        date_values=list(date_count.values()),
        plate_labels=list(plate_count.keys()),
        plate_values=list(plate_count.values())
    )

@app.route("/violation_images/<filename>")
def violation_images(filename):
    return send_from_directory(VEHICLE_DIR, filename)

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
