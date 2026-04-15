import os
import warnings
import torch
import cv2
import numpy as np
from pathlib import Path

from utils.plate_crop_saver import save_plate_crops_from_detections

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

# ==============================================================================

INPUT_SOURCE = 'test3.jpg'
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent

MODEL_PATH = 'runs/train/ket_qua_3202/weights/best.pt'
CONF_THRESHOLD = 0.5
ENABLE_PLATE_CROP = True
PLATE_CROP_DIR = str(ROOT_DIR / 'runs' / 'demo' / 'plate_crops')
MAX_CROPS_PER_FRAME = None

print("⏳ Đang tải YOLOv5 model...")
try:
    yolo_model = torch.hub.load('.', 'custom', path=MODEL_PATH, source='local')
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    exit()

def draw_boxes(img, detections, enable_plate_crop=False, source_name='source', frame_idx=None):
    found = False
    saved_paths = []

    if enable_plate_crop:
        saved_paths = save_plate_crops_from_detections(
            image=img,
            detections=detections,
            output_dir=PLATE_CROP_DIR,
            source_name=source_name,
            conf_threshold=CONF_THRESHOLD,
            frame_idx=frame_idx,
            max_crops=MAX_CROPS_PER_FRAME,
        )

    for box in detections:
        x1, y1, x2, y2, conf, cls = box

        if conf < CONF_THRESHOLD:
            continue

        found = True
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        conf_text = f"{conf:.1%}"
        label = f"Bien so: {conf_text}"
        color = (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + w_text, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return img, found, saved_paths

def process_image(source):
    if not os.path.exists(source):
        print(f"❌ Không tìm thấy ảnh: {source}")
        return

    print(f"📷 Đang xử lý ảnh: {source}")
    img = cv2.imread(source)

    results = yolo_model(img)
    detections = results.xyxy[0].cpu().numpy()

    img_result, found, saved_paths = draw_boxes(
        img,
        detections,
        enable_plate_crop=ENABLE_PLATE_CROP,
        source_name=Path(source).stem,
    )

    if found:
        print("✅ Đã tìm thấy biển số!")
    else:
        print("❌ Không tìm thấy biển số nào.")

    if ENABLE_PLATE_CROP:
        print(f"💾 Đã lưu {len(saved_paths)} ảnh biển số vào: {PLATE_CROP_DIR}")

    screen_res = 1000
    scale = screen_res / img_result.shape[1]
    w, h = int(img_result.shape[1] * scale), int(img_result.shape[0] * scale)
    img_resized = cv2.resize(img_result, (w, h))

    cv2.imshow("KET QUA ANH - Nhan phim bat ky de thoat", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================= XỬ LÝ VIDEO / WEBCAM =======================
def process_video(source):
    if source == '0':
        cap = cv2.VideoCapture(0)
        print("🎥 Đang bật webcam...")
    else:
        if not os.path.exists(source):
            print(f"❌ Không tìm thấy video: {source}")
            return
        cap = cv2.VideoCapture(source)
        print(f"🎥 Đang xử lý video: {source}")

    screen_res = 1000

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kết thúc video.")
            break

        frame_idx += 1

        results = yolo_model(frame)
        detections = results.xyxy[0].cpu().numpy()

        frame_result, _, saved_paths = draw_boxes(
            frame,
            detections,
            enable_plate_crop=ENABLE_PLATE_CROP,
            source_name=Path(source).stem if source != '0' else 'webcam',
            frame_idx=frame_idx,
        )

        if ENABLE_PLATE_CROP and saved_paths:
            print(f"💾 Frame {frame_idx}: đã lưu {len(saved_paths)} crop")

        scale = screen_res / frame_result.shape[1]
        w, h = int(frame_result.shape[1] * scale), int(frame_result.shape[0] * scale)
        frame_resized = cv2.resize(frame_result, (w, h))

        cv2.imshow("KET QUA VIDEO - Nhan 'q' de thoat", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ======================= MAIN =======================
if __name__ == "__main__":
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    file_ext = Path(INPUT_SOURCE).suffix.lower()

    if INPUT_SOURCE == '0':
        process_video(INPUT_SOURCE)
    elif file_ext in image_exts:
        process_image(INPUT_SOURCE)
    else:
        process_video(INPUT_SOURCE)
