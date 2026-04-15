import torch
import cv2
import easyocr
import time
import os
from pathlib import Path

from utils.plate_crop_saver import save_plate_crop

# --- CẤU HÌNH ---
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv('MODEL_PATH', r'runs\train\ket_qua_xin\weights\best.pt')
IMAGE_NAME = os.getenv('IMAGE_NAME', 'test_xe.jpg')           # Tên ảnh đầu vào
OUTPUT_NAME = os.getenv('OUTPUT_NAME', 'ket_qua_cuoi_cung.jpg') # Tên ảnh kết quả sẽ lưu ra
ENABLE_PLATE_CROP = True
PLATE_CROP_DIR = str(ROOT_DIR / 'runs' / 'demo' / 'plate_crops')

def process_image():
    # Kiểm tra file
    if not os.path.exists(IMAGE_NAME):
        print(f"LỖI: Không tìm thấy ảnh '{IMAGE_NAME}'")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy model '{MODEL_PATH}'")
        return

    print("--- Đang khởi động hệ thống... ---")
    
    # 1. Load Model
    try:
        model = torch.hub.load('.', 'custom', path=MODEL_PATH, source='local')
    except Exception as e:
        print("Lỗi load model:", e)
        return

    # 2. Load EasyOCR
    print("--- Đang tải OCR... ---")
    reader = easyocr.Reader(['en'], gpu=True) # Để gpu=False nếu máy không có card rời mạnh

    # 3. Xử lý ảnh
    img = cv2.imread(IMAGE_NAME)
    start_time = time.time()

    results = model(img)
    detections = results.pandas().xyxy[0]

    found = False
    print(f"\n--- KẾT QUẢ ---")
    
    if len(detections) > 0:
        # Lấy kết quả tốt nhất
        best = detections.iloc[0]
        xmin, ymin, xmax, ymax = int(best['xmin']), int(best['ymin']), int(best['xmax']), int(best['ymax'])
        conf = best['confidence']

        # HẠ THẤP ĐỘ TIN CẬY XUỐNG 0.01 ĐỂ TEST
        if conf > 0.01:
            found = True
            
            # Cắt và đọc chữ
            plate_img = img[ymin:ymax, xmin:xmax]
            ocr_res = reader.readtext(plate_img, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
            text = "".join(ocr_res)

            if ENABLE_PLATE_CROP:
                saved_path = save_plate_crop(
                    image=img,
                    box=(xmin, ymin, xmax, ymax),
                    output_dir=PLATE_CROP_DIR,
                    source_name=Path(IMAGE_NAME).stem,
                    index=1,
                    confidence=float(conf),
                )
                if saved_path:
                    print(f"--> Đã lưu ảnh biển số cắt ra: {saved_path}")
                else:
                    print("--> Không thể lưu ảnh crop (bbox không hợp lệ).")
            
            # Vẽ lên ảnh
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(img, f"{text} ({conf:.2f})", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            print(f"--> Tìm thấy: {text}")
            print(f"--> Độ tin cậy: {conf:.2f}")
    
    if not found:
        print("--> Không tìm thấy biển số (Model cần train thêm).")

    # Tính giờ
    run_time = time.time() - start_time
    cv2.putText(img, f"Time: {run_time:.3f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(f"--> Thời gian: {run_time:.3f}s")

    # --- QUAN TRỌNG: LƯU RA FILE THAY VÌ HIỆN CỬA SỔ ---
    cv2.imwrite(OUTPUT_NAME, img)
    print(f"\n--> Đã lưu ảnh kết quả vào: {OUTPUT_NAME}")
    
    # Tự động mở ảnh lên xem (Chỉ chạy trên Windows)
    os.system(f"start {OUTPUT_NAME}")

if __name__ == "__main__":
    process_image()