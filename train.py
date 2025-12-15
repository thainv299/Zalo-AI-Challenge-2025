from ultralytics import YOLO
import os

if __name__ == "__main__":

    # 1. SỬA ĐƯỜNG DẪN: Đã chuyển sang định dạng Windows (dấu \) và dùng string thô (r"...")
    DATA_PATH = r"dataset\data.yaml"

    # Khởi tạo mô hình (kiểm tra lại tên file: 'yolo12s.pt' hoặc 'yolov12s.pt')
    model = YOLO("yolo12s.pt")  

    # Điều chỉnh tham số training
    results = model.train(
        data=DATA_PATH,
        epochs=300,        # Tăng số epoch cho 3xx class
        imgsz=640,
        batch=16,          # TĂNG BATCH
        device=0,          # Giữ device=0 (GPU đầu tiên)
        
        # ===== TĂNG CƯỜNG DỮ LIỆU =====
        fliplr=0.0,
        flipud=0.0,
        degrees=10.0,      # Tăng độ xoay
        translate=0.1,
        scale=0.15,
        hsv_h=0.015,
        hsv_s=0.7,         # Tăng bão hòa/sắc độ
        hsv_v=0.4,
        mosaic=0.5,        # Kích hoạt Mosaic
        mixup=0.1,         # Sử dụng Mixup nhẹ
        copy_paste=0.3,    # Tăng Copy-Paste cho vật thể nhỏ
        erasing=0.03,

        # ===== TRAINING =====
        optimizer="AdamW",
        lr0=0.001,         # LR ban đầu tối ưu cho AdamW/fine-tuning
        lrf=0.01,
        warmup_epochs=5,
        patience=50,       # Tăng Patience
        close_mosaic=10,

        project="traffic_signs_vietnam",
        name="yolo12s_finetune_v2",
        exist_ok=True,
        val=True,
        verbose=True,
    )

    print(f"\n✅ Training completed!")
    print(f"📁 Results: {results.save_dir}")

    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val()
    print(f"\n📊 Best mAP50: {metrics.box.map50:.4f}")
    print(f"📊 Best mAP50-95: {metrics.box.map:.4f}")