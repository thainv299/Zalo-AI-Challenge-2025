import cv2
import os
import numpy as np
from queue import Queue

def extract_frames_to_queue(
    video_path: str,
    base_fps: float = 2,
    high_fps: float = 10,
    motion_threshold: float = 30,
    motion_persist_time: float = 1.5,
    frame_diff_interval: int = 1,
    max_queue_size: int = 100,
    sharpness_threshold: float = 50.0
) -> Queue:
    """
    Trích xuất các khung hình video vào Queue với lọc chuyển động và độ sắc nét
    """
    q = Queue(maxsize=max_queue_size)

    if not os.path.exists(video_path):
        print(f"❌ Video không tồn tại: {video_path}")
        q.put(None)
        return q

    print(f"🎥 Đang xử lý: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened() or fps == 0:
        print(f"❌ Không thể mở video hoặc FPS = 0")
        q.put(None)
        return q

    # Tính toán khoảng cách frame
    frame_interval = int(max(1, fps / base_fps))
    high_interval = int(max(1, fps / high_fps))

    prev_gray = None
    frame_id = 0
    motion_mode = False
    motion_countdown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện chuyển động
        if prev_gray is not None and frame_id % frame_diff_interval == 0:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.mean(diff)

            if motion_score > motion_threshold:
                motion_mode = True
                motion_countdown = int(fps * motion_persist_time)
            else:
                motion_countdown -= 1
                if motion_countdown <= 0:
                    motion_mode = False

        # Chọn interval dựa trên motion mode
        interval = high_interval if motion_mode else frame_interval

        if frame_id % interval == 0:
            # Lọc độ sắc nét trong motion mode
            if motion_mode:
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < sharpness_threshold:
                    prev_gray = gray
                    continue
            
            # Đưa frame vào queue
            q.put(frame)

        prev_gray = gray

    cap.release()
    q.put(None)
    return q