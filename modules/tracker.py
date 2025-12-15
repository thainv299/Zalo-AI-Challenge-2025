import cv2
from data_types import FrameData
from data_types import BoxInfo
from utils.SaveFrame import save_frame
import random

class BestFrameTracker:
    def __init__(self):
        self.best_frames: dict[int, FrameData] = {}
        
    def update_track(self, frame, track_id, bbox, confidence, cls_name):
        x1, y1, x2, y2 = bbox
        object_region = frame[y1:y2, x1:x2]
        if object_region.size == 0:
            return None
        
        sharpness = self.calculate_sharpness(object_region)
        
        # Áp dụng Lọc Cứng: Loại bỏ nếu Confidence quá thấp HOẶC Sharpness quá mờ (Tối ưu cho VLM/OCR)
        if confidence < 0.4 or sharpness < 100.0:
            return False
            
        quality_score = self.calculate_quality_score(confidence, sharpness, max_sharpness=5000)

        # Cập nhật frame tốt nhất nếu có điểm chất lượng cao hơn
        if track_id not in self.best_frames or quality_score > self.best_frames[track_id].score:
            self.best_frames[track_id] = FrameData(
                id=random.randint(100000, 999999),
                frame=frame.copy(),
                score=quality_score,
                box_info=BoxInfo(
                    bbox=bbox,
                    confidence=confidence,
                    class_name=cls_name,
                    sharpness=sharpness
                )
            )
            return True
        
        return False
    
    def calculate_sharpness(self, image_region):
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        return sharpness

    def calculate_quality_score(self, confidence, sharpness, max_sharpness=5000):
        # max_sharpness được đặt là 5000 trực tiếp trong định nghĩa hàm, không cần hằng số toàn cục
        sharp_norm = min(sharpness / max_sharpness, 1.0)
        conf_norm = min(max(confidence, 0.0), 1.0)
        
        # Ưu tiên Sharpness 60% > Confidence 40% cho đầu vào VLM/OCR
        quality_score = 0.4 * conf_norm + 0.6 * sharp_norm
        
        return quality_score