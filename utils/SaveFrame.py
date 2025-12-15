import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from data_types import FrameData

def save_frame(*, frameData: FrameData, track_id, output_path, font_path="C:/Windows/Fonts/arial.ttf"):
    """
    Vẽ bbox + nhãn UTF-8 và lưu frame ra file.
    
    Args:
        frame: numpy array (BGR) - ảnh gốc từ OpenCV
        box_info: dict có các key:
            {
                'bbox': (x1, y1, x2, y2),
                'class_name': str,
                'confidence': float,
                'sharpness': float,
                'score': float
            }
        track_id: int
        output_path: đường dẫn lưu file .jpg
        font_path: font hỗ trợ Unicode (ví dụ Arial, Roboto, NotoSans, ...)
    """
    frame = frameData.frame.copy()
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    font = ImageFont.truetype(font_path, 22)

    x1, y1, x2, y2 = frameData.box_info.bbox
    cls_name = frameData.box_info.class_name
    conf = frameData.box_info.confidence
    sharp = frameData.box_info.sharpness
    score = frameData.score

    # ====== Label chính ======
    label = f"{cls_name} (ID: {track_id})"
    bbox_label = draw.textbbox((0, 0), label, font=font)
    text_w = bbox_label[2] - bbox_label[0]
    text_h = bbox_label[3] - bbox_label[1]

    # Nền cho nhãn
    draw.rectangle(
        [(x1, y1 - text_h - 8), (x1 + text_w + 8, y1)],
        fill=(0, 255, 0, 200)
    )
    draw.text((x1 + 4, y1 - text_h - 6), label, font=font, fill=(0, 0, 0))

    # ====== Chi tiết nhỏ (conf, sharpness, score) ======
    detail = f"Score: {score:.3f} | Conf: {conf:.3f} | Sharp: {sharp:.1f}"
    bbox2 = draw.textbbox((0, 0), detail, font=font)
    detail_w = bbox2[2] - bbox2[0]
    detail_h = bbox2[3] - bbox2[1]

    draw.rectangle(
        [(x1, y2), (x1 + detail_w + 8, y2 + detail_h + 8)],
        fill=(0, 100, 255, 200)
    )
    draw.text((x1 + 4, y2 + 4), detail, font=font, fill=(255, 255, 255))

    # ====== Khung bbox ======
    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)

    # Chuyển lại sang BGR và lưu
    frame_final = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame_final)
