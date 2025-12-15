from dataclasses import dataclass

@dataclass
class BoxInfo:
    bbox: tuple
    confidence: float
    class_name: str
    sharpness: float

@dataclass
class FrameData:
    id: int
    frame: any
    score: float
    box_info: BoxInfo