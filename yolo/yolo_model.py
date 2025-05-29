from ultralytics import YOLO

from utils.image_utils import cropped_banner, resize_with_padding
from yolo.yolo_utils import save_cord
import torch

class YOLOModel:
    def __init__(self, model_path=None):
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print("Model loaded succeessfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
    
    def predict(self, image):
        return self.model.predict(source = image, device=self.device)
    
    def detect_banners(self, image, banner_data):
        """YOLO 모델을 사용하여 현수막을 탐지하고 좌표를 반환"""
        height, width = image.shape[:2]
        padded_image, scale, pad_x, pad_y = resize_with_padding(image)

        predictions = self.predict(padded_image)
        if predictions[0].masks is None or not predictions[0].masks.xy:
            return [], []
        class_id = predictions[0].boxes.cls.cpu().numpy()
        boxes = predictions[0].boxes.xywh.cpu().numpy()
        masks = predictions[0].masks.xy     # mask 추가
        banners, banner_holder, bus = save_cord(
            class_id, boxes, masks, 640, 640, scale, pad_x, pad_y, height, width
        )

        return cropped_banner(image, banners, banner_holder, bus, banner_data)