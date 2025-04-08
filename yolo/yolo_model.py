from inference_sdk import InferenceHTTPClient
import cv2
from ultralytics import YOLO

from utils.image_utils import cropped_banner, resize_with_padding
from yolo.yolo_utils import save_cord

class YOLOModel:
    def __init__(self, model_path=None):
        self.model = YOLO(model_path)
        #self.client = InferenceHTTPClient(api_url, api_key)
    
    def predict(self, image):
        return self.model.predict(source = image)
    
    def detect_banners(self, image, banner_data):
        """YOLO 모델을 사용하여 현수막을 탐지하고 좌표를 반환"""
        padded_image, scale, pad_x, pad_y = resize_with_padding(image)

        predictions = self.model.predict(padded_image)
        class_id = predictions[0].boxes.cls.cpu().numpy()
        boxes = predictions[0].boxes.xywh.cpu().numpy()
        banners, banner_holder = save_cord(
            class_id, boxes, 640, 640, scale, pad_x, pad_y
        )

        return cropped_banner(image, banners, banner_holder, banner_data)
    
    
"""def detect_banners(yolo_model, image):
    # YOLO 모델을 사용하여 현수막을 탐지하고 좌표를 반환
    padded_image, scale, pad_x, pad_y = resize_with_padding(image)

    predictions = yolo_model.predict(padded_image)
    class_id = predictions[0].boxes.cls.cpu().numpy()
    boxes = predictions[0].boxes.xywh.cpu().numpy()
    banners, banner_holder = save_cord(
        class_id, boxes, 640, 640, scale, pad_x, pad_y
    )

    return cropped_banner(image, banners, banner_holder)"""