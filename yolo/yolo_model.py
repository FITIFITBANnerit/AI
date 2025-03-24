from inference_sdk import InferenceHTTPClient
import cv2

class YOLOModel:
    def __init__(self, api_url, api_key):
        self.client = InferenceHTTPClient(api_url, api_key)
    
    def predict(self, image_url, model_id):
        return self.client.infer(image_url, model_id)