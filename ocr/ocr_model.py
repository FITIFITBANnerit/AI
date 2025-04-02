from paddleocr import PaddleOCR
import requests
import numpy as np

from ocr.ocr_utils import OCRPreprocessing
from utils.image_utils import crop_image

class OCRModel:
    def __init__(self, lang: str = "korean", **kwargs):
        self.lang = lang
        self._ocr = PaddleOCR(lang="korean", show_log=False)

    def run_ocr(self, original_image, images_cord):
        ocr_results = {}
        for i, cord in enumerate(images_cord):
            
            image = crop_image(original_image, cord[0], cord[1], cord[2], cord[3])  # banner 부분만 추출
            
            ocr_preprocessing = OCRPreprocessing(image)
            preprocessing_image = ocr_preprocessing.image_preprocessing()
            result = self._ocr.ocr(preprocessing_image, cls=False)
            ocr_results[i] = result[0]
            ocr_results[i].append((cord[2], cord[3]))
        return ocr_results
        
        