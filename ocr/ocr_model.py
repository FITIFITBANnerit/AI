from paddleocr import PaddleOCR
import requests
import numpy as np

class OCRModel:
    def __init__(self, lang: str = "korean", **kwargs):
        self.lang = lang
        self._ocr = PaddleOCR(lang="korean", show_log=False)
        self.img_path = None
        self.ocr_result = {}

    def run_ocr(self, images):
        ocr_dic = {}
        for i, image in enumerate(images):
            ocr_text = []
            result = self._ocr.ocr(image, cls=False)
            self.ocr_result = result[0]

            if self.ocr_result:
                for r in result[0]:
                    ocr_text.append((r[1][0]))
                    
            ocr_dic[i] = ocr_text

        return ocr_dic
        
        