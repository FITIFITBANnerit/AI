from paddleocr import PaddleOCR

from ocr.ocr_utils import OCRPreprocessing
from utils.image_utils import crop_image

class OCRModel:
    def __init__(self, lang: str = "korean", **kwargs):
        self.lang = lang
        self._ocr = PaddleOCR(lang="korean", show_log=False)

    def run_ocr(self, original_image, images_cord, banner_data):
        ocr_results = {}
        for i, cord in enumerate(images_cord):
            
            image = crop_image(original_image, cord[0], cord[1], cord[2], cord[3])  # banner 부분만 추출
            
            ocr_preprocessing = OCRPreprocessing(image)
            preprocessing_image = ocr_preprocessing.image_preprocessing()
            result = self._ocr.ocr(preprocessing_image, cls=False)
            
            if result[0] != None:
                ocr_results[i] = result[0]
                ocr_results[i].append((cord[2], cord[3])) # 배너 너비, 높이 추가
            else:
                banner_data.append({
                    "status": "UNKNOWN",
                    "category": "banner",
                    "company_name": "",
                    "phone_number": "",
                    "coordinates":{
                        "x": cord[0],
                        "y": cord[1],
                        "width": cord[2],
                        "height": cord[3],
                    }
                })
                ocr_results[i] = "NO_TEXT"
        return ocr_results, banner_data
        
        