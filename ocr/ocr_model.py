from paddleocr import PaddleOCR

from ocr.ocr_utils import OCRPreprocessing
from utils.image_utils import crop_image
import numpy as np

from yolo.yolo_utils import warp_and_mask_auto_size

class OCRModel:
    def __init__(self, lang: str = "korean", **kwargs):
        self.lang = lang
        self._ocr = PaddleOCR(lang="korean", show_log=False, use_gpu=False)

    def line_length(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def summarize_ocr_result(self, ocr_result):
        summarized = []

        for item in ocr_result:
            box, (text, conf) = item
            if not text.strip():
                continue

            # 중심 좌표
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            cx = np.mean(x_coords)
            cy = np.mean(y_coords)

            # font 크기 추정 (좌우 세로 edge 평균 길이)
            left_height = self.line_length(box[0], box[3])
            right_height = self.line_length(box[1], box[2])
            avg_font_size = (left_height + right_height) / 2

            summarized.append({
                "text": text,
                "font_size": round(float(avg_font_size), 2),
                "center": [round(float(cx), 1), round(float(cy), 1)],
                "conf": round(float(conf), 2)
            })

        # 폰트 크기 기준 정렬 (큰 글씨 우선)
        summarized.sort(key=lambda x: -x["font_size"])
        return summarized

    def run_ocr(self, original_image, images_cord, banner_data, cropped_info):
        ocr_results = {}
        for i, cord in enumerate(images_cord):
            
            image = crop_image(original_image, cord)  # banner 부분만 추출
            try:
                final_image = warp_and_mask_auto_size(original_image, cord)
            except Exception as e:
                print(f"warp_and_mask_auto_size 실패: {e}")
                final_image = image
            
            ocr_preprocessing = OCRPreprocessing(final_image)
            preprocessing_image = ocr_preprocessing.image_preprocessing()
            result = self._ocr.ocr(preprocessing_image, cls=False)
            
            if result[0] != None:
                res = self.summarize_ocr_result(result[0])
                ocr_results[i] = res
            else:
                banner_data.append({
                    "status": "UNKNOWN",
                    "category": "banner",
                    "company_name": "",
                    "phone_number": "",
                    "center": [float(cropped_info[i][0]), float(cropped_info[i][1])],
                    'width': float(cropped_info[i][2]),
                    'height': float(cropped_info[i][3]), # 해당 배너의 좌표 추가
                })
                ocr_results[i] = "NO_TEXT"
        return ocr_results, banner_data
        
        