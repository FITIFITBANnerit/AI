import cv2
from config import API_URL, API_KEY, IMAGE_URL, OUTPUT_DIR, LLM_NAME
import json
from llm.llm_model import BannerTextClassifier
from llm.llm_utils import select_text
from ocr.ocr_model import OCRModel
from yolo.yolo_model import YOLOModel
from utils.image_utils import cropped_banner, resize_with_padding
from yolo.yolo_utils import save_cord

import math
import time


def main():
    # load model
    yolo_model = YOLOModel(api_key=API_KEY, api_url=API_URL)
    ocr = OCRModel()
    llm = BannerTextClassifier(LLM_NAME)
    
    image = cv2.imread(IMAGE_URL)
    padded_image, scale, pad_x, pad_y = resize_with_padding(image)  # 이미지 패딩 추가 & 640 x 640 리사이징
    
    predictions = yolo_model.predict(padded_image, "-mbbh7/2") # yolo 모델을 통해 감지된 결과
    
    banners, banner_holder = save_cord(predictions["predictions"], 640, 640, scale, pad_x, pad_y) # banner와 banner_holder 좌표 실제 이미지에 맞게 수정 후 banners와 banner_holder 정보 분리
    
    cropped = cropped_banner(image, banners, banner_holder, OUTPUT_DIR) # 지정게시대에 있는 현수막을 제외한 나머지 현수막에 대한 정보
    
    if not cropped:
            print("No illegal banners detected.")
            return json.dumps({"banners": []})  # 빈 리스트 반환
    
    banner_data = []
    
    results = ocr.run_ocr(image, cropped)   # cropped에는 x, y, width, height이 저장

    if not results:
            print("No text detected in banners.")
            return json.dumps({"banners": []})
        
    for i, key in enumerate(results):
        if len(results[key]) > 1:
            select, all_select = select_text(results[key])
            print(select, all_select)
            classification, info, category = llm.process_banner_text(' '.join(select), ' '.join(all_select))
            print(f"Legal/Illigal: {classification}, company info: {info}")
        
        # JSON 데이터 구조 생성
            banner_info = {
                "text": ' '.join(select),
                "classification": classification,
                "category": category,
                "info": info,
                "coordinates": cropped[i]  # 해당 배너의 좌표 추가
            }
            banner_data.append(banner_info)
        
    json_result = json.dumps({"banners": banner_data}, ensure_ascii=False, indent=4)
    print(json_result)  # JSON 출력
    return json_result
    
if __name__ == "__main__":
    
    start = time.time()
    main()
    end = time.time()
    print(f"{end - start:.5f} sec")