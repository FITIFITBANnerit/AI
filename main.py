import cv2
import json
import time
import re
import math

from config import API_URL, API_KEY, IMAGE_URL, OUTPUT_DIR, LLM_NAME
from llm.llm_model import BannerTextClassifier
from llm.llm_utils import extract_company_info, select_text
from ocr.ocr_model import OCRModel
from yolo.yolo_model import YOLOModel
from utils.image_utils import cropped_banner, resize_with_padding
from yolo.yolo_utils import save_cord


def detect_banners(image):
    """YOLO 모델을 사용하여 현수막을 탐지하고 좌표를 반환"""
    yolo_model = YOLOModel(api_key=API_KEY, api_url=API_URL)
    padded_image, scale, pad_x, pad_y = resize_with_padding(image)

    predictions = yolo_model.predict(padded_image, "-mbbh7/2")
    banners, banner_holder = save_cord(
        predictions["predictions"], 640, 640, scale, pad_x, pad_y
    )

    return cropped_banner(image, banners, banner_holder, OUTPUT_DIR)


def analyze_banner_text(ocr_texts, llm, cropped):
    """LLM을 이용하여 현수막의 불법 여부, 카테고리, 전화번호 및 회사명을 추출"""
    banner_data = []

    for i, key in enumerate(ocr_texts):
        if len(ocr_texts[key]) > 1:
            select, all_select = select_text(ocr_texts[key])
            classification, category, info = llm.process_banner_text(
                " ".join(select), " ".join(all_select)
            )
            company_name, phone_number = extract_company_info(info)
            print(classification)
            if classification == "illegal":
                banner_data.append(
                    {
                        "text": " ".join(select),
                        "classification": classification,
                        "category": category,
                        "company_name": company_name,
                        "phone_number": str(phone_number),
                        "coordinates": {
                            "x": cropped[i][0],
                            "y": cropped[i][1],
                            "width": cropped[i][2],
                            "height": cropped[i][3],
                        },  # 해당 배너의 좌표 추가
                    },
                )

    return banner_data


def main():
    """메인 실행 함수"""
    start_time = time.time()

    # 모델 초기화
    ocr = OCRModel()
    llm = BannerTextClassifier(LLM_NAME)

    # 이미지 로드 및 현수막 탐지
    image = cv2.imread(IMAGE_URL)
    cropped = detect_banners(image)

    if not cropped:
        print("No illegal banners detected.")
        return json.dumps({"banners": []})

    # OCR 실행
    results = ocr.run_ocr(image, cropped)

    if not results:
        print("No text detected in banners.")
        return json.dumps({"banners": []})

    # 현수막 분석
    banner_data = analyze_banner_text(results, llm, cropped)

    # JSON 변환 및 출력
    json_result = json.dumps({"banners": banner_data}, ensure_ascii=False, indent=4)
    print(json_result)

    print(f"Execution Time: {time.time() - start_time:.5f} sec")
    return json_result


if __name__ == "__main__":
    main()
