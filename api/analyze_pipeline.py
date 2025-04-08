import cv2
import numpy as np

from config import API_URL, API_KEY, IMAGE_URL, MODEL_DIR, LLM_NAME
from llm.llm_model import BannerTextClassifier
from llm.llm_utils import analyze_banner_text
from ocr.ocr_model import OCRModel
from yolo.yolo_model import YOLOModel
from network.image_loader import load_image_from_url

def analyze_banner_from_url(image_url: str):
    """메인 실행 함수"""
    # 모델 초기화
    yolo_model = YOLOModel(model_path=MODEL_DIR)
    ocr = OCRModel()
    llm = BannerTextClassifier(LLM_NAME)

    # 이미지 로드 및 현수막 탐지
    image = load_image_from_url(image_url)
    #image = cv2.imread(IMAGE_URL)
    image = np.array(image)
    image = image[:, :, ::-1]
    
    banner_data = []
    
    cropped = yolo_model.detect_banners(image, banner_data)
    if not cropped:
        print("No banners detected.")
        return banner_data

    # OCR 실행
    results = ocr.run_ocr(image, cropped)

    if not results:
        print("No text detected in banners.")
        return banner_data

    # 현수막 분석
    analyze_banner_text(results, llm, cropped, banner_data)
    
    return banner_data