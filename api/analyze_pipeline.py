import numpy as np
from llm.llm_model import BannerTextClassifier
from llm.llm_utils import analyze_banner_text
from ocr.ocr_model import OCRModel
from yolo.yolo_model import YOLOModel
from network.image_loader import load_image_from_url

def analyze_banner_from_url(image_url: str, app):
    """메인 실행 함수"""
    
    # 미리 초기화된 모델 가져오기
    yolo_model = app.state.yolo
    ocr = app.state.ocr
    llm = app.state.llm

    # 이미지 로드 및 현수막 탐지
    image = load_image_from_url(image_url)
    image = np.array(image)
    image = image[:, :, ::-1]
    
    banner_data = []
    
    cropped, cropped_info = yolo_model.detect_banners(image, banner_data)
    
    if not cropped:
        print("No banners detected.")
        return banner_data
    
    if len(cropped) != len(cropped_info):
        print("Not same Length!!!!!")
    

    # OCR 실행
    results, banner_data = ocr.run_ocr(image, cropped, banner_data)

    if not results:
        print("No text detected in banners.")
        return banner_data

    # 현수막 분석
    analyze_banner_text(results, llm, cropped, banner_data, cropped_info)
    
    return banner_data