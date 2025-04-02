import cv2
from config import API_URL, API_KEY, IMAGE_URL, OUTPUT_DIR

from llm.llm_utils import select_text
from ocr.ocr_model import OCRModel
from yolo.yolo_model import YOLOModel
from utils.image_utils import cropped_banner, resize_with_padding
from yolo.yolo_utils import save_cord


import math
import time



def main():
    yolo_model = YOLOModel(api_key=API_KEY, api_url=API_URL)
    ocr = OCRModel()
    
    image = cv2.imread(IMAGE_URL)
    padded_image, scale, pad_x, pad_y = resize_with_padding(image)  # 이미지 패딩 추가 & 640 x 640 리사이징
    
    predictions = yolo_model.predict(padded_image, "-mbbh7/2")
    
    banners, banner_holder = save_cord(predictions["predictions"], 640, 640, scale, pad_x, pad_y)
    
    cropped = cropped_banner(image, banners, banner_holder, OUTPUT_DIR)
    
    if cropped:
        results = ocr.run_ocr(image, cropped)
        if results:
            for key in results:
                select, all_selected = select_text(results[key])
                
        else:
            print("No text detected")
    
    return
    
if __name__ == "__main__":
    
    start = time.time()
    main()
    end = time.time()
    print(f"{end - start:.5f} sec")