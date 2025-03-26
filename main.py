import cv2
from config import API_URL, API_KEY, IMAGE_URL, OUTPUT_DIR

from ocr.ocr_model import OCRModel
from yolo.yolo_model import YOLOModel
from utils.image_utils import cropped_banner, resize_with_padding
from yolo.yolo_utils import save_cord




def main():
    yolo_model = YOLOModel(api_key=API_KEY, api_url=API_URL)
    ocr = OCRModel()
    
    image = cv2.imread(IMAGE_URL)
    padded_image, scale, pad_x, pad_y = resize_with_padding(image)
    
    predictions = yolo_model.predict(padded_image, "-mbbh7/2")
    
    banners, banner_holder = save_cord(predictions["predictions"], 640, 640, scale, pad_x, pad_y)
    
    cropped = cropped_banner(image, banners, banner_holder, OUTPUT_DIR)
    
    if cropped:
        results = ocr.run_ocr(cropped)
    
        if results:
            for result in results.items():
                combined_text = " ".join(r for r in result[1])
                print(combined_text, "\n")
            
        else:
            print("No text detected")
    
if __name__ == "__main__":
    main()