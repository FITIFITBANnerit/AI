import cv2
from config import API_URL, API_KEY, IMAGE_URL, OUTPUT_DIR
from crop_banners import cropped_banner
from yolo.yolo_model import YOLOModel
from yolo.image_utils import convert_yolo_to_orginal, resize_with_padding
from yolo.yolo_utils import save_cord




def main():
    model = YOLOModel(api_key=API_KEY, api_url=API_URL)
    
    image = cv2.imread(IMAGE_URL)
    padded_image, scale, pad_x, pad_y = resize_with_padding(image)
    
    predictions = model.predict(padded_image, "-mbbh7/2")
    
    banners, banner_holder = save_cord(predictions["predictions"], 640, 640, scale, pad_x, pad_y)
    
    cropped_banner(image, banners, banner_holder, OUTPUT_DIR)
    
if __name__ == "__main__":
    main()