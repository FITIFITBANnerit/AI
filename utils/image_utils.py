import cv2
import numpy as np
from yolo.yolo_utils import is_inside

def resize_with_padding(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
        
    resized = cv2.resize(image, (new_w, new_h))
        
    pad_x = (target_size[0] - new_w) // 2
    pad_y = (target_size[1] - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, 
        pad_y, target_size[1] - new_h - pad_y,
        pad_x, target_size[0] - new_w - pad_x,
        cv2.BORDER_CONSTANT, 
        value=(0, 0, 0)
    )
        
    return padded, scale, pad_x, pad_y

def crop_image(image, mask_polygon):
    mask_polygon_int = mask_polygon.astype(np.int32)
    
    h_img, w_img = image.shape[:2]
    background = np.zeros((h_img, w_img), dtype=np.uint8)

    # 복원된 폴리곤으로 마스크 채우기
    cv2.fillPoly(background, [mask_polygon], 255)

    # 원본 이미지에 마스크 적용
    masked_img = cv2.bitwise_and(image, image, mask=background)

    # 마스크 영역만 크롭
    x, y, w, h = cv2.boundingRect(mask_polygon)
    cropped_masked_img = masked_img[y:y+h, x:x+w]
    
    return cropped_masked_img

def cropped_banner(original_image, banners, holders, bus, banner_data):
    cropped_banners = []
    cropped_banners_coods = []
    for banner in banners:
        is_legal = False
        for holder in holders:
            if is_inside(banner, holder):
                banner_data.append(
                    {
                        "status": "LEGAL",
                        "category": "banner_in_holder",
                        "company_name": "",
                        "phone_number": "",
                        "center": [float(banner['x']), float(banner['y'])],
                        'width': float(banner['width']),
                        'height': float(banner['height'])
                    }
                )
                is_legal = True
                break

        for bus_item in bus:
            if is_inside(banner, bus_item):
              is_legal = True
              break    
                    
        if not is_legal:
            cropped_banners.append(banner['mask'])
            cropped_banners_coods.append([float(banner['x']), float(banner['y']), float(banner['width']), float(banner['height'])])
    return cropped_banners, cropped_banners_coods