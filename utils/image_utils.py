import cv2
from yolo.yolo_utils import is_inside

def resize_with_padding(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
        
    resized = cv2.resize(image, (new_w, new_h))
        
    pad_x = (target_size[0] - new_w) // 2
    pad_y = (target_size[1] - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, pad_y, target_size[1] - new_h - pad_y,
        pad_x, target_size[0] - new_w - pad_x,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
        
    return padded, scale, pad_x, pad_y

def crop_image(image, x, y, w, h):
    h_img, w_img = image.shape[:2]
    
    x1 = max(0, int(x - w / 2))
    y1 = max(0, int(y - h / 2))
    x2 = min(w_img, int(x + w / 2))
    y2 = min(h_img, int(y + h / 2))
    return image[y1:y2, x1:x2]

def cropped_banner(original_image, banners, holders, banner_data):
    cropped_banners = []
        
    for banner in banners:
        for holder in holders:
            if is_inside(banner, holder):
                banner_data.append(
                    {
                        "status": "LEGAL",
                        "category": "banner_in_holder",
                        "company_name": "",
                        "phone_number": "",
                        "coordinates":{
                            "x": banner['x'],
                            "y": banner['y'],
                            "width": banner['width'],
                            "height": banner['height'],
                        }
                    }
                )
                break
                    
        else:
            cropped_banners.append([banner['x'], banner['y'], banner['width'], banner['height']])
    
    return cropped_banners