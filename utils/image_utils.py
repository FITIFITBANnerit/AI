import cv2
from yolo.yolo_utils import is_inside


def resize_with_padding(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
        
    resized = cv2.resize(image, (new_w, new_h))
        
    pad_x = (target_size[0] - new_w) // 2
    pad_y = (target_size[1] - new_h) // 2
    padded = cv2.copyMakeBorder(resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
    return padded, scale, pad_x, pad_y

def crop_image(image, x, y, w, h):
    return image[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

def cropped_banner(original_image, banners, holders, output_dir):
    cropped_banners = []
        
    for banner in banners:
        for holder in holders:
            if is_inside(banner, holder):
                banner['lebal'] = True
                break
                    
        else:
            banner['legal'] = None
            cropped_banners.append([banner['x'], banner['y'], banner['width'], banner['height']])
    
    return cropped_banners