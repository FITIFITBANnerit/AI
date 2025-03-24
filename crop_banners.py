import cv2
from yolo.yolo_utils import is_inside

def crop_image(image, x, y, w, h):
    return image[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

def cropped_banner(original_image, banners, holders, output_dir):
    height, width, _ = original_image.shape
        
    for banner in banners:
        for holder in holders:
            if is_inside(banner, holder):
                banner['legal'] = True
                break
                    
        else:
            cropped = crop_image(original_image, banner['x'], banner['y'], banner['width'], banner['height'])
            save_path = f"{output_dir}/{banner['detection_id']}.jpg"
            cv2.imwrite(save_path, cropped)