import numpy as np

def is_inside(banner, holder):
    expand_ratio = 0.4
    expanded_width = holder['width'] * (1 + expand_ratio)
    expanded_height = holder['height'] * (1 + expand_ratio)

    bx_min, bx_max = banner['x'] - banner['width'] / 2, banner['x'] + banner['width'] / 2
    by_min, by_max = banner['y'] - banner['height'] / 2, banner['y'] + banner['height'] / 2
    hx_min = holder['x'] - expanded_width / 2
    hx_max = holder['x'] + expanded_width / 2
    hy_min = holder['y'] - expanded_height / 2
    hy_max = holder['y'] + expanded_height / 2 
    
    return bx_min >= hx_min and bx_max <= hx_max and by_min >= hy_min and by_max <= hy_max

# Bounding box 좌표변환
def convert_yolo_to_orginal(box, orginal_size, resized_size, scale, pad_x, pad_y):
    x, y, w ,h = box[0], box[1], box[2], box[3]
    W_org, H_org = orginal_size
    W_resized, H_resized = resized_size
            
    x = (x - pad_x) / scale
    y = (y - pad_y) / scale
    w = w / scale
    h = h / scale
        
    return x, y, w, h

# Segmentation 복원 좌표 계산
def restore_coords(polygon, pad_x, pad_y, scale, original_width, original_height):
    restored_polygon = []
    for x, y in polygon:
        restored_x = (x - pad_x) / scale
        restored_y = (y - pad_y) / scale
        
        restored_x = min(max(restored_x, 0), original_width - 1)
        restored_y = min(max(restored_y, 0), original_height - 1)
        
        restored_polygon.append([restored_x, restored_y])
    return np.array(restored_polygon, dtype=np.int32)
    
def save_cord(class_id, boxes, masks, dw, dh, scaled, pad_x, pad_y, original_height, original_width):    # mask 추가
        banners = []
        banner_holder = []
        bus = []
        for i, box in enumerate(boxes):
            prediction = {}
            x_center, y_center, w, h = convert_yolo_to_orginal(box, (dw, dh), (640, 640), scaled, pad_x, pad_y)
            
            prediction['x'] = x_center
            prediction['y'] = y_center
            prediction['width'] = w
            prediction['height'] = h
            
            if class_id[i] == 0:
                prediction['class'] = 'banner_holder'
                banner_holder.append(prediction)
            elif class_id[i] == 1:
                prediction['class'] = 'banner'
                prediction['mask'] = restore_coords(masks[i], pad_x, pad_y, scaled, original_height, original_width)
                banners.append(prediction)
            elif class_id[i] == 2:
                prediction['class'] = 'bus'
                bus.append(prediction)
                
        return banners, banner_holder, bus