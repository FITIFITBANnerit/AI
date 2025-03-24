def is_inside(banner, holder):
    bx_min, bx_max = banner['x'] - banner['width'] / 2, banner['x'] + banner['width'] / 2
    by_min, by_max = banner['y'] - banner['height'] / 2, banner['y'] + banner['height'] / 2
    hx_min, hx_max = holder['x'] - holder['width'] / 2, holder['x'] + holder['width'] / 2  
    hy_min, hy_max = holder['y'] - holder['height'] / 2, holder['y'] + holder['height'] / 2  
    
    return bx_min >= hx_min and bx_max <= hx_max and by_min >= hy_min and by_max <= hy_max

def convert_yolo_to_orginal(box, orginal_size, resized_size, scale, pad_x, pad_y):
    x, y, w ,h = box['x'], box['y'], box['width'], box['height']
    W_org, H_org = orginal_size
    W_resized, H_resized = resized_size
            
    x = (x - pad_x) / scale
    y = (y - pad_y) / scale
    w = w / scale
    h = h / scale
        
    return x, y, w, h

def save_cord(predictions, dw, dh, scaled, pad_x, pad_y):
        banners = []
        banner_holder = []
        for prediction in predictions:
            x_center, y_center, w, h = convert_yolo_to_orginal(prediction, (dw, dh), (640, 640), scaled, pad_x, pad_y)
            
            prediction['x'] = x_center
            prediction['y'] = y_center
            prediction['width'] = w
            prediction['height'] = h
            
            if prediction['class'] == 'banner':
                banners.append(prediction)
            elif prediction['class'] == 'banner_holder':
                banner_holder.append(prediction)
        return banners, banner_holder