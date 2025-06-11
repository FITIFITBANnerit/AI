import numpy as np
import cv2

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

def order_points_final(pts):
    sums = pts.sum(axis=1)
    tl_index = np.argmin(sums)
    ordered_pts = np.roll(pts, -tl_index, axis=0)
    edge1 = ordered_pts[1] - ordered_pts[0]
    edge2 = ordered_pts[3] - ordered_pts[0]

    # 이미지 좌표계(Y축이 아래로 향함)에 맞게 부등호 방향 수정
    if (edge1[0] * edge2[1] - edge1[1] * edge2[0]) < 0:
        ordered_pts[[1, 3]] = ordered_pts[[3, 1]]

    return ordered_pts

def warp_and_mask_auto_size(image, polygon, max_output_width=1000):
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    box = order_points_final(box) # 수정된 함수 호출

    width = np.linalg.norm(box[0] - box[1])
    height = np.linalg.norm(box[0] - box[3])
    if width < 1 or height < 1: return None
    aspect_ratio = width / height
    output_w = min(int(width), max_output_width)
    output_h = int(output_w / aspect_ratio)
    if output_w == 0 or output_h == 0: return None

    dst_pts = np.array([[0, 0], [output_w - 1, 0], [output_w - 1, output_h - 1], [0, output_h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(box, dst_pts)
    warped_img = cv2.warpPerspective(image, M, (output_w, output_h))

    polygon_reshaped = polygon.reshape(-1, 1, 2).astype(np.float32)
    transformed_polygon = cv2.perspectiveTransform(polygon_reshaped, M)
    mask = np.zeros((output_h, output_w), dtype=np.uint8)
    cv2.fillPoly(mask, [transformed_polygon.astype(np.int32)], 255)
    warped_masked = cv2.bitwise_and(warped_img, warped_img, mask=mask)

    return warped_masked

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