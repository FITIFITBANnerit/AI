def select_text(text_list):
    selected = []
    all_text = []
    banner_height = text_list.pop()[1]
    
    for bbox, (text, confidence) in text_list:
        # 네 개의 좌표 분리
        x_coords, y_coords = zip(*bbox)
        # 사각형의 최소 x, y 좌표 및 너비, 높이 계산
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)
        height = max_y - min_y
        if height >  banner_height * 0.1:
            selected.append(text)
        all_text.append(text)
    return selected, all_text

