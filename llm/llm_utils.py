import re

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
    if len(selected) == 0:
        selected = all_text
        
    return selected, all_text

def extract_company_info(info):
    """LLM 결과에서 회사명과 전화번호를 추출"""
    
    # 기본값 설정 (None 대신 'Not Found'로 초기화)
    company_name = "Not Found"
    phone_number = "Not Found"
    
    if info:
        # 정규식 패턴
        company_pattern = r'Company:\s*(.+)'
        phone_pattern = r'Phone Number:\s*"?([\d\s-]+)"?'

        # 정규식 매칭
        company_match = re.search(company_pattern, info)
        phone_match = re.search(phone_pattern, info)

        # 값이 존재하면 변수에 저장
        if company_match:
            company_name = company_match.group(1).strip()

        if phone_match:
            phone_number = re.sub(r'\D', '', phone_match.group(1))  # 숫자만 추출

    return company_name, phone_number

def analyze_banner_text(ocr_texts, llm, cropped, banner_data):
    """LLM을 이용하여 현수막의 불법 여부, 카테고리, 전화번호 및 회사명을 추출"""
    for i, key in enumerate(ocr_texts):
        if len(ocr_texts[key]) > 1:
            select, all_select = select_text(ocr_texts[key])
            classification, category, info = llm.process_banner_text(
                " ".join(select), " ".join(all_select)
            )
            company_name, phone_number = extract_company_info(info)
            banner_data.append(
                        {
                            "status": classification,
                            "category": category,
                            "company_name": company_name,
                            "phone_number": str(phone_number),
                            "coordinates": {
                                "x": cropped[i][0],
                                "y": cropped[i][1],
                                "width": cropped[i][2],
                                "height": cropped[i][3],
                            },  # 해당 배너의 좌표 추가
                        },
                    )

