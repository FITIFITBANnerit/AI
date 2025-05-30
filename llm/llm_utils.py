import re

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

def analyze_banner_text(ocr_texts, llm, cropped, banner_data, cropped_info):
    """LLM을 이용하여 현수막의 불법 여부, 카테고리, 전화번호 및 회사명을 추출"""
    for i, key in enumerate(ocr_texts):
        if ocr_texts[key] == "NO_TEXT":
            continue
        elif len(ocr_texts[key]) > 1:
            classification, category, info = llm.process_banner_text(
                ocr_texts[key]
            )
            company_name, phone_number = extract_company_info(info)
            print(f"Company Name: {company_name}, Phone Number: {phone_number}")
            banner_data.append(
                        {
                            "status": classification,
                            "category": category,
                            "company_name": company_name,
                            "phone_number": str(phone_number),
                            "center": [float(cropped_info[i][0]), float(cropped_info[i][1])],
                            'width': float(cropped_info[i][2]),
                            'height': float(cropped_info[i][3]), # 해당 배너의 좌표 추가
                        },
                    )

