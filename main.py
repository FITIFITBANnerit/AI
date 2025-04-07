
from api.analyze_pipeline import analyze_banner_from_url
from config import IMAGE_URL

if __name__ == "__main__":
    # URL에서 현수막 탐지
    result = analyze_banner_from_url(IMAGE_URL)
    
    # 결과 출력
    print(result)

