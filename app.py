from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from api.analyze_pipeline import analyze_banner_from_url

from llm.llm_download import get_openai_api_key_from_aws
from llm.llm_model import BannerTextClassifier
from ocr.ocr_model import OCRModel
from yolo.yolo_model import YOLOModel
from config import MODEL_DIR

# lifespan 컨텍스트 매니저 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting...")

    # api_key 가져오기
    ssm_param_name_for_openai_key = "/bannerit/api_keys/openai"
    retrieved_api_key = get_openai_api_key_from_aws(ssm_param_name_for_openai_key)

    # 모델 초기화
    app.state.yolo = YOLOModel(model_path=MODEL_DIR)
    app.state.ocr = OCRModel()
    
    if retrieved_api_key:
        app.state.llm = BannerTextClassifier(api_key=retrieved_api_key, model_name="gpt-3.5-turbo-0125")  # 객체 반환하도록 수정 필요
        print("✅ BannerTextClassifier (LLM) initialized with OpenAI API key.")
    else:
        app.state.llm = None
        print("⚠️ BannerTextClassifier (LLM) could not be initialized: OpenAI API key not found.")

    print("✅ All models initialized.")
    yield
    print("🛑 Server shutdown...")

# lifespan 적용
app = FastAPI(lifespan=lifespan)

class ImageRequest(BaseModel):
    report_id: int
    image_urls: List[str]
    
@app.post("/analyze")
def analyze(req: ImageRequest):
    results = []
    for url in req.image_urls:
        result = analyze_banner_from_url(url, app) # req.image_url에 URL 저장
        print(result)
        results.append(result)
    
    return {
        "report_id": req.report_id,
        "banner_list": results[0] if len(results) == 1 else results
    }
