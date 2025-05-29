from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from api.analyze_pipeline import analyze_banner_from_url

from llm.llm_download import download_model_from_s3
from llm.llm_model import BannerTextClassifier
from ocr.ocr_model import OCRModel
from yolo.yolo_model import YOLOModel
from config import MODEL_DIR, LLM_BASE_DIR, LLM_ADAPTER_DIR

# lifespan 컨텍스트 매니저 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting...")

    # S3에서 모델 다운로드
    download_model_from_s3()

    # 모델 초기화
    app.state.yolo = YOLOModel(model_path=MODEL_DIR)
    app.state.ocr = OCRModel()
    app.state.llm = BannerTextClassifier(LLM_BASE_DIR, LLM_ADAPTER_DIR)  # 객체 반환하도록 수정 필요

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
        print("Result:", result)
        results.append(result)
    return {
        "report_id": req.report_id,
        "banner_list": results[0] if len(results) == 1 else results
    }
