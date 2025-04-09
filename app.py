from fastapi import FastAPI
from pydantic import BaseModel
from api.analyze_pipeline import analyze_banner_from_url
from typing import List
app = FastAPI()

class ImageRequest(BaseModel):
    report_id: int
    image_urls: List[str]
    
@app.post("/analyze")
def analyze(req: ImageRequest):
    results = []
    for url in req.image_urls:
        result = analyze_banner_from_url(url) # req.image_url에 URL 저장
        results.append(result)
    
    return {
        "report_id": req.report_id,
        "banner_list": results[0] if len(results) == 1 else results
    }
