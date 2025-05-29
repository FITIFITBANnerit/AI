from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from peft import PeftModel
import re

class BannerTextClassifier:
    def __init__(self, llm_base_path, llm_adapter_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            llm_base_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # 원본 모델에 맞는 데이터 타입을 지정하는 것이 좋습니다.
            device_map="auto" if device == "cuda" else None # 여러 GPU에 모델을 분산 로드
            ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_base_path, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(
            self.base_model,
            llm_adapter_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
        self.device = device
        
    def classify_banner_text(self, full_text):
        prompt = """
                    Classify a banner into:
                    - Politics
                    - Public interest
                    - Commercial purposes
                    - Other

                    Input(OCR blocks with text, font size, center (x, y), and confidence): {banner_info}

                    Guidelines:
                    - Politics: mentions of politicians, parties, elections (e.g., 국민의힘, 이재명)
                    - Public interest: events or announcements (e.g., 축제, 헌혈, 환경)
                    - Commercial purposes: ads or services (e.g., 세일, 병원, 학원)
                    - Other: anything unclear or unrelated

                    Instructions:
                    - Prioritize blocks with largest font size and highest confidence
                    - Use center (x, y) to determine visual reading order (top-to-bottom, left-to-right)
                    - Nearby centers imply strong contextual continuity
                    - Group texts as humans would read
                    - Output only one category, no explanation or copied text

                    Answer: or <|assistant|>
                """
        
        full_prompt = prompt.format(banner_info=full_text)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=20, # 분류 결과만 받으면 되므로 길게 설정할 필요 없음
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        response_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return response_text.strip()

    def extract_info(self, full_text):
        no_info_placeholder_value = "Not detected"
        
        prompt = """
                    
                    You are an expert data extractor specializing in analyzing unstructured text from OCR (Optical Character Recognition) scans. Your task is to accurately extract the company/store name and phone number from the provided text. The text may contain OCR errors, so be prepared to correct common mistakes.

                    **Source Text:**
                    `{banner_info}`

                    **Instructions:**

                    1.  **Company/Store Name:**
                        * Identify the most likely name of the business, store, restaurant, or service.
                        * This is often indicated by larger font size, a logo, or keywords like "마트" (Mart), "의원" (Clinic), "가구" (Furniture), "컴퍼니" (Company), etc.
                        * If multiple potential names exist, choose the most prominent one.

                    2.  **Phone Number:**
                        * Scan for any sequence of digits that resembles a phone number (e.g., `XXX-XXXX-XXXX`, `(0XX) XXX-XXXX`, `010.XXXX.XXXX`).
                        * Correct common OCR errors, such as mistaking 'O' for '0', 'l' for '1', or ignoring spaces/hyphens. For example, if you see `O1O-1234-5b78`, you should interpret it as `010-1234-5678`.
                        * If there are multiple numbers, prioritize the main business landline or mobile number over fax or secondary numbers.

                    3.  **Output Format:**
                        * You MUST strictly follow the format below.
                        * If a piece of information cannot be found, use the value `{no_info}`.

                    **Output:**
                    ```
                    "Company": "Company Name",
                    "Phone Number": "Phone Number"
                    
                """
                
        full_prompt = prompt.format(banner_info=full_text)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        outputs = self.base_model.generate(
            **inputs,
            max_new_tokens=100, 
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        response_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return response_text.strip()
    
    def normalize(self, text):
        low_text = text.lower()
        if 'politics' in low_text:
            return 'Politics'
        elif 'public interest' in low_text:
            return 'Public interest'
        elif 'commercial purposes' in low_text:
            return 'Commercial purposes'
        else:
            return 'Other'
    
    def process_banner_text(self, full_text):
        """현수막 텍스트를 분석하고 불법이라면 추가 정보를 추출."""
        classification_result = self.classify_banner_text(full_text)
        info = None

        # 찾으면 저장, 없으면 "Unknown"으로 설정
        category = self.normalize(classification_result)
        
        if category == "Commercial purposes":
            result = "ILLEGAL"
            info = self.extract_info(full_text)

        else:
            result = "LEGAL"
        
        return result, category, info
                
                