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
        prompt = f"""<|user|>
                    Classify a banner into:
                    - Politics
                    - Public interest
                    - Commercial purposes
                    - Other

                    Input(OCR blocks with text, font size, center (x, y), and confidence): {full_text}

                    Guidelines:
                    - Politics: mentions of politicians, parties, elections (e.g., 국민의힘, 이재명)
                    - Public interest: events or announcements (e.g. 축제, 헌혈, 환경)
                    - Commercial purposes: ads or services (e.g. 세일, 병원, 학원, 일반 분양, 씽크대, 가구마트, 인테리어, 도장, 수강생,모집, 실측,오픈기념, 테이블당, 공짜)
                    - Other: anything unclear or unrelated

                    **Important**:
                    - If any word in the input is similar to the commercial keywords(e.g. 세일, 병원, 학원, 일반 분양, 씽크대, 가구마트, 인테리어, 도장, 수강생,모집, 실측) above (even if the spelling is slightly incorrect, e.g., "찡크대", "쒸크대", "가굳마트"), you must **unconditionally classify** the banner as **Commercial purposes**.
                    - Consider common OCR errors such as:
                    * 'ㅅ' ↔ 'ㅊ' (e.g., 씽 ↔ 찡)
                    * 'ㄱ' ↔ 'ㅋ', 'ㄷ' ↔ 'ㅌ'
                    * spacing and hyphen issues
                    * Latin letter confusion (e.g., 'O' ↔ '0', 'l' ↔ '1')
  
                    Instructions:
                    - Prioritize blocks with largest font size and highest confidence
                    - Use center (x, y) to determine visual reading order (top-to-bottom, left-to-right)
                    - Nearby centers imply strong contextual continuity
                    - Group texts as humans would read
                    - Output only one category, no explanation or copied text
                    
                    Output only one category, no explanation or copied text
                    Answer: 
                <|assistant|>
                """
        
        #full_prompt = prompt.format(banner_info=full_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10, # 분류 결과만 받으면 되므로 길게 설정할 필요 없음
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in response_text:
            classification_result = response_text.split("<|assistant|>")[-1].strip()
        else:
            classification_result = response_text.strip()
        
        print("Classification Result: ", classification_result)
        return classification_result

    def extract_info(self, full_text):
        no_info_text = "Not detected"
        
        prompt = f"""
                    
                    You are an expert data extractor specializing in analyzing banner advertisements from OCR scans. The text you receive has already been sorted by visual position: top to bottom, and left to right within each line. Your task is to extract key business information from the text.

                    **Source Text (line-sorted OCR output):**
                    {full_text}

                    **Instructions:**

                    1. **Company/Store Name**
                    - Find the most likely name of the company, store, restaurant, or service.
                    - It may contain business-related keywords such as "마트", "가구", "의원", "센터", "건설", "치과", etc.
                    - If multiple candidates exist, pick the one that appears near the top of the text or has a promotional tone.

                    2. **Phone Number**
                    - Detect any phone number in common formats (e.g., `010-XXXX-XXXX`, `(02) XXXX-XXXX`, etc).
                    - Correct common OCR mistakes: e.g., 'O' → '0', 'l' → '1', etc.
                    - If there are multiple numbers, choose the main contact number, not fax or secondary lines.

                    3. **Output Format**
                    You MUST return the result strictly in the format below:

                    "Company": "Extracted Company Name or {no_info_text}",
                    "Phone Number": "Extracted Phone Number or {no_info_text}"
                    
                    
                """
                
        #full_prompt = prompt.format(banner_info=full_text, no_info=no_info_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.base_model.generate(
            **inputs,
            max_new_tokens=80, # 분류 결과만 받으면 되므로 길게 설정할 필요 없음
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("extract info: ", response_text)
        return response_text
    
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
        print("ocr_text(Non-processing): ", full_text)
        classification_result = self.classify_banner_text(full_text)
        info = None

        # 찾으면 저장, 없으면 "Unknown"으로 설정
        category = self.normalize(classification_result)
        
        if category == "Commercial purposes":
            result = "ILLEGAL"
            sorted_items = sorted(full_text, key=lambda o: (round(o['center'][1] / 10), o['center'][0]))

            # 줄 단위로 하나의 문자열 만들기
            lines = []
            current_y = None
            line = []

            for item in sorted_items:
                y = round(item['center'][1] / 10)  # y값 비슷한 건 같은 줄로 처리
                if current_y is None or y == current_y:
                    line.append(item['text'])
                else:
                    lines.append(' '.join(line))
                    line = [item['text']]
                current_y = y

            # 마지막 줄 추가
            if line:
                lines.append(' '.join(line))

            # 최종 텍스트
            text = '\n'.join(lines)
            print("ocr_text(process): ", text)
            info = self.extract_info(text)

        else:
            result = "LEGAL"
        
        return result, category, info
                
                