import torch
import openai
import re
import json

class BannerTextClassifier:
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = openai.OpenAI(api_key = api_key)
        self.model_name = model_name
        print(print(f"BannerTextClassifier initialized with OpenAI model: {self.model_name}"))
        
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

                    Answer:
                """
        user_prompt = prompt.format(banner_info=full_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                     "content": user_prompt}
                ],
                temperature=0,
                max_tokens=20
            )
            result = response.choices[0].message.content.strip()
            return result
        
        except Exception as e:
            print(f"Error during classification API call: {e}")
            return "Error"

    def extract_info(self, full_text):
        no_info = "Not detected"
        
        prompt = """
                    
                    You are an expert data extractor specializing in analyzing unstructured text from OCR (Optical Character Recognition) scans. Your task is to accurately extract the company/store name and phone number from the provided text. The text may contain OCR errors, so be prepared to correct common mistakes.

                    **Source Text:**
                    `{banner_text}`

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

                   Example of the exact output format:
                   {{"Company": "추출된 회사명", "Phone Number": "추출된 전화번호"}}
                    
                """
                
        user_prompt = prompt.format(banner_text=full_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                     "content": user_prompt}
                ],
                temperature=0,
                max_tokens=150
            )
            extracted_data = json.loads(response.choices[0].message.content)
            return extracted_data
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from API response: {e}")
            print(f"Raw response content: {response.choices[0].message.content if 'response' in locals() and response.choices else 'N/A'}")
            return {"Company": no_info, "Phone Number": no_info, "Error_Details": "JSON Decode Error"}
        except Exception as e:
            print(f"Error during extraction API call: {e}")
            return {"Company": no_info, "Phone Number": no_info, "Error_Details": str(e)}
    
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
        info_data = None

        # 찾으면 저장, 없으면 "Unknown"으로 설정
        category = self.normalize(classification_result)
        
        if category == "Commercial purposes":
            result = "ILLEGAL"
            
            info_data = self.extract_info(" ".join(full_text))
        elif category == "Error": # 분류 단계에서 오류가 발생한 경우
            result = "CLASSIFICATION_ERROR"
        else:
            result = "LEGAL"
        
        return result, category, info_data
                
                