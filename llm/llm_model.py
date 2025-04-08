import torch
import re
from transformers import AutoTokenizer, Gemma3ForCausalLM

class BannerTextClassifier:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Gemma3ForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16)
        
    def classify_banner_text(self, selected_text):
        messages = [
                    [
                        {
                            "role": "system",
                            "content":[
                                {
                                    "type": "text",
                                    "text": "You are an expert in analyzing the purpose of text. Your task is to classify the given text into the most relevant category and determine its legality based on strict criteria."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""
                                        
                                                Analyze the meaning of the following banner text and classify it into the most relevant category from the list below.

                                                **Banner Text:** {selected_text}

                                                ### Categories:
                                                1. Politics
                                                2. Public interest
                                                3. Commercial purposes
                                                4. Other

                                                ### Judgment Criteria:
                                                - If the category is Politics, Public interest then:
                                                **Judgment: "legal"**
                                                
                                                - If the category is **Commercial purposess**, then:
                                                **Judgment: "illegal"**
                                                

                                                ### Output Format (Strictly Follow This):

                                                ```
                                                Category: "Politics"
                                                Judgment: "legal"
                                                ```
                                            """
                                            
                                }
                            ]
                            
                        }
                    ],
                ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict = True, return_tensors="pt"
        ).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        
        decoded = self.tokenizer.decode(generation, skip_sepcial_tokens=True)
        
        return decoded

    def extract_info(self, full_text):
        no_info = "Not detected"
        
        messages = [
                    [
                        {
                            "role": "system",
                            "content":[
                                {
                                    "type": "text",
                                    "text": "You are a helpful assistant who extract phone numbers and company name from text extracted from banners."
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content":[
                                {
                                    "type": "text",
                                    "text": f"""
                                                Extract phone numbers and a company name or store name from {full_text}. If there are none, answer {no_info}.
                                                The output is printed as an example.
                                                
                                                ### Output Format (Strictly Follow This):

                                                ```
                                                Company: "Company Name"
                                                Phone Number: "Phone Number"
                                                ```
                                            """
                                }
                            ]
                        },
                    ],
                ]
        inputs = self.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

        decoded = self.tokenizer.decode(generation, skip_sepcial_tokens=True)
        
        return decoded

    def process_banner_text(self, selected_text, full_text):
        """현수막 텍스트를 분석하고 불법이라면 추가 정보를 추출."""
        classification_result = self.classify_banner_text(selected_text)
        info = None
        
        # 정규식으로 카테고리 추출
        match = re.search(r'Category:\s*"([^"]+)"', classification_result)

        # 찾으면 저장, 없으면 "Unknown"으로 설정
        category = match.group(1) if match else "Unknown"
        
        if category == "Commercial purposes":
            result = "illegal"
            info = self.extract_info(full_text)
        else:
            result = "legal"
        
        return result, category, info
                
                