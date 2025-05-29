# llm/download_model.py

import boto3
import os

def download_model_from_s3():
    bucket_name = "bannerit-images"
    
    s3_base_model_prefix = "models/hyperclovax-3b-base/"
    local_base_model_dir = "model/clovax-base-3b"  # app.py 기준 상대 경로
    
    s3_adapter_model_prefix = "models/hyperclovax-3b-adapter/"
    local_adapter_model_dir = "model/clovax-adapter-3b" 
    
    os.makedirs(local_base_model_dir, exist_ok=True)
    os.makedirs(local_adapter_model_dir, exist_ok=True)
    
    s3 = boto3.client("s3")

    base_model_files = [
        "config.json",
        "configuration_hyperclovax.py",
        "generation_config.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.safetensors.index.json",
        "modeling_hyperclovax.py",
        "preprocessor_config.json",
        "preprocessor.py",
        "README.md",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    
    adapter_model_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "added_tokens.json",
        "merges.txt",
        "README.md",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json"
    ]

    for file_name in base_model_files:
        local_path = os.path.join(local_base_model_dir, file_name)
        s3_key = s3_base_model_prefix + file_name # S3 전체 키
        print(f"Attempting to download from S3: Bucket='{bucket_name}', Key='{s3_key}' to Local='{local_path}'") # 추가된 로그
        if os.path.exists(local_path):
            print(f"Skipping, already exists: {local_path}") # 이미 존재 시 스킵 로그
            continue
        try:
            s3.download_file(
                Bucket=bucket_name,
                Key=s3_key, # 수정된 부분
                Filename=local_path
            )
            print(f"✅ Downloaded {file_name}")
        except Exception as e:
            print(f"❌ FAILED to download {file_name} from S3 key {s3_key}. Error: {e}") # 오류 발생 시 로그
            raise
        
    for file_name in adapter_model_files:
        local_path = os.path.join(local_adapter_model_dir, file_name)
        s3_key = s3_adapter_model_prefix + file_name # S3 전체 키
        print(f"Attempting to download from S3: Bucket='{bucket_name}', Key='{s3_key}' to Local='{local_path}'") # 추가된 로그
        if os.path.exists(local_path):
            print(f"Skipping, already exists: {local_path}") # 이미 존재 시 스킵 로그
            continue
        try:
            s3.download_file(
                Bucket=bucket_name,
                Key=s3_key, # 수정된 부분
                Filename=local_path
            )
            print(f"✅ Downloaded {file_name}")
        except Exception as e:
            print(f"❌ FAILED to download {file_name} from S3 key {s3_key}. Error: {e}") # 오류 발생 시 로그
            raise

    return local_base_model_dir, local_adapter_model_dir

