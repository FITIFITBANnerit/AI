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
        "merges_txt",
        "README.md",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json"
    ]

    for file_name in base_model_files:
        local_path = os.path.join(local_base_model_dir, file_name)
        if os.path.exists(local_path):
            continue  # 이미 존재하면 생략
        s3.download_file(
            Bucket=bucket_name,
            Key=s3_base_model_prefix + file_name,
            Filename=local_path
        )
        print(f"✅ Downloaded {file_name}")
        
    for file_name in adapter_model_files:
        local_path = os.path.join(local_adapter_model_dir, file_name)
        if os.path.exists(local_path):
            continue  # 이미 존재하면 생략
        s3.download_file(
            Bucket=bucket_name,
            Key=s3_adapter_model_prefix + file_name,
            Filename=local_path
        )
        print(f"✅ Downloaded {file_name}")

    return local_base_model_dir, local_adapter_model_dir

