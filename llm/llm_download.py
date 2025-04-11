# llm/download_model.py

import boto3
import os

def download_model_from_s3():
    bucket_name = "bannerit-images"
    s3_model_prefix = "models/gemma3/"
    local_model_dir = "model/gemma-3b"  # app.py 기준 상대 경로
    os.makedirs(local_model_dir, exist_ok=True)
    s3 = boto3.client("s3")

    model_files = [
        "config.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "model.safetensors",
        "added_tokens.json",
        "generation_config.json",
        "gitattributes",
        "special_tokens_map.json",
        "tokenizer.json",
    ]

    for file_name in model_files:
        local_path = os.path.join(local_model_dir, file_name)
        if os.path.exists(local_path):
            continue  # 이미 존재하면 생략
        s3.download_file(
            Bucket=bucket_name,
            Key=s3_model_prefix + file_name,
            Filename=local_path
        )
        print(f"✅ Downloaded {file_name}")

    return local_model_dir

