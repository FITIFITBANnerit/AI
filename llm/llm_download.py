# llm/download_model.py

import boto3
import os

def get_openai_api_key_from_aws(parameter_name: str, region_name: str = "ap-northeast-2") -> str | None:
    
    try:
        ssm_client = boto3.client('ssm', region_name=region_name)
        response = ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=True  # API 키가 SecureString으로 저장된 경우 복호화 필요
        )
        api_key = response['Parameter']['Value'] # 이것이 API 키 문자열입니다.
        print(f"✅ OpenAI API Key ('{parameter_name}') successfully retrieved from AWS Parameter Store.")
        return api_key # API 키 문자열 반환
    except Exception as e:
        print(f"🚨 Error retrieving API Key ('{parameter_name}') from AWS Parameter Store: {e}")
        return None

