# ---- Builder Stage ----
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y git git-lfs && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN git lfs install

WORKDIR /app
    
COPY . .
RUN git lfs pull
    
# ---- Final Stage ----
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
        
WORKDIR /app
    
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*
    
RUN python3 -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
RUN python3 -m pip install paddleocr==2.10.0
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install ultralytics transformers==4.51.1 peft==0.14.0 trl==0.17.0

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
    
COPY --from=builder /app /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
    