# ---- Builder Stage ----
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y git git-lfs && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN git lfs install --global
 
WORKDIR /BANner_it_AI  # 현재 디렉토리 맞추기

COPY . .
RUN git lfs pull

# ---- Final Stage ----
FROM python:3.10-slim
    
WORKDIR /BANner_it_AI

RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /BANner_it_AI /BANner_it_AI 

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    
    

    