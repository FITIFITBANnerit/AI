FROM python:3.10-slim

WORKDIR /BANner_it_AI

RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git-lfs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN git lfs install

COPY . .

RUN git lfs pull

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]