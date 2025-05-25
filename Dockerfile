FROM python:3.9-slim

# 시스템 패키지 업데이트 및 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 패키지 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 모델 캐시 디렉토리 생성
RUN mkdir -p /root/.cache/torch/sentence_transformers

# 포트 노출 (Jupyter notebook용, 선택사항)
EXPOSE 8888

# 기본 명령어
CMD ["python", "embedding_test.py"] 