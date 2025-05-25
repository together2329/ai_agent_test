FROM python:3.9-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 패키지 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# NLTK 데이터 다운로드
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('punkt_tab'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('maxent_ne_chunker'); \
    nltk.download('words')"

# 애플리케이션 파일 복사
COPY . .

# 모델 캐시 디렉토리 생성
RUN mkdir -p /root/.cache/torch/sentence_transformers

# 포트 노출
EXPOSE 8501

# Streamlit 실행
CMD ["streamlit", "run", "pdf_search_app.py", "--server.address", "0.0.0.0"] 