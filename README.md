# PDF 문장 검색 시스템

AI 기반 임베딩을 활용한 PDF 문서 내 문장 검색 시스템입니다. 웹 인터페이스와 CLI를 모두 제공합니다.

## 주요 기능

- PDF 문서 내 문장 단위 검색
- AI 기반 의미론적 검색 (SentenceTransformer 활용)
- 웹 인터페이스 (Streamlit)
- 명령행 인터페이스 (CLI)
- 검색 결과 컨텍스트 제공
- 문서 통계 및 시각화
- 검색 결과 저장 (JSON)

## 시스템 요구사항

- Docker
- Docker Compose
- Python 3.8 이상 (Docker 컨테이너 내부)

## 설치 및 실행

1. 저장소 클론:
```bash
git clone <repository-url>
cd pdf-search
```

2. 실행 스크립트 실행:
```bash
./run_pdf_search.sh
```

3. 메뉴에서 실행할 모드 선택:
   - 웹 애플리케이션
   - CLI
   - 종료

## 웹 애플리케이션 사용법

1. 웹 브라우저에서 `http://localhost:8501` 접속
2. PDF 파일 업로드
3. 검색어 입력 및 검색 옵션 설정
4. 검색 결과 확인

## CLI 사용법

### 기본 검색
```bash
./pdf_search_cli.py <PDF_파일> -q <검색어>
```

### 대화형 모드
```bash
./pdf_search_cli.py <PDF_파일>
```

### 통계 보기
```bash
./pdf_search_cli.py <PDF_파일> --stats
```

### 추가 옵션
- `-k, --top-k`: 반환할 결과 개수 (기본값: 5)
- `-s, --min-similarity`: 최소 유사도 (기본값: 0.1)
- `-o, --output`: 결과 저장 경로 (JSON)

## 프로젝트 구조

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run_pdf_search.sh
├── pdf_search_engine.py
├── pdf_search_app.py
├── pdf_search_cli.py
├── results/
└── temp/
```

## 기술 스택

- Python
- Docker
- Streamlit
- SentenceTransformer
- PyPDF2
- pdfplumber
- NLTK
- Rich (CLI)
- Plotly (시각화)

## 라이선스

MIT License 