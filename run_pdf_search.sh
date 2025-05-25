#!/bin/bash

# PDF 검색 시스템 실행 스크립트

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 디렉토리 생성
mkdir -p results
mkdir -p temp

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo -e "${RED}오류: Docker가 설치되어 있지 않습니다.${NC}"
    exit 1
fi

# Docker Compose 설치 확인
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}오류: Docker Compose가 설치되어 있지 않습니다.${NC}"
    exit 1
fi

# 메뉴 표시
echo -e "${GREEN}PDF 검색 시스템${NC}"
echo "1. 웹 애플리케이션 실행"
echo "2. CLI 실행"
echo "3. 종료"
echo

# 사용자 입력 받기
read -p "선택하세요 (1-3): " choice

case $choice in
    1)
        echo -e "${YELLOW}웹 애플리케이션을 시작합니다...${NC}"
        docker-compose up pdf-search
        ;;
    2)
        echo -e "${YELLOW}CLI를 시작합니다...${NC}"
        echo "사용법:"
        echo "  기본 검색: ./pdf_search_cli.py <PDF_파일> -q <검색어>"
        echo "  대화형 모드: ./pdf_search_cli.py <PDF_파일>"
        echo "  통계 보기: ./pdf_search_cli.py <PDF_파일> --stats"
        echo
        echo "예시:"
        echo "  ./pdf_search_cli.py document.pdf -q '검색어' -k 10 -s 0.2 -o results/output.json"
        echo
        docker-compose run --rm pdf-search-cli
        ;;
    3)
        echo -e "${GREEN}프로그램을 종료합니다.${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}잘못된 선택입니다.${NC}"
        exit 1
        ;;
esac 