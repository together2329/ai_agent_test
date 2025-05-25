#!/bin/bash

echo "🐳 UVM Embedding 테스트 Docker 환경"
echo "=================================="

# Docker 및 Docker Compose 설치 확인
if ! command -v docker &> /dev/null; then
    echo "❌ Docker가 설치되어 있지 않습니다."
    echo "Docker를 먼저 설치해주세요: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose가 설치되어 있지 않습니다."
    echo "Docker Compose를 먼저 설치해주세요."
    exit 1
fi

# 결과 디렉토리 생성
mkdir -p results

echo "📦 Docker 이미지 빌드 중..."
docker-compose build embedding-test

echo "🚀 Embedding 테스트 실행 중..."
docker-compose run --rm embedding-test

echo "✅ 테스트 완료!"
echo "📊 결과는 ./results 디렉토리에서 확인할 수 있습니다." 