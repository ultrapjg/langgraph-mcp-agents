# 1단계: 베이스 이미지 및 의존성 설치
FROM python:3.11-slim-bookworm AS base
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2단계: 애플리케이션 코드 복사 및 비루트 사용자 설정
COPY . .
RUN useradd --create-home streamlituser
USER streamlituser

# Streamlit 포트 노출
EXPOSE 8585

# 컨테이너 시작 시 실행 명령
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8585", "--server.address=0.0.0.0"]
