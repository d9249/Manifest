"""
ClearML 로컬 설정 스크립트
==========================
이 스크립트를 실행하면 ~/.clearml/clearml.conf 파일이 자동 생성됩니다.
"""
import os
from pathlib import Path

# 설정 값 (ngrok URL로 변경하세요)
API_HOST = "https://83f837d6923d.ngrok-free.app"
WEB_HOST = "http://localhost:8080"
FILES_HOST = "http://localhost:8081"
ACCESS_KEY = "Kj7mNp2xQw9rTs5vYb3uLc8h"
SECRET_KEY = "Xf4kMn7pQr2sTv5wYb8zCd3eGh6jKm9nPq2rSt5uVx8y"

config_content = f"""
api {{
    web_server: {WEB_HOST}
    api_server: {API_HOST}
    files_server: {FILES_HOST}
    credentials {{
        access_key: "{ACCESS_KEY}"
        secret_key: "{SECRET_KEY}"
    }}
}}
"""

# 설정 파일 경로
config_dir = Path.home() / ".clearml"
config_file = config_dir / "clearml.conf"

# 디렉토리 생성
config_dir.mkdir(exist_ok=True)

# 설정 파일 작성
with open(config_file, "w") as f:
    f.write(config_content)

print(f"✅ ClearML 설정 파일 생성 완료!")
print(f"   경로: {config_file}")
print(f"   API Server: {API_HOST}")
