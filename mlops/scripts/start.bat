@echo off
REM ClearML Server 시작 스크립트
REM Windows용

echo ============================================
echo ClearML MLOps Server Starting...
echo ============================================

cd /d "%~dp0.."

REM .env 파일 존재 확인
if not exist ".env" (
    echo [WARNING] .env 파일이 없습니다!
    echo .env.example 파일을 .env로 복사하고 설정을 수정하세요.
    copy .env.example .env
    echo .env 파일이 생성되었습니다. 설정을 수정한 후 다시 실행하세요.
    pause
    exit /b 1
)

REM 데이터 디렉토리 생성
echo [1/4] 데이터 디렉토리 생성 중...
if not exist "data\elastic_7" mkdir "data\elastic_7"
if not exist "data\mongo_4\db" mkdir "data\mongo_4\db"
if not exist "data\mongo_4\configdb" mkdir "data\mongo_4\configdb"
if not exist "data\redis" mkdir "data\redis"
if not exist "data\fileserver" mkdir "data\fileserver"
if not exist "data\logs" mkdir "data\logs"
if not exist "data\config" mkdir "data\config"
if not exist "data\agent" mkdir "data\agent"

REM Docker Compose 실행
echo [2/4] Docker 컨테이너 시작 중...
docker-compose up -d

REM 상태 확인
echo [3/4] 컨테이너 상태 확인 중...
timeout /t 10 /nobreak > nul
docker-compose ps

echo.
echo ============================================
echo [4/4] ClearML Server 시작 완료!
echo ============================================
echo.
echo Web UI:     http://localhost:8080
echo API Server: http://localhost:8008
echo File Server: http://localhost:8081
echo.
echo ngrok 터널링 시작하려면:
echo   ngrok http 8080 --region jp
echo   ngrok http 8008 --region jp  (별도 터미널)
echo   ngrok http 8081 --region jp  (별도 터미널)
echo.
echo 또는 ngrok.yml 설정으로 한번에 실행:
echo   ngrok start --all --config scripts\ngrok.yml
echo.
pause
