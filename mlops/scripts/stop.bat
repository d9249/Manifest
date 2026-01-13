@echo off
REM ClearML Server 중지 스크립트
REM Windows용

echo ============================================
echo ClearML MLOps Server Stopping...
echo ============================================

cd /d "%~dp0.."

docker-compose down

echo.
echo ClearML Server가 중지되었습니다.
echo.
echo 데이터를 유지한 채 중지됨
echo 데이터까지 삭제하려면: docker-compose down -v
echo.
pause
