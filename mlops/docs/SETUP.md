# ClearML MLOps 서버 설치 가이드

## 📋 사전 요구사항

### 필수 소프트웨어
- **Docker Desktop**: [다운로드](https://www.docker.com/products/docker-desktop)
- **ngrok**: [설치 가이드](https://ngrok.com/download)
- **최소 8GB RAM** (16GB 권장)

### 포트 요구사항
| 포트 | 서비스 | 설명 |
|------|--------|------|
| 8080 | Web UI | 대시보드 |
| 8008 | API Server | REST API |
| 8081 | File Server | 아티팩트 저장 |

---

## 🚀 설치 단계

### Step 1: 환경 변수 설정

```bash
# .env.example을 복사하여 .env 생성
copy .env.example .env
```

`.env` 파일을 열고 다음 값들을 수정하세요:

```bash
# 관리자 계정 (반드시 변경!)
CLEARML_ADMIN_USER=admin
CLEARML_ADMIN_PASS=your_secure_password

# Agent 인증 키 (랜덤 문자열 권장)
CLEARML_AGENT_ACCESS_KEY=your_random_access_key_123
CLEARML_AGENT_SECRET_KEY=your_random_secret_key_456
```

### Step 2: Docker Compose 실행

```bash
# scripts 폴더의 start.bat 실행 또는:
cd mlops
docker-compose up -d
```

### Step 3: 서버 상태 확인

```bash
# 컨테이너 상태 확인
docker-compose ps

# 모든 서비스가 "Up" 상태여야 함
```

### Step 4: 웹 UI 접속

브라우저에서 `http://localhost:8080` 접속

**첫 로그인:**
- Username: `.env`의 `CLEARML_ADMIN_USER`
- Password: `.env`의 `CLEARML_ADMIN_PASS`

---

## 🌐 ngrok 터널링 설정

Colab에서 접근하려면 외부 URL이 필요합니다.

### 옵션 A: 개별 터널 (무료 계정)

```bash
# 터미널 1
ngrok http 8080

# 터미널 2
ngrok http 8008

# 터미널 3
ngrok http 8081
```

### 옵션 B: 다중 터널 (유료 계정)

```bash
# scripts/ngrok.yml에 authtoken 설정 후
ngrok start --all --config scripts\ngrok.yml
```

### ngrok URL 설정

ngrok 실행 후 발급된 URL을 `.env`에 반영:

```bash
# 예시 (실제 URL로 변경)
CLEARML_WEB_HOST=https://abc123.ngrok-free.app
CLEARML_API_HOST=https://def456.ngrok-free.app
CLEARML_FILES_HOST=https://ghi789.ngrok-free.app
```

> ⚠️ **무료 ngrok**은 재시작 시 URL이 변경됩니다. 세션마다 `.env` 업데이트 필요.

---

## ✅ 설치 확인

### API 서버 테스트
```bash
curl http://localhost:8008/debug.ping
# 응답: {"data": {"msg": "pong"}, ...}
```

### Web UI 테스트
브라우저에서 `http://localhost:8080` 접속

---

## 🔧 트러블슈팅

### Elasticsearch 메모리 오류
```bash
# Windows에서 WSL2 메모리 제한
# .wslconfig 파일 생성 (사용자 폴더)
[wsl2]
memory=8GB
```

### Docker 권한 오류
Docker Desktop에서 "Expose daemon on tcp://localhost:2375" 활성화

### 포트 충돌
```bash
# 사용 중인 포트 확인
netstat -ano | findstr :8080
```

---

## 📁 파일 구조

```
mlops/
├── docker-compose.yml    # 메인 설정 파일
├── .env.example          # 환경 변수 템플릿
├── .env                  # 실제 환경 변수 (생성 필요)
├── data/                 # 데이터 저장 (자동 생성)
│   ├── elastic_7/
│   ├── mongo_4/
│   ├── redis/
│   ├── fileserver/
│   └── logs/
└── scripts/
    ├── start.bat
    ├── stop.bat
    └── ngrok.yml
```
