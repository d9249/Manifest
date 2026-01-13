# ClearML 사용자 가이드

**연구원을 위한 ClearML MLOps 플랫폼 사용 가이드**

---

## 📌 목차

1. [시작하기](#1-시작하기)
2. [실험 생성 및 추적](#2-실험-생성-및-추적)
3. [원격 실행 (Colab)](#3-원격-실행-colab)
4. [결과 분석](#4-결과-분석)
5. [데이터셋 관리](#5-데이터셋-관리)

---

## 1. 시작하기

### ClearML SDK 설치

```bash
pip install clearml
```

### 인증 설정

```python
from clearml import Task

Task.set_credentials(
    api_host="API_SERVER_URL",      # ngrok API URL
    web_host="WEB_UI_URL",          # ngrok Web URL
    files_host="FILE_SERVER_URL",   # ngrok Files URL
    key="YOUR_ACCESS_KEY",
    secret="YOUR_SECRET_KEY"
)
```

> 관리자에게 API 인증 키를 요청하세요.

---

## 2. 실험 생성 및 추적

### 기본 사용법

학습 코드에 **2줄만 추가**하면 됩니다:

```python
from clearml import Task

# 실험 초기화 (프로젝트명, 실험명)
task = Task.init(
    project_name="My-Project",
    task_name="Experiment-001"
)

# 이후 모든 학습이 자동으로 추적됩니다!
```

### 하이퍼파라미터 기록

```python
# 딕셔너리로 연결 → UI에서 수정 가능
params = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10
}
task.connect(params)
```

### 메트릭 로깅

```python
from clearml import Logger
logger = Logger.current_logger()

# 스칼라 값 기록
logger.report_scalar("loss", "train", value=0.5, iteration=100)
logger.report_scalar("accuracy", "test", value=0.95, iteration=100)
```

### 아티팩트 업로드

```python
# 모델 파일
task.upload_artifact("best_model", artifact_object="model.pth")

# 테이블 데이터
import pandas as pd
df = pd.DataFrame({"metric": [0.9, 0.95], "epoch": [1, 2]})
logger.report_table("Results", "metrics", table_plot=df)
```

---

## 3. 원격 실행 (Colab)

### 방법 1: Clone & Enqueue (UI)

1. Web UI에서 실험 선택
2. 우클릭 → **Clone**
3. 클론된 실험 선택 → **Enqueue**
4. 큐 선택 (`default`, `vision`, `nlp`)

### 방법 2: 코드에서 원격 실행

```python
task = Task.init(project_name="My-Project", task_name="Remote-Run")

# 이 줄 이후 Colab에서 실행됨
task.execute_remotely(queue_name="default")

# 아래 코드는 Colab에서 실행됨
model.train()
```

### 방법 3: CLI로 태스크 생성

```bash
clearml-task \
    --project My-Project \
    --name training-run \
    --script train.py \
    --queue vision
```

---

## 4. 결과 분석

### Web UI 주요 기능

| 메뉴 | 기능 |
|------|------|
| **Projects** | 프로젝트별 실험 관리 |
| **Experiments** | 실험 목록, 비교, 필터링 |
| **Scalars** | 메트릭 그래프 시각화 |
| **Artifacts** | 모델, 파일 다운로드 |
| **Workers** | Agent 상태 모니터링 |

### 실험 비교

1. 여러 실험 선택 (체크박스)
2. **Compare** 버튼 클릭
3. 메트릭, 파라미터 비교 차트 확인

---

## 5. 데이터셋 관리

### 데이터셋 생성

```python
from clearml import Dataset

# 로컬 파일에서 생성
dataset = Dataset.create(
    dataset_name="MNIST-Data",
    dataset_project="Manifest-Datasets"
)
dataset.add_files(path="./data/mnist")
dataset.upload()
dataset.finalize()
```

### 데이터셋 사용

```python
# 학습 코드에서
dataset = Dataset.get(
    dataset_name="MNIST-Data",
    dataset_project="Manifest-Datasets"
)
local_path = dataset.get_local_copy()
```

---

## 📞 도움이 필요하면

- **Web UI 문서**: http://localhost:8080/docs
- **ClearML 공식 문서**: https://clear.ml/docs
- **관리자 문의**: 인증 키 발급, 큐 생성 요청
