"""
ClearML YOLOv11 Training Example
================================
YOLOv11 객체 탐지 모델 학습 예제

Features:
- ClearML 실험 추적 및 아티팩트 관리
- 원격 실행 지원 (ClearML Agent)
- 하이퍼파라미터 UI 수정 가능

사용법:
1. 로컬 실행: python yolov11_training.py
2. 원격 대기열 추가: 
   clearml-task --project Manifest-Vision --name yolov11-training --script examples/yolov11_training.py --queue vision
"""

import os
import shutil
from pathlib import Path

# ClearML 임포트
from clearml import Task, Logger, Dataset

# ===========================================
# ClearML 서버 인증 설정 (ngrok 사용 시)
# ===========================================
Task.set_credentials(
    api_host="https://83f837d6923d.ngrok-free.app",
    web_host="http://localhost:8080",
    files_host="http://localhost:8081",
    key="Kj7mNp2xQw9rTs5vYb3uLc8h",
    secret="Xf4kMn7pQr2sTv5wYb8zCd3eGh6jKm9nPq2rSt5uVx8y"
)

# Ultralytics YOLO 임포트
from ultralytics import YOLO
import torch
import numpy as np

# ===========================================
# ClearML Task 초기화
# ===========================================
task = Task.init(
    project_name="Manifest-Vision",
    task_name="YOLOv11-Object-Detection",
    task_type=Task.TaskTypes.training,
    tags=["yolov11", "object-detection", "vision", "ultralytics"]
)

# ===========================================
# 하이퍼파라미터 설정 (ClearML UI에서 수정 가능)
# ===========================================
params = {
    # 모델 설정
    "model_variant": "yolo11s.pt",  # yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    
    # 학습 설정
    "epochs": 200,
    "batch_size": 32,
    "imgsz": 640,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "patience": 20,  # Early stopping patience
    
    # 데이터 설정
    "data_yaml": "data/dataset.yaml",  # 데이터셋 YAML 경로
    "cache": False,
    "workers": 2,
    
    # 출력 설정
    "project_dir": "./runs",
    "experiment_name": "yolov11_train"
}
task.connect(params)

# Logger 초기화
logger = Logger.current_logger()


# ===========================================
# 한글 폰트 설정 (선택적)
# ===========================================
def setup_korean_font():
    """한글 폰트 설정 (Linux/Colab 환경용)"""
    try:
        import matplotlib
        from matplotlib import font_manager, rcParams
        
        # 시스템에서 나눔 폰트 찾기
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/nanum/NanumGothic.ttf",
        ]
        
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path:
            font_manager.fontManager.addfont(font_path)
            rcParams["font.family"] = "NanumGothic"
            rcParams["axes.unicode_minus"] = False
            
            # Ultralytics 폰트 교체
            ultra_font = os.path.expanduser("~/.config/Ultralytics/Arial.ttf")
            os.makedirs(os.path.dirname(ultra_font), exist_ok=True)
            shutil.copyfile(font_path, ultra_font)
            print(f"✓ Korean font configured: {ultra_font}")
        else:
            print("⚠ Korean font not found, using default font")
            
    except Exception as e:
        print(f"⚠ Font setup failed: {e}")


# ===========================================
# 데이터셋 준비
# ===========================================
def prepare_dataset():
    """데이터셋 준비 및 검증"""
    data_yaml = params["data_yaml"]
    
    # ClearML Dataset에서 다운로드 (옵션)
    # dataset = Dataset.get(dataset_project="Manifest-Vision", dataset_name="my-dataset")
    # data_path = dataset.get_local_copy()
    
    if not os.path.exists(data_yaml):
        print(f"⚠ Warning: Dataset YAML not found at {data_yaml}")
        print("  Please ensure the dataset is properly configured.")
        print("  Expected YAML structure:")
        print("  ---")
        print("  path: /path/to/dataset")
        print("  train: images/train")
        print("  val: images/val")
        print("  names:")
        print("    0: class1")
        print("    1: class2")
        return None
    
    print(f"✓ Dataset YAML found: {data_yaml}")
    return data_yaml


# ===========================================
# 학습 결과 로깅
# ===========================================
def log_training_results(results, epoch=None):
    """학습 결과를 ClearML에 로깅"""
    if results is None:
        return
    
    # 학습 메트릭 로깅
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                logger.report_scalar(
                    title="Metrics",
                    series=key,
                    value=value,
                    iteration=epoch or 0
                )


# ===========================================
# 메인 학습 함수
# ===========================================
def main():
    # 디바이스 설정
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 한글 폰트 설정 (선택적)
    setup_korean_font()
    
    # 데이터셋 준비
    data_yaml = prepare_dataset()
    if data_yaml is None:
        print("❌ Dataset not configured. Exiting.")
        return
    
    # 출력 디렉토리 설정
    project_dir = Path(params["project_dir"])
    project_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("🚀 Starting YOLOv11 Training")
    print("="*50)
    print(f"  Model: {params['model_variant']}")
    print(f"  Epochs: {params['epochs']}")
    print(f"  Batch Size: {params['batch_size']}")
    print(f"  Image Size: {params['imgsz']}")
    print(f"  Optimizer: {params['optimizer']}")
    print("="*50 + "\n")
    
    # 모델 로드
    model = YOLO(params["model_variant"])
    
    # 학습 실행
    results = model.train(
        data=data_yaml,
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        batch=params["batch_size"],
        device=device,
        workers=params["workers"],
        cache=params["cache"],
        optimizer=params["optimizer"],
        patience=params["patience"],
        project=str(project_dir),
        name=params["experiment_name"],
        exist_ok=True,
        
        # ClearML 자동 로깅 활성화
        plots=True,
        save=True,
    )
    
    # 학습 완료 후 결과 로깅
    print("\n" + "="*50)
    print("📊 Training Results")
    print("="*50)
    
    # 최종 메트릭 기록
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                logger.report_single_value(key, value)
                print(f"  {key}: {value:.4f}")
    
    # 모델 아티팩트 업로드
    best_model_path = project_dir / params["experiment_name"] / "weights" / "best.pt"
    last_model_path = project_dir / params["experiment_name"] / "weights" / "last.pt"
    
    if best_model_path.exists():
        task.upload_artifact("best_model", artifact_object=str(best_model_path))
        print(f"\n✓ Best model uploaded: {best_model_path}")
    
    if last_model_path.exists():
        task.upload_artifact("last_model", artifact_object=str(last_model_path))
        print(f"✓ Last model uploaded: {last_model_path}")
    
    # 학습 결과 이미지 업로드
    results_dir = project_dir / params["experiment_name"]
    for img_file in results_dir.glob("*.png"):
        logger.report_image(
            title="Training Results",
            series=img_file.stem,
            local_path=str(img_file)
        )
    
    print("\n" + "="*50)
    print("✅ Training completed!")
    print("="*50)


if __name__ == "__main__":
    main()
