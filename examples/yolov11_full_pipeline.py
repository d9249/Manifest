"""
ClearML YOLOv11 Full Pipeline Example
=====================================
ClearML의 모든 핵심 기능을 활용하는 종합 예제

Features:
├── 1. Task Management (실험 추적)
├── 2. Dataset Versioning (데이터셋 버전 관리)
├── 3. Hyperparameter Optimization (HPO)
├── 4. Pipeline Orchestration (파이프라인)
├── 5. Model Registry (모델 레지스트리)
├── 6. Model Serving (모델 서빙)
├── 7. Artifacts Management (아티팩트 관리)
├── 8. Scalars & Plots (메트릭 시각화)
├── 9. Debug Samples (디버그 샘플)
└── 10. Remote Execution (원격 실행)

사용법:
1. 전체 파이프라인 실행:
   python yolov11_full_pipeline.py --mode pipeline

2. 데이터셋 업로드:
   python yolov11_full_pipeline.py --mode dataset --data-path /path/to/dataset

3. 학습만 실행:
   python yolov11_full_pipeline.py --mode train

4. HPO 실행:
   python yolov11_full_pipeline.py --mode hpo

5. 모델 서빙:
   python yolov11_full_pipeline.py --mode serve --model-id <model_id>

6. 원격 실행:
   clearml-task --project Manifest-Vision --name yolov11-pipeline \
                --script examples/yolov11_full_pipeline.py --queue vision
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# ===========================================
# ClearML Imports
# ===========================================
from clearml import (
    Task,
    Logger,
    Dataset,
    Model,
    PipelineController,
    PipelineDecorator,
    TaskTypes,
    OutputModel,
    InputModel,
)
from clearml.automation import (
    UniformParameterRange,
    UniformIntegerParameterRange,
    DiscreteParameterRange,
    HyperParameterOptimizer,
    GridSearch,
    RandomSearch,
    OptimizerBOHB,
)

# ===========================================
# ClearML 서버 인증 설정
# ===========================================
Task.set_credentials(
    api_host="https://83f837d6923d.ngrok-free.app",
    web_host="http://localhost:8080",
    files_host="http://localhost:8081",
    key="Kj7mNp2xQw9rTs5vYb3uLc8h",
    secret="Xf4kMn7pQr2sTv5wYb8zCd3eGh6jKm9nPq2rSt5uVx8y"
)


# ###########################################
# #  1. DATASET VERSIONING (데이터셋 버전 관리)
# ###########################################

def create_dataset(
    data_path: str,
    dataset_name: str = "YOLOv11-Dataset",
    dataset_project: str = "Manifest-Vision/Datasets",
    description: str = "YOLOv11 Object Detection Dataset"
) -> str:
    """
    데이터셋을 ClearML에 업로드하고 버전 관리
    
    ClearML Dataset 기능:
    - 데이터 버전 관리 (Git처럼 변경 추적)
    - 대용량 파일 효율적 저장
    - 데이터 계보(lineage) 추적
    - 팀 간 데이터 공유
    """
    print("\n" + "="*60)
    print("📦 Creating ClearML Dataset")
    print("="*60)
    
    # 새 데이터셋 생성
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        description=description,
        dataset_tags=["yolov11", "object-detection", "v1"]
    )
    
    # 데이터 파일 추가
    data_path = Path(data_path)
    if data_path.is_dir():
        dataset.add_files(
            path=str(data_path),
            dataset_path="data/",
            recursive=True
        )
        print(f"  ✓ Added directory: {data_path}")
    else:
        dataset.add_files(path=str(data_path))
        print(f"  ✓ Added file: {data_path}")
    
    # 메타데이터 추가
    dataset.set_metadata({
        "created_by": os.environ.get("USER", "unknown"),
        "created_at": datetime.now().isoformat(),
        "format": "YOLO",
        "source": str(data_path)
    })
    
    # 데이터셋 업로드 및 완료
    dataset.upload(
        output_url=None,  # 기본 파일 서버 사용
        verbose=True
    )
    dataset.finalize()
    
    dataset_id = dataset.id
    print(f"\n✅ Dataset created successfully!")
    print(f"   ID: {dataset_id}")
    print(f"   Name: {dataset_name}")
    print(f"   Project: {dataset_project}")
    
    return dataset_id


def get_dataset(
    dataset_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_project: str = "Manifest-Vision/Datasets"
) -> str:
    """
    데이터셋 다운로드 및 로컬 경로 반환
    """
    print("\n📥 Fetching dataset from ClearML...")
    
    if dataset_id:
        dataset = Dataset.get(dataset_id=dataset_id)
    else:
        dataset = Dataset.get(
            dataset_project=dataset_project,
            dataset_name=dataset_name,
            only_published=True  # 배포된 버전만
        )
    
    # 로컬에 캐시된 경로 반환
    local_path = dataset.get_local_copy()
    print(f"  ✓ Dataset cached at: {local_path}")
    
    return local_path


# ###########################################
# #  2. TRAINING WITH FULL LOGGING
# ###########################################

def train_model(
    dataset_id: Optional[str] = None,
    data_yaml: Optional[str] = None,
    parent_task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    ClearML의 모든 로깅 기능을 활용한 학습
    
    ClearML Logging 기능:
    - Scalars: Loss, Accuracy 등 수치 메트릭
    - Plots: Confusion Matrix, PR Curve 등 차트
    - Debug Samples: 이미지, 오디오, 비디오 샘플
    - Artifacts: 모델 체크포인트, 설정 파일 등
    - Console Logs: 실시간 콘솔 출력
    """
    
    # Task 초기화
    task = Task.init(
        project_name="Manifest-Vision",
        task_name="YOLOv11-Training-Full",
        task_type=TaskTypes.training,
        tags=["yolov11", "full-pipeline", "production"],
        reuse_last_task_id=False,  # 항상 새 Task 생성
        auto_connect_frameworks={
            "pytorch": True,
            "matplotlib": True,
            "tensorboard": True,
        }
    )
    
    # 부모 Task 연결 (파이프라인용)
    if parent_task_id:
        task.set_parent(parent_task_id)
    
    # ===========================================
    # 하이퍼파라미터 설정 (UI에서 수정 가능)
    # ===========================================
    model_config = {
        "variant": "yolo11s.pt",
        "pretrained": True,
    }
    
    training_config = {
        "epochs": 100,
        "batch_size": 16,
        "imgsz": 640,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "patience": 50,
        "workers": 4,
        "cache": False,
    }
    
    augmentation_config = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    }
    
    # ClearML에 파라미터 연결
    task.connect(model_config, name="model")
    task.connect(training_config, name="training")
    task.connect(augmentation_config, name="augmentation")
    
    # Logger 초기화
    logger = Logger.current_logger()
    
    # ===========================================
    # 시스템 정보 로깅
    # ===========================================
    import torch
    
    system_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    task.connect(system_info, name="system")
    
    print("\n" + "="*60)
    print("🚀 Starting YOLOv11 Training with Full ClearML Integration")
    print("="*60)
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # ===========================================
    # 데이터셋 로드
    # ===========================================
    if dataset_id:
        data_path = get_dataset(dataset_id=dataset_id)
        data_yaml = Path(data_path) / "data" / "dataset.yaml"
    elif data_yaml:
        data_yaml = Path(data_yaml)
    else:
        print("⚠ No dataset specified. Using default.")
        data_yaml = Path("data/dataset.yaml")
    
    if not data_yaml.exists():
        print(f"❌ Dataset YAML not found: {data_yaml}")
        # 데모용 더미 데이터 생성
        print("  Creating dummy dataset for demo...")
        task.set_parameter("training/epochs", 1)
        training_config["epochs"] = 1
    
    # ===========================================
    # Ultralytics 콜백으로 ClearML 로깅
    # ===========================================
    from ultralytics import YOLO
    from ultralytics.utils.callbacks.clearml import callbacks as clearml_callbacks
    
    # 커스텀 콜백 정의
    def on_train_epoch_end(trainer):
        """에폭 종료 시 추가 로깅"""
        epoch = trainer.epoch
        
        # 학습 메트릭 로깅
        if hasattr(trainer, 'metrics'):
            for key, value in trainer.metrics.items():
                logger.report_scalar(
                    title="Training Metrics",
                    series=key,
                    value=float(value),
                    iteration=epoch
                )
        
        # GPU 메모리 로깅
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            logger.report_scalar(
                title="System",
                series="GPU Memory (GB)",
                value=gpu_memory,
                iteration=epoch
            )
    
    def on_val_end(validator):
        """검증 종료 시 추가 로깅"""
        # Confusion Matrix 로깅
        if hasattr(validator, 'confusion_matrix'):
            cm = validator.confusion_matrix.matrix
            logger.report_confusion_matrix(
                title="Confusion Matrix",
                series="Validation",
                matrix=cm.tolist(),
                xlabels=validator.names,
                ylabels=validator.names,
            )
    
    def on_train_end(trainer):
        """학습 완료 시 최종 로깅"""
        # 최종 결과 요약
        logger.report_text(
            "Training completed successfully!",
            level=Logger.LogLevel.INFO
        )
    
    # 모델 로드
    model = YOLO(model_config["variant"])
    
    # 콜백 등록
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_train_end", on_train_end)
    
    # ===========================================
    # 학습 실행
    # ===========================================
    device = 0 if torch.cuda.is_available() else "cpu"
    
    results = model.train(
        data=str(data_yaml) if data_yaml.exists() else None,
        epochs=training_config["epochs"],
        imgsz=training_config["imgsz"],
        batch=training_config["batch_size"],
        device=device,
        workers=training_config["workers"],
        cache=training_config["cache"],
        optimizer=training_config["optimizer"],
        lr0=training_config["lr0"],
        lrf=training_config["lrf"],
        momentum=training_config["momentum"],
        weight_decay=training_config["weight_decay"],
        warmup_epochs=training_config["warmup_epochs"],
        patience=training_config["patience"],
        # Augmentation
        hsv_h=augmentation_config["hsv_h"],
        hsv_s=augmentation_config["hsv_s"],
        hsv_v=augmentation_config["hsv_v"],
        degrees=augmentation_config["degrees"],
        translate=augmentation_config["translate"],
        scale=augmentation_config["scale"],
        flipud=augmentation_config["flipud"],
        fliplr=augmentation_config["fliplr"],
        mosaic=augmentation_config["mosaic"],
        mixup=augmentation_config["mixup"],
        # Output
        project="runs/train",
        name=f"exp_{task.id[:8]}",
        exist_ok=True,
        plots=True,
        save=True,
    )
    
    # ===========================================
    # 결과 아티팩트 업로드
    # ===========================================
    exp_dir = Path(f"runs/train/exp_{task.id[:8]}")
    
    # 모델 체크포인트 업로드
    best_model = exp_dir / "weights" / "best.pt"
    last_model = exp_dir / "weights" / "last.pt"
    
    if best_model.exists():
        # OutputModel로 모델 레지스트리에 등록
        output_model = OutputModel(
            task=task,
            name="YOLOv11-Best",
            framework="PyTorch",
            tags=["yolov11", "best", "production-ready"]
        )
        output_model.update_weights(
            weights_filename=str(best_model),
            auto_delete_file=False
        )
        output_model.update_design(config_dict=model_config)
        
        task.upload_artifact("best_model", artifact_object=str(best_model))
        print(f"✓ Best model uploaded to Model Registry")
    
    if last_model.exists():
        task.upload_artifact("last_model", artifact_object=str(last_model))
    
    # 학습 결과 이미지 업로드
    for img_file in exp_dir.glob("*.png"):
        logger.report_image(
            title="Training Results",
            series=img_file.stem,
            local_path=str(img_file)
        )
    
    # 학습 설정 아티팩트
    config_artifact = {
        "model": model_config,
        "training": training_config,
        "augmentation": augmentation_config,
    }
    task.upload_artifact("training_config", artifact_object=config_artifact)
    
    # ===========================================
    # 최종 결과
    # ===========================================
    final_results = {
        "task_id": task.id,
        "model_path": str(best_model) if best_model.exists() else None,
        "metrics": results.results_dict if hasattr(results, 'results_dict') else {},
    }
    
    # Summary 메트릭
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                logger.report_single_value(key, float(value))
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print(f"   Task ID: {task.id}")
    print(f"   Best Model: {best_model}")
    print("="*60)
    
    return final_results


# ###########################################
# #  3. HYPERPARAMETER OPTIMIZATION (HPO)
# ###########################################

def run_hpo(
    base_task_id: Optional[str] = None,
    max_trials: int = 20,
    concurrent_trials: int = 2
):
    """
    ClearML Hyperparameter Optimization
    
    HPO 기능:
    - Grid Search: 모든 조합 탐색
    - Random Search: 무작위 샘플링
    - BOHB: Bayesian Optimization + Hyperband
    - Optuna: Optuna 백엔드 지원
    """
    print("\n" + "="*60)
    print("🔍 Starting Hyperparameter Optimization")
    print("="*60)
    
    # HPO Controller Task 생성
    task = Task.init(
        project_name="Manifest-Vision",
        task_name="YOLOv11-HPO",
        task_type=TaskTypes.optimizer,
        tags=["hpo", "yolov11", "optimization"]
    )
    
    # 기본 학습 Task (클론할 템플릿)
    if not base_task_id:
        # 이전 학습 Task 찾기
        tasks = Task.get_tasks(
            project_name="Manifest-Vision",
            task_name="YOLOv11-Training-Full",
            task_filter={"status": ["completed"]}
        )
        if tasks:
            base_task_id = tasks[0].id
        else:
            print("❌ No base task found. Please train a model first.")
            return
    
    # HPO 설정
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        
        # 탐색할 하이퍼파라미터 정의
        hyper_parameters=[
            # 학습률 탐색
            UniformParameterRange(
                name="training/lr0",
                min_value=0.0001,
                max_value=0.01,
                step_size=0.0001
            ),
            # 배치 사이즈 탐색
            DiscreteParameterRange(
                name="training/batch_size",
                values=[8, 16, 32, 64]
            ),
            # 옵티마이저 선택
            DiscreteParameterRange(
                name="training/optimizer",
                values=["SGD", "Adam", "AdamW"]
            ),
            # 이미지 사이즈
            DiscreteParameterRange(
                name="training/imgsz",
                values=[416, 512, 640]
            ),
            # Augmentation 강도
            UniformParameterRange(
                name="augmentation/mosaic",
                min_value=0.0,
                max_value=1.0,
                step_size=0.1
            ),
        ],
        
        # 최적화 목표
        objective_metric_title="metrics",
        objective_metric_series="mAP50-95",
        objective_metric_sign="max",  # 최대화
        
        # 탐색 전략
        optimizer_class=OptimizerBOHB,  # Bayesian Optimization
        
        # 실행 설정
        max_number_of_concurrent_tasks=concurrent_trials,
        total_max_jobs=max_trials,
        min_iteration_per_job=10,
        max_iteration_per_job=100,
        
        # 실행 큐
        execution_queue="vision",
        
        # 리소스
        compute_time_limit=None,
        pool_period_min=1,
    )
    
    # HPO 시작
    optimizer.start()
    
    print(f"✓ HPO started with {max_trials} trials")
    print(f"  Concurrent trials: {concurrent_trials}")
    print(f"  Base task: {base_task_id}")
    
    # 완료 대기 (옵션)
    # optimizer.wait()
    
    # 상위 결과 확인
    # top_experiments = optimizer.get_top_experiments(top_k=5)
    
    return optimizer


# ###########################################
# #  4. PIPELINE ORCHESTRATION
# ###########################################

@PipelineDecorator.component(
    return_values=["dataset_id"],
    cache=True,
    task_type=TaskTypes.data_processing
)
def pipeline_step_prepare_data(data_path: str) -> str:
    """파이프라인 Step 1: 데이터 준비"""
    dataset_id = create_dataset(data_path)
    return dataset_id


@PipelineDecorator.component(
    return_values=["model_path", "metrics"],
    cache=False,
    task_type=TaskTypes.training
)
def pipeline_step_train(dataset_id: str) -> tuple:
    """파이프라인 Step 2: 모델 학습"""
    results = train_model(dataset_id=dataset_id)
    return results["model_path"], results["metrics"]


@PipelineDecorator.component(
    return_values=["eval_results"],
    cache=False,
    task_type=TaskTypes.testing
)
def pipeline_step_evaluate(model_path: str, dataset_id: str) -> dict:
    """파이프라인 Step 3: 모델 평가"""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    data_path = get_dataset(dataset_id=dataset_id)
    
    results = model.val(
        data=Path(data_path) / "data" / "dataset.yaml",
        split="test"
    )
    
    eval_results = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }
    
    return eval_results


@PipelineDecorator.component(
    return_values=["model_id"],
    cache=False,
    task_type=TaskTypes.custom
)
def pipeline_step_register_model(
    model_path: str,
    metrics: dict,
    min_map: float = 0.5
) -> Optional[str]:
    """파이프라인 Step 4: 모델 레지스트리 등록"""
    
    # 품질 게이트
    if metrics.get("mAP50-95", 0) < min_map:
        print(f"⚠ Model quality below threshold ({min_map}). Skipping registration.")
        return None
    
    # 모델 등록
    from clearml import Model
    
    model = Model.create(
        name="YOLOv11-Production",
        project="Manifest-Vision/Models",
        tags=["production", "yolov11", "approved"],
        framework="PyTorch"
    )
    
    model.update_weights(weights_filename=model_path)
    model.update_labels({"classes": ["class1", "class2"]})  # 클래스 목록
    
    # 배포 가능으로 마킹
    model.publish()
    
    print(f"✓ Model registered: {model.id}")
    return model.id


@PipelineDecorator.pipeline(
    name="YOLOv11-Training-Pipeline",
    project="Manifest-Vision/Pipelines",
    version="1.0.0",
    pipeline_execution_queue="vision",
    default_queue="vision"
)
def run_training_pipeline(data_path: str, min_map: float = 0.5):
    """
    전체 학습 파이프라인
    
    Pipeline:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Prepare    │───▶│   Train     │───▶│  Evaluate   │───▶│  Register   │
    │   Data      │    │   Model     │    │   Model     │    │   Model     │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
    """
    print("\n" + "="*60)
    print("🔄 Starting YOLOv11 Training Pipeline")
    print("="*60)
    
    # Step 1: 데이터 준비
    dataset_id = pipeline_step_prepare_data(data_path)
    
    # Step 2: 모델 학습
    model_path, metrics = pipeline_step_train(dataset_id)
    
    # Step 3: 모델 평가
    eval_results = pipeline_step_evaluate(model_path, dataset_id)
    
    # Step 4: 모델 등록 (품질 게이트 통과 시)
    model_id = pipeline_step_register_model(model_path, eval_results, min_map)
    
    return model_id


def run_pipeline_controller(data_path: str):
    """
    PipelineController를 사용한 파이프라인 실행
    (더 세밀한 제어가 필요한 경우)
    """
    pipe = PipelineController(
        name="YOLOv11-Pipeline-Controlled",
        project="Manifest-Vision/Pipelines",
        version="1.0.0",
        add_pipeline_tags=True,
    )
    
    # Step 정의
    pipe.add_step(
        name="prepare_data",
        base_task_project="Manifest-Vision",
        base_task_name="Data-Preparation",
        parameter_override={
            "General/data_path": data_path
        }
    )
    
    pipe.add_step(
        name="train_model",
        base_task_project="Manifest-Vision",
        base_task_name="YOLOv11-Training-Full",
        parents=["prepare_data"],
        parameter_override={
            "General/dataset_id": "${prepare_data.dataset_id}"
        }
    )
    
    pipe.add_step(
        name="evaluate",
        base_task_project="Manifest-Vision",
        base_task_name="Model-Evaluation",
        parents=["train_model"],
        parameter_override={
            "General/model_path": "${train_model.model_path}"
        }
    )
    
    # 파이프라인 시작
    pipe.start(queue="vision")
    
    return pipe


# ###########################################
# #  5. MODEL SERVING
# ###########################################

def serve_model(
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    port: int = 8000
):
    """
    ClearML Model Serving
    
    서빙 옵션:
    1. ClearML Serving: 프로덕션 서빙 인프라
    2. Triton Inference Server: 고성능 추론
    3. Custom FastAPI: 커스텀 REST API
    """
    print("\n" + "="*60)
    print("🚀 Starting Model Serving")
    print("="*60)
    
    # 모델 로드
    if model_id:
        # ClearML Model Registry에서 로드
        model = Model(model_id=model_id)
        model_path = model.get_local_copy()
        print(f"✓ Model loaded from registry: {model_id}")
    elif model_path:
        print(f"✓ Using local model: {model_path}")
    else:
        print("❌ No model specified")
        return
    
    # FastAPI 서빙 예제
    try:
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import JSONResponse
        import uvicorn
        from PIL import Image
        import io
        
        app = FastAPI(
            title="YOLOv11 Inference API",
            description="ClearML Model Serving Example",
            version="1.0.0"
        )
        
        # 모델 로드
        from ultralytics import YOLO
        yolo_model = YOLO(model_path)
        
        @app.get("/health")
        def health_check():
            return {"status": "healthy", "model": model_path}
        
        @app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            """이미지 추론"""
            # 이미지 읽기
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # 추론
            results = yolo_model(image)
            
            # 결과 파싱
            predictions = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    predictions.append({
                        "class": int(box.cls),
                        "class_name": yolo_model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()[0]
                    })
            
            return JSONResponse({
                "predictions": predictions,
                "count": len(predictions)
            })
        
        @app.post("/batch_predict")
        async def batch_predict(files: List[UploadFile] = File(...)):
            """배치 추론"""
            all_predictions = []
            
            for file in files:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                results = yolo_model(image)
                
                file_predictions = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        file_predictions.append({
                            "class": int(box.cls),
                            "class_name": yolo_model.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy.tolist()[0]
                        })
                
                all_predictions.append({
                    "filename": file.filename,
                    "predictions": file_predictions
                })
            
            return JSONResponse({"results": all_predictions})
        
        print(f"\n🌐 Starting server on http://0.0.0.0:{port}")
        print("   Endpoints:")
        print("   - GET  /health        - Health check")
        print("   - POST /predict       - Single image inference")
        print("   - POST /batch_predict - Batch inference")
        print("\n   Press Ctrl+C to stop")
        
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except ImportError:
        print("⚠ FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart")
        print("\nAlternative: Use ClearML Serving")
        print("  clearml-serving create --name yolov11-serving")
        print("  clearml-serving model add --model-id", model_id or "<model_id>")


# ###########################################
# #  6. MODEL COMPARISON & A/B TESTING
# ###########################################

def compare_models(model_ids: List[str]):
    """
    여러 모델 성능 비교
    """
    print("\n" + "="*60)
    print("📊 Model Comparison")
    print("="*60)
    
    task = Task.init(
        project_name="Manifest-Vision",
        task_name="Model-Comparison",
        task_type=TaskTypes.testing
    )
    
    logger = Logger.current_logger()
    
    comparison_results = []
    
    for model_id in model_ids:
        model = Model(model_id=model_id)
        model_path = model.get_local_copy()
        
        from ultralytics import YOLO
        yolo = YOLO(model_path)
        
        # 검증 실행
        results = yolo.val()
        
        metrics = {
            "model_id": model_id,
            "model_name": model.name,
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
        comparison_results.append(metrics)
        
        # 차트에 추가
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.report_scalar(
                    title="Model Comparison",
                    series=model.name,
                    value=value,
                    iteration=model_ids.index(model_id)
                )
    
    # 비교 테이블 생성
    import pandas as pd
    df = pd.DataFrame(comparison_results)
    
    logger.report_table(
        title="Model Comparison Table",
        series="All Models",
        table_plot=df
    )
    
    # 최고 성능 모델 선택
    best_model = max(comparison_results, key=lambda x: x["mAP50-95"])
    logger.report_single_value("Best Model ID", best_model["model_id"])
    
    print(f"\n✓ Best model: {best_model['model_name']}")
    print(f"  mAP50-95: {best_model['mAP50-95']:.4f}")
    
    return comparison_results


# ###########################################
# #  7. REMOTE EXECUTION HELPERS
# ###########################################

def execute_remotely(queue: str = "vision"):
    """
    현재 Task를 원격 Agent로 전송
    
    이 함수 호출 시 스크립트가 즉시 종료되고,
    지정된 큐의 Agent가 스크립트를 다시 실행함
    """
    task = Task.init(
        project_name="Manifest-Vision",
        task_name="Remote-Execution-Example"
    )
    
    # 원격 실행 모드 전환
    # 이 시점에서 스크립트가 종료되고 Agent가 이어받음
    task.execute_remotely(queue_name=queue)
    
    print("This code runs on the remote agent!")
    # ... 이후 코드는 원격 Agent에서 실행됨


# ###########################################
# #  MAIN ENTRY POINT
# ###########################################

def main():
    parser = argparse.ArgumentParser(
        description="ClearML YOLOv11 Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 데이터셋 업로드
  python yolov11_full_pipeline.py --mode dataset --data-path ./data

  # 모델 학습
  python yolov11_full_pipeline.py --mode train

  # HPO 실행
  python yolov11_full_pipeline.py --mode hpo --max-trials 10

  # 파이프라인 실행
  python yolov11_full_pipeline.py --mode pipeline --data-path ./data

  # 모델 서빙
  python yolov11_full_pipeline.py --mode serve --model-id <id>

  # 모델 비교
  python yolov11_full_pipeline.py --mode compare --model-ids id1,id2,id3
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["dataset", "train", "hpo", "pipeline", "serve", "compare"],
        help="실행 모드"
    )
    parser.add_argument("--data-path", type=str, help="데이터셋 경로")
    parser.add_argument("--data-yaml", type=str, help="데이터셋 YAML 경로")
    parser.add_argument("--dataset-id", type=str, help="ClearML 데이터셋 ID")
    parser.add_argument("--model-id", type=str, help="모델 ID")
    parser.add_argument("--model-ids", type=str, help="비교할 모델 ID 목록 (쉼표 구분)")
    parser.add_argument("--max-trials", type=int, default=20, help="HPO 최대 시도 횟수")
    parser.add_argument("--port", type=int, default=8000, help="서빙 포트")
    
    args = parser.parse_args()
    
    if args.mode == "dataset":
        if not args.data_path:
            print("❌ --data-path required for dataset mode")
            sys.exit(1)
        create_dataset(args.data_path)
    
    elif args.mode == "train":
        train_model(
            dataset_id=args.dataset_id,
            data_yaml=args.data_yaml
        )
    
    elif args.mode == "hpo":
        run_hpo(max_trials=args.max_trials)
    
    elif args.mode == "pipeline":
        if not args.data_path:
            print("❌ --data-path required for pipeline mode")
            sys.exit(1)
        # Decorator 기반 파이프라인 실행
        PipelineDecorator.run_locally()  # 로컬에서 테스트
        run_training_pipeline(args.data_path)
    
    elif args.mode == "serve":
        serve_model(
            model_id=args.model_id,
            port=args.port
        )
    
    elif args.mode == "compare":
        if not args.model_ids:
            print("❌ --model-ids required for compare mode")
            sys.exit(1)
        model_ids = args.model_ids.split(",")
        compare_models(model_ids)


if __name__ == "__main__":
    main()
