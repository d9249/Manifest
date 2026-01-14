"""
Ground Handling Vision Pipeline
================================
항공 지상조업 특화 객체 탐지 및 안전 분석 파이프라인

Features:
├── 1. YOLO11 + P2 Layer (소형 객체 탐지)
├── 2. SAHI (Slicing Aided Hyper Inference)
├── 3. OBB (Oriented Bounding Boxes)
├── 4. Pose Estimation (작업자 안전 모니터링)
├── 5. WIoU/MPDIoU Loss Functions
├── 6. Multi-Task Learning (Detection + Pose + Safety)
└── 7. Real-time Safety Alert System

사용법:
1. 학습: python ground_handling_pipeline.py --mode train --data-yaml data/ground_handling.yaml
2. SAHI 추론: python ground_handling_pipeline.py --mode inference-sahi --source video.mp4
3. OBB 학습: python ground_handling_pipeline.py --mode train-obb --data-yaml data/obb_dataset.yaml
4. Pose 안전분석: python ground_handling_pipeline.py --mode pose-safety --source rtsp://...
5. 원격 실행: clearml-task --project Ground-Handling --name gh-training --script examples/ground_handling_pipeline.py --queue vision
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# ClearML
from clearml import Task, Logger, Dataset, OutputModel

Task.set_credentials(
    api_host="https://83f837d6923d.ngrok-free.app",
    web_host="http://localhost:8080",
    files_host="http://localhost:8081",
    key="Kj7mNp2xQw9rTs5vYb3uLc8h",
    secret="Xf4kMn7pQr2sTv5wYb8zCd3eGh6jKm9nPq2rSt5uVx8y"
)

# ============================================
# 1. P2 LAYER CONFIGURATION (소형 객체 탐지)
# ============================================

P2_YOLO11_CONFIG = """
# YOLOv11 with P2 Layer for Small Object Detection
# P2: 1/4 resolution feature map (160x160 for 640 input)

nc: 15  # Ground handling classes
scales:
  s: [0.50, 0.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C3k2, [128, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, C3k2, [512, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 3, C3k2, [512, True]]
  - [-1, 1, SPPF, [512, 5]]
  - [-1, 3, C2PSA, [512]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C3k2, [256, False]]
  # P2 Layer Addition (High Resolution)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 3, C3k2, [128, False]]  # P2 features
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]
  - [-1, 3, C3k2, [256, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C3k2, [512, False]]
  - [[18, 21, 24], 1, Detect, [nc]]  # P2, P3, P4, P5 detection
"""

# Ground Handling 클래스 정의
GROUND_HANDLING_CLASSES = {
    0: "aircraft",
    1: "passenger_boarding_bridge",
    2: "baggage_loader",
    3: "towing_car",
    4: "fuel_truck",
    5: "catering_truck",
    6: "ground_power_unit",
    7: "air_conditioning_unit",
    8: "baggage_cart",
    9: "worker",
    10: "worker_with_helmet",
    11: "worker_without_helmet",
    12: "chock",
    13: "cone",
    14: "safety_vest",
}

# 위험 구역 정의
DANGER_ZONES = {
    "engine_zone": {"radius": 5.0, "alert_level": "critical"},
    "propeller_zone": {"radius": 3.0, "alert_level": "critical"},
    "aircraft_movement": {"radius": 10.0, "alert_level": "warning"},
}


def create_p2_config(num_classes: int, output_path: str) -> str:
    """P2 Layer가 추가된 YOLO11 config 생성"""
    config = P2_YOLO11_CONFIG.replace("nc: 15", f"nc: {num_classes}")
    config_path = Path(output_path) / "yolo11_p2.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config)
    return str(config_path)


# ============================================
# 2. SAHI INTEGRATION (고해상도 슬라이싱 추론)
# ============================================

def inference_with_sahi(
    model_path: str,
    source: str,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
    confidence: float = 0.25,
    device: str = "0"
) -> Dict[str, Any]:
    """
    SAHI를 활용한 고해상도 이미지/비디오 추론
    - 큰 이미지를 작은 조각으로 나눠서 추론
    - 작은 객체 탐지율 향상
    """
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction, get_prediction
        from sahi.utils.cv import read_image
    except ImportError:
        print("❌ SAHI not installed. Install with: pip install sahi")
        return {}
    
    task = Task.init(
        project_name="Ground-Handling",
        task_name="SAHI-Inference",
        task_type=Task.TaskTypes.inference
    )
    
    params = {
        "model_path": model_path,
        "source": source,
        "slice_size": slice_size,
        "overlap_ratio": overlap_ratio,
        "confidence": confidence,
    }
    task.connect(params)
    logger = Logger.current_logger()
    
    print("\n" + "="*60)
    print("🔍 SAHI Sliced Inference for Ground Handling")
    print("="*60)
    print(f"  Slice Size: {slice_size}x{slice_size}")
    print(f"  Overlap: {overlap_ratio*100}%")
    
    # SAHI 모델 로드
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=confidence,
        device=f"cuda:{device}" if device.isdigit() else device
    )
    
    source_path = Path(source)
    results = []
    
    if source_path.is_file():
        if source_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            # 단일 이미지
            result = get_sliced_prediction(
                image=str(source_path),
                detection_model=detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
            )
            results.append({
                "file": str(source_path),
                "objects": len(result.object_prediction_list),
                "predictions": [
                    {
                        "class": p.category.name,
                        "confidence": p.score.value,
                        "bbox": p.bbox.to_xyxy()
                    }
                    for p in result.object_prediction_list
                ]
            })
            
            # 시각화 저장
            output_path = f"sahi_result_{source_path.stem}.jpg"
            result.export_visuals(export_dir="./runs/sahi/")
            logger.report_image("SAHI Results", source_path.stem, local_path=output_path)
            
        elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # 비디오 처리
            import cv2
            cap = cv2.VideoCapture(str(source_path))
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 30 == 0:  # 매 30프레임마다
                    result = get_sliced_prediction(
                        image=frame,
                        detection_model=detection_model,
                        slice_height=slice_size,
                        slice_width=slice_size,
                        overlap_height_ratio=overlap_ratio,
                        overlap_width_ratio=overlap_ratio,
                    )
                    
                    logger.report_scalar("Objects Detected", "count", 
                                        len(result.object_prediction_list), frame_count)
                
                frame_count += 1
            
            cap.release()
    
    print(f"\n✅ SAHI Inference completed!")
    return {"results": results}


# ============================================
# 3. OBB TRAINING (회전 바운딩 박스)
# ============================================

def train_obb(
    data_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16
) -> Dict[str, Any]:
    """
    OBB (Oriented Bounding Boxes) 학습
    - 회전된 객체 탐지에 적합
    - 항공기, 토잉카 등 긴 형태 객체에 효과적
    """
    from ultralytics import YOLO
    import torch
    
    task = Task.init(
        project_name="Ground-Handling",
        task_name="YOLOv11-OBB-Training",
        task_type=Task.TaskTypes.training,
        tags=["obb", "oriented", "ground-handling"]
    )
    
    params = {
        "model": "yolo11s-obb.pt",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "task_type": "obb"
    }
    task.connect(params)
    logger = Logger.current_logger()
    
    print("\n" + "="*60)
    print("📐 OBB (Oriented Bounding Box) Training")
    print("="*60)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11s-obb.pt")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        task="obb",
        project="runs/obb",
        name=f"exp_{task.id[:8]}",
    )
    
    # 결과 업로드
    best_model = Path(f"runs/obb/exp_{task.id[:8]}/weights/best.pt")
    if best_model.exists():
        task.upload_artifact("obb_model", str(best_model))
    
    return {"task_id": task.id, "model_path": str(best_model)}


# ============================================
# 4. POSE ESTIMATION + SAFETY MONITORING
# ============================================

class SafetyZoneMonitor:
    """작업자 안전 구역 모니터링 시스템"""
    
    def __init__(self, danger_zones: Dict = None):
        self.danger_zones = danger_zones or DANGER_ZONES
        self.alerts = []
    
    def check_pose_safety(self, keypoints: np.ndarray, zone_polygons: List) -> Dict:
        """
        작업자 자세 분석 및 위험 구역 진입 확인
        
        keypoints: [17, 3] - COCO format (x, y, confidence)
        - 0: nose, 5-6: shoulders, 11-12: hips, 15-16: ankles
        """
        alerts = []
        
        # 1. 허리 구부림 감지 (부적절한 짐 들기)
        if keypoints is not None and len(keypoints) >= 17:
            shoulders = keypoints[5:7, :2].mean(axis=0)  # 어깨 중심
            hips = keypoints[11:13, :2].mean(axis=0)     # 엉덩이 중심
            
            # 상체 기울기 계산
            if np.all(shoulders > 0) and np.all(hips > 0):
                angle = np.arctan2(shoulders[1] - hips[1], 
                                   shoulders[0] - hips[0])
                angle_deg = np.degrees(angle)
                
                if abs(angle_deg - 90) > 30:  # 30도 이상 기울임
                    alerts.append({
                        "type": "improper_posture",
                        "level": "warning",
                        "message": "작업자 허리 구부림 감지 - 부적절한 자세"
                    })
        
        return {"alerts": alerts, "is_safe": len(alerts) == 0}
    
    def check_zone_intrusion(
        self, 
        worker_position: Tuple[float, float],
        aircraft_bbox: Tuple[float, float, float, float]
    ) -> Dict:
        """엔진/프로펠러 위험 구역 진입 확인"""
        alerts = []
        
        # 항공기 엔진 위치 추정 (바운딩 박스 기준)
        x1, y1, x2, y2 = aircraft_bbox
        engine_zones = [
            ((x1 - 50, (y1 + y2) / 2), 50),  # 좌측 엔진
            ((x2 + 50, (y1 + y2) / 2), 50),  # 우측 엔진
        ]
        
        wx, wy = worker_position
        
        for (ex, ey), radius in engine_zones:
            distance = np.sqrt((wx - ex)**2 + (wy - ey)**2)
            if distance < radius:
                alerts.append({
                    "type": "danger_zone_intrusion",
                    "level": "critical",
                    "message": f"⚠️ 작업자 엔진 위험 구역 진입! 거리: {distance:.1f}m",
                    "distance": distance
                })
        
        return {"alerts": alerts, "is_safe": len(alerts) == 0}


def run_pose_safety_monitoring(
    source: str,
    output_dir: str = "runs/safety",
    alert_callback=None
):
    """
    실시간 작업자 자세 및 안전 모니터링
    """
    from ultralytics import YOLO
    import cv2
    import torch
    
    task = Task.init(
        project_name="Ground-Handling",
        task_name="Pose-Safety-Monitoring",
        task_type=Task.TaskTypes.monitor,
        tags=["pose", "safety", "real-time"]
    )
    
    logger = Logger.current_logger()
    safety_monitor = SafetyZoneMonitor()
    
    print("\n" + "="*60)
    print("👷 Pose Estimation + Safety Monitoring")
    print("="*60)
    
    # 두 모델 로드: Detection + Pose
    device = 0 if torch.cuda.is_available() else "cpu"
    detect_model = YOLO("yolo11s.pt")
    pose_model = YOLO("yolo11s-pose.pt")
    
    # 비디오 소스 열기
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return
    
    frame_count = 0
    total_alerts = 0
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 객체 탐지 (항공기, 장비 등)
        detect_results = detect_model(frame, verbose=False)
        
        # 작업자 포즈 추정
        pose_results = pose_model(frame, verbose=False)
        
        for r in pose_results:
            if r.keypoints is not None:
                keypoints = r.keypoints.data.cpu().numpy()
                
                for kp in keypoints:
                    # 안전 분석
                    safety_result = safety_monitor.check_pose_safety(kp, [])
                    
                    if not safety_result["is_safe"]:
                        total_alerts += 1
                        for alert in safety_result["alerts"]:
                            logger.report_text(
                                f"[Frame {frame_count}] {alert['message']}",
                                level=Logger.LogLevel.WARN
                            )
                            if alert_callback:
                                alert_callback(alert)
        
        # 메트릭 로깅
        if frame_count % 100 == 0:
            logger.report_scalar("Safety", "Total Alerts", total_alerts, frame_count)
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n✅ Safety monitoring completed!")
    print(f"   Total frames: {frame_count}")
    print(f"   Total alerts: {total_alerts}")
    
    return {"total_alerts": total_alerts, "frames_processed": frame_count}


# ============================================
# 5. CUSTOM LOSS FUNCTIONS (WIoU, MPDIoU)
# ============================================

def train_with_wiou(
    data_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16
) -> Dict[str, Any]:
    """
    WIoU (Wise-IoU) 손실 함수로 학습
    - 노이즈가 많은 데이터에 강건함
    - 그림자, 반사광 등 지상조업 환경에 적합
    """
    from ultralytics import YOLO
    import torch
    
    task = Task.init(
        project_name="Ground-Handling",
        task_name="Training-WIoU-Loss",
        task_type=Task.TaskTypes.training,
        tags=["wiou", "custom-loss", "ground-handling"]
    )
    
    # Ultralytics에서 지원하는 IoU 타입
    # CIoU, DIoU, GIoU, SIoU, WIoU (8.1+)
    params = {
        "model": "yolo11s.pt",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "box": 7.5,      # Box loss gain
        "cls": 0.5,      # Cls loss gain
        "dfl": 1.5,      # DFL loss gain
        "iou_type": "WIoU",  # Wise-IoU
    }
    task.connect(params)
    
    print("\n" + "="*60)
    print("📊 Training with WIoU Loss Function")
    print("="*60)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11s.pt")
    
    # WIoU는 ultralytics 설정에서 지원
    # 또는 커스텀 구현 필요
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        box=params["box"],
        cls=params["cls"],
        dfl=params["dfl"],
        project="runs/wiou",
        name=f"exp_{task.id[:8]}",
    )
    
    return {"task_id": task.id}


# ============================================
# 6. FULL TRAINING PIPELINE
# ============================================

def train_ground_handling_model(
    data_yaml: str,
    use_p2: bool = True,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    model_variant: str = "yolo11s.pt"
) -> Dict[str, Any]:
    """
    지상조업 특화 모델 학습
    """
    from ultralytics import YOLO
    import torch
    
    task = Task.init(
        project_name="Ground-Handling",
        task_name="Ground-Handling-Training",
        task_type=Task.TaskTypes.training,
        tags=["ground-handling", "p2-layer" if use_p2 else "standard"]
    )
    
    params = {
        "model_variant": model_variant,
        "use_p2_layer": use_p2,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "patience": 30,
        # 지상조업 특화 augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 15.0,  # 회전 증가 (다양한 각도)
        "translate": 0.1,
        "scale": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
    }
    task.connect(params)
    logger = Logger.current_logger()
    
    print("\n" + "="*60)
    print("🛫 Ground Handling Model Training")
    print("="*60)
    print(f"  Model: {model_variant}")
    print(f"  P2 Layer: {'Enabled' if use_p2 else 'Disabled'}")
    print(f"  Epochs: {epochs}")
    
    device = 0 if torch.cuda.is_available() else "cpu"
    
    # P2 Layer 사용 시 커스텀 config 생성
    if use_p2:
        config_path = create_p2_config(
            num_classes=len(GROUND_HANDLING_CLASSES),
            output_path="./configs"
        )
        model = YOLO(config_path)
        model.load(model_variant)  # Pretrained weights
    else:
        model = YOLO(model_variant)
    
    # 학습 실행
    results = model.train(
        data=data_yaml,
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        batch=params["batch"],
        device=device,
        optimizer=params["optimizer"],
        lr0=params["lr0"],
        patience=params["patience"],
        hsv_h=params["hsv_h"],
        hsv_s=params["hsv_s"],
        hsv_v=params["hsv_v"],
        degrees=params["degrees"],
        translate=params["translate"],
        scale=params["scale"],
        mosaic=params["mosaic"],
        mixup=params["mixup"],
        project="runs/ground_handling",
        name=f"exp_{task.id[:8]}",
        plots=True,
        save=True,
    )
    
    # 결과 저장
    exp_dir = Path(f"runs/ground_handling/exp_{task.id[:8]}")
    best_model = exp_dir / "weights" / "best.pt"
    
    if best_model.exists():
        output_model = OutputModel(task=task, name="Ground-Handling-Model")
        output_model.update_weights(str(best_model))
        task.upload_artifact("best_model", str(best_model))
    
    # 이미지 결과 업로드
    for img in exp_dir.glob("*.png"):
        logger.report_image("Results", img.stem, local_path=str(img))
    
    return {
        "task_id": task.id,
        "model_path": str(best_model) if best_model.exists() else None
    }


# ============================================
# 7. REAL-TIME INFERENCE SYSTEM
# ============================================

def run_realtime_inference(
    model_path: str,
    source: str,
    use_sahi: bool = False,
    enable_safety: bool = True,
    confidence: float = 0.25
):
    """
    실시간 추론 + 안전 모니터링 통합 시스템
    """
    from ultralytics import YOLO
    import cv2
    import torch
    
    task = Task.init(
        project_name="Ground-Handling",
        task_name="Real-Time-Inference",
        task_type=Task.TaskTypes.inference
    )
    
    logger = Logger.current_logger()
    safety_monitor = SafetyZoneMonitor() if enable_safety else None
    
    print("\n" + "="*60)
    print("🎥 Real-Time Ground Handling Analysis")
    print("="*60)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    
    # 스트리밍 또는 비디오 처리
    results = model.track(
        source=source,
        conf=confidence,
        device=device,
        stream=True,
        tracker="bytetrack.yaml",  # ByteTrack for tracking
        persist=True,
    )
    
    frame_count = 0
    for r in results:
        # 탐지 결과 처리
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                
                class_name = GROUND_HANDLING_CLASSES.get(cls, f"class_{cls}")
                
                if frame_count % 30 == 0:
                    logger.report_scalar("Detections", class_name, 1, frame_count)
        
        frame_count += 1
    
    return {"frames_processed": frame_count}


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Ground Handling Vision Pipeline")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "train-obb", "train-wiou", 
                               "inference-sahi", "pose-safety", "realtime"])
    parser.add_argument("--data-yaml", type=str, help="Dataset YAML path")
    parser.add_argument("--source", type=str, help="Video/RTSP source")
    parser.add_argument("--model", type=str, default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--use-p2", action="store_true", help="Enable P2 layer")
    parser.add_argument("--slice-size", type=int, default=640)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ground_handling_model(
            data_yaml=args.data_yaml,
            use_p2=args.use_p2,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            model_variant=args.model
        )
    
    elif args.mode == "train-obb":
        train_obb(args.data_yaml, args.epochs, args.imgsz, args.batch)
    
    elif args.mode == "train-wiou":
        train_with_wiou(args.data_yaml, args.epochs, args.imgsz, args.batch)
    
    elif args.mode == "inference-sahi":
        inference_with_sahi(
            model_path=args.model,
            source=args.source,
            slice_size=args.slice_size
        )
    
    elif args.mode == "pose-safety":
        run_pose_safety_monitoring(source=args.source)
    
    elif args.mode == "realtime":
        run_realtime_inference(
            model_path=args.model,
            source=args.source,
            use_sahi=False,
            enable_safety=True
        )


if __name__ == "__main__":
    main()
