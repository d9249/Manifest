"""
ClearML Vision Training Example
===============================
MNIST 손글씨 분류 모델 학습 예제

Features:
- wget을 통한 데이터셋 다운로드
- ClearML 실험 추적
- 원격 실행 지원 (Colab Agent)

사용법:
1. 로컬 실행: python vision_training.py
2. 원격 대기열 추가: 
   clearml-task --project Manifest-Vision --name mnist-training --script vision_training.py --queue vision
"""

import os
import subprocess
from pathlib import Path

# ClearML 임포트
from clearml import Task, Logger, Dataset

# 서버 인증 설정 (ngrok 사용 시)
Task.set_credentials(
    api_host="https://83f837d6923d.ngrok-free.app",
    web_host="http://localhost:8080",
    files_host="http://localhost:8081",
    key="Kj7mNp2xQw9rTs5vYb3uLc8h",
    secret="Xf4kMn7pQr2sTv5wYb8zCd3eGh6jKm9nPq2rSt5uVx8y"
)

# PyTorch 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ===========================================
# ClearML Task 초기화
# ===========================================
task = Task.init(
    project_name="Manifest-Vision",
    task_name="MNIST-CNN-Training",
    task_type=Task.TaskTypes.training,
    tags=["mnist", "cnn", "vision", "example"]
)

# 하이퍼파라미터 설정 (ClearML UI에서 수정 가능)
params = {
    "batch_size": 64,
    "epochs": 5,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "log_interval": 100
}
task.connect(params)

# Logger 초기화
logger = Logger.current_logger()


# ===========================================
# 데이터셋 다운로드 (wget 방식)
# ===========================================
def download_dataset_wget():
    """wget을 사용하여 MNIST 데이터셋 다운로드"""
    data_dir = Path("./data/mnist")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # MNIST 데이터 URL (Yann LeCun's website)
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    for filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            subprocess.run([
                "wget", "-q", "-O", str(filepath),
                f"{base_url}{filename}"
            ], check=True)
            print(f"  ✓ {filename} downloaded")
    
    return data_dir


# ===========================================
# 모델 정의
# ===========================================
class SimpleCNN(nn.Module):
    """간단한 CNN 모델 for MNIST"""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


# ===========================================
# 학습 함수
# ===========================================
def train(model, device, train_loader, optimizer, epoch):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # 로그 출력 및 ClearML 기록
        if batch_idx % params["log_interval"] == 0:
            step = epoch * len(train_loader) + batch_idx
            logger.report_scalar("loss", "train", value=loss.item(), iteration=step)
            
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    
    # 에폭별 평균 메트릭
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    
    logger.report_scalar("epoch_loss", "train", value=avg_loss, iteration=epoch)
    logger.report_scalar("accuracy", "train", value=accuracy, iteration=epoch)
    
    return avg_loss, accuracy


def test(model, device, test_loader, epoch):
    """테스트 평가"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # ClearML에 메트릭 기록
    logger.report_scalar("epoch_loss", "test", value=test_loss, iteration=epoch)
    logger.report_scalar("accuracy", "test", value=accuracy, iteration=epoch)
    
    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    
    return test_loss, accuracy


# ===========================================
# 메인 실행
# ===========================================
def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터셋 로드 (torchvision 자동 다운로드 사용)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 모델, 옵티마이저 설정
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), 
                          lr=params["learning_rate"], 
                          momentum=params["momentum"])
    
    # 학습 루프
    best_accuracy = 0
    for epoch in range(1, params["epochs"] + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader, epoch)
        
        # 최고 성능 모델 저장
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            model_path = "best_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # ClearML에 모델 아티팩트 업로드
            task.upload_artifact("best_model", artifact_object=model_path)
            print(f"  ✓ New best model saved (accuracy: {best_accuracy:.2f}%)")
    
    # 최종 결과 요약
    print("\n" + "="*50)
    print(f"Training completed!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print("="*50)
    
    # ClearML의 summary에 최종 결과 기록
    logger.report_single_value("best_accuracy", best_accuracy)


if __name__ == "__main__":
    main()
