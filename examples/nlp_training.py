"""
ClearML NLP Training Example
============================
텍스트 감정 분류 모델 학습 예제

Features:
- wget을 통한 IMDB 데이터셋 다운로드
- ClearML 실험 추적
- 테이블 형태 결과 로깅
- 원격 실행 지원 (Colab Agent)

사용법:
1. 로컬 실행: python nlp_training.py
2. 원격 대기열 추가:
   clearml-task --project Manifest-NLP --name sentiment-analysis --script nlp_training.py --queue nlp
"""

import os
import subprocess
import tarfile
from pathlib import Path

# ClearML 임포트
from clearml import Task, Logger

# PyTorch 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ===========================================
# ClearML Task 초기화
# ===========================================
task = Task.init(
    project_name="Manifest-NLP",
    task_name="Sentiment-LSTM-Training",
    task_type=Task.TaskTypes.training,
    tags=["nlp", "sentiment", "lstm", "example"]
)

# 하이퍼파라미터 설정
params = {
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 0.001,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "vocab_size": 10000,
    "max_length": 200,
    "num_layers": 2,
    "dropout": 0.5
}
task.connect(params)

logger = Logger.current_logger()


# ===========================================
# 데이터셋 다운로드
# ===========================================
def download_imdb_dataset():
    """wget으로 IMDB 데이터셋 다운로드"""
    data_dir = Path("./data/imdb")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Stanford AI Lab의 IMDB 데이터셋
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = data_dir / "aclImdb_v1.tar.gz"
    
    if not (data_dir / "aclImdb").exists():
        print("Downloading IMDB dataset...")
        subprocess.run([
            "wget", "-q", "-O", str(tar_path), url
        ], check=True)
        
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        
        os.remove(tar_path)
        print("✓ IMDB dataset ready")
    
    return data_dir / "aclImdb"


# ===========================================
# 간단한 토크나이저
# ===========================================
class SimpleTokenizer:
    """어휘 기반 간단한 토크나이저"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    def fit(self, texts):
        """텍스트에서 어휘 생성"""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # 빈도 상위 단어로 어휘 구축
        for word, _ in word_counts.most_common(self.vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text, max_length=200):
        """텍스트를 인덱스로 변환"""
        words = text.lower().split()[:max_length]
        indices = [self.word2idx.get(w, 1) for w in words]
        
        # 패딩
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        
        return indices


# ===========================================
# 데이터셋 클래스
# ===========================================
class IMDBDataset(Dataset):
    """IMDB 감정 분류 데이터셋"""
    
    def __init__(self, data_dir, split="train", tokenizer=None, max_length=200):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 리뷰 로드
        for label, folder in [(1, "pos"), (0, "neg")]:
            folder_path = Path(data_dir) / split / folder
            if folder_path.exists():
                for file_path in list(folder_path.glob("*.txt"))[:2500]:  # 각 클래스 2500개
                    with open(file_path, "r", encoding="utf-8") as f:
                        self.texts.append(f.read())
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            indices = self.tokenizer.encode(text, self.max_length)
            return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        
        return text, label


# ===========================================
# LSTM 모델
# ===========================================
class SentimentLSTM(nn.Module):
    """감정 분류를 위한 LSTM 모델"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 양방향 LSTM의 마지막 hidden state 결합
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        output = self.fc(hidden)
        return self.sigmoid(output)


# ===========================================
# 학습 함수
# ===========================================
def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (texts, labels) in enumerate(dataloader):
        texts, labels = texts.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 50 == 0:
            step = epoch * len(dataloader) + batch_idx
            logger.report_scalar("loss", "train", value=loss.item(), iteration=step)
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    logger.report_scalar("epoch_loss", "train", value=avg_loss, iteration=epoch)
    logger.report_scalar("accuracy", "train", value=accuracy, iteration=epoch)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, epoch):
    """평가"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device).float()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    logger.report_scalar("epoch_loss", "test", value=avg_loss, iteration=epoch)
    logger.report_scalar("accuracy", "test", value=accuracy, iteration=epoch)
    
    return avg_loss, accuracy


# ===========================================
# 메인 실행
# ===========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 준비 (간단한 예제 데이터 사용)
    print("Preparing data...")
    
    # 간단한 예제 데이터 생성 (실제 환경에서는 download_imdb_dataset() 사용)
    sample_texts = [
        "This movie was great and I loved it",
        "Terrible film, waste of time",
        "Amazing performance by the actors",
        "Boring and predictable plot",
        "Highly recommend this masterpiece",
        "Worst movie I have ever seen",
    ] * 500
    
    sample_labels = [1, 0, 1, 0, 1, 0] * 500
    
    # 토크나이저 학습
    tokenizer = SimpleTokenizer(vocab_size=params["vocab_size"])
    tokenizer.fit(sample_texts)
    
    # 데이터 인코딩
    encoded_data = [
        (torch.tensor(tokenizer.encode(text, params["max_length"]), dtype=torch.long),
         torch.tensor(label, dtype=torch.long))
        for text, label in zip(sample_texts, sample_labels)
    ]
    
    # Train/Test 분할
    split = int(len(encoded_data) * 0.8)
    train_data = encoded_data[:split]
    test_data = encoded_data[split:]
    
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=params["batch_size"], shuffle=False)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # 모델 초기화
    model = SentimentLSTM(
        vocab_size=params["vocab_size"],
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    
    # 학습 루프
    best_accuracy = 0
    for epoch in range(1, params["epochs"] + 1):
        print(f"\nEpoch {epoch}/{params['epochs']}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, epoch)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), "best_sentiment_model.pth")
            task.upload_artifact("best_model", artifact_object="best_sentiment_model.pth")
    
    # 결과 요약
    print("\n" + "="*50)
    print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")
    print("="*50)
    
    logger.report_single_value("best_accuracy", best_accuracy)


if __name__ == "__main__":
    main()
