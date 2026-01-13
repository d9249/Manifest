"""
Dataset Utilities for ClearML
==============================
wget을 통한 데이터셋 다운로드 및 ClearML Data 등록 유틸리티

주요 기능:
- wget으로 외부 데이터셋 다운로드
- ClearML Dataset으로 버전 관리
- 다양한 데이터 소스 지원 (URL, S3, GCS)
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Union

# ClearML 임포트
from clearml import Dataset, Task


class DatasetDownloader:
    """wget 기반 데이터셋 다운로더"""
    
    def __init__(self, download_dir: str = "./data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def download(
        self, 
        url: str, 
        filename: Optional[str] = None,
        extract: bool = False
    ) -> Path:
        """
        wget으로 파일 다운로드
        
        Args:
            url: 다운로드 URL
            filename: 저장할 파일명 (None이면 URL에서 추출)
            extract: tar/zip 압축 해제 여부
            
        Returns:
            다운로드된 파일 경로
        """
        if filename is None:
            filename = url.split("/")[-1]
        
        filepath = self.download_dir / filename
        
        if not filepath.exists():
            print(f"Downloading {filename}...")
            subprocess.run([
                "wget", "-q", "--show-progress",
                "-O", str(filepath),
                url
            ], check=True)
            print(f"  ✓ Downloaded: {filepath}")
        else:
            print(f"  ⏭ Already exists: {filepath}")
        
        if extract and filepath.exists():
            self._extract(filepath)
        
        return filepath
    
    def _extract(self, filepath: Path) -> Path:
        """압축 파일 해제"""
        import tarfile
        import zipfile
        
        extract_dir = filepath.parent / filepath.stem
        
        if filepath.suffix == ".gz" or str(filepath).endswith(".tar.gz"):
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(filepath.parent)
            print(f"  ✓ Extracted: {filepath.parent}")
            
        elif filepath.suffix == ".zip":
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(filepath.parent)
            print(f"  ✓ Extracted: {filepath.parent}")
        
        return extract_dir
    
    def download_multiple(self, urls: List[str]) -> List[Path]:
        """여러 URL 다운로드"""
        paths = []
        for url in urls:
            path = self.download(url)
            paths.append(path)
        return paths


class ClearMLDatasetManager:
    """ClearML Dataset 관리자"""
    
    def __init__(self, project_name: str = "Manifest-Datasets"):
        self.project_name = project_name
    
    def create_from_files(
        self,
        dataset_name: str,
        files_path: Union[str, Path],
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """
        로컬 파일에서 ClearML Dataset 생성
        
        Args:
            dataset_name: 데이터셋 이름
            files_path: 파일/디렉토리 경로
            description: 데이터셋 설명
            tags: 태그 목록
            
        Returns:
            생성된 Dataset ID
        """
        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=self.project_name
        )
        
        if tags:
            dataset.add_tags(tags)
        
        files_path = Path(files_path)
        
        if files_path.is_file():
            dataset.add_files(path=str(files_path))
        elif files_path.is_dir():
            dataset.add_files(path=str(files_path))
        
        dataset.upload()
        dataset.finalize()
        
        print(f"✓ Dataset created: {dataset_name}")
        print(f"  ID: {dataset.id}")
        print(f"  Project: {self.project_name}")
        
        return dataset.id
    
    def create_from_url(
        self,
        dataset_name: str,
        url: str,
        description: str = "",
        extract: bool = True
    ) -> str:
        """
        URL에서 다운로드하여 ClearML Dataset 생성
        
        Args:
            dataset_name: 데이터셋 이름
            url: 다운로드 URL
            description: 데이터셋 설명
            extract: 압축 해제 여부
            
        Returns:
            생성된 Dataset ID
        """
        # 다운로드
        downloader = DatasetDownloader()
        filepath = downloader.download(url, extract=extract)
        
        # Dataset 생성
        dataset_id = self.create_from_files(
            dataset_name=dataset_name,
            files_path=filepath.parent if extract else filepath,
            description=description,
            tags=["auto-downloaded", url.split("/")[-1]]
        )
        
        return dataset_id
    
    @staticmethod
    def get_dataset(
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_project: Optional[str] = None
    ) -> Path:
        """
        ClearML Dataset 가져오기
        
        Args:
            dataset_id: 데이터셋 ID
            dataset_name: 데이터셋 이름
            dataset_project: 프로젝트 이름
            
        Returns:
            로컬 캐시 경로
        """
        dataset = Dataset.get(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_project=dataset_project
        )
        
        local_path = dataset.get_local_copy()
        print(f"Dataset loaded: {local_path}")
        
        return Path(local_path)
    
    @staticmethod
    def list_datasets(project_name: Optional[str] = None) -> List[dict]:
        """프로젝트의 데이터셋 목록 조회"""
        datasets = Dataset.list_datasets(dataset_project=project_name)
        
        print(f"\nDatasets in '{project_name or 'all projects'}':")
        print("-" * 60)
        
        for ds in datasets:
            print(f"  • {ds.get('name', 'N/A')} (ID: {ds.get('id', 'N/A')[:8]}...)")
        
        return datasets


# ===========================================
# 편의 함수
# ===========================================
def download_and_register(
    url: str,
    dataset_name: str,
    project_name: str = "Manifest-Datasets",
    extract: bool = True
) -> str:
    """
    URL에서 다운로드하고 ClearML에 등록하는 편의 함수
    
    Examples:
        # MNIST 데이터셋
        dataset_id = download_and_register(
            url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            dataset_name="MNIST-NPZ",
            project_name="Manifest-Vision"
        )
        
        # IMDB 데이터셋
        dataset_id = download_and_register(
            url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            dataset_name="IMDB-Reviews",
            project_name="Manifest-NLP",
            extract=True
        )
    """
    manager = ClearMLDatasetManager(project_name)
    return manager.create_from_url(
        dataset_name=dataset_name,
        url=url,
        extract=extract
    )


def get_dataset_path(
    dataset_name: str,
    project_name: str = "Manifest-Datasets"
) -> Path:
    """
    데이터셋 로컬 경로 가져오기
    
    Examples:
        path = get_dataset_path("MNIST-NPZ", "Manifest-Vision")
        data = np.load(path / "mnist.npz")
    """
    return ClearMLDatasetManager.get_dataset(
        dataset_name=dataset_name,
        dataset_project=project_name
    )


# ===========================================
# 테스트
# ===========================================
if __name__ == "__main__":
    # 테스트: MNIST 데이터셋 다운로드
    print("="*60)
    print("Dataset Utilities Test")
    print("="*60)
    
    # 다운로더 테스트
    downloader = DatasetDownloader("./data/test")
    
    # 작은 테스트 파일 다운로드
    test_url = "https://www.sample-videos.com/text/Sample-text-file-10kb.txt"
    
    try:
        path = downloader.download(test_url, "sample.txt")
        print(f"\n✓ Download test passed: {path}")
    except Exception as e:
        print(f"\n⚠ Download test (wget not available on Windows by default)")
        print(f"  Use Python requests library as alternative")
    
    print("\n" + "="*60)
    print("Usage Examples:")
    print("="*60)
    print("""
# 1. URL에서 다운로드 및 ClearML 등록
from dataset_utils import download_and_register

dataset_id = download_and_register(
    url="https://example.com/dataset.zip",
    dataset_name="My-Dataset",
    project_name="My-Project"
)

# 2. 등록된 데이터셋 가져오기
from dataset_utils import get_dataset_path

local_path = get_dataset_path("My-Dataset", "My-Project")
""")
