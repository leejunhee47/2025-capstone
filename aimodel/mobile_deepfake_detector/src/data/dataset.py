"""
Dataset classes for deepfake detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

from .preprocessing import ShortsPreprocessor


class MMSBDataset(Dataset):
    """
    MMMS-BA 학습용 Dataset (preprocessed_mmms-ba/ 폴더 구조)

    구조:
    preprocessed_mmms-ba/
    ├── train_index.json
    ├── val_index.json
    ├── test_index.json
    ├── real/
    │   └── *.npz
    └── fake/
        └── *.npz
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        config: Optional[Dict] = None,
        augmentation = None
    ):
        """
        Initialize MMMS-BA dataset

        Args:
            data_root: preprocessed_mmms-ba/ 경로
            split: 'train', 'val', 'test'
            config: Configuration dictionary
            augmentation: Augmentation function (train split only)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.config = config or {}
        self.augmentation = augmentation if split == "train" else None

        self.logger = logging.getLogger(__name__)

        # Index JSON 로드
        index_file = self.data_root / f"{split}_index.json"

        if not index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: {index_file}\n"
                f"Run create_mmms_ba_splits.py first!"
            )

        with open(index_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        self.logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        self.logger.info(f"Data root: {self.data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        NPZ 파일에서 데이터 로드

        Args:
            idx: 샘플 인덱스

        Returns:
            item: Dictionary with tensors
        """
        sample = self.samples[idx]

        # NPZ 파일 경로 (real/00000.npz 또는 fake/00000.npz)
        npz_path = self.data_root / sample['npz_path']

        # NPZ 파일 로드
        try:
            data = np.load(npz_path)

            # Frames: (N, H, W, 3) -> (N, 3, H, W)
            frames = torch.from_numpy(data['frames']).float()
            frames = frames.permute(0, 3, 1, 2)  # (N, 3, H, W)

            # Audio: (T, n_mfcc)
            audio = torch.from_numpy(data['audio']).float()

            # Lip: (N, lip_H, lip_W, 3) -> (N, 3, lip_H, lip_W)
            lip = torch.from_numpy(data['lip']).float()
            lip = lip.permute(0, 3, 1, 2)  # (N, 3, lip_H, lip_W)

            # Label (0: REAL, 1: FAKE)
            label = torch.tensor(sample['label'], dtype=torch.long)

            item = {
                'frames': frames,
                'audio': audio,
                'lip': lip,
                'label': label,
                'video_id': sample['video_id']
            }

            # Augmentation (train only)
            if self.augmentation is not None:
                item = self.augmentation(item)

            return item

        except Exception as e:
            self.logger.error(f"Error loading {npz_path}: {e}")
            raise


class PreprocessedDeepfakeDataset(Dataset):
    """
    전처리된 .npz 파일을 직접 로드하는 Dataset
    비디오 처리 없이 빠른 학습 가능
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        config: Optional[Dict] = None,
        augmentation = None
    ):
        """
        Initialize preprocessed dataset

        Args:
            data_root: 전처리 데이터 루트 디렉토리 (preprocessed_data_real/)
            split: 'train', 'val', 'test'
            config: Configuration dictionary (사용 안 함, 호환성용)
            augmentation: Augmentation function (train split only)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.config = config or {}
        self.augmentation = augmentation

        self.logger = logging.getLogger(__name__)
        
        # .npz 파일 디렉토리
        self.npz_dir = self.data_root / split
        
        # 전처리 인덱스 로드
        index_file = self.data_root / f"{split}_preprocessed_index.json"
        
        if not index_file.exists():
            raise FileNotFoundError(
                f"Preprocessed index not found: {index_file}\n"
                f"Run preprocessing first!"
            )
        
        with open(index_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        self.logger.info(f"Loaded {len(self.samples)} preprocessed samples for {split} split")
        self.logger.info(f"NPZ directory: {self.npz_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        npz 파일에서 직접 로드 (매우 빠름!)
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            item: Dictionary with tensors
        """
        sample = self.samples[idx]
        
        # npz 파일 경로 (output_path는 상대 경로)
        npz_path = self.data_root / sample['output_path']
        
        # npz 파일 로드 (매우 빠름 - 이미 전처리됨)
        try:
            data = np.load(npz_path)
            
            # Frames: (N, H, W, 3) -> (N, 3, H, W)
            frames = torch.from_numpy(data['frames']).float()
            frames = frames.permute(0, 3, 1, 2)  # (N, 3, H, W)
            
            # Audio: (T, n_mfcc)
            audio = torch.from_numpy(data['audio']).float()
            
            # Lip: (N, lip_H, lip_W, 3) -> (N, 3, lip_H, lip_W)
            lip = torch.from_numpy(data['lip']).float()
            lip = lip.permute(0, 3, 1, 2)  # (N, 3, lip_H, lip_W)
            
            # Label
            label = torch.tensor(int(data['label']), dtype=torch.long)

            item = {
                'frames': frames,
                'audio': audio,
                'lip': lip,
                'label': label
            }

            # Apply augmentation (train split only)
            if self.augmentation is not None:
                item = self.augmentation(item)

            return item
            
        except Exception as e:
            self.logger.error(f"Failed to load {npz_path}: {e}")
            # 에러 시 더미 데이터 반환 (건너뛰기)
            return {
                'frames': None,
                'audio': None,
                'lip': None,
                'label': torch.tensor(0, dtype=torch.long)
            }


class DeepfakeDataset(Dataset):
    """
    Deepfake detection dataset
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        config: Optional[Dict] = None,
        preload: bool = False
    ):
        """
        Initialize dataset

        Args:
            data_root: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            config: Configuration dictionary
            preload: Whether to preload all data into memory
        """
        self.data_root = Path(data_root)
        self.split = split
        self.config = config or {}
        self.preload = preload

        self.logger = logging.getLogger(__name__)

        # Initialize preprocessor
        self.preprocessor = ShortsPreprocessor(self.config)

        # Load dataset index
        self.samples = self._load_samples()

        self.logger.info(f"Loaded {len(self.samples)} samples for {split} split")

        # Preload data if requested
        self.preloaded_data = {}
        if self.preload:
            self._preload_data()

    def _load_samples(self) -> List[Dict]:
        """
        Load dataset samples

        Returns:
            samples: List of sample dictionaries
        """
        samples = []

        # Load from JSON index if exists
        index_file = self.data_root / f"{self.split}_index.json"

        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 리스트 형태 또는 딕셔너리 형태 모두 처리
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict):
                    samples = data.get('samples', [])
                else:
                    samples = []

            self.logger.info(f"Loaded index from {index_file}")
        else:
            # Scan directory structure
            self.logger.info(f"Index file not found, scanning directory...")
            samples = self._scan_directory()

            # Save index
            self._save_index(samples, index_file)

        return samples

    def _scan_directory(self) -> List[Dict]:
        """
        Scan directory for video files

        Returns:
            samples: List of sample dictionaries
        """
        samples = []

        # Expected structure: data_root/split/label/*.mp4
        split_dir = self.data_root / self.split

        if not split_dir.exists():
            self.logger.warning(f"Split directory not found: {split_dir}")
            return samples

        # Scan real and fake directories
        for label in ['real', 'fake']:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue

            video_files = list(label_dir.glob("*.mp4"))
            video_files.extend(list(label_dir.glob("*.avi")))

            for video_path in video_files:
                samples.append({
                    'video_path': str(video_path),
                    'label': 0 if label == 'real' else 1,
                    'label_name': label
                })

        self.logger.info(f"Found {len(samples)} videos")

        return samples

    def _save_index(self, samples: List[Dict], index_file: Path):
        """Save dataset index"""
        index_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'split': self.split,
            'num_samples': len(samples),
            'samples': samples
        }

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved index to {index_file}")

    def _preload_data(self):
        """Preload all data into memory"""
        self.logger.info("Preloading data...")

        for idx in range(len(self.samples)):
            try:
                data = self._load_sample(idx)
                self.preloaded_data[idx] = data
            except Exception as e:
                self.logger.error(f"Failed to preload sample {idx}: {e}")

        self.logger.info(f"Preloaded {len(self.preloaded_data)} samples")

    def _load_sample(self, idx: int) -> Dict:
        """
        Load single sample

        Args:
            idx: Sample index

        Returns:
            data: Dictionary with processed data
        """
        sample = self.samples[idx]
        video_path = sample['video_path']
        
        # 상대 경로를 절대 경로로 변환
        if not Path(video_path).is_absolute():
            # preprocessed_data의 부모 디렉토리 기준으로 변환
            video_path = self.data_root.parent / video_path
        
        video_path = str(video_path)

        # Process video
        result = self.preprocessor.process_video(
            video_path,
            extract_audio=True,
            extract_lip=True
        )

        # Add label (문자열 또는 숫자 형식 모두 처리)
        label = sample['label']
        if isinstance(label, str):
            result['label'] = 0 if label.lower() == 'real' else 1
        else:
            result['label'] = label

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single item

        Args:
            idx: Sample index

        Returns:
            item: Dictionary with tensors
                - frames: (T, 3, H, W)
                - audio: (T_audio, n_mfcc)
                - lip: (T, 3, lip_H, lip_W)
                - label: scalar
        """
        # Load from preloaded data or process on-the-fly
        if self.preload and idx in self.preloaded_data:
            data = self.preloaded_data[idx]
        else:
            data = self._load_sample(idx)

        # Convert to tensors
        item = {}

        # Frames: (N, H, W, 3) -> (N, 3, H, W)
        if data['frames'] is not None:
            frames = torch.from_numpy(data['frames']).float()
            frames = frames.permute(0, 3, 1, 2)  # (N, 3, H, W)
            item['frames'] = frames
        else:
            item['frames'] = None

        # Audio: (T, n_mfcc)
        if data['audio'] is not None:
            audio = torch.from_numpy(data['audio']).float()
            item['audio'] = audio
        else:
            item['audio'] = None

        # Lip: (N, lip_H, lip_W, 3) -> (N, 3, lip_H, lip_W)
        if data['lip'] is not None:
            lip = torch.from_numpy(data['lip']).float()
            lip = lip.permute(0, 3, 1, 2)
            item['lip'] = lip
        else:
            item['lip'] = None

        # Label
        item['label'] = torch.tensor(data['label'], dtype=torch.long)

        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences

    Args:
        batch: List of items

    Returns:
        batched: Dictionary with batched tensors
    """
    # 길이가 0인 샘플 필터링 (None이거나 길이 0인 경우)
    valid_batch = []
    for item in batch:
        # frames, audio, lip 모두 유효한지 확인
        if (item['frames'] is not None and item['frames'].size(0) > 0 and
            item['audio'] is not None and item['audio'].size(0) > 0 and
            item['lip'] is not None and item['lip'].size(0) > 0):
            valid_batch.append(item)
    
    # 유효한 샘플이 없으면 빈 배치 반환
    if len(valid_batch) == 0:
        return None
    
    batch = valid_batch  # 유효한 샘플만 사용
    
    # Get max sequence lengths
    max_frames = max(item['frames'].size(0) for item in batch if item['frames'] is not None)
    max_audio = max(item['audio'].size(0) for item in batch if item['audio'] is not None)

    batch_size = len(batch)

    # Initialize tensors
    frames_batch = torch.zeros(batch_size, max_frames, 3, 224, 224)
    audio_batch = torch.zeros(batch_size, max_audio, 40)  # n_mfcc=40
    lip_batch = torch.zeros(batch_size, max_frames, 3, 112, 112)
    labels_batch = torch.zeros(batch_size, dtype=torch.long)

    # Masks
    frames_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    audio_mask = torch.zeros(batch_size, max_audio, dtype=torch.bool)

    # Fill tensors
    for i, item in enumerate(batch):
        # Frames
        if item['frames'] is not None:
            seq_len = item['frames'].size(0)
            frames_batch[i, :seq_len] = item['frames']
            frames_mask[i, :seq_len] = True

        # Audio
        if item['audio'] is not None:
            seq_len = item['audio'].size(0)
            audio_batch[i, :seq_len] = item['audio']
            audio_mask[i, :seq_len] = True

        # Lip
        if item['lip'] is not None:
            seq_len = item['lip'].size(0)
            lip_batch[i, :seq_len] = item['lip']

        # Label
        labels_batch[i] = item['label']

    return {
        'frames': frames_batch,
        'audio': audio_batch,
        'lip': lip_batch,
        'label': labels_batch,
        'frames_mask': frames_mask,
        'audio_mask': audio_mask
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    sampler=None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2
) -> DataLoader:
    """
    Create dataloader with advanced optimizations

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if sampler is provided)
        num_workers: Number of workers
        sampler: Custom sampler (for distributed training)
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        dataloader: DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,  # sampler와 함께 shuffle 사용 불가
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True if shuffle else False  # 학습 시 마지막 배치 버림
    )
