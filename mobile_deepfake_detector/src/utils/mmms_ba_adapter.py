"""
MMMS-BA 모델 데이터 어댑터

Phase 2 of Temporal Visualization Implementation
- npz 메타데이터 + 원본 영상 → 전체 프레임 데이터 변환
- add_timeline_metadata.py의 영상 검색 로직 재사용
- MMMS-BA 모델 학습/추론용 데이터 준비
"""

import sys
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Optional, List
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mobile_deepfake_detector.src.utils.full_frame_extractor import FullFrameExtractor


class MMSBAdapter:
    """
    MMMS-BA 모델 데이터 어댑터

    npz 파일의 메타데이터를 사용하여 원본 영상에서 전체 프레임을 추출하고
    MMMS-BA 모델이 사용할 수 있는 형식으로 변환합니다.

    특징:
    - npz 메타데이터 기반 전체 프레임 추출
    - 원본 영상 자동 검색 (재귀적 탐색)
    - 기존 audio (MFCC) 재사용
    - MMMS-BA 모델 입력 형식으로 변환
    """

    def __init__(self, original_video_root: str):
        """
        Initialize MMMS-BA adapter

        Args:
            original_video_root: 원본 영상 루트 디렉토리
                예: "E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터"
        """
        self.original_video_root = Path(original_video_root)
        self.frame_extractor = FullFrameExtractor()
        self.logger = logging.getLogger(__name__)

        if not self.original_video_root.exists():
            raise ValueError(f"Original video root not found: {original_video_root}")

    def load_npz_with_full_frames(
        self,
        npz_path: str,
        extract_lip: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        npz 파일 + 원본 영상에서 전체 프레임 데이터 추출

        Args:
            npz_path: 전처리된 npz 파일 경로
            extract_lip: 립 영역 추출 여부

        Returns:
            data: Dictionary with:
                - frames: (total_frames, 224, 224, 3) - 전체 프레임
                - audio: (T, 40) - MFCC features (기존 데이터 재사용)
                - lip: (total_frames, 112, 112, 3) - 전체 립 프레임 (optional)
                - label: 0 (Real) or 1 (Fake)
                - timestamps: (total_frames,) - 프레임별 타임스탬프
                - video_id: str - 영상 ID
                - total_frames: int - 총 프레임 수
                - video_fps: float - 영상 FPS
        """
        npz_path = Path(npz_path)
        self.logger.info(f"\nLoading: {npz_path.name}")

        # 1. Load npz metadata
        try:
            npz_data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            raise ValueError(f"Failed to load npz: {e}")

        # 2. Check metadata exists
        required_keys = ['full_frame_indices', 'full_timestamps', 'total_frames', 'video_fps']
        missing_keys = [k for k in required_keys if k not in npz_data.keys()]

        if missing_keys:
            raise ValueError(
                f"Missing metadata in npz: {missing_keys}\n"
                f"Please run add_timeline_metadata.py first!"
            )

        # 3. Extract metadata
        video_id = str(npz_data['video_id'])
        total_frames = int(npz_data['total_frames'])
        video_fps = float(npz_data['video_fps'])

        self.logger.info(f"  Video ID: {video_id}")
        self.logger.info(f"  Total frames: {total_frames}")
        self.logger.info(f"  FPS: {video_fps:.2f}")

        metadata = {
            'total_frames': total_frames,
            'video_fps': video_fps
        }

        # 4. Find original video
        original_video_path = self._find_original_video(video_id)

        if original_video_path is None:
            raise ValueError(f"Original video not found: {video_id}")

        self.logger.info(f"  Original video: {original_video_path}")

        # 5. Extract all frames
        all_frames = self.frame_extractor.extract_all_frames(
            video_path=str(original_video_path),
            metadata=metadata,
            preprocess=True
        )

        # 6. Extract lip regions (optional)
        all_lips = None
        if extract_lip:
            all_lips = self.frame_extractor.extract_all_lip_regions(
                frames=all_frames,
                lip_size=(112, 112)
            )

        # 7. Prepare result
        result = {
            'frames': all_frames,
            'audio': npz_data['audio'],  # Reuse existing MFCC
            'lip': all_lips,
            'label': int(npz_data['label']),
            'timestamps': npz_data['full_timestamps'],
            'video_id': video_id,
            'total_frames': total_frames,
            'video_fps': video_fps
        }

        self.logger.info(f"[OK] Data loaded successfully")
        self.logger.info(f"  Frames: {all_frames.shape}")
        self.logger.info(f"  Audio: {npz_data['audio'].shape}")
        if all_lips is not None:
            self.logger.info(f"  Lip: {all_lips.shape}")
        self.logger.info(f"  Label: {result['label']} ({'Fake' if result['label'] == 1 else 'Real'})")

        return result

    def _find_original_video(self, video_id: str) -> Optional[Path]:
        """
        video_id로 원본 영상 찾기 (재귀적 탐색)
        Reuse logic from add_timeline_metadata.py

        Args:
            video_id: 영상 ID (e.g., "00000.mp4" or "00000")

        Returns:
            original_path: 원본 영상 경로 or None
        """
        # Ensure .mp4 extension
        if not video_id.endswith('.mp4'):
            video_id = f"{video_id}.mp4"

        # 1. Try simple paths first (01.원본, 02.변조)
        for category in ['01.원본', '02.변조']:
            candidate = self.original_video_root / category / video_id
            if candidate.exists():
                return candidate

        # 2. Recursive search if not found
        for video_file in self.original_video_root.rglob(video_id):
            if video_file.is_file():
                return video_file

        return None

    def batch_load_split(
        self,
        npz_dir: str,
        split: str = 'test',
        max_samples: Optional[int] = None,
        extract_lip: bool = True
    ) -> List[Dict]:
        """
        Split 전체 데이터 배치 로드

        Args:
            npz_dir: npz 파일 디렉토리 (e.g., "preprocessed_data_phoneme")
            split: Split name ('train', 'val', 'test')
            max_samples: 최대 샘플 수 (None이면 전체)
            extract_lip: 립 영역 추출 여부

        Returns:
            data_list: List of data dictionaries
        """
        npz_dir = Path(npz_dir) / split

        if not npz_dir.exists():
            raise ValueError(f"Split directory not found: {npz_dir}")

        npz_files = sorted(npz_dir.glob("*.npz"))

        if max_samples is not None:
            npz_files = npz_files[:max_samples]

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Batch loading {split} split ({len(npz_files)} files)")
        self.logger.info(f"{'='*60}")

        data_list = []
        failed_count = 0

        for npz_path in tqdm(npz_files, desc=f"Loading {split}"):
            try:
                data = self.load_npz_with_full_frames(
                    npz_path=str(npz_path),
                    extract_lip=extract_lip
                )
                data_list.append(data)

            except Exception as e:
                self.logger.error(f"[ERROR] Failed to load {npz_path.name}: {e}")
                failed_count += 1
                continue

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Batch loading complete")
        self.logger.info(f"  Success: {len(data_list)}/{len(npz_files)}")
        self.logger.info(f"  Failed: {failed_count}")
        self.logger.info(f"{'='*60}")

        return data_list

    def convert_to_mmms_ba_format(
        self,
        data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        MMMS-BA 모델 입력 형식으로 변환

        MMMS-BA expects:
        - visual: (T, 224, 224, 3)
        - audio: (T', 40) - MFCC features
        - lip: (T, 112, 112, 3)
        - label: 0 or 1

        Args:
            data: load_npz_with_full_frames() output

        Returns:
            mmms_ba_data: MMMS-BA 형식 데이터
        """
        mmms_ba_data = {
            'visual': data['frames'],  # (T, 224, 224, 3)
            'audio': data['audio'],    # (T', 40)
            'lip': data['lip'],        # (T, 112, 112, 3) or None
            'label': data['label']     # 0 or 1
        }

        return mmms_ba_data

    def get_statistics(self, data_list: List[Dict]) -> Dict:
        """
        데이터셋 통계 계산

        Args:
            data_list: List of data dictionaries

        Returns:
            stats: Statistics dictionary
        """
        total_frames = sum(d['total_frames'] for d in data_list)
        total_duration = sum(d['total_frames'] / d['video_fps'] for d in data_list)
        real_count = sum(1 for d in data_list if d['label'] == 0)
        fake_count = sum(1 for d in data_list if d['label'] == 1)

        stats = {
            'total_videos': len(data_list),
            'total_frames': total_frames,
            'total_duration_sec': total_duration,
            'avg_frames_per_video': total_frames / len(data_list) if data_list else 0,
            'avg_duration_per_video': total_duration / len(data_list) if data_list else 0,
            'real_count': real_count,
            'fake_count': fake_count,
            'real_ratio': real_count / len(data_list) if data_list else 0,
            'fake_ratio': fake_count / len(data_list) if data_list else 0
        }

        return stats


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)

    # Test initialization
    original_root = "E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터"

    adapter = MMSBAdapter(original_video_root=original_root)

    print("\nMMSBAdapter initialized successfully!")
    print(f"Original video root: {adapter.original_video_root}")
