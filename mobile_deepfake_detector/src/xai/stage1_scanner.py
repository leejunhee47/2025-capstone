import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List

from ..models.teacher import MMMSBA
from ..utils.config import load_config

# Setup logger
logger = logging.getLogger(__name__)

# ========================================
# Persistence-based Suspicious Interval Detection
# ========================================
# 지속성 기반 의심구간 탐지 파라미터
# - Fake: plateau 패턴 (높은 값 연속 유지)
# - Real: spiky 패턴 (빠른 상승/하락)
# - 연속 N프레임 이상 threshold 초과 시 의심구간으로 판정
PERSISTENCE_THRESHOLD = 0.6       # 고정 threshold (Real 스파이크 필터링)
MIN_CONSECUTIVE_FRAMES = 20       # 최소 연속 프레임 (2초 at 10fps, 엄격한 설정)

class Stage1Scanner:
    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/train_teacher_korean.yaml",
        device: str = "cuda"
    ):
        logger.info("Initializing Stage1Scanner (MMMS-BA)...")
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        # Load configuration
        self.config = load_config(config_path)
        
        self.mmms_model: Optional[nn.Module] = None
        
        # Load MMMS-BA model
        if model_path != "dummy":
            self.mmms_model = self._load_mmms_model(model_path)
        else:
            logger.warning("MMMS-BA model skipped (dummy path)")

    def _load_mmms_model(self, mmms_model_path: str) -> nn.Module:
        """Load MMMS-BA model from checkpoint."""
        model_config = self.config.get('model', {})
        
        model = MMMSBA(
            audio_dim=self.config.get('dataset', {}).get('audio', {}).get('n_mfcc', 40),
            visual_dim=256,
            lip_dim=128,
            gru_hidden_dim=model_config.get('gru', {}).get('hidden_size', 300),
            gru_num_layers=model_config.get('gru', {}).get('num_layers', 3),
            gru_dropout=model_config.get('gru', {}).get('dropout', 0.3),
            dense_hidden_dim=model_config.get('dense', {}).get('hidden_size', 100),
            dense_dropout=model_config.get('dense', {}).get('dropout', 0.5),
            attention_type=model_config.get('attention', {}).get('type', 'bimodal'),
            num_classes=model_config.get('num_classes', 2)
        ).to(self.device)

        try:
            checkpoint = torch.load(mmms_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            logger.error(f"Failed to load MMMS-BA model: {e}")
            # Handle failure gracefully or raise depending on requirement
            # raise e 

        model.eval()
        return model

    def predict_full_video(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fake probability for the full video using MMMS-BA.
        
        Args:
            features: Dict containing extracted features (frames_10fps, audio, lip_10fps)
            
        Returns:
            result: Dict with verdict, confidence, probabilities, etc.
        """
        if self.mmms_model is None:
            return {'verdict': 'unknown', 'confidence': 0.0, 'probabilities': {'fake': 0.0, 'real': 0.0}}

        logger.info("[Stage1] Running MMMS-BA prediction...")
        
        # Prepare inputs
        # MMMS-BA expects (1, T, C, H, W) for visual/lip and (1, T_audio, C) for audio
        frames = features['frames_10fps'] # (T, 224, 224, 3)
        audio = features['audio']         # (T_audio, 40)
        lip = features['lip_10fps']       # (T, 96, 96, 3)
        
        if len(frames) == 0:
             logger.warning("No frames for MMMS-BA prediction")
             return {'verdict': 'unknown', 'confidence': 0.0, 'probabilities': {'fake': 0.0, 'real': 0.0}}

        # Convert to tensors
        # Assuming frames/lip are already normalized [0,1] float32
        frames_tensor = torch.from_numpy(frames).float()
        lip_tensor = torch.from_numpy(lip).float()
        
        # Check range and permute
        if frames_tensor.ndim == 4 and frames_tensor.shape[-1] == 3: # (T, H, W, C) -> (1, T, C, H, W)
            frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0)
            lip_tensor = lip_tensor.permute(0, 3, 1, 2).unsqueeze(0)
            
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0) # (1, T_audio, 40)
        
        # Move to device
        frames_tensor = frames_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)
        lip_tensor = lip_tensor.to(self.device)
        
        with torch.no_grad():
            # Forward pass (frame_level=True to get per-frame scores)
            logits = self.mmms_model(
                audio=audio_tensor, 
                frames=frames_tensor, 
                lip=lip_tensor,
                frame_level=True
            )
            
            # logits: (1, T, 2)
            frame_probs = torch.softmax(logits, dim=-1)
            fake_probs = frame_probs[0, :, 1].cpu().numpy()
            
        # Aggregation
        mean_fake_prob = float(np.mean(fake_probs))
        
        verdict = 'fake' if mean_fake_prob > 0.5 else 'real'
        confidence = mean_fake_prob if verdict == 'fake' else 1.0 - mean_fake_prob
        
        logger.info(f"  MMMS-BA Result: {verdict.upper()} ({confidence:.2%})")
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'probabilities': {
                'real': 1.0 - mean_fake_prob,
                'fake': mean_fake_prob
            },
            'frame_probs': fake_probs,
            'mean_fake_prob': mean_fake_prob,
            'threshold': PERSISTENCE_THRESHOLD,  # 고정 threshold 사용
            'min_consecutive_frames': MIN_CONSECUTIVE_FRAMES
        }

    def find_suspicious_intervals(
        self,
        frame_probs: np.ndarray,
        timestamps: np.ndarray,
        min_candidates: int = 5,
        max_candidates: int = 10,
        min_duration: float = 3.0,  # 최소 3초 이상 확보 (PIA phoneme coverage 보장)
        merge_gap: float = 1.0,
        padding_sec: float = 0.5  # Interval에 앞뒤 패딩 추가 (phoneme coverage 증가)
    ) -> List[Dict[str, Any]]:
        """
        지속성 기반 의심구간 탐지 (Persistence-based Detection).

        핵심 아이디어:
        - Fake 영상: plateau 패턴 (높은 값이 연속 유지)
        - Real 영상: spiky 패턴 (빠르게 상승 후 급하강)
        - 고정 threshold (0.6) + 최소 연속 프레임 조건 (20프레임 = 2초)으로
          스파이크 필터링, plateau만 의심구간으로 선정

        Args:
            frame_probs: MMMS-BA frame-level fake probabilities (T,)
            timestamps: Corresponding timestamps in seconds (T,)
            min_candidates: (unused in persistence mode)
            max_candidates: 최대 구간 개수
            min_duration: 최소 구간 길이 (초)
            merge_gap: 이 시간(초) 이내 구간 병합
            padding_sec: Interval 앞뒤 패딩 (초)

        Returns:
            List of intervals sorted by mean_prob descending:
            [
                {
                    'start': 3.2,
                    'end': 5.8,
                    'mean_prob': 0.92,
                    'duration': 2.6,
                    'interval_id': 'interval_001',
                    'frame_count': 26
                },
                ...
            ]
        """
        total_frames = len(frame_probs)

        if total_frames == 0:
            logger.warning("No frames for suspicious interval detection")
            return []

        mean_prob = float(np.mean(frame_probs))
        max_prob = float(np.max(frame_probs))

        logger.info(f"[Stage1] Persistence-based suspicious interval detection:")
        logger.info(f"  Total frames: {total_frames}, Mean P(Fake): {mean_prob:.3f}, Max P(Fake): {max_prob:.3f}")
        logger.info(f"  Threshold: {PERSISTENCE_THRESHOLD}, Min consecutive frames: {MIN_CONSECUTIVE_FRAMES}")

        # Step 1: 고정 threshold로 마스크 생성
        high_prob_mask = frame_probs >= PERSISTENCE_THRESHOLD

        # Step 2: 연속 구간 찾기 (numpy diff 기반)
        padded = np.concatenate([[False], high_prob_mask, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            logger.info(f"  No frames exceed threshold {PERSISTENCE_THRESHOLD}")
            return []

        # Step 3: 지속성 조건으로 필터링 (최소 N프레임 연속)
        intervals = []
        for start_idx, end_idx in zip(starts, ends):
            frame_count = end_idx - start_idx

            # 지속성 조건: 최소 MIN_CONSECUTIVE_FRAMES 연속
            if frame_count >= MIN_CONSECUTIVE_FRAMES:
                start_time = float(timestamps[start_idx])
                end_time = float(timestamps[min(end_idx - 1, len(timestamps) - 1)])
                duration = end_time - start_time
                interval_mean_prob = float(np.mean(frame_probs[start_idx:end_idx]))

                intervals.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': duration,
                    'mean_prob': interval_mean_prob,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'frame_count': frame_count
                })

        if len(intervals) == 0:
            # 지속성 조건을 충족하는 구간 없음 → Real 영상으로 추정
            logger.info(f"  No intervals with {MIN_CONSECUTIVE_FRAMES}+ consecutive frames (likely Real)")
            return []

        # Step 4: 인접 구간 병합 (merge_gap 이내)
        merged_intervals = []
        current = intervals[0]

        for next_interval in intervals[1:]:
            gap = next_interval['start'] - current['end']

            if gap <= merge_gap:
                # Merge
                current['end'] = next_interval['end']
                current['duration'] = current['end'] - current['start']
                current['end_idx'] = next_interval['end_idx']
                current['frame_count'] = current['end_idx'] - current['start_idx']
                current['mean_prob'] = float(
                    np.mean(frame_probs[current['start_idx']:current['end_idx']])
                )
            else:
                merged_intervals.append(current)
                current = next_interval

        merged_intervals.append(current)

        # Step 5: min_duration 필터링
        filtered_intervals = [
            interval for interval in merged_intervals
            if interval['duration'] >= min_duration
        ]

        if len(filtered_intervals) == 0:
            logger.warning(f"No intervals >= {min_duration}s duration after persistence filter")
            # Fallback: 가장 긴 구간 하나 선택
            if merged_intervals:
                filtered_intervals = sorted(merged_intervals, key=lambda x: x['frame_count'], reverse=True)[:1]

        # Step 6: mean_prob 기준 정렬 및 top N 선택
        filtered_intervals.sort(key=lambda x: x['mean_prob'], reverse=True)
        top_intervals = filtered_intervals[:max_candidates]

        # Step 7: 패딩 추가 및 interval_id 부여
        video_duration = float(timestamps[-1])
        final_intervals = []

        for i, interval in enumerate(top_intervals):
            padded_start = max(0.0, interval['start'] - padding_sec)
            padded_end = min(video_duration, interval['end'] + padding_sec)

            final_intervals.append({
                'start': padded_start,
                'end': padded_end,
                'mean_prob': interval['mean_prob'],
                'duration': padded_end - padded_start,
                'interval_id': f'interval_{i+1:03d}',
                'frame_count': interval['frame_count'],
                'original_start': interval['start'],
                'original_end': interval['end']
            })

        logger.info(f"  Found {len(final_intervals)} suspicious intervals (persistence-based)")
        for i, interval in enumerate(final_intervals[:3]):
            logger.info(f"    #{i+1}: {interval['start']:.1f}~{interval['end']:.1f}s "
                       f"(prob={interval['mean_prob']:.3f}, dur={interval['duration']:.1f}s, frames={interval['frame_count']})")

        return final_intervals
