import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List

from ..models.teacher import MMMSBA
from ..utils.config import load_config

# Setup logger
logger = logging.getLogger(__name__)

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
            'threshold': mean_fake_prob * 0.8 if mean_fake_prob >= 0.5 else None
        }

    def find_suspicious_intervals(
        self,
        frame_probs: np.ndarray,
        timestamps: np.ndarray,
        min_candidates: int = 5,
        max_candidates: int = 10,
        min_duration: float = 3.0,  # 최소 3초 이상 확보 (PIA phoneme coverage 보장)
        merge_gap: float = 1.0,
        padding_sec: float = 0.5  # [NEW] Interval에 앞뒤 패딩 추가 (phoneme coverage 증가)
    ) -> List[Dict[str, Any]]:
        """
        Min/Max Candidates 전략으로 의심 구간 추출.

        Strategy:
        - 영상 길이 무관하게 안정적 개수 (5~10개)
        - Relative threshold (상대적 percentile)
        - Consecutive grouping + Merge
        - 안전장치: 200 프레임 이하 짧은 영상 대응

        Args:
            frame_probs: MMMS-BA frame-level fake probabilities (T,)
            timestamps: Corresponding timestamps in seconds (T,)
            min_candidates: 최소 구간 개수 (짧은 영상 대응)
            max_candidates: 최대 구간 개수 (긴 영상 과다 방지)
            min_duration: 최소 구간 길이 (초)
            merge_gap: 이 시간(초) 이내 구간 병합
            padding_sec: Interval 앞뒤 패딩 (초) - phoneme coverage 증가

        Returns:
            List of intervals sorted by mean_prob descending:
            [
                {
                    'start': 3.2,
                    'end': 5.8,
                    'mean_prob': 0.92,
                    'duration': 2.6,
                    'interval_id': 'interval_001'
                },
                ...
            ]
        """
        total_frames = len(frame_probs)

        if total_frames == 0:
            logger.warning("No frames for suspicious interval detection")
            return []

        # Step 1: 동적 Percentile 계산 with 안전장치
        raw_target = total_frames // 40  # 20초 영상(200 frames) → 5개
        mean_prob = float(np.mean(frame_probs))
        if mean_prob < 0.5:
            logger.info(f"Mean fake probability {mean_prob:.3f} < 0.5 (likely real). Skipping interval detection.")
            return []

        absolute_threshold = max(0.0, min(1.0, mean_prob * 0.8))
        threshold = absolute_threshold

        logger.info(f"[Stage1] Finding suspicious intervals with absolute threshold:")
        logger.info(f"  Total frames: {total_frames}, Mean fake prob: {mean_prob:.3f}")
        logger.info(f"  Threshold: {threshold:.3f}")

        # Step 2: Consecutive Grouping
        high_prob_mask = frame_probs >= threshold

        # Find change points
        padded = np.concatenate([[False], high_prob_mask, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            logger.warning("No high probability regions found")
            return []

        # Step 3: Create initial intervals
        intervals = []
        for start_idx, end_idx in zip(starts, ends):
            start_time = float(timestamps[start_idx])
            end_time = float(timestamps[end_idx - 1])  # end_idx는 exclusive
            duration = end_time - start_time

            mean_prob = float(np.mean(frame_probs[start_idx:end_idx]))

            intervals.append({
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'mean_prob': mean_prob,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

        if len(intervals) == 0:
            logger.warning("No intervals created")
            return []

        # Step 4: Merge nearby intervals
        merged_intervals = []
        current = intervals[0]

        for next_interval in intervals[1:]:
            gap = next_interval['start'] - current['end']

            if gap <= merge_gap:
                # Merge
                current['end'] = next_interval['end']
                current['duration'] = current['end'] - current['start']
                current['end_idx'] = next_interval['end_idx']

                # Recalculate mean_prob
                current['mean_prob'] = float(
                    np.mean(frame_probs[current['start_idx']:current['end_idx']])
                )
            else:
                # Save current and move to next
                merged_intervals.append(current)
                current = next_interval

        # Add last interval
        merged_intervals.append(current)

        # Step 4.5: 짧은 구간 병합 → 최소 3초 확보 (PIA phoneme coverage 보장)
        SHORT_THRESHOLD = 1.0
        TARGET_MERGE_DURATION = max(min_duration, 3.0)

        enhanced_intervals = []
        i = 0
        while i < len(merged_intervals):
            current_interval = merged_intervals[i]

            # 짧은 구간이면 인접 구간들과 병합 시도
            if current_interval['duration'] < SHORT_THRESHOLD:
                # 앞/뒤 구간과 병합하여 1초 이상 만들기
                merge_candidates = [current_interval]
                accumulated_duration = current_interval['duration']

                # 앞으로 탐색
                j = i - 1
                while j >= 0 and accumulated_duration < TARGET_MERGE_DURATION:
                    if merged_intervals[j] not in [x for sublist in [m['merged_from'] if 'merged_from' in m else [m] for m in enhanced_intervals] for x in sublist]:
                        merge_candidates.insert(0, merged_intervals[j])
                        accumulated_duration += merged_intervals[j]['duration']
                    j -= 1

                # 뒤로 탐색
                j = i + 1
                while j < len(merged_intervals) and accumulated_duration < TARGET_MERGE_DURATION:
                    merge_candidates.append(merged_intervals[j])
                    accumulated_duration += merged_intervals[j]['duration']
                    j += 1

                # 병합 수행
                if len(merge_candidates) > 1:
                    merged = {
                        'start': merge_candidates[0]['start'],
                        'end': merge_candidates[-1]['end'],
                        'duration': merge_candidates[-1]['end'] - merge_candidates[0]['start'],
                        'start_idx': merge_candidates[0]['start_idx'],
                        'end_idx': merge_candidates[-1]['end_idx'],
                        'merged_from': merge_candidates
                    }
                    # Recalculate mean_prob
                    merged['mean_prob'] = float(
                        np.mean(frame_probs[merged['start_idx']:merged['end_idx']])
                    )
                    enhanced_intervals.append(merged)
                    i = i + len([m for m in merge_candidates if m['start'] >= current_interval['start']])
                else:
                    enhanced_intervals.append(current_interval)
                    i += 1
            else:
                enhanced_intervals.append(current_interval)
                i += 1

        # Step 5: Filter by min_duration
        filtered_intervals = [
            interval for interval in enhanced_intervals
            if interval['duration'] >= min_duration
        ]

        if len(filtered_intervals) == 0:
            logger.warning(f"No intervals >= {min_duration}s duration")
            # Fallback: return top interval regardless of duration
            filtered_intervals = sorted(merged_intervals, key=lambda x: x['mean_prob'], reverse=True)[:1]

        # Step 6: Sort by mean_prob and select top N
        filtered_intervals.sort(key=lambda x: x['mean_prob'], reverse=True)
        top_intervals = filtered_intervals[:max_candidates]

        # Step 7: Add padding and interval_id
        # [NEW] 0.5초 패딩으로 phoneme coverage 증가 (짧은 구간 보완)
        video_duration = float(timestamps[-1])
        final_intervals = []
        for i, interval in enumerate(top_intervals):
            # Apply padding with video boundary check
            padded_start = max(0.0, interval['start'] - padding_sec)
            padded_end = min(video_duration, interval['end'] + padding_sec)

            final_intervals.append({
                'start': padded_start,
                'end': padded_end,
                'mean_prob': interval['mean_prob'],
                'duration': padded_end - padded_start,
                'interval_id': f'interval_{i+1:03d}',
                'original_start': interval['start'],  # 디버깅용
                'original_end': interval['end']
            })

        logger.info(f"  Found {len(final_intervals)} suspicious intervals")
        for i, interval in enumerate(final_intervals[:3]):  # Log top 3
            logger.info(f"    #{i+1}: {interval['start']:.1f}~{interval['end']:.1f}s "
                       f"(prob={interval['mean_prob']:.3f}, dur={interval['duration']:.1f}s)")

        return final_intervals
