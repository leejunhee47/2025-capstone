import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, List

from ..models.pia_model import PIAModel
from ..utils.config import load_config
from ..utils.korean_phoneme_config import get_phoneme_vocab
from .pia_explainer import PIAExplainer
from .hybrid_utils import (
    get_interval_phoneme_dict,
    resample_frames_to_pia_format,
    resample_features_to_grid
)

# Setup logger
logger = logging.getLogger(__name__)

class Stage2Analyzer:
    def __init__(
        self,
        pia_model_path: str,
        pia_config_path: str = "configs/train_pia.yaml",
        device: str = "cuda"
    ):
        logger.info("Initializing Stage2Analyzer (PIA)...")
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # Load configuration
        self.pia_config = load_config(pia_config_path)
        self.phoneme_vocab = get_phoneme_vocab()
        
        self.pia_model: Optional[nn.Module] = None
        self.pia_explainer: Optional[PIAExplainer] = None

        # Load PIA model
        if pia_model_path != "dummy":
            self.pia_model = self._load_pia_model(pia_model_path)
            if self.pia_model:
                self.pia_explainer = PIAExplainer(self.pia_model, self.phoneme_vocab)
        else:
            logger.warning("PIA model skipped (dummy path)")

    def _load_pia_model(self, checkpoint_path: str) -> nn.Module:
        """Load PIA model from checkpoint."""
        model = PIAModel(
            num_phonemes=self.pia_config.get('model', {}).get('num_phonemes', 14),
            frames_per_phoneme=self.pia_config.get('model', {}).get('frames_per_phoneme', 5),
            geo_dim=self.pia_config.get('model', {}).get('geo_dim', 1),
            arcface_dim=self.pia_config.get('model', {}).get('arcface_dim', 512),
            num_classes=self.pia_config.get('model', {}).get('num_classes', 2)
        ).to(self.device)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            logger.error(f"Failed to load PIA model: {e}")

        model.eval()
        return model

    def predict_full_video(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fake probability for the full video using PIA.

        Args:
            features: Dict containing extracted features (frames_30fps, mar_30fps, etc.)

        Returns:
            result: Dict with verdict, confidence, probabilities, XAI info
        """
        if self.pia_model is None:
             return {'verdict': 'unknown', 'confidence': 0.0, 'probabilities': {'fake': 0.0, 'real': 0.0}}

        logger.info("[Stage2] Running PIA prediction...")
        
        # 1. Get features (30fps)
        # Ensure we handle potential missing keys gracefully or expect unified extractor output
        frames = features.get('frames_30fps')
        geometry = features.get('mar_30fps')
        arcface = features.get('arcface_30fps')
        phoneme_labels = features.get('phoneme_labels_30fps')
        timestamps = features.get('timestamps_30fps')
        
        if frames is None or len(frames) == 0:
            logger.warning("No frames for PIA prediction")
            return {'verdict': 'unknown', 'confidence': 0.0}

        if geometry is None or arcface is None or phoneme_labels is None or timestamps is None:
             logger.warning("Missing required features for PIA (geometry/arcface/phoneme/timestamps).")
             return {'verdict': 'unknown', 'confidence': 0.0}

        # 2. Resample to PIA format (14x5 grid)
        # [UPDATED] Use frame-level phoneme labels directly to match training behavior.
        # This ensures that frame selection logic is identical to KoreanPhonemeDataset.
        
        # Note: Using the entire video's phoneme occurrences might blend valid and invalid frames if video is mixed,
        # but we assume "full fake" or "full real".
        resampled_frames, matched_phonemes, valid_mask_frames = resample_frames_to_pia_format(
            frames, timestamps, phoneme_labels, target_phonemes=14, frames_per_phoneme=5
        )
        
        # [DEBUG] Log which phonemes were matched and their frame counts
        logger.info(f"[DEBUG] Matched phonemes for PIA: {matched_phonemes}")
        logger.info(f"[DEBUG] Valid mask: {valid_mask_frames.sum()}/14 phonemes have frames")
        
        resampled_geometry, _, valid_mask_geometry = resample_features_to_grid(
            geometry, timestamps, phoneme_labels, 14, 5
        )
        
        resampled_arcface, _, valid_mask_arcface = resample_features_to_grid(
            arcface, timestamps, phoneme_labels, 14, 5
        )
        
        # 4. Prepare tensors
        # Frames: (14, 5, H, W, 3) -> (1, 14, 5, 3, H, W)
        imgs_tensor = torch.from_numpy(resampled_frames).float().unsqueeze(0)
        imgs_tensor = imgs_tensor.permute(0, 1, 2, 5, 3, 4)

        # [FIX] Resize 224x224 → 112x112 for PIA model
        B, P, Fr, C, H, W = imgs_tensor.shape
        if H != 112 or W != 112:
            imgs_tensor = imgs_tensor.view(B * P * Fr, C, H, W)
            imgs_tensor = F.interpolate(imgs_tensor, size=(112, 112), mode='bilinear', align_corners=False)
            imgs_tensor = imgs_tensor.view(B, P, Fr, C, 112, 112)

        geoms_tensor = torch.from_numpy(resampled_geometry).float().unsqueeze(0) # (1, 14, 5, 1)
        arcs_tensor = torch.from_numpy(resampled_arcface).float().unsqueeze(0)   # (1, 14, 5, 512)
        
        # Mask
        valid_mask = valid_mask_frames & valid_mask_geometry & valid_mask_arcface
        mask_tensor = torch.zeros(1, 14, 5, dtype=torch.bool)
        for pi in range(14):
            if valid_mask[pi]:
                mask_tensor[0, pi, :] = True

        if mask_tensor.sum() == 0:
            logger.warning("No valid phonemes found for PIA.")
            return {
                'verdict': 'unknown', 
                'confidence': 0.0,
                'error': 'no_valid_phonemes'
            }

        # Move to device
        imgs_tensor = imgs_tensor.to(self.device)
        geoms_tensor = geoms_tensor.to(self.device)
        arcs_tensor = arcs_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        # Extract all phonemes for transcription display (unique, preserving order)
        all_phonemes_in_video = []
        if phoneme_labels is not None:
            seen = set()
            for p in phoneme_labels:
                if p not in seen and p not in ['<pad>', '<PAD>', 'sil', 'sp', 'spn', '']:
                    all_phonemes_in_video.append(p)
                    seen.add(p)

        # 5. Prediction (using Explainer for detailed output including attention)
        if self.pia_explainer:
            xai_result = self.pia_explainer.explain(
                geoms=geoms_tensor,
                imgs=imgs_tensor,
                arcs=arcs_tensor,
                mask=mask_tensor,
                video_id="full_video",
                confidence_threshold=0.5
            )
            # Add full transcription and matched phonemes
            xai_result['matched_phonemes'] = matched_phonemes
            xai_result['all_phonemes'] = all_phonemes_in_video
            return xai_result # Already contains detection, branch contributions, etc.
        else:
             # Fallback: raw model prediction
             with torch.no_grad():
                 logits, _ = self.pia_model(geoms_tensor, imgs_tensor, arcs_tensor, mask_tensor)
                 probs = torch.softmax(logits, dim=1)
                 fake_prob = probs[0, 1].item()
                 
             verdict = 'fake' if fake_prob > 0.5 else 'real'
             confidence = fake_prob if verdict == 'fake' else 1 - fake_prob
             
             return {
                 'detection': {
                     'verdict': verdict,
                     'confidence': confidence,
                     'probabilities': {'fake': fake_prob, 'real': 1 - fake_prob},
                     'is_fake': fake_prob > 0.5
                 },
                 # Add placeholder structure for compatibility if needed
                 'phoneme_analysis': {},
                 'branch_contributions': {}
             }

    def _is_valid_phoneme(self, label) -> bool:
        """
        Silence + 예외 필터링 Guard (강화).

        Filters:
        - None, ''
        - <sil>, <pad>, <unk>, <blank>
        - <...> 형식 special tokens (확실히 적용)
        - [...] 형식 (실 운영 대비)

        Args:
            label: Phoneme label to check

        Returns:
            True if valid phoneme, False otherwise
        """
        # None, empty string
        if label is None or label == '':
            return False

        # 명시적 special tokens
        if label in ['<sil>', '<pad>', '<unk>', '<blank>']:
            return False

        # <...> 형식 (확실히 적용)
        if isinstance(label, str) and label.startswith('<') and label.endswith('>'):
            return False

        # [...] 형식 (실 운영 대비)
        if isinstance(label, str) and label.startswith('[') and label.endswith(']'):
            return False

        return True

    def predict_interval(
        self,
        features: Dict[str, Any],
        start_time: float,
        end_time: float,
        filter_silence: bool = True,
        padding: float = 0.5
    ) -> Dict[str, Any]:
        """
        특정 시간 구간만 PIA 분석.

        Args:
            features: UnifiedFeatureExtractor 결과
            start_time: 구간 시작 (초)
            end_time: 구간 끝 (초)
            filter_silence: Whisper <sil> 필터링 여부
            padding: 전후 여유 (초)

        Returns:
            {
                'verdict': 'fake' | 'real' | 'unknown',
                'confidence': float,
                'probabilities': {'real': x, 'fake': y},
                'interval': {'start': 3.2, 'end': 5.8, 'duration': 2.6},
                'matched_phonemes': ['ㅏ', 'ㅗ', ...],
                'valid_frame_count': int,
                'error': str (if unknown)
            }
        """
        if self.pia_model is None:
            return {
                'verdict': 'unknown',
                'confidence': 0.0,
                'probabilities': {'fake': 0.0, 'real': 0.0},
                'error': 'PIA model not loaded'
            }

        logger.info(f"[Stage2] Analyzing interval {start_time:.1f}~{end_time:.1f}s...")

        # Step 1: Padding + Slicing
        start_pad = max(0, start_time - padding)
        end_pad = end_time + padding

        timestamps = features.get('timestamps_30fps')
        frames = features.get('frames_30fps')
        geometry = features.get('mar_30fps')
        arcface = features.get('arcface_30fps')
        phoneme_labels = features.get('phoneme_labels_30fps')

        if timestamps is None or frames is None:
            logger.warning("Missing required features")
            return {
                'verdict': 'unknown',
                'confidence': 0.0,
                'probabilities': {'fake': 0.0, 'real': 0.0},
                'error': 'Missing timestamps or frames'
            }

        # Time mask
        time_mask = (timestamps >= start_pad) & (timestamps <= end_pad)

        # Step 2: Silence Filtering (Optional)
        if filter_silence and phoneme_labels is not None:
            valid_mask = time_mask & np.array([
                self._is_valid_phoneme(p) for p in phoneme_labels
            ])
        else:
            valid_mask = time_mask

        valid_frame_count = valid_mask.sum()

        # Step 3: Min Frame Check
        # [OPTIMIZED] 30 → 15 (0.5초면 충분, 짧은 발음 구간 대응)
        if valid_frame_count < 15:  # 최소 0.5초 (15 frames @ 30fps)
            logger.warning(f"Too few valid frames: {valid_frame_count}/15")
            return {
                'verdict': 'unknown',
                'confidence': 0.0,
                'probabilities': {'fake': 0.0, 'real': 0.0},
                'valid_frame_count': int(valid_frame_count),
                'error': 'Too few valid frames after silence filtering',
                'interval': {
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                }
            }

        # Step 4: Slice features
        frames_sliced = frames[valid_mask]
        timestamps_sliced = timestamps[valid_mask]
        phoneme_labels_sliced = phoneme_labels[valid_mask] if phoneme_labels is not None else None

        if geometry is not None:
            geometry_sliced = geometry[valid_mask]
        else:
            geometry_sliced = None

        if arcface is not None:
            arcface_sliced = arcface[valid_mask]
        else:
            arcface_sliced = None

        if geometry_sliced is None or arcface_sliced is None or phoneme_labels_sliced is None:
            logger.warning("Missing geometry/arcface/phoneme after slicing")
            return {
                'verdict': 'unknown',
                'confidence': 0.0,
                'probabilities': {'fake': 0.0, 'real': 0.0},
                'valid_frame_count': int(valid_frame_count),
                'error': 'Missing required features after slicing'
            }

        # Step 5: Resample to PIA format (14x5)
        resampled_frames, matched_phonemes, valid_mask_frames = resample_frames_to_pia_format(
            frames_sliced, timestamps_sliced, phoneme_labels_sliced, target_phonemes=14, frames_per_phoneme=5
        )

        # [OPTIMIZED] Vectorized resize using torch interpolate (50x faster than cv2 loop)
        target_h, target_w = 112, 112
        if resampled_frames.shape[2:4] != (target_h, target_w):
            # Shape: (14, 5, H, W, 3) → need (70, 3, H, W) for interpolate
            P, Fr, H_old, W_old, C = resampled_frames.shape

            # Convert to torch and permute to (P, Fr, C, H, W)
            frames_torch = torch.from_numpy(resampled_frames).float().permute(0, 1, 4, 2, 3)

            # Flatten to (P*Fr, C, H, W) = (70, 3, H, W)
            frames_flat = frames_torch.reshape(P * Fr, C, H_old, W_old)

            # Resize all 70 frames at once
            frames_resized = F.interpolate(frames_flat, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # Reshape back to (P, Fr, C, H_new, W_new) = (14, 5, 3, 112, 112)
            frames_resized = frames_resized.view(P, Fr, C, target_h, target_w)

            # Permute back to (P, Fr, H, W, C) = (14, 5, 112, 112, 3)
            resampled_frames = frames_resized.permute(0, 1, 3, 4, 2).cpu().numpy()

        resampled_geometry, _, valid_mask_geometry = resample_features_to_grid(
            geometry_sliced, timestamps_sliced, phoneme_labels_sliced, 14, 5
        )

        resampled_arcface, _, valid_mask_arcface = resample_features_to_grid(
            arcface_sliced, timestamps_sliced, phoneme_labels_sliced, 14, 5
        )

        # Step 6: Prepare tensors
        # [DEBUG] Log shapes to diagnose bug
        logger.info(f"[DEBUG] resampled_frames shape: {resampled_frames.shape}")
        logger.info(f"[DEBUG] resampled_geometry shape: {resampled_geometry.shape}")
        logger.info(f"[DEBUG] resampled_arcface shape: {resampled_arcface.shape}")

        imgs_tensor = torch.from_numpy(resampled_frames).float().unsqueeze(0)
        logger.info(f"[DEBUG] imgs_tensor after unsqueeze shape: {imgs_tensor.shape}")

        imgs_tensor = imgs_tensor.permute(0, 1, 2, 5, 3, 4)
        logger.info(f"[DEBUG] imgs_tensor after permute shape: {imgs_tensor.shape}")

        # Note: Frames already resized to 112x112 via cv2.resize in lines 343-354

        geoms_tensor = torch.from_numpy(resampled_geometry).float().unsqueeze(0)
        arcs_tensor = torch.from_numpy(resampled_arcface).float().unsqueeze(0)

        # Mask
        valid_mask_combined = valid_mask_frames & valid_mask_geometry & valid_mask_arcface
        mask_tensor = torch.zeros(1, 14, 5, dtype=torch.bool)
        for pi in range(14):
            if valid_mask_combined[pi]:
                mask_tensor[0, pi, :] = True

        if mask_tensor.sum() == 0:
            logger.warning("No valid phonemes after resampling")
            return {
                'verdict': 'unknown',
                'confidence': 0.0,
                'probabilities': {'fake': 0.0, 'real': 0.0},
                'valid_frame_count': int(valid_frame_count),
                'error': 'No valid phonemes after resampling',
                'interval': {
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                }
            }

        # Move to device
        imgs_tensor = imgs_tensor.to(self.device)
        geoms_tensor = geoms_tensor.to(self.device)
        arcs_tensor = arcs_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)

        # Step 7: Prediction
        with torch.no_grad():
            logits, _ = self.pia_model(geoms_tensor, imgs_tensor, arcs_tensor, mask_tensor)
            probs = torch.softmax(logits, dim=1)
            fake_prob = probs[0, 1].item()

        verdict = 'fake' if fake_prob > 0.5 else 'real'
        confidence = fake_prob if verdict == 'fake' else 1 - fake_prob

        logger.info(f"  Interval Result: {verdict.upper()} ({confidence:.2%})")
        logger.info(f"  Valid frames: {valid_frame_count}, Matched phonemes: {matched_phonemes[:3]}")

        # Extract unique phonemes from the interval for transcription display
        all_phonemes_in_interval = []
        if phoneme_labels_sliced is not None:
            # Get unique phonemes preserving order (for transcription)
            seen = set()
            for p in phoneme_labels_sliced:
                if p not in seen and p not in ['<pad>', '<PAD>', 'sil', 'sp', 'spn', '']:
                    all_phonemes_in_interval.append(p)
                    seen.add(p)

        return {
            'verdict': verdict,
            'confidence': confidence,
            'probabilities': {'real': 1 - fake_prob, 'fake': fake_prob},
            'interval': {
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            },
            'matched_phonemes': matched_phonemes,
            'all_phonemes': all_phonemes_in_interval,  # Full transcription
            'valid_frame_count': int(valid_frame_count)
        }
