"""
Feature Extractor Module

Extracts frames, audio, and lip features from raw video files.
Handles preprocessing and downsampling for Stage 1 analysis.

OPTIMIZATION: Also extracts geometry/arcface/phoneme for Stage2 reuse.

Author: Claude
Date: 2025-11-17
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import logging

from ..data.preprocessing import ShortsPreprocessor

# Setup logger
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature Extractor for video preprocessing.
    
    Extracts frames, audio, and lip features from raw video files
    and handles FPS downsampling for efficient processing.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize FeatureExtractor with preprocessing configuration.

        Args:
            config: Preprocessing configuration dict. If None, uses defaults:
                - target_fps: 30
                - max_frames: 900 (30 seconds × 30fps)
                - frame_size: [224, 224]
                - max_duration: 30 seconds
                - min_duration: 1 second
                - sample_rate: 16000
                - n_mfcc: 40
                - hop_length: 512
                - n_fft: 2048
                - lip_size: [112, 112]
        """
        if config is None:
            # Stage1 preprocessing config (limited to 30 seconds)
            config = {
                'target_fps': 30,
                'max_frames': 900,  # 30 seconds × 30fps = 900 frames max
                'frame_size': [224, 224],
                'max_duration': 30,  # Limit to 30 seconds
                'min_duration': 1,   # Allow short clips (for testing)
                'sample_rate': 16000,
                'n_mfcc': 40,
                'hop_length': 512,
                'n_fft': 2048,
                'lip_size': [112, 112]  # Match MMMS-BA training (was 96)
            }

        self.preprocess_config = config

        # Initialize ShortsPreprocessor
        logger.info("Initializing ShortsPreprocessor for feature extraction...")
        self.preprocessor = ShortsPreprocessor(self.preprocess_config)

    def extract_features(
        self,
        video_path: str,
        target_fps: float = 10.0
    ) -> Dict[str, Any]:
        """
        Extract features from raw video file.

        Args:
            video_path: Path to raw video file (NOT NPZ)
            target_fps: Target FPS for downsampling (default: 10.0)

        Returns:
            features: Dict with keys:
                - 'frames': np.ndarray (T, 224, 224, 3)
                - 'audio': np.ndarray (T_audio, 40)
                - 'lip': np.ndarray (T, 112, 112, 3)
                - 'timestamps': np.ndarray (T,)
                - 'fps': float
                - 'total_frames': int
                - 'video_id': str
                - 'video_path': str
        """
        logger.info(f"Extracting features from video: {video_path}")

        # Get video info to optimize preprocessing
        cap_temp = cv2.VideoCapture(str(video_path))
        if not cap_temp.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        original_fps = cap_temp.get(cv2.CAP_PROP_FPS)
        total_frames_original = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames_original / original_fps if original_fps > 0 else 0
        cap_temp.release()

        # Calculate target max_frames based on target_fps
        # This makes preprocessing extract only the frames we need!
        max_duration = self.preprocess_config.get('max_duration', 30)
        effective_duration = min(duration, max_duration)
        target_max_frames = int(effective_duration * target_fps) + 1  # +1 for safety

        logger.info(f"  Video: {duration:.1f}s at {original_fps:.1f}fps ({total_frames_original} frames)")
        logger.info(f"  Target: {target_fps:.1f}fps → extracting ~{target_max_frames} frames (optimized preprocessing)")

        # Update preprocessor config to extract only target_fps frames
        # This dramatically speeds up preprocessing!
        original_max_frames = self.preprocessor.config.get('max_frames')
        self.preprocessor.config['max_frames'] = target_max_frames
        self.preprocessor.video_processor.max_frames = target_max_frames

        # Extract frames/audio/lip using ShortsPreprocessor
        logger.info("  Extracting features from video...")
        result = self.preprocessor.process_video(
            str(video_path),
            extract_audio=True,
            extract_lip=True
        )

        # Restore original max_frames config
        self.preprocessor.config['max_frames'] = original_max_frames
        self.preprocessor.video_processor.max_frames = original_max_frames

        if result['frames'] is None or result['audio'] is None or result['lip'] is None:
            raise RuntimeError(f"Failed to extract features from {video_path}")

        frames = result['frames']  # (T, 224, 224, 3)
        audio = result['audio']    # (T_audio, 40)
        lip = result['lip']        # (T, 112, 112, 3)

        logger.info(f"  Extracted features: {frames.shape[0]} frames")

        # No need to downsample anymore - already extracted at target_fps!
        # (VideoPreprocessor's uniform sampling handles it)

        video_id = Path(video_path).stem

        # Calculate timestamps based on uniform sampling
        total_frames = frames.shape[0]

        # Uniform sampling: frames are evenly distributed across the video
        if total_frames_original <= total_frames:
            # Edge case: extracted frames >= original frames
            frame_indices = np.arange(total_frames, dtype=np.float32)
        else:
            # Normal case: uniform sampling
            step = total_frames_original / total_frames
            frame_indices = np.array([int(i * step) for i in range(total_frames)], dtype=np.float32)

        # Timestamps in seconds
        timestamps = frame_indices / original_fps

        return {
            'frames': frames,
            'audio': audio,
            'lip': lip,
            'timestamps': timestamps,
            'fps': original_fps,  # Original video FPS
            'total_frames': total_frames,
            'video_id': video_id,
            'video_path': video_path
        }

    def _downsample_frames(
        self,
        frames: np.ndarray,
        lip: np.ndarray,
        original_fps: float,
        target_fps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downsample frames and lip features to target FPS.

        Args:
            frames: (T, H, W, 3) numpy array
            lip: (T, H_lip, W_lip, 3) numpy array
            original_fps: Original FPS of the video
            target_fps: Target FPS for downsampling

        Returns:
            Tuple of (downsampled_frames, downsampled_lip)
        """
        downsample_step = max(1, int(original_fps / target_fps))  # e.g., 30/10 = 3
        logger.info(f"  Downsampling from {original_fps:.1f}fps to {target_fps}fps (step={downsample_step})")

        downsampled_frames = frames[::downsample_step]
        downsampled_lip = lip[::downsample_step]

        logger.info(f"  Downsampled to {downsampled_frames.shape[0]} frames")

        return downsampled_frames, downsampled_lip

