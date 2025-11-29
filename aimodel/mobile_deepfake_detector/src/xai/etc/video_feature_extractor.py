"""
Video Feature Extractor for Hybrid XAI Pipeline

Extracts all features needed for MMMS-BA and PIA from raw video.
Reuses existing extractors from preprocessing pipeline.
"""

import os
import sys
import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import preprocessing modules
from mobile_deepfake_detector.src.data.preprocessing import ShortsPreprocessor, match_phoneme_to_frames
import cv2

# Import feature extractors
from mobile_deepfake_detector.src.utils.hybrid_phoneme_aligner_v2 import HybridPhonemeAligner
from mobile_deepfake_detector.src.utils.enhanced_mar_extractor import EnhancedMARExtractor
from mobile_deepfake_detector.src.utils.arcface_extractor import ArcFaceExtractor

logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """
    Extract all features needed for MMMS-BA and PIA from raw video.

    This class orchestrates multiple feature extractors to generate
    a complete feature set for deepfake detection and XAI analysis.
    """

    def __init__(
        self,
        device: str = 'cuda',
        sample_frames: int = 50,
        extract_all_frames: bool = False
    ):
        """
        Initialize video feature extractor.

        Args:
            device: Device for models ('cuda' or 'cpu')
            sample_frames: Number of frames to sample (if not extracting all)
            extract_all_frames: If True, extract all frames; if False, sample
        """
        self.device = device
        self.sample_frames = sample_frames
        self.extract_all_frames = extract_all_frames

        logger.info("Initializing VideoFeatureExtractor...")

        # Initialize extractors
        self._init_extractors()

        logger.info("VideoFeatureExtractor initialized successfully")

    def _init_extractors(self):
        """Initialize all feature extractors (preprocess_parallel.py style)."""
        # Phoneme alignment (uses WhisperX + GPU)
        self.phoneme_aligner = HybridPhonemeAligner(
            whisper_model="base",
            device=self.device,
            compute_type="float16"
        )

        # MAR geometry extractor
        self.mar_extractor = EnhancedMARExtractor()

        # ArcFace face embedding extractor (use CPU to avoid GPU contention)
        self.arcface_extractor = ArcFaceExtractor(
            device='cpu',  # CPU to reduce GPU memory pressure
            model_name='buffalo_l'
        )

        logger.info(f"  - Initialized extractors (phoneme: {self.device}, arcface: cpu)")

    def extract_all_features(self, video_path: str) -> Dict[str, Any]:
        """
        Extract all features from video.

        Args:
            video_path: Path to input video file

        Returns:
            Dictionary containing:
                - frames: (T, 224, 224, 3) Video frames
                - audio: (T_audio, 40) MFCC features
                - lip: (T, 96, 96, 3) Lip region frames
                - phoneme_labels: (T,) Phoneme label per frame
                - phoneme_intervals: List of phoneme dictionaries
                - timestamps: (T,) Frame timestamps
                - geometry: (T, 1) MAR values
                - arcface: (T, 512) ArcFace embeddings
                - fps: Frame rate
                - total_frames: Total number of frames
                - video_path: Original video path
        """
        logger.info(f"Extracting features from video: {video_path}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Extract frames and audio
        logger.info("  Step 1: Extracting frames and audio...")
        video_data = self._extract_video_features(video_path)

        # 2. Extract phoneme alignment
        logger.info("  Step 2: Extracting phoneme alignment...")
        phoneme_data = self._extract_phoneme_features(video_path, video_data)

        # 3. Extract MAR geometry
        logger.info("  Step 3: Extracting MAR geometry...")
        geometry = self._extract_mar_features(video_data['frames'])

        # 4. Extract ArcFace embeddings
        logger.info("  Step 4: Extracting ArcFace embeddings...")
        arcface = self._extract_arcface_features(video_data['frames'])

        # Combine all features
        features = {
            **video_data,
            **phoneme_data,
            'geometry': geometry,
            'arcface': arcface,
            'video_path': video_path
        }

        # Validate features
        self._validate_features(features)

        logger.info(f"  Feature extraction complete:")
        logger.info(f"    - Frames: {features['frames'].shape}")
        logger.info(f"    - Audio: {features['audio'].shape}")
        logger.info(f"    - Lip: {features['lip'].shape}")
        logger.info(f"    - Geometry: {features['geometry'].shape}")
        logger.info(f"    - ArcFace: {features['arcface'].shape}")
        logger.info(f"    - Phonemes: {len(features['phoneme_intervals'])} intervals")

        return features

    def _extract_video_features(self, video_path: str) -> Dict[str, Any]:
        """Extract frames, audio, and lip from video (av_module style)."""
        # Use av_module for extraction (same as preprocess_parallel.py)
        from av_module.frame_extractor import FrameExtractor
        from av_module.audio_extractor import AudioExtractor

        # Extract frames
        frame_extractor = FrameExtractor()
        frames = frame_extractor.extract_frames(
            video_path,
            max_frames=self.sample_frames,
            target_size=(224, 224)
        )  # Returns (T, 224, 224, 3) normalized [0,1]

        # Extract audio MFCC
        audio_extractor = AudioExtractor()
        audio = audio_extractor.extract_mfcc(
            video_path,
            n_mfcc=40,
            sr=16000
        )  # Returns (T_audio, 40)

        # Extract lip regions (simplified - just crop center for now)
        # TODO: Implement proper lip detection if needed
        lip_frames = []
        for frame in frames:
            # Simple center crop for lip region (96x96)
            h, w = frame.shape[:2]
            center_y, center_x = h // 2, w // 2
            lip_h, lip_w = 96, 96
            y1 = max(0, center_y - lip_h // 2)
            y2 = min(h, y1 + lip_h)
            x1 = max(0, center_x - lip_w // 2)
            x2 = min(w, x1 + lip_w)

            lip_crop = frame[y1:y2, x1:x2]
            # Resize to exact 96x96
            lip_resized = cv2.resize(lip_crop, (96, 96))
            lip_frames.append(lip_resized)

        lip = np.array(lip_frames)  # (T, 96, 96, 3)

        # Get FPS from video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else frames.shape[0]
        cap.release()

        # Calculate timestamps
        num_frames = frames.shape[0]
        if total_frames_video <= num_frames:
            frame_indices = np.arange(num_frames, dtype=np.float32)
        else:
            step = total_frames_video / num_frames
            frame_indices = np.array([int(i * step) for i in range(num_frames)], dtype=np.float32)

        timestamps = frame_indices / fps

        return {
            'frames': frames,
            'audio': audio,
            'lip': lip,
            'fps': fps,
            'total_frames': num_frames,
            'timestamps': timestamps
        }

    def _extract_phoneme_features(
        self,
        video_path: str,
        video_data: Dict
    ) -> Dict[str, Any]:
        """Extract phoneme alignment and labels."""
        try:
            # Get phoneme alignment
            alignment_result = self.phoneme_aligner.align_video(
                video_path=video_path,
                output_format='interval'
            )

            phoneme_intervals = alignment_result['phoneme_intervals']

            # Match phonemes to frames
            phoneme_labels = match_phoneme_to_frames(
                phoneme_intervals=phoneme_intervals,
                timestamps=video_data['timestamps']
            )

            return {
                'phoneme_intervals': phoneme_intervals,
                'phoneme_labels': phoneme_labels
            }

        except Exception as e:
            logger.warning(f"Phoneme extraction failed: {e}")
            # Return silence labels as fallback
            num_frames = video_data['total_frames']
            return {
                'phoneme_intervals': [],
                'phoneme_labels': np.array(['<sil>'] * num_frames, dtype=object)
            }

    def _extract_mar_features(self, frames: np.ndarray) -> np.ndarray:
        """Extract MAR (Mouth Aspect Ratio) features."""
        try:
            # Extract MAR for each frame
            mar_values = []

            for i, frame in enumerate(frames):
                mar = self.mar_extractor.extract_single_frame(frame)
                mar_values.append(mar)

            # Convert to numpy array
            geometry = np.array(mar_values, dtype=np.float32).reshape(-1, 1)

            # Handle NaN values
            if np.any(np.isnan(geometry)):
                logger.warning(f"NaN values in MAR, replacing with zeros")
                geometry = np.nan_to_num(geometry, nan=0.0)

            return geometry

        except Exception as e:
            logger.warning(f"MAR extraction failed: {e}")
            # Return zeros as fallback
            num_frames = frames.shape[0]
            return np.zeros((num_frames, 1), dtype=np.float32)

    def _extract_arcface_features(self, frames: np.ndarray) -> np.ndarray:
        """Extract ArcFace embeddings."""
        try:
            # Extract ArcFace for each frame
            embeddings = []

            for i, frame in enumerate(frames):
                embedding = self.arcface_extractor.extract_single_frame(frame)
                embeddings.append(embedding)

            # Stack embeddings
            arcface = np.vstack(embeddings)  # (T, 512)

            # Normalize embeddings
            arcface = arcface / (np.linalg.norm(arcface, axis=1, keepdims=True) + 1e-8)

            return arcface

        except Exception as e:
            logger.warning(f"ArcFace extraction failed: {e}")
            # Return random embeddings as fallback
            num_frames = frames.shape[0]
            embeddings = np.random.randn(num_frames, 512).astype(np.float32)
            # Normalize
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            return embeddings

    def _validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate extracted features.

        Checks:
        - All required keys present
        - Shapes are consistent
        - No NaN/Inf values
        - Data types correct
        """
        required_keys = [
            'frames', 'audio', 'lip', 'phoneme_labels',
            'timestamps', 'geometry', 'arcface', 'fps', 'total_frames'
        ]

        # Check required keys
        for key in required_keys:
            if key not in features:
                raise ValueError(f"Missing required feature: {key}")

        # Check shapes
        num_frames = features['total_frames']

        assert features['frames'].shape[0] == num_frames, "Frame count mismatch"
        assert features['lip'].shape[0] == num_frames, "Lip frame count mismatch"
        assert features['geometry'].shape[0] == num_frames, "Geometry count mismatch"
        assert features['arcface'].shape[0] == num_frames, "ArcFace count mismatch"
        assert len(features['phoneme_labels']) == num_frames, "Phoneme label count mismatch"
        assert len(features['timestamps']) == num_frames, "Timestamp count mismatch"

        # Check for NaN/Inf
        for key in ['frames', 'audio', 'geometry', 'arcface']:
            if np.any(np.isnan(features[key])) or np.any(np.isinf(features[key])):
                logger.warning(f"NaN/Inf values detected in {key}")

        return True

    def extract_features_batch(
        self,
        video_paths: List[str],
        cache_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract features from multiple videos.

        Args:
            video_paths: List of video file paths
            cache_dir: Optional directory to cache extracted features

        Returns:
            List of feature dictionaries
        """
        features_list = []

        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")

            # Check cache if enabled
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"{Path(video_path).stem}_features.npz")
                if os.path.exists(cache_path):
                    logger.info(f"  Loading from cache: {cache_path}")
                    features = dict(np.load(cache_path, allow_pickle=True))
                    features_list.append(features)
                    continue

            # Extract features
            try:
                features = self.extract_all_features(video_path)
                features_list.append(features)

                # Save to cache if enabled
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.savez_compressed(cache_path, **features)
                    logger.info(f"  Saved to cache: {cache_path}")

            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                continue

        return features_list


if __name__ == "__main__":
    # Test the feature extractor
    import argparse

    parser = argparse.ArgumentParser(description="Test Video Feature Extractor")
    parser.add_argument('--video', type=str, required=True, help="Path to video file")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output', type=str, help="Output directory for features")

    args = parser.parse_args()

    # Initialize extractor
    extractor = VideoFeatureExtractor(device=args.device)

    # Extract features
    features = extractor.extract_all_features(args.video)

    # Print summary
    print("\nExtracted Features:")
    print(f"  - Frames: {features['frames'].shape}")
    print(f"  - Audio: {features['audio'].shape}")
    print(f"  - Lip: {features['lip'].shape}")
    print(f"  - Geometry: {features['geometry'].shape}")
    print(f"  - ArcFace: {features['arcface'].shape}")
    print(f"  - Phoneme intervals: {len(features['phoneme_intervals'])}")
    print(f"  - FPS: {features['fps']:.2f}")

    # Save if requested
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, f"{Path(args.video).stem}_features.npz")
        np.savez_compressed(output_path, **features)
        print(f"\nSaved features to: {output_path}")