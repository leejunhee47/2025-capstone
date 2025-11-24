"""
Stage 2 Analyzer Module

PIA XAI analysis on suspicious intervals.
Extracts phoneme alignment, MAR, ArcFace features and runs PIA model.

Author: Claude
Date: 2025-11-17
"""

import json
import numpy as np
import torch
import torch.nn as nn
import cv2
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..models.pia_model import PIAModel
from ..utils.config import load_config
from ..utils.korean_phoneme_config import get_phoneme_vocab
from .pia_explainer import PIAExplainer
from .pia_visualizer import PIAVisualizer
from ..data.preprocessing import match_phoneme_to_frames
from ..utils.hybrid_phoneme_aligner_v2 import HybridPhonemeAligner
from ..utils.enhanced_mar_extractor import EnhancedMARExtractor
from ..utils.arcface_extractor import ArcFaceExtractor
from ..utils.feature_extraction_utils import extract_features_optimized
from .hybrid_utils import (
    get_interval_phoneme_dict,
    resample_frames_to_pia_format,
    resample_features_to_grid,
    format_stage2_to_interface
)

# Setup logger
logger = logging.getLogger(__name__)


class Stage2Analyzer:
    """
    Stage 2 Analyzer: PIA XAI Analysis

    Performs deep XAI analysis on suspicious intervals using PIA model.
    Extracts phoneme alignment, MAR geometry, and ArcFace identity features.

    Reuses:
    - PIAExplainer for branch contribution and phoneme attention
    - PIAVisualizer for visualization generation
    - hybrid_utils.py for resampling and formatting
    """

    def __init__(
        self,
        pia_model_path: str,
        pia_config_path: str = "configs/train_pia.yaml",
        device: str = "cuda"
    ):
        """
        Initialize Stage2Analyzer with PIA model.

        Args:
            pia_model_path: Path to PIA model checkpoint
            pia_config_path: PIA config file
            device: cuda or cpu
        """
        logger.info("Initializing Stage2Analyzer...")

        # CUDA device setup
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            if device == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
            else:
                logger.info("Using CPU device")

        # Load configuration
        self.pia_config = load_config(pia_config_path)

        # Load Korean phoneme vocabulary (14 kept phonemes for PIA)
        # Use get_phoneme_vocab() instead of full 42-phoneme vocab
        self.phoneme_vocab = get_phoneme_vocab()  # Returns 14 phonemes: ['A', 'B', 'BB', 'CHh', 'E', 'EU', 'I', 'M', 'O', 'Ph', 'U', 'iA', 'iO', 'iU']
        logger.info(f"Loaded {len(self.phoneme_vocab)} Korean phonemes for PIA: {self.phoneme_vocab}")

        # Load PIA model
        if pia_model_path != "dummy":
            logger.info("Loading PIA model...")
            self.pia_model = self._load_pia_model(pia_model_path)
            # Initialize PIA explainer and visualizer
            self.pia_explainer = PIAExplainer(self.pia_model, self.phoneme_vocab)
            self.pia_visualizer = PIAVisualizer()
        else:
            self.pia_model = None
            self.pia_explainer = None
            self.pia_visualizer = None
            logger.info("PIA model skipped (dummy path)")

        # Stage 2 extractors (lazy initialization - only when needed)
        self.phoneme_aligner = None
        self.mar_extractor = None
        self.arcface_extractor = None

        logger.info("Stage2Analyzer initialized successfully!")

    def _load_pia_model(self, checkpoint_path: str) -> nn.Module:
        """Load PIA model from checkpoint."""
        # Use config values or hardcoded defaults for PIA
        model = PIAModel(
            num_phonemes=self.pia_config.get('model', {}).get('num_phonemes', 14),  # 14 Korean phonemes
            frames_per_phoneme=self.pia_config.get('model', {}).get('frames_per_phoneme', 5),  # 5 frames per phoneme
            geo_dim=self.pia_config.get('model', {}).get('geo_dim', 1),  # MAR feature
            arcface_dim=self.pia_config.get('model', {}).get('arcface_dim', 512),  # ArcFace embedding dim
            num_classes=self.pia_config.get('model', {}).get('num_classes', 2)  # Binary classification
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def _extract_interval_frames_30fps(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        target_fps: float = 30.0
    ) -> tuple:
        """
        Extract frames from video interval at 30fps for better phoneme matching.

        Args:
            video_path: Path to video file
            start_time: Interval start time in seconds
            end_time: Interval end time in seconds
            target_fps: Target FPS for extraction (default: 30.0)

        Returns:
            frames: (T, 224, 224, 3) RGB float32 [0,1]
            timestamps: (T,) relative timestamps in seconds
        """
        cap = cv2.VideoCapture(str(video_path))

        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate frame indices
        start_frame_idx = int(start_time * original_fps)
        end_frame_idx = int(end_time * original_fps)

        frames = []
        timestamps = []

        # Frame sampling interval
        frame_interval = original_fps / target_fps
        current_frame_idx = start_frame_idx

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        while current_frame_idx <= end_frame_idx:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and normalize
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

            # Calculate relative timestamp (offset from start_time)
            timestamp = (current_frame_idx - start_frame_idx) / original_fps
            timestamps.append(timestamp)

            # Move to next frame
            current_frame_idx += frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame_idx))

        cap.release()

        logger.info(f"  [30FPS] Extracted {len(frames)} frames from {start_time:.2f}s-{end_time:.2f}s at {target_fps}fps")

        return np.array(frames), np.array(timestamps)

    def analyze_interval(
        self,
        interval: Dict,
        video_path: str,
        extracted_features: Dict,
        output_dir: Optional[str] = None,
        whisper_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run PIA XAI analysis on a single suspicious interval.

        30fps Optimization: If whisper_result provided, skip WhisperX transcription
        and use shared transcription from STEP 1.5 instead.

        Args:
            interval: Interval dict from Stage1Scanner with keys:
                - interval_id: int
                - start_frame: int
                - end_frame: int
                - start_time_sec: float
                - end_time_sec: float
                - duration_sec: float
                - frame_count: int
            video_path: Path to video file
            extracted_features: Dict with features from Stage1:
                - frames: np.ndarray
                - audio: np.ndarray
                - lip: np.ndarray
                - fps: float
                - timestamps: np.ndarray
                - total_frames: int
                - video_path: str
            output_dir: Optional directory to save visualizations
            whisper_result: Optional shared WhisperX transcription from STEP 1.5
                          If provided, saves ~20 seconds by skipping transcription

        Returns:
            interval_xai: Dictionary matching hybrid_xai_interface.ts format
        """
        logger.info(f"[Stage2Analyzer] Analyzing interval {interval['interval_id']}: "
                   f"{interval['start_time_sec']:.1f}s - {interval['end_time_sec']:.1f}s")

        # Use features from Stage1
        logger.info(f"  Using pre-extracted features from Stage1...")
        stage1_result = {
            'frames': extracted_features['frames'],
            'audio': extracted_features['audio'],
            'lip': extracted_features['lip'],
            'timestamps': extracted_features['timestamps'],
            'fps': extracted_features['fps'],
            'total_frames': extracted_features['total_frames'],
            'video_path': extracted_features['video_path']
        }

        # Pass geometry/arcface/phoneme_labels if available (for Stage2 reuse)
        if 'geometry' in extracted_features:
            stage1_result['geometry'] = extracted_features['geometry']
        if 'arcface' in extracted_features:
            stage1_result['arcface'] = extracted_features['arcface']
        if 'phoneme_labels' in extracted_features:
            stage1_result['phoneme_labels'] = extracted_features['phoneme_labels']

        # Convert Stage1 interval format to Stage2 expected format
        stage2_interval = {
            'interval_id': interval['interval_id'],
            'start_frame': interval['start_frame'],
            'end_frame': interval['end_frame'],
            'start_time': interval['start_time_sec'],
            'end_time': interval['end_time_sec'],
            'duration': interval['duration_sec'],
            'frame_count': interval['frame_count']
        }

        # Run Stage2 XAI analysis
        interval_xai_result = self.run_stage2_interval_xai(
            stage1_result=stage1_result,
            interval=stage2_interval,
            output_dir=output_dir,
            whisper_result=whisper_result  # Pass shared transcription for Phase 1 optimization
        )

        # Convert internal format to hybrid_xai_interface.ts format
        formatted_result = self._format_stage2_result(
            interval_xai=interval_xai_result,
            interval=interval,
            output_dir=output_dir
        )

        # Store raw PIA XAI result for visualization (needed for Step 4.5)
        formatted_result['_raw_pia_xai'] = interval_xai_result['pia_xai']

        return formatted_result

    def run_stage2_interval_xai(
        self,
        stage1_result: Dict,
        interval: Dict,
        output_dir: Optional[str] = None,
        whisper_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Stage 2: PIA XAI analysis on a single suspicious interval.

        Args:
            stage1_result: Result from Stage 1 with frames/timestamps/video_path
            interval: Interval dict with start_frame, end_frame, etc.
            output_dir: Optional directory to save visualization
            whisper_result: Optional shared WhisperX transcription from STEP 1.5
                          If provided, saves ~20 seconds by skipping transcription (Phase 1 optimization)

        Returns:
            interval_xai_result: {
                'interval_id': int,
                'interval_info': dict,
                'pia_xai': dict (full PIAExplainer output)
            }
        """
        interval_id = interval['interval_id']
        logger.info(f"[Stage 2] Running PIA XAI on interval {interval_id}: "
                   f"{interval['start_time']:.1f}s - {interval['end_time']:.1f}s")

        # Initialize Stage2 extractors if not already done
        if self.phoneme_aligner is None:
            logger.info("  Initializing HybridPhonemeAligner...")
            self.phoneme_aligner = HybridPhonemeAligner(
                whisper_model="large-v2",  # Changed from base to large-v2 for better transcription
                device=str(self.device),
                compute_type="float16"
            )
        if self.mar_extractor is None:
            logger.info("  Initializing EnhancedMARExtractor...")
            self.mar_extractor = EnhancedMARExtractor()
        if self.arcface_extractor is None:
            logger.info("  Initializing ArcFaceExtractor...")
            self.arcface_extractor = ArcFaceExtractor(
                device='cuda',  # GPU for optimal performance (matches preprocessing)
                model_name='buffalo_l',
                det_size=(320, 320)  # Smaller detection size for speed
            )

        # Get video path from stage1_result
        video_path = stage1_result['video_path']

        # Extract interval timestamps
        start_frame = interval['start_frame']
        end_frame = interval['end_frame']
        start_time = interval['start_time']
        end_time = interval['end_time']

        # Extract interval frames from stage1_result
        interval_frames = stage1_result['frames'][start_frame:end_frame+1]  # (T, 224, 224, 3) RGB [0,1]
        interval_timestamps = stage1_result['timestamps'][start_frame:end_frame+1]

        # CRITICAL FIX: Convert timestamps to relative coordinates IMMEDIATELY
        # This ensures all downstream extractors (phoneme, MAR, ArcFace) use correct coordinates
        interval_timestamps = interval_timestamps - start_time
        logger.info(f"  Converted interval timestamps to relative coordinates (offset={start_time:.2f}s)")
        logger.info(f"  Interval timestamps range: [{interval_timestamps[0]:.3f}, {interval_timestamps[-1]:.3f}]s ({len(interval_timestamps)} frames)")

        # ===== Feature Extraction (Optimized with Stage1 Reuse) =====
        fps_val = stage1_result['fps']  # Original Stage1 FPS (will be overridden for SLOW PATH)

        # Check if Stage1 has precomputed features (from preprocessed npz)
        has_precomputed = (
            'geometry' in stage1_result and
            'arcface' in stage1_result and
            'phoneme_labels' in stage1_result
        )

        if has_precomputed:
            # FAST PATH: Reuse precomputed features from Stage1
            logger.info("  [OPTIMIZATION] Reusing precomputed features from Stage1 (skip MAR/ArcFace/Phoneme extraction)...")

            # Slice interval features directly (no extraction needed)
            geometry = stage1_result['geometry'][start_frame:end_frame+1]  # (T, 1)
            arcface = stage1_result['arcface'][start_frame:end_frame+1]    # (T, 512)
            phoneme_labels = stage1_result['phoneme_labels'][start_frame:end_frame+1]  # (T,)

            # Only extract audio MFCC (not in Stage1 extracted_features)
            from ..data.preprocessing import AudioPreprocessor
            audio_processor = AudioPreprocessor(
                sample_rate=16000,
                n_mfcc=40,
                hop_length=512,
                n_fft=2048
            )

            # Extract audio for this interval only (trim by time)
            try:
                audio_mfcc = audio_processor.extract_audio(
                    str(video_path),
                    feature_type='mfcc',
                    max_duration=60,
                    start_time=start_time,
                    end_time=end_time
                )
                if audio_mfcc is None or audio_mfcc.shape[0] < 10:
                    logger.warning("Audio extraction failed, using zeros")
                    audio_mfcc = np.zeros((100, 40), dtype=np.float32)
            except Exception as e:
                logger.warning(f"Audio extraction failed: {e}, using zeros")
                audio_mfcc = np.zeros((100, 40), dtype=np.float32)

            logger.info(f"  ✓ Features reused - MAR: {geometry.shape}, ArcFace: {arcface.shape}, Phoneme: {phoneme_labels.shape}")
            logger.info(f"  ✓ Audio extracted - MFCC: {audio_mfcc.shape}")

        else:
            # SLOW PATH: Extract all features from scratch (raw video mode)
            logger.info("  [30FPS OPTIMIZATION] Re-extracting interval at 30fps for better phoneme matching...")

            # 🆕 PHASE 2: Extract interval at 30fps for improved phoneme-frame matching
            interval_frames, interval_timestamps = self._extract_interval_frames_30fps(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                target_fps=30.0
            )

            # Update fps_val to 30fps
            fps_val = 30.0

            logger.info(f"  ✓ 30fps extraction - Frames: {interval_frames.shape}, Timestamps: {interval_timestamps.shape}")

            # Prepare frames: Resize to 384x384 and convert to BGR uint8 for extractors
            frames_bgr_384 = []
            for frame in interval_frames:
                # Convert RGB [0,1] float32 to uint8 [0,255]
                frame_uint8 = (frame * 255).astype(np.uint8)
                # Convert RGB to BGR
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                # Resize to 384x384 for optimal MAR/ArcFace performance
                frame_384 = cv2.resize(frame_bgr, (384, 384))
                frames_bgr_384.append(frame_384)

            # Prepare extractors dict
            extractors = {
                'mar': self.mar_extractor,
                'arcface': self.arcface_extractor,
                'phoneme': self.phoneme_aligner
            }

            # Prepare config
            config = {
                'sample_rate': 16000,
                'n_mfcc': 40,
                'hop_length': 512,
                'n_fft': 2048
            }

            try:
                # 30fps OPTIMIZATION: Apply 0.8s padding for better phoneme coverage
                PADDING_SEC = 0.5
                padded_start_time = max(0, start_time - PADDING_SEC)
                padded_end_time = end_time + PADDING_SEC
                logger.info(f"  [PADDING] Interval: {start_time:.1f}s - {end_time:.1f}s")
                logger.info(f"  [PADDING] With padding: {padded_start_time:.1f}s - {padded_end_time:.1f}s (+/-{PADDING_SEC}s)")

                # 30fps OPTIMIZATION: Check if shared WhisperX transcription provided
                if whisper_result is not None:
                    logger.info("  [30FPS OPTIMIZATION] Using shared WhisperX transcription")

                    # Step A: Extract MAR/Audio/ArcFace WITHOUT phoneme (시니어 개발자 제안)
                    geometry, audio_mfcc, arcface, _, audio_wav_path = extract_features_optimized(
                        frames_bgr=frames_bgr_384,
                        video_path=video_path,
                        timestamps=interval_timestamps,
                        fps=fps_val,
                        config=config,
                        start_time=padded_start_time,  # 🔑 패딩 적용
                        end_time=padded_end_time,      # 🔑 패딩 적용
                        max_duration=60,
                        extractors=extractors,
                        extract_phonemes=False,  # 🔑 음소 추출 끄기
                        logger=logger
                    )

                    # Step B: Phoneme alignment using shared transcription (Stage2에서 직접 수행)
                    logger.info("  [30FPS OPTIMIZATION] Aligning phonemes from shared transcription...")
                    alignment = self.phoneme_aligner.align_from_transcription(
                        transcription_result=whisper_result,
                        timestamps=interval_timestamps,
                        start_time=start_time  # Interval offset for correct segment matching
                    )
                    phoneme_labels = alignment['phoneme_labels']

                    # Cleanup WAV file
                    if audio_wav_path and Path(audio_wav_path).exists():
                        Path(audio_wav_path).unlink()

                    logger.info(f"  ✓ Features extracted (optimized) - MAR: {geometry.shape}, ArcFace: {arcface.shape}, Phoneme: {phoneme_labels.shape}")

                else:
                    # ORIGINAL PATH: Extract all features including phoneme (standalone mode)
                    logger.info("  Extracting features with phoneme alignment (standalone mode)...")
                    geometry, audio_mfcc, arcface, phoneme_labels, audio_wav_path = extract_features_optimized(
                        frames_bgr=frames_bgr_384,
                        video_path=video_path,
                        timestamps=interval_timestamps,
                        fps=fps_val,
                        config=config,
                        start_time=padded_start_time,  # 🔑 패딩 적용
                        end_time=padded_end_time,      # 🔑 패딩 적용
                        max_duration=60,
                        extractors=extractors,
                        extract_phonemes=True,  # Original behavior
                        logger=logger
                    )

                    # Cleanup WAV file immediately
                    if audio_wav_path and Path(audio_wav_path).exists():
                        Path(audio_wav_path).unlink()

                    logger.info(f"  ✓ Features extracted - MAR: {geometry.shape}, ArcFace: {arcface.shape}, Phoneme: {phoneme_labels.shape}")

            except RuntimeError as e:
                logger.error(f"Feature extraction failed: {e}")
                raise

        # Now we have all interval features
        interval_features = {
            'frames': interval_frames,
            'geometry': geometry,
            'arcface': arcface,
            'phoneme_labels': phoneme_labels,
            'timestamps': interval_timestamps
        }

        # ===== DEBUG: Check phoneme_labels quality (BEFORE conversion) =====
        non_sil_count = np.sum(interval_features['phoneme_labels'] != '<sil>')
        total_count = len(interval_features['phoneme_labels'])
        matching_rate_raw = (non_sil_count / total_count * 100) if total_count > 0 else 0.0

        logger.info(f"\n[PHONEME MATCHING QUALITY] Interval {interval_id}")
        logger.info(f"  Raw phoneme_labels: {total_count} frames")
        logger.info(f"  Non-<sil> frames: {non_sil_count}/{total_count}")
        logger.info(f"  RAW MATCHING RATE: {matching_rate_raw:.1f}% (from match_phoneme_to_frames)")

        # Convert phoneme labels to interval format
        # Note: phonemes and timestamps are already in relative coordinates (converted at Line 272)
        phonemes = get_interval_phoneme_dict(
            phoneme_labels=interval_features['phoneme_labels'],
            timestamps=interval_features['timestamps']  # Already relative [0.0, 0.033, ...]
        )

        # ===== DEBUG: Phoneme Distribution in Suspicious Interval =====
        logger.info(f"\n[PHONEME DISTRIBUTION DEBUG] Interval {interval_id}")
        logger.info(f"  Interval time range: {start_time:.2f}s - {end_time:.2f}s (duration: {end_time - start_time:.2f}s)")
        logger.info(f"  Total frames in interval: {len(interval_features['timestamps'])}")
        logger.info(f"  Total phonemes detected: {len(phonemes)}")

        if len(phonemes) > 0:
            logger.info(f"\n  Phoneme sequence in this interval:")
            for idx, p_dict in enumerate(phonemes):
                duration = p_dict['end'] - p_dict['start']
                logger.info(f"    [{idx}] '{p_dict['phoneme']}' | {p_dict['start']:.3f}s - {p_dict['end']:.3f}s (duration: {duration:.3f}s)")
        else:
            logger.warning(f"  ⚠️ No phonemes detected in this interval (silence or low quality audio)")

        # Count frames matched to each phoneme
        phoneme_frame_counts = {p: 0 for p in self.phoneme_vocab}
        for frame_idx, ts in enumerate(interval_features['timestamps']):
            for p_dict in phonemes:
                if p_dict['start'] <= ts <= p_dict['end']:
                    phoneme = p_dict['phoneme']
                    if phoneme in phoneme_frame_counts:
                        phoneme_frame_counts[phoneme] += 1
                    break

        logger.info(f"\n  Frame distribution across phonemes (before resampling):")
        for phoneme in self.phoneme_vocab:
            count = phoneme_frame_counts[phoneme]
            if count > 0:
                logger.info(f"    '{phoneme}': {count} frames")

        total_matched = sum(phoneme_frame_counts.values())
        unmatched = len(interval_features['timestamps']) - total_matched
        logger.info(f"  Matched frames: {total_matched}/{len(interval_features['timestamps'])}")
        if unmatched > 0:
            logger.warning(f"  ⚠️ Unmatched frames: {unmatched} (no phoneme coverage)")
        logger.info(f"  ========================================\n")

        # Resample frames to PIA 14×5 format
        resampled_frames, matched_phonemes, valid_mask_frames = resample_frames_to_pia_format(
            frames=interval_features['frames'],
            timestamps=interval_features['timestamps'],
            phonemes=phonemes,
            target_phonemes=14,
            frames_per_phoneme=5
        )

        # Prepare tensors for PIA
        geometry = interval_features['geometry']  # (N, 1)
        arcface = interval_features['arcface']    # (N, 512)

        # Resample geometry and arcface to 14×5 using shared utility
        resampled_geometry, _, valid_mask_geometry = resample_features_to_grid(
            geometry, interval_features['timestamps'], phonemes, 14, 5
        )
        resampled_arcface, _, valid_mask_arcface = resample_features_to_grid(
            arcface, interval_features['timestamps'], phonemes, 14, 5
        )

        # DEBUG: Print feature statistics before converting to tensors
        logger.info(f"\n[STAGE2 FEATURE DEBUG] Interval {interval_id}")
        logger.info(f"  Geometry: shape={resampled_geometry.shape}, mean={resampled_geometry.mean():.6f}, std={resampled_geometry.std():.6f}, range=[{resampled_geometry.min():.6f}, {resampled_geometry.max():.6f}]")
        logger.info(f"  ArcFace: shape={resampled_arcface.shape}, mean={resampled_arcface.mean():.6f}, std={resampled_arcface.std():.6f}, range=[{resampled_arcface.min():.6f}, {resampled_arcface.max():.6f}]")
        logger.info(f"  Frames: shape={resampled_frames.shape}, mean={resampled_frames.mean():.6f}, std={resampled_frames.std():.6f}, range=[{resampled_frames.min():.6f}, {resampled_frames.max():.6f}]")
        logger.info(f"  Matched phonemes: {matched_phonemes}")

        # Convert to tensors
        geoms_tensor = torch.from_numpy(resampled_geometry).float().unsqueeze(0)  # (1, 14, 5, 1)

        # imgs_tensor: (1, 14, 5, H, W, 3) -> (1, 14, 5, 3, H, W) for PIA model
        imgs_tensor = torch.from_numpy(resampled_frames).float().unsqueeze(0)     # (1, 14, 5, H, W, 3)
        imgs_tensor = imgs_tensor.permute(0, 1, 2, 5, 3, 4)  # (1, 14, 5, 3, H, W) - (B, P, F, C, H, W)

        arcs_tensor = torch.from_numpy(resampled_arcface).float().unsqueeze(0)     # (1, 14, 5, 512)

        # Create proper mask using valid_mask from resampling
        # Combine all three masks (frames, geometry, arcface) - use AND logic
        valid_mask = valid_mask_frames & valid_mask_geometry & valid_mask_arcface  # (14,)

        mask_tensor = torch.zeros(1, 14, 5, dtype=torch.bool)
        for pi in range(14):
            if valid_mask[pi]:  # Only mark valid phonemes
                mask_tensor[0, pi, :] = True

        # DEBUG: Log valid phoneme mask
        logger.info(f"\n[VALID PHONEME MASK DEBUG] Interval {interval_id}")
        logger.info(f"  Valid phonemes mask: {valid_mask.astype(int)}")
        logger.info(f"  Valid phoneme count: {valid_mask.sum()}/14")
        logger.info(f"  Valid phoneme list: {[matched_phonemes[i] for i in range(14) if valid_mask[i]]}")
        logger.info(f"  Mask tensor sum: {mask_tensor.sum().item()}/70 frames")

        # Validate mask for NaN prevention (critical bug fix from training)
        mask_sum = mask_tensor.sum().item()
        if mask_sum == 0:
            logger.warning(f"Interval {interval_id}: No valid phonemes (all silence)")
            return {
                'interval_id': interval_id,
                'interval_info': interval,
                'pia_xai': {
                    'error': 'no_valid_phonemes',
                    'verdict': 'inconclusive',
                    'prediction': -1,
                    'confidence': 0.0
                },
                'visualization_path': None
            }

        # Run PIA XAI analysis
        if self.pia_explainer is None:
            raise RuntimeError("PIA explainer not initialized. Cannot perform analysis.")

        xai_result = self.pia_explainer.explain(
            geoms=geoms_tensor,
            imgs=imgs_tensor,
            arcs=arcs_tensor,
            mask=mask_tensor,
            video_id=f"interval_{interval_id}",
            confidence_threshold=0.5
        )

        # Note: Visualization will be generated later in hybrid_pipeline.py
        # after overall detection is computed (to show combined result)

        return {
            'interval_id': interval_id,
            'interval_info': interval,
            'pia_xai': xai_result,
            'visualization_path': None  # Will be set later
        }

    def visualize_stage2_interval(
        self,
        interval_xai: Dict,
        interval_id: int,
        video_path: str,
        output_dir: str,
        overall_detection: Optional[Dict] = None
    ) -> str:
        """
        Visualize Stage2 XAI analysis for a single suspicious interval.

        Creates a comprehensive 4-panel visualization:
        - Panel 1 (Top-Left): Branch Contribution Bar Chart
        - Panel 2 (Top-Right): Phoneme Attention Distribution
        - Panel 3 (Bottom-Left): Temporal Attention Heatmap (14 phonemes × 5 frames)
        - Panel 4 (Bottom-Right): Detection Summary with Overall Combined Result

        Args:
            interval_xai: Stage2 XAI result for one interval (from run_stage2_interval_xai)
            interval_id: Interval index (0, 1, 2...)
            video_path: Original video path for title
            output_dir: Directory to save visualization
            overall_detection: Overall combined detection result (Stage1 + Stage2) to display
                             instead of individual PIA prediction

        Returns:
            Path to saved visualization PNG
        """
        import matplotlib.pyplot as plt

        logger.info(f"[Stage2] Generating XAI visualization for interval {interval_id}...")

        if self.pia_visualizer is None:
            raise RuntimeError("PIA visualizer not initialized. Cannot generate visualization.")

        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Title with interval info
        video_name = Path(video_path).stem
        fig.suptitle(
            f"Stage2 XAI Analysis | Video: {video_name} | Interval {interval_id}",
            fontsize=14,
            fontweight='bold'
        )

        # Panel 1: Branch Contribution (Top-Left)
        ax1 = axes[0, 0]
        self.pia_visualizer.plot_branch_contributions(
            interval_xai,
            ax=ax1,
            title=f"Branch Contribution (Interval {interval_id})"
        )

        # Panel 2: Phoneme Attention (Top-Right)
        ax2 = axes[0, 1]
        self.pia_visualizer.plot_phoneme_attention(
            interval_xai,
            ax=ax2,
            title=f"Phoneme Attention Distribution"
        )

        # Panel 3: Temporal Heatmap (Bottom-Left)
        ax3 = axes[1, 0]
        self.pia_visualizer.plot_temporal_heatmap(
            interval_xai,
            ax=ax3,
            title=f"Temporal Attention Heatmap (14×5)"
        )

        # Panel 4: Detection Summary (Bottom-Right)
        ax4 = axes[1, 1]

        # If overall_detection provided, use it instead of PIA-only prediction
        if overall_detection is not None:
            # Create a modified result with overall detection for display
            display_result = interval_xai.copy()
            display_result['detection'] = {
                'prediction_label': overall_detection['verdict'].upper(),
                'confidence': overall_detection['confidence']
            }
            display_result['summary'] = {
                'overall': f"Overall: {overall_detection['verdict'].upper()} ({overall_detection['confidence']*100:.1f}%)"
            }
            self.pia_visualizer.plot_detection_summary(
                display_result,
                ax=ax4,
                title=f"Overall Detection (Combined)"
            )
        else:
            # Use PIA-only prediction (backward compatibility)
            self.pia_visualizer.plot_detection_summary(
                interval_xai,
                ax=ax4,
                title=f"Detection Summary (PIA Only)"
            )

        # Save visualization
        output_path = Path(output_dir) / f"{video_name}_interval_{interval_id}_xai.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved Stage2 XAI visualization to: {output_path}")

        return str(output_path)

    def _format_stage2_result(
        self,
        interval_xai: Dict,
        interval: Dict,
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        Convert internal PIA XAI result to hybrid_xai_interface.ts format.

        REFACTORED: Now delegates to hybrid_utils.format_stage2_to_interface()
        to reduce code duplication.

        Args:
            interval_xai: Result from run_stage2_interval_xai()
            interval: Original interval dict from Stage1
            output_dir: Output directory for visualization path

        Returns:
            Formatted result matching TypeScript interface
        """
        # Delegate formatting to hybrid_utils to reduce duplication
        return format_stage2_to_interface(
            interval_xai=interval_xai,
            interval=interval,
            output_dir=output_dir
        )

