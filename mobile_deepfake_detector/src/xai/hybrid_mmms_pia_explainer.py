"""
Hybrid MMMS-BA + PIA XAI Pipeline Orchestrator (Inference Version)

This module coordinates the 2-stage deepfake detection pipeline:
- Stage 1: MMMS-BA temporal scan for suspicious frame detection
- Stage 2: PIA deep XAI analysis on suspicious intervals

Pipeline Flow:
    Raw Video → Feature Extraction → Stage 1 (MMMS-BA) → Suspicious Intervals → Stage 2 (PIA XAI) → Final Results

Author: Claude
Date: 2025-11-17
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm

# Import existing models and utilities
import sys
import cv2
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.teacher import MMMSBA
from src.models.pia_model import PIAModel
from src.utils.config import load_config
from src.xai.pia_explainer import PIAExplainer
from src.xai.pia_visualizer import PIAVisualizer
from src.data.preprocessing import ShortsPreprocessor, match_phoneme_to_frames
from src.utils.hybrid_phoneme_aligner_v2 import HybridPhonemeAligner
from src.utils.enhanced_mar_extractor import EnhancedMARExtractor
from src.utils.arcface_extractor import ArcFaceExtractor
from src.xai.hybrid_utils import (
    group_consecutive_frames,
    get_interval_phoneme_dict,
    resample_frames_to_pia_format,
    aggregate_interval_insights,
    build_korean_summary,
    calculate_severity,
    calculate_risk_level,
    convert_for_json,
    format_stage1_to_interface,
    format_stage2_to_interface
)

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridMMSBAPIA:
    """
    Hybrid MMMS-BA + PIA XAI Pipeline

    Combines temporal scanning (MMMS-BA) with deep XAI analysis (PIA)
    for comprehensive deepfake detection and explanation.
    """

    def __init__(
        self,
        mmms_model_path: str,
        pia_model_path: str,
        config_path: str = "configs/train_teacher_korean.yaml",
        pia_config_path: str = "configs/train_pia.yaml",
        device: str = "cuda"
    ):
        """
        Initialize the hybrid pipeline with both models for inference.

        Args:
            mmms_model_path: Path to MMMS-BA checkpoint
            pia_model_path: Path to PIA checkpoint
            config_path: MMMS-BA config file
            pia_config_path: PIA config file
            device: cuda or cpu
        """
        # CUDA device setup with better logging
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = torch.device("cpu")
            if device == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
            else:
                logger.info("Using CPU device")

        # Load configurations
        self.mmms_config = load_config(config_path)
        self.pia_config = load_config(pia_config_path)

        # Load Korean phoneme vocabulary
        with open("configs/phoneme_vocab.json", "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            # Convert dict to list sorted by ID: {phoneme: id} → [phoneme_at_0, phoneme_at_1, ...]
            vocab_dict = vocab_data["vocab"]
            self.phoneme_vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda x: x[1])]

        # Stage1 preprocessing config (limited to 30 seconds)
        self.preprocess_config = {
            'target_fps': 30,
            'max_frames': 900,  # 30 seconds × 30fps = 900 frames max
            'frame_size': [224, 224],
            'max_duration': 30,  # Limit to 30 seconds
            'min_duration': 1,   # Allow short clips (for testing)
            'sample_rate': 16000,
            'n_mfcc': 40,
            'hop_length': 512,
            'n_fft': 2048,
            'lip_size': [96, 96]
        }

        # Stage1: ShortsPreprocessor for frames/audio/lip
        logger.info("Initializing ShortsPreprocessor for Stage 1...")
        self.preprocessor = ShortsPreprocessor(self.preprocess_config)

        # Stage 2 extractors (lazy initialization - only when needed)
        self.phoneme_aligner = None
        self.mar_extractor = None
        self.arcface_extractor = None

        # Initialize MMMS-BA model (if not dummy)
        if mmms_model_path != "dummy":
            logger.info("Loading MMMS-BA model...")
            self.mmms_model = self._load_mmms_model(mmms_model_path)
        else:
            self.mmms_model = None
            logger.info("MMMS-BA model skipped (dummy path)")

        # Initialize PIA model (if not dummy)
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

        logger.info("Hybrid pipeline initialized successfully!")

    def _load_mmms_model(self, checkpoint_path: str) -> nn.Module:
        """Load MMMS-BA model from checkpoint."""
        # Use config values where available, otherwise use hardcoded defaults from comments
        model = MMMSBA(
            audio_dim=self.mmms_config.get('dataset', {}).get('audio', {}).get('n_mfcc', 40),
            visual_dim=256,  # Hardcoded from ResNet feature extractor
            lip_dim=128,     # Hardcoded from lip ROI feature extractor
            gru_hidden_dim=self.mmms_config.get('model', {}).get('gru', {}).get('hidden_size', 300),
            dense_hidden_dim=self.mmms_config.get('model', {}).get('dense', {}).get('hidden_size', 100)
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

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

    def run_stage1_temporal_scan(
        self,
        video_path: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Stage 1: MMMS-BA temporal scan on raw video.

        Args:
            video_path: Path to raw video file (NOT NPZ)
            threshold: Fake probability threshold for suspicious frames

        Returns:
            stage1_result: {
                'fake_probs': np.ndarray (T,),
                'features': Dict with all extracted features,
                'timestamps': np.ndarray (T,),
                'fps': float,
                'total_frames': int,
                'suspicious_indices': np.ndarray,
                'overall_verdict': str,
                'overall_confidence': float,
                'video_path': str
            }
        """
        logger.info(f"[Stage 1] Running MMMS-BA temporal scan on video: {video_path}")

        # Extract frames/audio/lip using ShortsPreprocessor
        logger.info("  Extracting features from video...")
        result = self.preprocessor.process_video(
            str(video_path),
            extract_audio=True,
            extract_lip=True
        )

        if result['frames'] is None or result['audio'] is None or result['lip'] is None:
            raise RuntimeError(f"Failed to extract features from {video_path}")

        frames = result['frames']  # (T, 224, 224, 3)
        audio = result['audio']    # (T_audio, 40)
        lip = result['lip']        # (T, 96, 96, 3)

        logger.info(f"  Extracted features: {frames.shape[0]} frames")

        # Downsample to 10fps for 3x speedup (30fps → 10fps)
        # Get original FPS from video
        cap_temp = cv2.VideoCapture(str(video_path))
        original_fps = cap_temp.get(cv2.CAP_PROP_FPS) if cap_temp.isOpened() else 30.0
        cap_temp.release()

        target_fps = 10.0
        if original_fps > target_fps:
            downsample_step = max(1, int(original_fps / target_fps))  # e.g., 30/10 = 3
            logger.info(f"  Downsampling from {original_fps:.1f}fps to {target_fps}fps (step={downsample_step})")

            frames = frames[::downsample_step]
            lip = lip[::downsample_step]
            # Audio stays the same (already covers full timeline)

            logger.info(f"  Downsampled to {frames.shape[0]} frames")

        video_id = Path(video_path).stem

        # Get FPS and calculate timestamps
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else frames.shape[0]
        cap.release()

        # Calculate timestamps (same as preprocess_parallel.py)
        total_frames = frames.shape[0]
        if total_frames_video <= total_frames:
            frame_indices = np.arange(total_frames, dtype=np.float32)
        else:
            step = total_frames_video / total_frames
            frame_indices = np.array([int(i * step) for i in range(total_frames)], dtype=np.float32)

        timestamps = frame_indices / fps

        # Run MMMS-BA frame-level prediction
        logger.info(f"  Running frame-level prediction on {total_frames} frames...")
        fake_probs = self._predict_frame_level(frames, audio, lip)

        # Identify suspicious frames
        suspicious_indices = np.where(fake_probs > threshold)[0]
        suspicious_ratio = len(suspicious_indices) / total_frames * 100

        # Overall verdict
        mean_fake_prob = np.mean(fake_probs)
        overall_verdict = "fake" if mean_fake_prob > 0.5 else "real"
        overall_confidence = mean_fake_prob if overall_verdict == "fake" else 1 - mean_fake_prob

        logger.info(f"  Stage 1 Results:")
        logger.info(f"    - Total frames: {total_frames}")
        logger.info(f"    - Suspicious frames: {len(suspicious_indices)} ({suspicious_ratio:.1f}%)")
        logger.info(f"    - Overall verdict: {overall_verdict.upper()} (confidence: {overall_confidence:.2%})")

        return {
            'fake_probs': fake_probs,
            'frames': frames,
            'audio': audio,
            'lip': lip,
            'timestamps': timestamps,
            'fps': fps,
            'total_frames': total_frames,
            'suspicious_indices': suspicious_indices,
            'suspicious_ratio': suspicious_ratio,
            'overall_verdict': overall_verdict,
            'overall_confidence': overall_confidence,
            'video_path': video_path,
            'video_id': video_id
        }

    def _predict_frame_level(
        self,
        frames: np.ndarray,
        audio: np.ndarray,
        lip: np.ndarray
    ) -> np.ndarray:
        """
        Run MMMS-BA model with frame_level=True to get per-frame fake probabilities.

        Uses the model's built-in frame_level parameter for efficient GPU utilization.

        Args:
            frames: (T, 224, 224, 3) numpy array
            audio: (T_audio, 40) numpy array
            lip: (T, 96, 96, 3) numpy array

        Returns:
            fake_probs: (T,) array of fake probabilities
        """
        # Convert to tensors with correct shapes
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0) / 255.0  # (1, T, 3, H, W)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T_audio, 40)
        lip_tensor = torch.from_numpy(lip).float().permute(0, 3, 1, 2).unsqueeze(0) / 255.0  # (1, T, 3, lip_H, lip_W)

        # Move to device
        frames_tensor = frames_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)
        lip_tensor = lip_tensor.to(self.device)

        # Single forward pass with frame_level=True
        with torch.no_grad():
            logits = self.mmms_model(
                audio=audio_tensor,
                frames=frames_tensor,
                lip=lip_tensor,
                frame_level=True  # Returns (1, T, 2) instead of (1, 2)
            )

            # logits: (1, T, 2)
            frame_probs = torch.softmax(logits, dim=-1)  # (1, T, 2)
            fake_probs = frame_probs[0, :, 1].cpu().numpy()  # (T,)

            # Free GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return fake_probs

    def run_stage2_interval_xai(
        self,
        stage1_result: Dict,
        interval: Dict,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 2: PIA XAI analysis on a single suspicious interval.

        Args:
            stage1_result: Result from Stage 1 with frames/timestamps/video_path
            interval: Interval dict from group_consecutive_frames()
            output_dir: Optional directory to save visualization

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
                whisper_model="base",
                device=str(self.device),
                compute_type="float16"
            )
        if self.mar_extractor is None:
            logger.info("  Initializing EnhancedMARExtractor...")
            self.mar_extractor = EnhancedMARExtractor()
        if self.arcface_extractor is None:
            logger.info("  Initializing ArcFaceExtractor...")
            self.arcface_extractor = ArcFaceExtractor(
                device='cpu',  # CPU to reduce GPU memory
                model_name='buffalo_l'
            )

        # Get video path from stage1_result
        video_path = stage1_result['video_path']

        # Extract interval timestamps
        start_frame = interval['start_frame']
        end_frame = interval['end_frame']
        start_time = interval['start_time']
        end_time = interval['end_time']

        # Extract interval frames from stage1_result
        interval_frames = stage1_result['frames'][start_frame:end_frame+1]
        interval_timestamps = stage1_result['timestamps'][start_frame:end_frame+1]

        # Extract phoneme alignment for this video
        logger.info("  Extracting phoneme alignment...")
        alignment = self.phoneme_aligner.align_video(video_path)

        # Extract phoneme intervals from alignment
        phoneme_intervals = [
            {'phoneme': p, 'start': s, 'end': e}
            for p, (s, e) in zip(alignment['phonemes'], alignment['intervals'])
        ]

        # Match phonemes to interval frames
        phoneme_labels = match_phoneme_to_frames(phoneme_intervals, interval_timestamps)

        # Extract MAR for interval frames
        logger.info("  Extracting MAR features...")
        # Save interval frames to temp video for MAR extraction
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_video_path = tmp.name
            # Create temp video from interval frames
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps_val = stage1_result['fps']
            h, w = interval_frames.shape[1:3]
            out = cv2.VideoWriter(tmp_video_path, fourcc, fps_val, (w, h))
            for frame in interval_frames:
                out.write(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            out.release()

        try:
            mar_result = self.mar_extractor.extract_from_video(tmp_video_path, max_frames=len(interval_frames))
            geometry = np.array(mar_result['mar_vertical']).reshape(-1, 1)  # (T, 1)
        finally:
            import os
            os.unlink(tmp_video_path)

        # Extract ArcFace for interval frames
        logger.info("  Extracting ArcFace features...")
        # Save interval frames to temp video for ArcFace extraction
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_video_path = tmp.name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_video_path, fourcc, fps_val, (w, h))
            for frame in interval_frames:
                out.write(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            out.release()

        try:
            arcface = self.arcface_extractor.extract_from_video(
                tmp_video_path,
                max_frames=len(interval_frames),
                show_progress=False
            )  # (T, 512)
        finally:
            os.unlink(tmp_video_path)

        # Now we have all interval features
        interval_features = {
            'frames': interval_frames,
            'geometry': geometry,
            'arcface': arcface,
            'phoneme_labels': phoneme_labels,
            'timestamps': interval_timestamps
        }

        # Convert phoneme labels to interval format
        phonemes = get_interval_phoneme_dict(
            phoneme_labels=interval_features['phoneme_labels'],
            timestamps=interval_features['timestamps']
        )

        # Resample frames to PIA 14×5 format
        resampled_frames, matched_phonemes = resample_frames_to_pia_format(
            frames=interval_features['frames'],
            timestamps=interval_features['timestamps'],
            phonemes=phonemes,
            target_phonemes=14,
            frames_per_phoneme=5
        )

        # Prepare tensors for PIA
        geometry = interval_features['geometry']  # (N, 1)
        arcface = interval_features['arcface']    # (N, 512)

        # Resample geometry and arcface to 14×5
        resampled_geometry = self._resample_features_to_grid(
            geometry, interval_features['timestamps'], phonemes, 14, 5
        )
        resampled_arcface = self._resample_features_to_grid(
            arcface, interval_features['timestamps'], phonemes, 14, 5
        )

        # Convert to tensors
        geoms_tensor = torch.from_numpy(resampled_geometry).float().unsqueeze(0)  # (1, 14, 5, 1)

        # imgs_tensor: (1, 14, 5, H, W, 3) -> (1, 14, 5, 3, H, W) for PIA model
        imgs_tensor = torch.from_numpy(resampled_frames).float().unsqueeze(0)     # (1, 14, 5, H, W, 3)
        imgs_tensor = imgs_tensor.permute(0, 1, 2, 5, 3, 4)  # (1, 14, 5, 3, H, W) - (B, P, F, C, H, W)

        arcs_tensor = torch.from_numpy(resampled_arcface).float().unsqueeze(0)     # (1, 14, 5, 512)
        mask_tensor = torch.ones(1, 14, 5).bool()  # All frames valid

        # Run PIA XAI analysis
        xai_result = self.pia_explainer.explain(
            geoms=geoms_tensor,
            imgs=imgs_tensor,
            arcs=arcs_tensor,
            mask=mask_tensor,
            video_id=f"interval_{interval_id}",
            confidence_threshold=0.5
        )

        # Save visualization if requested
        if output_dir:
            viz_path = self.visualize_stage2_interval(
                interval_xai=xai_result,
                interval_id=interval_id,
                video_path=video_path,
                output_dir=output_dir
            )
        else:
            viz_path = None

        return {
            'interval_id': interval_id,
            'interval_info': interval,
            'pia_xai': xai_result,
            'visualization_path': viz_path
        }

    def _resample_features_to_grid(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        phonemes: List[Dict],
        target_phonemes: int = 14,
        frames_per_phoneme: int = 5
    ) -> np.ndarray:
        """
        Resample features to PIA 14×5 grid format.

        Args:
            features: (N, D) - Feature array
            timestamps: (N,) - Timestamps
            phonemes: List of phoneme dicts
            target_phonemes: 14
            frames_per_phoneme: 5

        Returns:
            resampled: (14, 5, D)
        """
        from ..utils.korean_phoneme_config import get_phoneme_vocab, is_kept_phoneme

        phoneme_vocab = get_phoneme_vocab()
        N, D = features.shape

        # Group features by phoneme
        by_phoneme = {p: [] for p in phoneme_vocab}

        for frame_idx, ts in enumerate(timestamps):
            for p_dict in phonemes:
                if p_dict['start'] <= ts <= p_dict['end']:
                    phoneme = p_dict['phoneme']
                    if is_kept_phoneme(phoneme):
                        by_phoneme[phoneme].append(features[frame_idx])
                    break

        # Build 14×5 grid
        resampled = np.zeros((target_phonemes, frames_per_phoneme, D), dtype=features.dtype)

        for pi, phoneme in enumerate(phoneme_vocab):
            frames_list = by_phoneme[phoneme][:frames_per_phoneme]
            for fi, frame_feat in enumerate(frames_list):
                resampled[pi, fi] = frame_feat

        return resampled

    def run_hybrid_pipeline(
        self,
        video_path: str,
        threshold: float = 0.5,
        min_interval_frames: int = 14,
        merge_gap_sec: float = 1.0,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main hybrid pipeline orchestration for inference.

        Pipeline Flow:
        1. Extract features from raw video
        2. Stage 1: MMMS-BA temporal scan
        3. Group suspicious frames into intervals
        4. Stage 2: PIA XAI analysis on each interval
        5. Aggregate results and build final report

        Args:
            video_path: Path to raw video file (NOT NPZ)
            threshold: Fake probability threshold for suspicious frames
            min_interval_frames: Minimum frames per interval (PIA needs 14+)
            merge_gap_sec: Gap in seconds to merge nearby intervals
            output_dir: Optional directory to save visualizations

        Returns:
            HybridXAIResult: Complete analysis result
        """
        logger.info("=" * 80)
        logger.info("Starting Hybrid MMMS-BA + PIA XAI Pipeline (Inference Mode)")
        logger.info("=" * 80)
        logger.info(f"Video: {video_path}")

        # === Stage 1: Feature Extraction + MMMS-BA Temporal Scan ===
        logger.info("\n[Stage 1] Extracting features and running MMMS-BA temporal scan...")
        stage1_result = self.run_stage1_temporal_scan(
            video_path=video_path,
            threshold=threshold
        )

        suspicious_indices = stage1_result['suspicious_indices']
        fps = stage1_result['fps']

        # === Group into intervals ===
        logger.info("\n[Stage 1] Grouping suspicious frames into intervals...")
        suspicious_intervals = group_consecutive_frames(
            suspicious_indices=suspicious_indices,
            fps=fps,
            min_interval_frames=min_interval_frames,
            merge_gap_sec=merge_gap_sec
        )

        logger.info(f"  Found {len(suspicious_intervals)} suspicious intervals")
        for interval in suspicious_intervals:
            logger.info(f"    Interval {interval['interval_id']}: "
                       f"{interval['start_time']:.1f}s - {interval['end_time']:.1f}s "
                       f"({interval['frame_count']} frames)")

        # === Stage 2: PIA XAI on each interval ===
        interval_xai_results = []

        if len(suspicious_intervals) > 0:
            logger.info(f"\n[Stage 2] Running PIA XAI on {len(suspicious_intervals)} intervals...")

            for interval in suspicious_intervals:
                interval_xai = self.run_stage2_interval_xai(
                    stage1_result=stage1_result,  # Pass stage1_result with frames/timestamps/video_path
                    interval=interval,
                    output_dir=output_dir
                )
                interval_xai_results.append(interval_xai)
        else:
            logger.info("\n[Stage 2] No suspicious intervals found - video appears clean")

        # === Aggregate results ===
        logger.info("\n[Aggregation] Combining results...")

        if interval_xai_results:
            # Extract PIA XAI results
            pia_xai_list = [r['pia_xai'] for r in interval_xai_results]

            # Aggregate insights
            aggregated_insights = aggregate_interval_insights(pia_xai_list)

            # Build Korean summary
            korean_summary = build_korean_summary(
                verdict=stage1_result['overall_verdict'],
                confidence=stage1_result['overall_confidence'],
                suspicious_intervals=suspicious_intervals,
                interval_xai_results=pia_xai_list
            )
        else:
            # Clean video - no suspicious content
            aggregated_insights = {
                'top_suspicious_phonemes': [],
                'branch_trends': {
                    'visual_avg': 0,
                    'geometry_avg': 0,
                    'identity_avg': 0,
                    'most_dominant': 'none'
                },
                'mar_summary': {
                    'intervals_with_anomalies': 0,
                    'total_anomalous_frames': 0,
                    'avg_deviation_percent': 0
                }
            }

            korean_summary = {
                'title': f"✓ 실제 영상으로 판정 (신뢰도: {stage1_result['overall_confidence']*100:.1f}%)",
                'risk_level': 'low',
                'primary_reason': '의심 구간이 탐지되지 않았습니다.',
                'suspicious_interval_count': 0,
                'top_suspicious_phonemes': [],
                'detailed_explanation': '딥페이크 패턴이 감지되지 않았습니다.'
            }

        # === Build final result ===
        final_result = {
            'metadata': {
                'video_path': video_path,
                'pipeline_version': '1.0',
                'analysis_timestamp': str(np.datetime64('now')),
                'parameters': {
                    'threshold': threshold,
                    'min_interval_frames': min_interval_frames,
                    'merge_gap_sec': merge_gap_sec
                }
            },
            'video_info': {
                'video_id': stage1_result['video_id'],
                'total_frames': stage1_result['total_frames'],
                'fps': stage1_result['fps'],
                'duration_sec': stage1_result['total_frames'] / stage1_result['fps']
            },
            'stage1_results': {
                'overall_verdict': stage1_result['overall_verdict'],
                'overall_confidence': stage1_result['overall_confidence'],
                'suspicious_frame_count': len(suspicious_indices),
                'suspicious_ratio': stage1_result['suspicious_ratio'],
                'risk_level': calculate_risk_level(stage1_result['suspicious_ratio']),
                'suspicious_intervals': suspicious_intervals
            },
            'stage2_results': interval_xai_results,
            'aggregated_insights': aggregated_insights,
            'korean_summary': korean_summary
        }

        # Save final result if output directory specified
        if output_dir:
            output_path = Path(output_dir) / f"{stage1_result['video_id']}_hybrid_result.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                # Convert numpy arrays to lists for JSON serialization
                json.dump(self._convert_for_json(final_result), f, indent=2, ensure_ascii=False)

            logger.info(f"\nSaved final result to {output_path}")

        # === Final summary ===
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"Overall Verdict: {stage1_result['overall_verdict'].upper()}")
        logger.info(f"Confidence: {stage1_result['overall_confidence']:.2%}")
        logger.info(f"Risk Level: {korean_summary['risk_level'].upper()}")
        logger.info(f"Korean Summary: {korean_summary['title']}")

        return final_result

    def visualize_stage2_interval(
        self,
        interval_xai: Dict,
        interval_id: int,
        video_path: str,
        output_dir: str
    ) -> str:
        """
        Visualize Stage2 XAI analysis for a single suspicious interval.

        Creates a comprehensive 4-panel visualization:
        - Panel 1 (Top-Left): Branch Contribution Bar Chart
        - Panel 2 (Top-Right): Phoneme Attention Distribution
        - Panel 3 (Bottom-Left): Temporal Attention Heatmap (14 phonemes × 5 frames)
        - Panel 4 (Bottom-Right): Detection Summary with Geometry Analysis

        Args:
            interval_xai: Stage2 XAI result for one interval (from run_stage2_interval_xai)
            interval_id: Interval index (0, 1, 2...)
            video_path: Original video path for title
            output_dir: Directory to save visualization

        Returns:
            Path to saved visualization PNG
        """
        import matplotlib.pyplot as plt

        logger.info(f"[Stage2] Generating XAI visualization for interval {interval_id}...")

        # Initialize visualizer
        visualizer = PIAVisualizer(figsize=(16, 12), dpi=150)

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
        visualizer.plot_branch_contributions(
            interval_xai,
            ax=ax1,
            title=f"Branch Contribution (Interval {interval_id})"
        )

        # Panel 2: Phoneme Attention (Top-Right)
        ax2 = axes[0, 1]
        visualizer.plot_phoneme_attention(
            interval_xai,
            ax=ax2,
            title=f"Phoneme Attention Distribution"
        )

        # Panel 3: Temporal Heatmap (Bottom-Left)
        ax3 = axes[1, 0]
        visualizer.plot_temporal_heatmap(
            interval_xai,
            ax=ax3,
            title=f"Temporal Attention Heatmap (14×5)"
        )

        # Panel 4: Detection Summary (Bottom-Right)
        ax4 = axes[1, 1]
        visualizer.plot_detection_summary(
            interval_xai,
            ax=ax4,
            title=f"Detection Summary"
        )

        # Save visualization
        output_path = Path(output_dir) / f"{video_name}_interval_{interval_id}_xai.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved Stage2 XAI visualization to: {output_path}")

        return str(output_path)

    def _convert_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj


class Stage1Scanner:
    """
    Stage 1 Scanner: MMMS-BA Temporal Scan Wrapper

    Wraps HybridMMSBAPIA.run_stage1_temporal_scan() to provide
    clean interface matching hybrid_xai_interface.ts

    Reuses:
    - HybridMMSBAPIA.run_stage1_temporal_scan() for frame-level prediction
    - group_consecutive_frames() from hybrid_utils.py for interval detection
    - Visualization code adapted from visualize_temporal_prediction.py
    """

    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/train_teacher_korean.yaml",
        device: str = "cuda"
    ):
        """
        Initialize Stage1Scanner with MMMS-BA model.

        Args:
            model_path: Path to MMMS-BA checkpoint
            config_path: MMMS-BA config file
            device: cuda or cpu
        """
        logger.info("Initializing Stage1Scanner...")

        # Initialize HybridMMSBAPIA with dummy PIA path (not used in Stage1)
        self.hybrid_pipeline = HybridMMSBAPIA(
            mmms_model_path=model_path,
            pia_model_path="dummy",  # Not used in Stage1
            config_path=config_path,
            device=device
        )

        logger.info("Stage1Scanner initialized successfully!")

    def scan_video(
        self,
        video_path: str,
        threshold: float = 0.6,
        min_interval_frames: int = 14,
        merge_gap_sec: float = 1.0,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run MMMS-BA temporal scan on video.

        Args:
            video_path: Path to raw video file (NOT NPZ)
            threshold: Fake probability threshold for suspicious frames
            min_interval_frames: Minimum frames per interval (PIA needs 14+)
            merge_gap_sec: Gap in seconds to merge nearby intervals
            output_dir: Optional directory to save visualization

        Returns:
            stage1_timeline: Dict matching hybrid_xai_interface.ts format
                {
                    'frame_probabilities': [...],
                    'suspicious_intervals': [...],
                    'statistics': {...},
                    'visualization': {'timeline_plot_url': ...}
                }
        """
        logger.info(f"[Stage1Scanner] Scanning video: {video_path}")

        # Step 1: Run existing MMMS-BA temporal scan
        stage1_result = self.hybrid_pipeline.run_stage1_temporal_scan(
            video_path=video_path,
            threshold=threshold
        )

        # Step 2: Group consecutive frames into intervals (reuse existing function)
        suspicious_intervals_raw = group_consecutive_frames(
            suspicious_indices=stage1_result['suspicious_indices'],
            fps=stage1_result['fps'],
            min_interval_frames=min_interval_frames,
            merge_gap_sec=merge_gap_sec
        )

        # Step 3: Convert to TypeScript interface format
        stage1_timeline = self._format_stage1_timeline(
            stage1_result=stage1_result,
            suspicious_intervals_raw=suspicious_intervals_raw,
            threshold=threshold
        )

        # Step 4: Generate visualization if requested
        if output_dir:
            viz_path = self.visualize_stage1_timeline(
                stage1_timeline=stage1_timeline,
                video_path=video_path,
                output_dir=output_dir
            )
            stage1_timeline['visualization']['timeline_plot_url'] = viz_path

        # Step 5: Store extracted features for Stage2 (FIX: Add features to result)
        stage1_timeline['extracted_features'] = {
            'frames': stage1_result['frames'],
            'audio': stage1_result['audio'],
            'lip': stage1_result['lip'],
            'fps': stage1_result['fps'],
            'total_frames': stage1_result['total_frames'],
            'timestamps': stage1_result['timestamps'],
            'video_path': stage1_result['video_path']
        }

        # Store stage1_result in instance variable for visualization access
        self.stage1_result = stage1_result

        logger.info(f"[Stage1Scanner] Scan complete!")
        logger.info(f"  Total frames: {len(stage1_timeline['frame_probabilities'])}")
        logger.info(f"  Suspicious intervals: {len(stage1_timeline['suspicious_intervals'])}")

        return stage1_timeline

    def _format_stage1_timeline(
        self,
        stage1_result: Dict,
        suspicious_intervals_raw: List[Dict],
        threshold: float
    ) -> Dict[str, Any]:
        """
        Convert internal format to hybrid_xai_interface.ts stage1_timeline format.

        REFACTORED: Now delegates to hybrid_utils.format_stage1_to_interface()
        to reduce code duplication.

        Args:
            stage1_result: Output from run_stage1_temporal_scan()
            suspicious_intervals_raw: Output from group_consecutive_frames()
            threshold: Threshold used for detection

        Returns:
            stage1_timeline: Formatted dict matching TypeScript interface
        """
        # Delegate formatting to hybrid_utils to reduce duplication
        return format_stage1_to_interface(
            stage1_result=stage1_result,
            suspicious_intervals_raw=suspicious_intervals_raw,
            threshold=threshold
        )

    def visualize_stage1_timeline(
        self,
        stage1_timeline: Dict,
        video_path: str,
        output_dir: str,
        num_sample_frames: int = 6
    ) -> str:
        """
        Generate timeline visualization plot with sample frames.

        Reuses visualization code from visualize_temporal_prediction.py.

        Args:
            stage1_timeline: Formatted stage1_timeline dict
            video_path: Path to video (for title)
            output_dir: Output directory
            num_sample_frames: Number of sample frames to display (default: 6)

        Returns:
            timeline_plot_url: Path to saved PNG file
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Extract data
        frame_probs = stage1_timeline['frame_probabilities']
        suspicious_intervals = stage1_timeline['suspicious_intervals']
        stats = stage1_timeline['statistics']

        timestamps = np.array([fp['timestamp_sec'] for fp in frame_probs])
        fake_probs = np.array([fp['fake_probability'] for fp in frame_probs])
        threshold = stats['threshold_used']

        # Get frames from stage1_result (stored during Stage1)
        # Note: frames are stored in self.stage1_result during process_video()
        frames = self.stage1_result['frames']  # (T, H, W, 3) numpy array [0, 1]
        total_frames = len(frames)

        # Create figure with 3 rows
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1.5, 0.5], hspace=0.3)

        # === Row 1: Sample Frames ===
        ax_frames = fig.add_subplot(gs[0])
        ax_frames.set_title(
            f"Video: {Path(video_path).stem} | Duration: {timestamps[-1]:.2f}s ({total_frames} frames)",
            fontsize=14,
            fontweight='bold'
        )

        # Select evenly spaced frames
        frame_indices = np.linspace(0, total_frames-1, num_sample_frames, dtype=int)

        # Create montage
        montage_frames = []
        for idx in frame_indices:
            frame = frames[idx]  # (224, 224, 3), [0, 1]
            frame_uint8 = (frame * 255).astype(np.uint8)
            # Add frame number text
            cv2.putText(
                frame_uint8,
                f"#{idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame_uint8,
                f"{timestamps[idx]:.1f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            # Add fake probability
            prob = fake_probs[idx]
            color = (255, 0, 0) if prob > threshold else (0, 255, 0)
            cv2.putText(
                frame_uint8,
                f"P(Fake)={prob:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            montage_frames.append(frame_uint8)

        # Concatenate horizontally
        montage = np.concatenate(montage_frames, axis=1)
        ax_frames.imshow(montage)
        ax_frames.axis('off')

        # === Row 2: Temporal Probability Graph ===
        ax_graph = fig.add_subplot(gs[1])
        ax_graph.set_title(
            f"MMMS-BA Frame-Level Fake Probability | Video: {Path(video_path).stem}",
            fontsize=14,
            fontweight='bold'
        )

        # Plot probability line
        ax_graph.plot(timestamps, fake_probs, linewidth=2, color='darkblue', alpha=0.8)
        ax_graph.fill_between(timestamps, fake_probs, alpha=0.3, color='lightblue')

        # Threshold line
        ax_graph.axhline(
            y=threshold,
            color='red',
            linestyle='--',
            linewidth=1,
            alpha=0.7,
            label=f'Threshold ({threshold})'
        )

        # Highlight suspicious regions
        high_risk_mask = fake_probs > threshold
        if high_risk_mask.any():
            ax_graph.fill_between(
                timestamps,
                0,
                1,
                where=high_risk_mask,
                alpha=0.2,
                color='red',
                label='Suspicious Frames'
            )

        # Mark suspicious intervals
        for interval in suspicious_intervals:
            ax_graph.axvspan(
                interval['start_time_sec'],
                interval['end_time_sec'],
                alpha=0.15,
                color='orange',
                label=f"Interval {interval['interval_id']}" if interval['interval_id'] == 0 else ""
            )

        # Styling
        ax_graph.set_xlabel("Time (seconds)", fontsize=11)
        ax_graph.set_ylabel("P(Fake)", fontsize=11)
        ax_graph.set_ylim([0, 1])
        ax_graph.set_xlim([timestamps[0], timestamps[-1]])
        ax_graph.grid(True, alpha=0.3, linestyle=':')
        ax_graph.legend(loc='upper right', fontsize=10)

        # === Row 3: Temporal Heatmap ===
        ax_heatmap = fig.add_subplot(gs[2])
        ax_heatmap.set_title("Temporal Heatmap", fontsize=11)

        # Create heatmap (1D)
        heatmap_data = fake_probs.reshape(1, -1)  # (1, T)

        im = ax_heatmap.imshow(
            heatmap_data,
            cmap='RdYlGn_r',  # Red (high fake prob) to Green (low fake prob)
            aspect='auto',
            vmin=0,
            vmax=1,
            extent=[timestamps[0], timestamps[-1], 0, 1]
        )

        ax_heatmap.set_xlabel("Time (seconds)", fontsize=11)
        ax_heatmap.set_yticks([])
        ax_heatmap.set_xlim([timestamps[0], timestamps[-1]])

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, orientation='horizontal', pad=0.15, fraction=0.05)
        cbar.set_label('P(Fake)', fontsize=10)

        # === Statistics text ===
        high_risk_count = (fake_probs > threshold).sum()
        high_risk_ratio = high_risk_count / len(fake_probs) * 100

        stats_text = (
            f"Statistics:\n"
            f"Suspicious frames: {high_risk_count}/{len(fake_probs)} ({high_risk_ratio:.1f}%)\n"
            f"Mean P(Fake): {stats['mean_fake_prob']:.3f}\n"
            f"Max P(Fake): {stats['max_fake_prob']:.3f}\n"
            f"Suspicious intervals: {len(suspicious_intervals)}"
        )

        fig.text(
            0.98, 0.02,
            stats_text,
            fontsize=10,
            ha='right',
            va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # Save
        output_path = Path(output_dir) / f"{Path(video_path).stem}_stage1_timeline.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved Stage1 timeline visualization to: {output_path}")

        return str(output_path)

    def visualize_stage2_interval(
        self,
        interval_xai: Dict,
        interval_id: int,
        video_path: str,
        output_dir: str
    ) -> str:
        """
        Visualize Stage2 XAI analysis for a single suspicious interval.

        Creates a comprehensive 4-panel visualization:
        - Panel 1 (Top-Left): Branch Contribution Bar Chart
        - Panel 2 (Top-Right): Phoneme Attention Distribution
        - Panel 3 (Bottom-Left): Temporal Attention Heatmap (14 phonemes × 5 frames)
        - Panel 4 (Bottom-Right): Detection Summary with Geometry Analysis

        Args:
            interval_xai: Stage2 XAI result for one interval (from run_stage2_interval_xai)
            interval_id: Interval index (0, 1, 2...)
            video_path: Original video path for title
            output_dir: Directory to save visualization

        Returns:
            Path to saved visualization PNG
        """
        logger.info(f"[Stage2] Generating XAI visualization for interval {interval_id}...")

        # Import PIAVisualizer
        from .pia_visualizer import PIAVisualizer

        # Initialize visualizer
        visualizer = PIAVisualizer(figsize=(16, 12), dpi=150)

        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Title with interval info
        interval_info = interval_xai['interval']
        video_name = Path(video_path).stem
        fig.suptitle(
            f"Stage2 XAI Analysis | Video: {video_name} | Interval {interval_id}: {interval_info['start_time']:.2f}s - {interval_info['end_time']:.2f}s",
            fontsize=14,
            fontweight='bold'
        )

        # Panel 1: Branch Contribution (Top-Left)
        ax1 = axes[0, 0]
        visualizer.plot_branch_contributions(
            interval_xai,
            ax=ax1,
            title=f"Branch Contribution (Interval {interval_id})"
        )

        # Panel 2: Phoneme Attention (Top-Right)
        ax2 = axes[0, 1]
        visualizer.plot_phoneme_attention(
            interval_xai,
            ax=ax2,
            title=f"Phoneme Attention Distribution"
        )

        # Panel 3: Temporal Heatmap (Bottom-Left)
        ax3 = axes[1, 0]
        visualizer.plot_temporal_heatmap(
            interval_xai,
            ax=ax3,
            title=f"Temporal Attention Heatmap (14×5)"
        )

        # Panel 4: Detection Summary (Bottom-Right)
        ax4 = axes[1, 1]
        visualizer.plot_detection_summary(
            interval_xai,
            ax=ax4,
            title=f"Detection Summary"
        )

        # Save visualization
        output_path = Path(output_dir) / f"{video_name}_interval_{interval_id}_xai.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved Stage2 XAI visualization to: {output_path}")

        return str(output_path)


class Stage2Analyzer:
    """
    Stage 2 Analyzer: PIA XAI Analysis Wrapper

    Wraps HybridMMSBAPIA.run_stage2_interval_xai() to provide
    clean interface matching hybrid_xai_interface.ts

    Reuses:
    - HybridMMSBAPIA.run_stage2_interval_xai() for PIA XAI analysis
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
        Initialize Stage2Analyzer with PIA model only.

        Args:
            pia_model_path: Path to PIA model checkpoint
            pia_config_path: PIA config file
            device: cuda or cpu
        """
        logger.info("Initializing Stage2Analyzer...")

        # Initialize HybridMMSBAPIA for Stage2 only
        # MMMS model not needed for Stage2, so use dummy path
        self.hybrid_pipeline = HybridMMSBAPIA(
            mmms_model_path="dummy",  # Not used in Stage2
            pia_model_path=pia_model_path,
            config_path=pia_config_path,
            pia_config_path=pia_config_path,
            device=device
        )

        logger.info("Stage2Analyzer initialized successfully!")

    def analyze_interval(
        self,
        interval: Dict,
        video_path: str,
        extracted_features: Dict,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run PIA XAI analysis on a single suspicious interval.

        Args:
            interval: Interval dict from Stage1Scanner with keys:
                - interval_id: int
                - start_frame: int
                - end_frame: int
                - start_time_sec: float
                - end_time_sec: float
                - duration_sec: float
                - frame_count: int
                - mean_fake_prob: float
                - severity: str
            video_path: Path to video file
            extracted_features: Dict with features from Stage1 (FIX: Use passed features)
                - frames: np.ndarray
                - audio: np.ndarray
                - lip: np.ndarray
                - fps: float
                - timestamps: np.ndarray
                - total_frames: int
                - video_path: str
            output_dir: Optional directory to save visualizations

        Returns:
            interval_xai: Dictionary matching hybrid_xai_interface.ts format:
                - interval_id: int
                - time_range: str (e.g., "8.2s-10.4s")
                - prediction: {verdict, confidence, probabilities}
                - branch_contributions: {visual, geometry, identity, top_branch, explanation}
                - phoneme_analysis: {phoneme_scores, top_phoneme, total_phonemes}
                - temporal_analysis: {heatmap_data, high_risk_points, statistics}
                - geometry_analysis: {statistics, phoneme_mar, anomalous_frames}
                - korean_explanation: {summary, key_findings, detailed_analysis}
                - visualization: {xai_plot_url}
        """
        logger.info(f"[Stage2Analyzer] Analyzing interval {interval['interval_id']}: "
                   f"{interval['start_time_sec']:.1f}s - {interval['end_time_sec']:.1f}s")

        # FIX: Use features from Stage1 instead of re-extracting
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

        # Call existing run_stage2_interval_xai method
        interval_xai_result = self.hybrid_pipeline.run_stage2_interval_xai(
            stage1_result=stage1_result,
            interval=stage2_interval,
            output_dir=output_dir
        )

        # Convert internal format to hybrid_xai_interface.ts format
        formatted_result = self._format_stage2_result(
            interval_xai=interval_xai_result,
            interval=interval,
            output_dir=output_dir
        )

        return formatted_result

    def _format_stage2_result(
        self,
        interval_xai: Dict,
        interval: Dict,
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        Convert internal PIA XAI result to hybrid_xai_interface.ts format.

        REFACTORED: Now delegates to hybrid_utils.format_stage2_to_interface()
        to reduce code duplication (saved ~175 lines).

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


class HybridXAIPipeline:
    """
    HybridXAIPipeline Orchestrator

    Coordinates the complete 2-stage deepfake detection pipeline:
    - Stage 1: MMMS-BA temporal scan (Stage1Scanner)
    - Stage 2: PIA XAI analysis per interval (Stage2Analyzer)
    - Aggregation: Combine insights across all intervals
    - Korean Summary: User-friendly explanation

    Output: HybridDeepfakeXAIResult matching hybrid_xai_interface.ts
    """

    def __init__(
        self,
        mmms_model_path: str,
        pia_model_path: str,
        mmms_config_path: str = "configs/train_teacher_korean.yaml",
        pia_config_path: str = "configs/train_pia.yaml",
        device: str = "cuda"
    ):
        """
        Initialize HybridXAIPipeline with Stage1 and Stage2 components.

        Args:
            mmms_model_path: Path to MMMS-BA checkpoint
            pia_model_path: Path to PIA checkpoint
            mmms_config_path: MMMS-BA config file
            pia_config_path: PIA config file
            device: cuda or cpu
        """
        logger.info("=" * 80)
        logger.info("Initializing HybridXAIPipeline Orchestrator...")
        logger.info("=" * 80)

        self.mmms_model_path = mmms_model_path
        self.pia_model_path = pia_model_path
        self.device = device

        # Stage 1: MMMS-BA Temporal Scanner
        logger.info("\n[1/2] Initializing Stage 1: MMMS-BA Temporal Scanner...")
        self.stage1 = Stage1Scanner(
            model_path=mmms_model_path,
            config_path=mmms_config_path,
            device=device
        )

        # Stage 2: PIA XAI Analyzer
        logger.info("\n[2/2] Initializing Stage 2: PIA XAI Analyzer...")
        self.stage2 = Stage2Analyzer(
            pia_model_path=pia_model_path,
            pia_config_path=pia_config_path,
            device=device
        )

        logger.info("\n" + "=" * 80)
        logger.info("HybridXAIPipeline Orchestrator Initialized Successfully!")
        logger.info("=" * 80 + "\n")

    def process_video(
        self,
        video_path: str,
        video_id: str = None,
        output_dir: str = None,
        threshold: float = 0.6,
        min_interval_frames: int = 14,
        save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Full E2E pipeline: Stage1 → Stage2 → Aggregation → Korean Summary.

        Args:
            video_path: Path to raw video file
            video_id: Video identifier (default: extracted from filename)
            output_dir: Base output directory (default: outputs/xai/hybrid/{video_id})
            threshold: Suspicious frame threshold (0.0-1.0)
            min_interval_frames: Minimum frames per interval (PIA needs 14+)
            save_visualizations: Whether to save visualization plots

        Returns:
            HybridDeepfakeXAIResult: Dict matching hybrid_xai_interface.ts
        """
        import time
        import uuid
        from datetime import datetime
        import cv2

        start_time = time.time()

        # Extract video_id from filename if not provided
        if video_id is None:
            video_id = Path(video_path).stem

        # Setup output directory
        if output_dir is None:
            output_dir = f"outputs/xai/hybrid/{video_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info(f"Processing Video: {video_path}")
        logger.info(f"Video ID: {video_id}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info(f"Threshold: {threshold}")
        logger.info("=" * 80 + "\n")

        # ========================================
        # Step 1: Stage 1 - MMMS-BA Temporal Scan
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Running Stage 1 - MMMS-BA Temporal Scan")
        logger.info("=" * 80)

        stage1_output_dir = output_dir if save_visualizations else None
        stage1_timeline = self.stage1.scan_video(
            video_path=video_path,
            threshold=threshold,
            min_interval_frames=min_interval_frames,
            merge_gap_sec=1.0,
            output_dir=stage1_output_dir
        )

        logger.info(f"\n[Stage1 Complete]")
        logger.info(f"  Total frames analyzed: {len(stage1_timeline['frame_probabilities'])}")
        logger.info(f"  Suspicious intervals found: {len(stage1_timeline['suspicious_intervals'])}")
        logger.info(f"  Mean fake probability: {stage1_timeline['statistics']['mean_fake_prob']:.3f}")

        # ========================================
        # Step 2: Stage 2 - PIA XAI Analysis (per interval)
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Running Stage 2 - PIA XAI Analysis on Suspicious Intervals")
        logger.info("=" * 80)

        stage2_interval_analysis = []

        if len(stage1_timeline['suspicious_intervals']) == 0:
            logger.warning("No suspicious intervals detected by Stage1. Skipping Stage2 analysis.")
        else:
            # FIX: Get features from Stage1 to pass to Stage2
            extracted_features = stage1_timeline.get('extracted_features', {})

            for interval in stage1_timeline['suspicious_intervals']:
                logger.info(f"\n  Analyzing Interval {interval['interval_id']}: "
                           f"{interval['start_time_sec']:.1f}s - {interval['end_time_sec']:.1f}s "
                           f"({interval['frame_count']} frames)")

                stage2_output_dir = output_dir if save_visualizations else None
                interval_xai = self.stage2.analyze_interval(
                    interval=interval,
                    video_path=video_path,
                    extracted_features=extracted_features,  # FIX: Pass features from Stage1
                    output_dir=stage2_output_dir
                )

                stage2_interval_analysis.append(interval_xai)

                logger.info(f"    Prediction: {interval_xai['prediction']['verdict'].upper()} "
                           f"({interval_xai['prediction']['confidence']:.1%})")
                logger.info(f"    Top branch: {interval_xai['branch_contributions']['top_branch']}")
                logger.info(f"    Top phoneme: {interval_xai['phoneme_analysis']['top_phoneme']}")

        logger.info(f"\n[Stage2 Complete] Analyzed {len(stage2_interval_analysis)} intervals")

        # ========================================
        # Step 3: Aggregate Insights
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Aggregating Insights Across All Intervals")
        logger.info("=" * 80)

        aggregated_insights = self._aggregate_insights(stage2_interval_analysis)

        logger.info(f"\n[Aggregation Complete]")
        logger.info(f"  Top suspicious phonemes: {[p['phoneme'] for p in aggregated_insights['top_suspicious_phonemes'][:3]]}")
        logger.info(f"  Most dominant branch: {aggregated_insights['branch_trends']['most_dominant']}")
        logger.info(f"  Intervals with MAR anomalies: {aggregated_insights['mar_summary']['intervals_with_anomalies']}")

        # ========================================
        # Step 4: Compute Overall Detection
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Computing Overall Detection Verdict")
        logger.info("=" * 80)

        detection = self._compute_overall_detection(stage1_timeline, stage2_interval_analysis)

        logger.info(f"\n[Overall Detection]")
        logger.info(f"  Verdict: {detection['verdict'].upper()}")
        logger.info(f"  Confidence: {detection['confidence']:.1%}")
        logger.info(f"  Suspicious frames: {detection['suspicious_frame_count']}/{len(stage1_timeline['frame_probabilities'])} "
                   f"({detection['suspicious_frame_ratio']:.1f}%)")

        # ========================================
        # Step 5: Generate Korean Summary
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Generating Korean Summary")
        logger.info("=" * 80)

        summary = self._generate_korean_summary(
            detection=detection,
            aggregated=aggregated_insights,
            stage1_timeline=stage1_timeline
        )

        logger.info(f"\n[Korean Summary]")
        logger.info(f"  Title: {summary['title']}")
        logger.info(f"  Risk Level: {summary['risk_level'].upper()}")
        logger.info(f"  Primary Reason: {summary['primary_reason']}")

        # ========================================
        # Step 6: Assemble Final Result
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Assembling Final HybridDeepfakeXAIResult")
        logger.info("=" * 80)

        processing_time_ms = (time.time() - start_time) * 1000

        # Extract video metadata
        video_info = self._extract_video_info(video_path)

        # Build final result
        result = self._build_final_result(
            video_path=video_path,
            video_id=video_id,
            output_dir=output_dir,
            stage1_timeline=stage1_timeline,
            stage2_interval_analysis=stage2_interval_analysis,
            aggregated_insights=aggregated_insights,
            detection=detection,
            summary=summary,
            video_info=video_info,
            processing_time_ms=processing_time_ms
        )

        # Save JSON result
        json_path = Path(output_dir) / "result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(convert_for_json(result), f, indent=2, ensure_ascii=False)

        logger.info(f"\n[Pipeline Complete]")
        logger.info(f"  Processing time: {processing_time_ms:.0f} ms")
        logger.info(f"  Result saved to: {json_path}")
        logger.info("=" * 80 + "\n")

        return result

    def _aggregate_insights(self, stage2_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate insights across all Stage2 intervals.

        Args:
            stage2_results: List of Stage2 interval analysis results

        Returns:
            aggregated_insights: Dict with top_suspicious_phonemes, branch_trends, mar_summary
        """
        if len(stage2_results) == 0:
            return {
                'top_suspicious_phonemes': [],
                'branch_trends': {
                    'visual_avg': 0.0,
                    'geometry_avg': 0.0,
                    'identity_avg': 0.0,
                    'most_dominant': 'visual'
                },
                'mar_summary': {
                    'intervals_with_anomalies': 0,
                    'total_anomalous_frames': 0,
                    'avg_deviation_percent': 0.0
                }
            }

        # ========================================
        # 1. Top Suspicious Phonemes (across all intervals)
        # ========================================
        phoneme_aggregation = {}  # {phoneme: {'attentions': [], 'intervals': []}}

        for idx, result in enumerate(stage2_results):
            interval_id = result['interval_id']
            for phoneme_score in result['phoneme_analysis']['phoneme_scores']:
                phoneme = phoneme_score['phoneme_korean']
                attention = phoneme_score['attention_weight']

                if phoneme not in phoneme_aggregation:
                    phoneme_aggregation[phoneme] = {'attentions': [], 'intervals': []}

                phoneme_aggregation[phoneme]['attentions'].append(attention)
                if interval_id not in phoneme_aggregation[phoneme]['intervals']:
                    phoneme_aggregation[phoneme]['intervals'].append(interval_id)

        # Calculate average attention and sort
        top_suspicious_phonemes = []
        for phoneme, data in phoneme_aggregation.items():
            avg_attention = np.mean(data['attentions'])
            appearance_count = len(data['intervals'])
            top_suspicious_phonemes.append({
                'phoneme': phoneme,
                'avg_attention': float(avg_attention),
                'appearance_count': appearance_count,
                'intervals': data['intervals']
            })

        # Sort by average attention (descending)
        top_suspicious_phonemes.sort(key=lambda x: x['avg_attention'], reverse=True)

        # ========================================
        # 2. Branch Contribution Trends
        # ========================================
        visual_contributions = []
        geometry_contributions = []
        identity_contributions = []

        for result in stage2_results:
            bc = result['branch_contributions']
            visual_contributions.append(bc['visual']['contribution_percent'])
            geometry_contributions.append(bc['geometry']['contribution_percent'])
            identity_contributions.append(bc['identity']['contribution_percent'])

        visual_avg = float(np.mean(visual_contributions))
        geometry_avg = float(np.mean(geometry_contributions))
        identity_avg = float(np.mean(identity_contributions))

        # Determine most dominant branch
        avg_contributions = {
            'visual': visual_avg,
            'geometry': geometry_avg,
            'identity': identity_avg
        }
        most_dominant = max(avg_contributions, key=avg_contributions.get)

        branch_trends = {
            'visual_avg': visual_avg,
            'geometry_avg': geometry_avg,
            'identity_avg': identity_avg,
            'most_dominant': most_dominant
        }

        # ========================================
        # 3. MAR Anomaly Summary
        # ========================================
        intervals_with_anomalies = 0
        total_anomalous_frames = 0
        all_deviations = []

        for result in stage2_results:
            anomalous_frames = result['geometry_analysis']['anomalous_frames']

            if len(anomalous_frames) > 0:
                intervals_with_anomalies += 1
                total_anomalous_frames += len(anomalous_frames)

                for frame in anomalous_frames:
                    all_deviations.append(frame['deviation_percent'])

        avg_deviation_percent = float(np.mean(all_deviations)) if all_deviations else 0.0

        mar_summary = {
            'intervals_with_anomalies': intervals_with_anomalies,
            'total_anomalous_frames': total_anomalous_frames,
            'avg_deviation_percent': avg_deviation_percent
        }

        return {
            'top_suspicious_phonemes': top_suspicious_phonemes,
            'branch_trends': branch_trends,
            'mar_summary': mar_summary
        }

    def _compute_overall_detection(
        self,
        stage1_timeline: Dict[str, Any],
        stage2_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute overall detection verdict combining Stage1 and Stage2 evidence.

        Args:
            stage1_timeline: Stage1 timeline result
            stage2_results: List of Stage2 interval results

        Returns:
            detection: Dict with verdict, confidence, probabilities, frame counts
        """
        # Method 1: Stage1 suspicious frame ratio
        total_frames = len(stage1_timeline['frame_probabilities'])
        suspicious_frames = [fp for fp in stage1_timeline['frame_probabilities'] if fp['is_suspicious']]
        suspicious_frame_count = len(suspicious_frames)
        suspicious_frame_ratio = (suspicious_frame_count / total_frames) * 100 if total_frames > 0 else 0.0

        # Calculate suspicious ratio as 0.0-1.0
        suspicious_ratio_score = suspicious_frame_count / total_frames if total_frames > 0 else 0.0

        # Method 2: Stage2 PIA confidence average (if available)
        if len(stage2_results) > 0:
            pia_fake_confidences = [r['prediction']['probabilities']['fake'] for r in stage2_results]
            pia_avg_confidence = float(np.mean(pia_fake_confidences))
        else:
            # No intervals analyzed - use Stage1 mean fake probability
            pia_avg_confidence = stage1_timeline['statistics']['mean_fake_prob']

        # Combined score (weighted combination)
        # 60% weight on Stage1 suspicious ratio, 40% on Stage2 PIA confidence
        combined_score = 0.6 * suspicious_ratio_score + 0.4 * pia_avg_confidence

        # Final verdict (threshold: 0.5)
        verdict = 'fake' if combined_score > 0.5 else 'real'

        # Probabilities
        probabilities = {
            'real': float(1.0 - combined_score),
            'fake': float(combined_score)
        }

        return {
            'verdict': verdict,
            'confidence': float(combined_score),
            'probabilities': probabilities,
            'suspicious_frame_count': suspicious_frame_count,
            'suspicious_frame_ratio': float(suspicious_frame_ratio)
        }

    def _generate_korean_summary(
        self,
        detection: Dict[str, Any],
        aggregated: Dict[str, Any],
        stage1_timeline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate user-friendly Korean summary.

        Args:
            detection: Overall detection result
            aggregated: Aggregated insights
            stage1_timeline: Stage1 timeline

        Returns:
            summary: Dict with title, risk_level, primary_reason, etc.
        """
        confidence = detection['confidence']
        verdict = detection['verdict']

        # Risk level mapping
        risk_level = self._compute_risk_level(confidence)

        # Title
        if verdict == 'fake':
            title = f"⚠️ 딥페이크 영상 의심됨 (신뢰도: {confidence*100:.1f}%)"
        else:
            title = f"✅ 진짜 영상으로 판정 (신뢰도: {(1-confidence)*100:.1f}%)"

        # Primary reason
        suspicious_intervals = stage1_timeline.get('suspicious_intervals', [])
        top_phonemes = aggregated.get('top_suspicious_phonemes', [])

        if verdict == 'fake' and len(suspicious_intervals) > 0 and len(top_phonemes) > 0:
            # Get first interval time range
            first_interval = suspicious_intervals[0]
            start_time = first_interval['start_time_sec']
            end_time = first_interval['end_time_sec']

            # Get top phoneme
            top_phoneme = top_phonemes[0]['phoneme']

            primary_reason = f"{start_time:.1f}초~{end_time:.1f}초 구간에서 '{top_phoneme}' 발음 시 입 움직임이 부자연스럽습니다"
        elif verdict == 'fake' and len(suspicious_intervals) > 0:
            primary_reason = f"영상 전체에서 {len(suspicious_intervals)}개 구간에 부자연스러운 음성-입모양 불일치가 감지되었습니다"
        elif verdict == 'real':
            primary_reason = "전체 영상에서 부자연스러운 음성-입모양 불일치가 감지되지 않았습니다"
        else:
            primary_reason = "분석 결과를 기반으로 진짜 영상으로 판정되었습니다"

        # Detailed explanation
        detailed_explanation = self._build_detailed_explanation(
            verdict=verdict,
            detection=detection,
            aggregated=aggregated,
            suspicious_intervals=suspicious_intervals
        )

        return {
            'title': title,
            'risk_level': risk_level,
            'primary_reason': primary_reason,
            'suspicious_interval_count': len(suspicious_intervals),
            'top_suspicious_phonemes': [p['phoneme'] for p in top_phonemes[:3]],
            'detailed_explanation': detailed_explanation
        }

    def _compute_risk_level(self, confidence: float) -> str:
        """Compute risk level from confidence score."""
        if confidence > 0.9:
            return 'critical'
        elif confidence > 0.8:
            return 'high'
        elif confidence > 0.65:
            return 'medium'
        else:
            return 'low'

    def _build_detailed_explanation(
        self,
        verdict: str,
        detection: Dict[str, Any],
        aggregated: Dict[str, Any],
        suspicious_intervals: List[Dict[str, Any]]
    ) -> str:
        """Build detailed Korean explanation (under 200 characters)."""
        if verdict == 'fake':
            branch_trends = aggregated['branch_trends']
            mar_summary = aggregated['mar_summary']

            # Build explanation
            parts = []
            parts.append(f"전체 영상 중 {detection['suspicious_frame_ratio']:.1f}%가 의심스러운 것으로 판정되었습니다.")
            parts.append(f"{branch_trends['most_dominant']} 특징이 가장 두드러지게 나타났습니다.")

            if mar_summary['intervals_with_anomalies'] > 0:
                parts.append(f"{mar_summary['intervals_with_anomalies']}개 구간에서 입 움직임 비정상 패턴이 발견되었습니다.")

            explanation = " ".join(parts)
        else:
            explanation = f"전체 {len(suspicious_intervals)}개 구간을 분석한 결과, 부자연스러운 음성-입모양 불일치 패턴이 감지되지 않았습니다. 신뢰도 {(1-detection['confidence'])*100:.1f}%로 진짜 영상으로 판정되었습니다."

        # Truncate to 200 characters if needed
        if len(explanation) > 200:
            explanation = explanation[:197] + "..."

        return explanation

    def _extract_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        cap.release()

        return {
            'duration_sec': float(duration_sec),
            'total_frames': total_frames,
            'fps': float(fps),
            'resolution': f"{width}x{height}",
            'original_path': str(video_path)
        }

    def _build_final_result(
        self,
        video_path: str,
        video_id: str,
        output_dir: str,
        stage1_timeline: Dict[str, Any],
        stage2_interval_analysis: List[Dict[str, Any]],
        aggregated_insights: Dict[str, Any],
        detection: Dict[str, Any],
        summary: Dict[str, Any],
        video_info: Dict[str, Any],
        processing_time_ms: float
    ) -> Dict[str, Any]:
        """
        Assemble final HybridDeepfakeXAIResult matching hybrid_xai_interface.ts.

        Returns:
            Complete result with all 9 sections
        """
        import uuid
        from datetime import datetime

        # Generate request ID
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        # Metadata
        metadata = {
            'video_id': video_id,
            'request_id': request_id,
            'processed_at': datetime.utcnow().isoformat() + 'Z',
            'processing_time_ms': processing_time_ms,
            'pipeline_version': 'hybrid_v1.0'
        }

        # Model info
        model_info = {
            'mmms_ba': {
                'name': 'MMMS-BA Teacher',
                'checkpoint': self.mmms_model_path,
                'training_accuracy': 0.9771,
                'stage': 'stage1_scanner'
            },
            'pia': {
                'name': 'PIA v1.0',
                'checkpoint': self.pia_model_path,
                'training_accuracy': 0.9771,
                'stage': 'stage2_analyzer',
                'supported_phonemes': 14,
                'xai_methods': ['branch_contribution', 'phoneme_attention', 'temporal_heatmap', 'mar_analysis']
            },
            'pipeline': {
                'version': 'hybrid_v1.0',
                'threshold': 0.6,
                'resampling_strategy': 'uniform'
            }
        }

        # Output file paths
        outputs = {
            'stage1_timeline': str(Path(output_dir) / "stage1_timeline.png"),
            'stage2_intervals': [
                str(Path(output_dir) / f"interval_{i}_xai.png")
                for i in range(len(stage2_interval_analysis))
            ],
            'combined_json': str(Path(output_dir) / "result.json"),
            'combined_summary': str(Path(output_dir) / "summary.png")
        }

        # Assemble complete result
        return {
            'metadata': metadata,
            'video_info': video_info,
            'detection': detection,
            'summary': summary,
            'stage1_timeline': stage1_timeline,
            'stage2_interval_analysis': stage2_interval_analysis,
            'aggregated_insights': aggregated_insights,
            'model_info': model_info,
            'outputs': outputs
        }


def main():
    """Example usage of the hybrid pipeline for inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid MMMS-BA + PIA XAI Pipeline (Inference)")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--mmms-model', type=str, required=True, help='MMMS-BA model checkpoint')
    parser.add_argument('--pia-model', type=str, required=True, help='PIA model checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/hybrid_xai', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Suspicious frame threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = HybridXAIPipeline(
        mmms_model_path=args.mmms_model,
        pia_model_path=args.pia_model,
        device=args.device
    )

    # Run pipeline on video
    result = pipeline.process_video(
        video_path=args.video,
        threshold=args.threshold,
        output_dir=args.output_dir
    )

    print(f"\nPipeline complete! Results saved to {args.output_dir}")
    print(f"Overall verdict: {result['detection']['verdict'].upper()}")
    print(f"Confidence: {result['detection']['confidence']:.2%}")


if __name__ == "__main__":
    main()