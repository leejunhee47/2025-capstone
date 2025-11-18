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
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.pia_model import PIAModel
from src.utils.config import load_config
from .pia_explainer import PIAExplainer
from .pia_visualizer import PIAVisualizer
from src.data.preprocessing import match_phoneme_to_frames
from src.utils.hybrid_phoneme_aligner_v2 import HybridPhonemeAligner
from src.utils.enhanced_mar_extractor import EnhancedMARExtractor
from src.utils.arcface_extractor import ArcFaceExtractor
from .hybrid_utils import (
    get_interval_phoneme_dict,
    resample_frames_to_pia_format,
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

        # Load Korean phoneme vocabulary
        with open("configs/phoneme_vocab.json", "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            vocab_dict = vocab_data["vocab"]
            self.phoneme_vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda x: x[1])]

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
            output_dir=output_dir
        )

        # Convert internal format to hybrid_xai_interface.ts format
        formatted_result = self._format_stage2_result(
            interval_xai=interval_xai_result,
            interval=interval,
            output_dir=output_dir
        )

        return formatted_result

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
            interval: Interval dict with start_frame, end_frame, etc.
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
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_video_path = tmp.name
            # Create temp video from interval frames
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

        # Save visualization if requested
        if output_dir and self.pia_visualizer is not None:
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
        from src.utils.korean_phoneme_config import get_phoneme_vocab, is_kept_phoneme

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
        self.pia_visualizer.plot_detection_summary(
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

