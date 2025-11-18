"""
Stage 1 Scanner Module

MMMS-BA temporal scan for suspicious frame detection.
Uses FeatureExtractor for preprocessing and MMMS-BA model for prediction.

Author: Claude
Date: 2025-11-17
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..models.teacher import MMMSBA
from ..utils.config import load_config
from .feature_extractor import FeatureExtractor
from .hybrid_utils import format_stage1_to_interface, group_consecutive_frames

# Setup logger
logger = logging.getLogger(__name__)


class Stage1Scanner:
    """
    Stage 1 Scanner: MMMS-BA Temporal Scan

    Performs frame-level fake probability prediction using MMMS-BA model
    and identifies suspicious frames/intervals.

    Reuses:
    - FeatureExtractor for video preprocessing
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
        self.config = load_config(config_path)

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Load MMMS-BA model
        if model_path != "dummy":
            logger.info("Loading MMMS-BA model...")
            self.mmms_model = self._load_mmms_model(model_path)
        else:
            self.mmms_model = None
            logger.info("MMMS-BA model skipped (dummy path)")

        logger.info("Stage1Scanner initialized successfully!")

    def _load_mmms_model(self, mmms_model_path: str) -> nn.Module:
        """Load MMMS-BA model from checkpoint."""
        # Use config values where available, otherwise use hardcoded defaults
        model = MMMSBA(
            audio_dim=self.config.get('dataset', {}).get('audio', {}).get('n_mfcc', 40),
            visual_dim=256,  # Hardcoded from ResNet feature extractor
            lip_dim=128,     # Hardcoded from lip ROI feature extractor
            gru_hidden_dim=self.config.get('model', {}).get('gru', {}).get('hidden_size', 300),
            dense_hidden_dim=self.config.get('model', {}).get('dense', {}).get('hidden_size', 100)
        ).to(self.device)

        checkpoint = torch.load(mmms_model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

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
                    'visualization': {'timeline_plot_url': ...},
                    'extracted_features': {...}
                }
        """
        logger.info(f"[Stage1Scanner] Scanning video: {video_path}")

        # Step 1: Extract features
        features = self.feature_extractor.extract_features(video_path, target_fps=10.0)

        # Step 2: Run MMMS-BA frame-level prediction
        logger.info(f"  Running frame-level prediction on {features['total_frames']} frames...")
        fake_probs = self._predict_frame_level(
            features['frames'],
            features['audio'],
            features['lip']
        )

        # Step 3: Identify suspicious frames
        suspicious_indices = np.where(fake_probs > threshold)[0]
        suspicious_ratio = len(suspicious_indices) / features['total_frames'] * 100

        # Overall verdict
        mean_fake_prob = np.mean(fake_probs)
        overall_verdict = "fake" if mean_fake_prob > 0.5 else "real"
        overall_confidence = mean_fake_prob if overall_verdict == "fake" else 1 - mean_fake_prob

        logger.info(f"  Stage 1 Results:")
        logger.info(f"    - Total frames: {features['total_frames']}")
        logger.info(f"    - Suspicious frames: {len(suspicious_indices)} ({suspicious_ratio:.1f}%)")
        logger.info(f"    - Overall verdict: {overall_verdict.upper()} (confidence: {overall_confidence:.2%})")

        # Build stage1_result dict
        stage1_result = {
            'fake_probs': fake_probs,
            'frames': features['frames'],
            'audio': features['audio'],
            'lip': features['lip'],
            'timestamps': features['timestamps'],
            'fps': features['fps'],
            'total_frames': features['total_frames'],
            'suspicious_indices': suspicious_indices,
            'suspicious_ratio': suspicious_ratio,
            'overall_verdict': overall_verdict,
            'overall_confidence': overall_confidence,
            'video_path': features['video_path'],
            'video_id': features['video_id']
        }

        # Step 4: Group consecutive frames into intervals
        suspicious_intervals_raw = group_consecutive_frames(
            suspicious_indices=suspicious_indices,
            fps=features['fps'],
            min_interval_frames=min_interval_frames,
            merge_gap_sec=merge_gap_sec
        )

        # Step 5: Convert to TypeScript interface format
        stage1_timeline = self._format_stage1_timeline(
            stage1_result=stage1_result,
            suspicious_intervals_raw=suspicious_intervals_raw,
            threshold=threshold
        )

        # Step 6: Store extracted features for Stage2 (before visualization)
        stage1_timeline['extracted_features'] = {
            'frames': stage1_result['frames'],
            'audio': stage1_result['audio'],
            'lip': stage1_result['lip'],
            'fps': stage1_result['fps'],
            'total_frames': stage1_result['total_frames'],
            'timestamps': stage1_result['timestamps'],
            'video_path': stage1_result['video_path']
        }

        # Step 7: Generate visualization if requested (after extracted_features is added)
        if output_dir:
            viz_path = self.visualize_stage1_timeline(
                stage1_timeline=stage1_timeline,
                video_path=video_path,
                output_dir=output_dir
            )
            stage1_timeline['visualization']['timeline_plot_url'] = viz_path

        logger.info(f"[Stage1Scanner] Scan complete!")
        logger.info(f"  Total frames: {len(stage1_timeline['frame_probabilities'])}")
        logger.info(f"  Suspicious intervals: {len(stage1_timeline['suspicious_intervals'])}")

        return stage1_timeline

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
        if self.mmms_model is None:
            raise RuntimeError("MMMS-BA model not loaded. Cannot perform prediction.")

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
            stage1_result: Output from scan_video()
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
        # Extract data
        frame_probs = stage1_timeline['frame_probabilities']
        suspicious_intervals = stage1_timeline['suspicious_intervals']
        stats = stage1_timeline['statistics']

        timestamps = np.array([fp['timestamp_sec'] for fp in frame_probs])
        fake_probs = np.array([fp['fake_probability'] for fp in frame_probs])
        threshold = stats['threshold_used']

        # Get frames from extracted_features in stage1_timeline
        extracted_features = stage1_timeline.get('extracted_features', {})
        if 'frames' not in extracted_features:
            raise ValueError("extracted_features not found in stage1_timeline. Make sure scan_video() was called first.")
        frames = extracted_features['frames']  # (T, H, W, 3) numpy array [0, 1]
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

