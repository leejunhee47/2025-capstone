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
        # Use config values - matches evaluate_mmms_ba_test.py
        model_config = self.config.get('model', {})

        model = MMMSBA(
            audio_dim=self.config.get('dataset', {}).get('audio', {}).get('n_mfcc', 40),
            visual_dim=256,  # From feature extractor
            lip_dim=128,     # From feature extractor
            gru_hidden_dim=model_config.get('gru', {}).get('hidden_size', 300),
            gru_num_layers=model_config.get('gru', {}).get('num_layers', 3),
            gru_dropout=model_config.get('gru', {}).get('dropout', 0.3),
            dense_hidden_dim=model_config.get('dense', {}).get('hidden_size', 100),
            dense_dropout=model_config.get('dense', {}).get('dropout', 0.5),
            attention_type=model_config.get('attention', {}).get('type', 'bimodal'),
            num_classes=model_config.get('num_classes', 2)
        ).to(self.device)

        checkpoint = torch.load(mmms_model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def _load_from_npz(
        self,
        npz_path: str,
        target_fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Load preprocessed features from npz file.

        Args:
            npz_path: Path to preprocessed .npz file
            target_fps: Target FPS for downsampling (default: 10.0)

        Returns:
            features: Dict containing:
                - 'frames': np.ndarray (T, 224, 224, 3) [0, 1]
                - 'audio': np.ndarray (T_audio, 40)
                - 'lip': np.ndarray (T, 112, 112, 3) [0, 1]
                - 'geometry': np.ndarray (T, 1) - MAR features
                - 'arcface': np.ndarray (T, 512) - ArcFace embeddings
                - 'phoneme_labels': np.ndarray (T,) - Phoneme labels (object array)
                - 'timestamps': np.ndarray (T,)
                - 'fps': float
                - 'total_frames': int
                - 'video_id': str
                - 'video_path': str (npz_path)
        """
        logger.info(f"Loading preprocessed features from: {npz_path}")

        # Load npz file
        data = np.load(npz_path, allow_pickle=True)  # allow_pickle for phoneme_labels (object array)

        # Extract arrays
        frames = data['frames']  # (50, 224, 224, 3) uint8 [0, 255]
        audio = data['audio']    # (3125, 40) float32
        lip = data['lip']        # (50, 112, 112, 3) uint8 [0, 255]
        geometry = data['geometry']  # (50, 1) float32 - MAR features
        arcface = data['arcface']    # (50, 512) float32 - ArcFace embeddings
        phoneme_labels = data['phoneme_labels']  # (50,) object - Phoneme labels
        video_id = str(data['video_id'])

        # Load timestamps from npz if available, otherwise compute
        if 'timestamps' in data:
            timestamps_from_npz = data['timestamps']
            original_fps = 30.0  # Default fps for preprocessed data
        else:
            # Fallback: compute timestamps
            original_fps = 30.0
            timestamps_from_npz = np.arange(frames.shape[0], dtype=np.float32) / original_fps

        # Normalize frames and lip to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        lip = lip.astype(np.float32) / 255.0

        # Note: lip is (112, 112) - mmms-ba_fulldata_best.pth is trained on this size

        original_total_frames = frames.shape[0]
        timestamps = timestamps_from_npz
        total_frames = original_total_frames
        fps = original_fps

        # Downsample to target FPS if needed
        if target_fps < fps:
            downsample_step = max(1, int(fps / target_fps))
            logger.info(f"  Downsampling from {fps:.1f}fps to {target_fps}fps (step={downsample_step})")

            # Select frames at uniform intervals
            selected_indices = np.arange(0, original_total_frames, downsample_step)
            frames = frames[selected_indices]
            lip = lip[selected_indices]
            geometry = geometry[selected_indices]  # Downsample geometry
            arcface = arcface[selected_indices]    # Downsample arcface
            phoneme_labels = phoneme_labels[selected_indices]  # Downsample phoneme_labels
            timestamps = timestamps[selected_indices]  # Keep original timestamps!
            # Audio stays the same (covers full timeline)

            total_frames = frames.shape[0]
            fps = target_fps

            logger.info(f"  Downsampled to {total_frames} frames")

        logger.info(f"  Loaded {total_frames} frames from npz (with geometry/arcface/phoneme features)")

        return {
            'frames': frames,
            'audio': audio,
            'lip': lip,
            'geometry': geometry,        # NEW: MAR features
            'arcface': arcface,          # NEW: ArcFace embeddings
            'phoneme_labels': phoneme_labels,  # NEW: Phoneme labels
            'timestamps': timestamps,
            'fps': fps,
            'total_frames': total_frames,
            'video_id': video_id,
            'video_path': npz_path
        }

    def scan_video(
        self,
        video_path: str,
        threshold: float = 0.6,
        min_interval_frames: int = 14,
        merge_gap_sec: float = 1.0,
        output_dir: Optional[str] = None,
        use_preprocessed: bool = None,
        target_fps: float = 10.0
    ) -> Dict[str, Any]:
        """
        Run MMMS-BA temporal scan on video or preprocessed npz file.

        Args:
            video_path: Path to video file (.mp4, .avi) OR preprocessed .npz file
            threshold: Fake probability threshold for suspicious frames
            min_interval_frames: Minimum frames per interval (PIA needs 14+)
            merge_gap_sec: Gap in seconds to merge nearby intervals
            output_dir: Optional directory to save visualization
            use_preprocessed: Whether to load from preprocessed npz file.
                             If None, auto-detects based on file extension (.npz).
            target_fps: Target FPS for downsampling raw video (default: 10.0)
                       Only applies to raw video mode (use_preprocessed=False)

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

        # Auto-detect preprocessed mode based on file extension
        if use_preprocessed is None:
            use_preprocessed = video_path.endswith('.npz')

        # Step 1: Extract features (from npz or raw video)
        if use_preprocessed:
            logger.info("  Using preprocessed npz file (fast mode)")
            features = self._load_from_npz(video_path, target_fps=10.0)
        else:
            logger.info("  Extracting features from raw video (slow mode)")
            features = self.feature_extractor.extract_features(video_path, target_fps=target_fps)

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

        # Add geometry/arcface/phoneme_labels if available (from preprocessed npz)
        if 'geometry' in features:
            stage1_result['geometry'] = features['geometry']
        if 'arcface' in features:
            stage1_result['arcface'] = features['arcface']
        if 'phoneme_labels' in features:
            stage1_result['phoneme_labels'] = features['phoneme_labels']

        # Step 4: Group consecutive frames into intervals
        suspicious_intervals_raw = group_consecutive_frames(
            suspicious_indices=suspicious_indices,
            fps=features['fps'],
            min_interval_frames=min_interval_frames,
            merge_gap_sec=merge_gap_sec,
            timestamps=features['timestamps']  # FIX: Use actual timestamps for accurate time calculation
        )

        # Step 4.5: Full feature extraction for raw video (always for accurate phoneme alignment)
        num_intervals = len(suspicious_intervals_raw)
        has_precomputed = (
            'geometry' in stage1_result and
            'arcface' in stage1_result and
            'phoneme_labels' in stage1_result
        )

        if not has_precomputed and not use_preprocessed:
            # Raw video mode → Extract geometry/arcface only (MMMS-BA doesn't need phonemes)
            # Stage2 will extract phonemes per interval with 30fps optimization
            logger.info(f"  [RAW VIDEO MODE] Extracting geometry/arcface for full video...")
            logger.info(f"  [OPTIMIZATION] Skipping phoneme extraction (MMMS-BA doesn't need it)")

            try:
                geometry, arcface, phoneme_labels = self._extract_full_features(
                    frames=stage1_result['frames'],
                    video_path=video_path,
                    timestamps=stage1_result['timestamps'],
                    fps=stage1_result['fps'],
                    skip_phonemes=True  # 🔑 MMMS-BA doesn't need phonemes (~10s saved)
                )

                # Add to stage1_result for Stage2 reuse
                stage1_result['geometry'] = geometry
                stage1_result['arcface'] = arcface
                # Note: phoneme_labels is None when skip_phonemes=True
                # Stage2 will extract phonemes per interval with 30fps optimization

                logger.info(f"  ✓ Geometry/ArcFace extracted → Stage2 will extract phonemes per interval")
            except Exception as e:
                logger.warning(f"  Full feature extraction failed: {e}")
                logger.warning(f"  Stage2 will extract features per interval")
        elif has_precomputed:
            logger.info(f"  [NPZ MODE] Precomputed features available → Stage2 will slice by interval")
        else:
            logger.info(f"  WARNING: No features available - Stage2 will extract per interval")

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
            'video_path': stage1_result['video_path'],
            'fake_probs': stage1_result['fake_probs'],  # For visualization only (not in JSON output)
            'threshold': threshold  # For visualization
        }

        # Add geometry/arcface/phoneme_labels if available (for Stage2 reuse)
        if 'geometry' in stage1_result:
            stage1_timeline['extracted_features']['geometry'] = stage1_result['geometry']
        if 'arcface' in stage1_result:
            stage1_timeline['extracted_features']['arcface'] = stage1_result['arcface']
        if 'phoneme_labels' in stage1_result:
            stage1_timeline['extracted_features']['phoneme_labels'] = stage1_result['phoneme_labels']

        # Step 7: Generate visualization if requested (after extracted_features is added)
        if output_dir:
            viz_path = self.visualize_stage1_timeline(
                stage1_timeline=stage1_timeline,
                video_path=video_path,
                output_dir=output_dir
            )
            stage1_timeline['visualization']['timeline_plot_url'] = viz_path

        logger.info(f"[Stage1Scanner] Scan complete!")
        logger.info(f"  Total frames: {stage1_timeline['statistics']['total_frames']}")
        logger.info(f"  Suspicious intervals: {len(stage1_timeline['suspicious_intervals'])}")

        return stage1_timeline

    def _extract_full_features(
        self,
        frames: np.ndarray,
        video_path: str,
        timestamps: np.ndarray,
        fps: float,
        skip_phonemes: bool = False
    ) -> tuple:
        """
        Extract geometry/arcface/phoneme for all frames.

        Called when many suspicious intervals detected (3+) for optimization.

        Args:
            frames: (T, 224, 224, 3) RGB [0, 1]
            video_path: Video file path
            timestamps: (T,) timestamps in seconds
            fps: Video FPS
            skip_phonemes: If True, skip phoneme extraction (MMMS-BA doesn't need it)

        Returns:
            geometry: (T, 1) MAR features
            arcface: (T, 512) ArcFace embeddings
            phoneme_labels: (T,) object array or None if skip_phonemes=True
        """
        from ..utils.feature_extraction_utils import extract_features_optimized, initialize_extractors

        logger.info(f"    Initializing extractors...")

        # Initialize extractors once
        extractors = initialize_extractors(device=str(self.device))

        # Prepare frames: 224×224 RGB → 384×384 BGR uint8 for extractors
        logger.info(f"    Preparing {frames.shape[0]} frames for extraction...")
        frames_bgr_384 = []
        for frame in frames:
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            frame_384 = cv2.resize(frame_bgr, (384, 384))
            frames_bgr_384.append(frame_384)

        # Config for audio
        config = {
            'sample_rate': 16000,
            'n_mfcc': 40,
            'hop_length': 512,
            'n_fft': 2048
        }

        # Extract features
        if skip_phonemes:
            logger.info(f"    [OPTIMIZATION] Extracting MAR/ArcFace only (MMMS-BA doesn't need phonemes)...")
        else:
            logger.info(f"    Extracting MAR/ArcFace/Phoneme...")

        geometry, audio_mfcc, arcface, phoneme_labels, audio_wav_path = extract_features_optimized(
            frames_bgr=frames_bgr_384,
            video_path=video_path,
            timestamps=timestamps,
            fps=fps,
            config=config,
            max_duration=60,
            extractors=extractors,
            extract_phonemes=not skip_phonemes,  # 🔑 Skip phoneme if MMMS-BA
            logger=logger
        )

        # Cleanup WAV file
        from pathlib import Path as P
        if audio_wav_path and P(audio_wav_path).exists():
            P(audio_wav_path).unlink()

        return geometry, arcface, phoneme_labels

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
            frames: (T, 224, 224, 3) numpy array [0, 1]
            audio: (T_audio, 40) numpy array
            lip: (T, H, W, 3) numpy array [0, 1] - Accepts 96x96 or 112x112

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
        # Extract data from extracted_features (frame_probabilities removed for JSON size reduction)
        extracted_features = stage1_timeline.get('extracted_features', {})
        if 'fake_probs' not in extracted_features:
            raise ValueError("fake_probs not found in extracted_features. Make sure scan_video() was called first.")

        timestamps = extracted_features['timestamps']
        fake_probs = extracted_features['fake_probs']
        threshold = extracted_features['threshold']

        suspicious_intervals = stage1_timeline['suspicious_intervals']
        stats = stage1_timeline['statistics']

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

