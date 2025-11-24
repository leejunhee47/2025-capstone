"""
Hybrid Pipeline Orchestrator

Coordinates the complete 2-stage deepfake detection pipeline:
- Stage 1: MMMS-BA temporal scan (Stage1Scanner)
- Stage 2: PIA XAI analysis per interval (Stage2Analyzer)
- Aggregation: Combine insights across all intervals
- Korean Summary: User-friendly explanation

Output: HybridDeepfakeXAIResult matching hybrid_xai_interface.ts

Author: Claude
Date: 2025-11-17
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Any
import logging

from .stage1_scanner import Stage1Scanner
from .stage2_analyzer import Stage2Analyzer
from .result_aggregator import ResultAggregator
from .hybrid_utils import convert_for_json
from .thumbnail_generator import generate_detection_card

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

        # Result Aggregator
        self.aggregator = ResultAggregator()

        # 30fps Optimization: Shared Phoneme Aligner for STEP 1.5
        logger.info("\n[30FPS OPTIMIZATION] Initializing Shared Phoneme Aligner...")
        from ..utils.hybrid_phoneme_aligner_v2 import HybridPhonemeAligner
        self.phoneme_aligner = HybridPhonemeAligner(
            whisper_model="large-v2",  # 일관성 위해 large-v2 사용
            device=device,
            compute_type="float16"
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
        save_visualizations: bool = True,
        use_preprocessed: bool = None
    ) -> Dict[str, Any]:
        """
        Full E2E pipeline: Stage1 → Stage2 → Aggregation → Korean Summary.

        Args:
            video_path: Path to video file (.mp4/.avi) OR preprocessed .npz file
            video_id: Video identifier (default: extracted from filename)
            output_dir: Base output directory (default: outputs/xai/hybrid/{video_id})
            threshold: Suspicious frame threshold (0.0-1.0)
            min_interval_frames: Minimum frames per interval (PIA needs 14+)
            save_visualizations: Whether to save visualization plots
            use_preprocessed: Whether to use preprocessed npz file.
                             If None, auto-detects based on file extension (.npz).

        Returns:
            HybridDeepfakeXAIResult: Dict matching hybrid_xai_interface.ts
        """
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
            output_dir=stage1_output_dir,
            use_preprocessed=use_preprocessed
        )

        logger.info(f"\n[Stage1 Complete]")
        logger.info(f"  Total frames analyzed: {stage1_timeline['statistics']['total_frames']}")
        logger.info(f"  Suspicious intervals found: {len(stage1_timeline['suspicious_intervals'])}")
        logger.info(f"  Mean fake probability: {stage1_timeline['statistics']['mean_fake_prob']:.3f}")

        # ========================================
        # STEP 1.5: Shared WhisperX Transcription (30fps Optimization)
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1.5: WhisperX Transcription (30fps Optimization)")
        logger.info("=" * 80)

        # Transcribe video once for entire pipeline (saves ~20s per interval!)
        logger.info("  Running WhisperX transcription (1 time only)...")
        transcription_start = time.time()

        try:
            whisper_result = self.phoneme_aligner.transcribe_only(
                video_path=str(video_path)
            )
            transcription_time = time.time() - transcription_start

            logger.info(f"  ✓ Transcription complete in {transcription_time:.1f}s")
            logger.info(f"  ✓ Transcription: {whisper_result['transcription'][:100]}...")
            logger.info(f"  ✓ Segments: {len(whisper_result['segments'])}")
            logger.info(f"  ✓ Audio shape: {whisper_result['audio'].shape}")

        except Exception as e:
            logger.warning(f"  Transcription failed: {e}, Stage2 will use fallback mode")
            whisper_result = None

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
            # Extract features from stage1_timeline to memory variable (for Stage2 use)
            # This prevents extracted_features from being saved to JSON (bloat reduction)
            extracted_features = stage1_timeline.pop('extracted_features', {})

            for interval in stage1_timeline['suspicious_intervals']:
                logger.info(f"\n  Analyzing Interval {interval['interval_id']}: "
                           f"{interval['start_time_sec']:.1f}s - {interval['end_time_sec']:.1f}s "
                           f"({interval['frame_count']} frames)")

                stage2_output_dir = output_dir if save_visualizations else None
                interval_xai = self.stage2.analyze_interval(
                    interval=interval,
                    video_path=video_path,
                    extracted_features=extracted_features,
                    output_dir=stage2_output_dir,
                    whisper_result=whisper_result  # 🔑 30fps Optimization: Pass shared transcription
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

        aggregated_insights = self.aggregator.aggregate_insights(stage2_interval_analysis)

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

        detection = self.aggregator.compute_overall_detection(stage1_timeline, stage2_interval_analysis)

        logger.info(f"\n[Overall Detection]")
        logger.info(f"  Verdict: {detection['verdict'].upper()}")
        logger.info(f"  Confidence: {detection['confidence']:.1%}")
        logger.info(f"  Suspicious frames: {detection['suspicious_frame_count']}/{stage1_timeline['statistics']['total_frames']} "
                   f"({detection['suspicious_frame_ratio']:.1f}%)")

        # ========================================
        # Step 4.5: Generate Stage2 Visualizations with Overall Detection
        # ========================================
        if save_visualizations and len(stage2_interval_analysis) > 0:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4.5: Generating Stage2 Visualizations with Overall Detection")
            logger.info("=" * 80)

            for interval_result in stage2_interval_analysis:
                interval_id = interval_result['interval_id']
                raw_pia_xai = interval_result.get('_raw_pia_xai')

                if raw_pia_xai is None:
                    logger.warning(f"  Interval {interval_id}: No raw PIA XAI data - skipping visualization")
                    continue

                logger.info(f"  Generating visualization for Interval {interval_id}")

                # Generate visualization with overall detection (Combined result)
                # Use raw PIA XAI result for visualizer compatibility
                viz_path = self.stage2.visualize_stage2_interval(
                    interval_xai=raw_pia_xai,  # Raw PIA XAI result (PIAExplainer format)
                    interval_id=interval_id,
                    video_path=video_path,
                    output_dir=output_dir,
                    overall_detection=detection  # Pass overall detection result
                )

                # Update the visualization path in the visualization dict
                if 'visualization' not in interval_result:
                    interval_result['visualization'] = {}
                interval_result['visualization']['xai_plot_url'] = viz_path

            logger.info(f"[Visualization Complete] Generated {len(stage2_interval_analysis)} visualizations")

        # ========================================
        # Step 5: Generate Korean Summary
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Generating Korean Summary")
        logger.info("=" * 80)

        # Extract video metadata (needed for summary detail_view)
        video_info = self.aggregator.extract_video_info(video_path)

        summary = self.aggregator.generate_korean_summary(
            detection=detection,
            aggregated=aggregated_insights,
            stage1_timeline=stage1_timeline,
            stage2_interval_analysis=stage2_interval_analysis,
            video_info=video_info
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

        # Generate thumbnail (NEW)
        thumbnail_path = None
        if output_dir:
            thumbnail_path = str(Path(output_dir) / 'thumbnail.png')
            try:
                generate_detection_card(
                    video_path=video_path,
                    detection=detection,
                    output_path=thumbnail_path
                )
                logger.info(f"Generated detection card: {thumbnail_path}")
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail: {e}")
                thumbnail_path = None

        # Build final result
        result = self.aggregator.build_final_result(
            video_path=video_path,
            video_id=video_id,
            output_dir=output_dir,
            stage1_timeline=stage1_timeline,
            stage2_interval_analysis=stage2_interval_analysis,
            aggregated_insights=aggregated_insights,
            detection=detection,
            summary=summary,
            video_info=video_info,
            processing_time_ms=processing_time_ms,
            mmms_model_path=self.mmms_model_path,
            pia_model_path=self.pia_model_path,
            thumbnail_path=thumbnail_path
        )

        # Save JSON result (remove internal _raw_pia_xai field first)
        for interval_result in result.get('stage2_interval_analysis', []):
            interval_result.pop('_raw_pia_xai', None)  # Remove raw data before JSON serialization

        json_path = Path(output_dir) / "result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(convert_for_json(result), f, indent=2, ensure_ascii=False)

        logger.info(f"\n[Pipeline Complete]")
        logger.info(f"  Processing time: {processing_time_ms:.0f} ms")
        logger.info(f"  Result saved to: {json_path}")
        logger.info("=" * 80 + "\n")

        return result


def main():
    """Example usage of the hybrid pipeline for inference.

    Usage:
        python -m src.xai.hybrid_pipeline \\
            --video path/to/video.mp4 \\
            --mmms-model models/checkpoints/mmms-ba_fulldata_best.pth \\
            --pia-model outputs/pia_aug50/checkpoints/best.pth
    """
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid MMMS-BA + PIA XAI Pipeline (Inference)")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--mmms-model', type=str,
                        default='models/checkpoints/mmms-ba_fulldata_best.pth',
                        help='MMMS-BA model checkpoint (default: mmms-ba_fulldata_best.pth, 97.71%% test acc)')
    parser.add_argument('--pia-model', type=str,
                        default='outputs/pia_aug50/checkpoints/best.pth',
                        help='PIA model checkpoint')
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

