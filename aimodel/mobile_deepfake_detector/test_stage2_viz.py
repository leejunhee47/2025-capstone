"""
Stage2 XAI Visualization Test Script

Tests the visualize_stage2_interval() method to verify 4-panel XAI visualization.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.xai.hybrid_mmms_pia_explainer import HybridMMSBAPIA

def main():
    """Test Stage2 XAI visualization with existing interval result."""

    print("=" * 80)
    print("Testing Stage2 XAI Visualization")
    print("=" * 80)

    # Model paths
    mmms_model = r"E:\capstone\mobile_deepfake_detector\models\kfold\fold_1\mmms-ba_best_val_94_57.pth"
    pia_model = "outputs/pia_aug50/checkpoints/best.pth"

    # Test video (FAKE sample from dataset)
    video_path = r"E:\capstone\dataset_sample\원천데이터\train_변조\106792\106792_40068_2_1070.mp4"

    # Output directory for visualizations
    output_dir = "test_outputs/stage2_viz_test"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[Config]")
    print(f"  MMMS Model: {mmms_model}")
    print(f"  PIA Model: {pia_model}")
    print(f"  Test Video: {video_path}")
    print(f"  Output Dir: {output_dir}")

    # Initialize hybrid pipeline
    print(f"\n[1/3] Initializing HybridMMSBAPIA...")
    pipeline = HybridMMSBAPIA(
        mmms_model_path=mmms_model,
        pia_model_path=pia_model,
        device="cuda"
    )

    # Run Stage1 temporal scan
    print(f"\n[2/3] Running Stage1 temporal scan...")
    stage1_result = pipeline.run_stage1_temporal_scan(
        video_path=video_path,
        threshold=0.6
    )

    print(f"  Suspicious frames: {len(stage1_result['suspicious_indices'])}/{stage1_result['total_frames']}")

    # Group into intervals
    from src.xai.hybrid_utils import group_consecutive_frames

    suspicious_intervals = group_consecutive_frames(
        suspicious_indices=stage1_result['suspicious_indices'],
        fps=stage1_result['fps'],
        min_interval_frames=14,
        merge_gap_sec=1.0
    )

    print(f"  Suspicious intervals: {len(suspicious_intervals)}")

    if len(suspicious_intervals) == 0:
        print("\n[!] No suspicious intervals found - cannot test Stage2 visualization")
        return

    # Test Stage2 on first interval
    interval = suspicious_intervals[0]
    print(f"\n[3/3] Running Stage2 XAI on Interval 0: {interval['start_time']:.1f}s - {interval['end_time']:.1f}s")

    interval_xai = pipeline.run_stage2_interval_xai(
        stage1_result=stage1_result,
        interval=interval,
        output_dir=output_dir
    )

    # Check visualization path
    viz_path = interval_xai.get('visualization_path')

    if viz_path and Path(viz_path).exists():
        print(f"\n[OK] Stage2 XAI visualization saved successfully!")
        print(f"  Path: {viz_path}")
        print(f"  Size: {Path(viz_path).stat().st_size / 1024:.1f} KB")

        # Print XAI insights
        pia_xai = interval_xai['pia_xai']
        print(f"\n[XAI Insights]")
        print(f"  Prediction: {pia_xai['prediction']['verdict']} ({pia_xai['prediction']['confidence']:.2%})")
        print(f"  Branch Contributions:")
        for branch, score in pia_xai['branch_contributions'].items():
            print(f"    - {branch}: {score:.3f}")
        print(f"  Top 3 Phonemes:")
        for i, (phoneme, score) in enumerate(list(pia_xai['phoneme_attention'].items())[:3]):
            print(f"    {i+1}. {phoneme}: {score:.3f}")
    else:
        print(f"\n[ERROR] Visualization was not created!")
        print(f"  Expected path: {viz_path}")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()