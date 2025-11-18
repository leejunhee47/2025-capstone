"""
Debug PIA Attention Scores

Checks if the model is giving proper attention distribution or just focusing on one phoneme.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from src.xai.hybrid_mmms_pia_explainer import HybridMMSBAPIA

def main():
    print("=" * 80)
    print("Debugging PIA Attention Scores")
    print("=" * 80)

    # Model paths
    mmms_model = r"E:\capstone\mobile_deepfake_detector\models\kfold\fold_1\mmms-ba_best_val_94_57.pth"
    pia_model = "outputs/pia_aug50/checkpoints/best.pth"

    # Test video
    video_path = r"E:\capstone\dataset_sample\원천데이터\train_변조\106792\106792_40068_2_1070.mp4"

    print(f"\n[1/3] Initializing pipeline...")
    pipeline = HybridMMSBAPIA(
        mmms_model_path=mmms_model,
        pia_model_path=pia_model,
        device="cuda"
    )

    print(f"\n[2/3] Running Stage1...")
    stage1_result = pipeline.run_stage1_temporal_scan(
        video_path=video_path,
        threshold=0.6
    )

    from src.xai.hybrid_utils import group_consecutive_frames
    suspicious_intervals = group_consecutive_frames(
        suspicious_indices=stage1_result['suspicious_indices'],
        fps=stage1_result['fps'],
        min_interval_frames=14,
        merge_gap_sec=1.0
    )

    if len(suspicious_intervals) == 0:
        print("[!] No suspicious intervals")
        return

    interval = suspicious_intervals[0]
    print(f"\n[3/3] Running Stage2 on Interval 0: {interval['start_time']:.1f}s - {interval['end_time']:.1f}s")

    # Extract interval frames
    start_frame = interval['start_frame']
    end_frame = interval['end_frame']
    interval_frames = stage1_result['frames'][start_frame:end_frame+1]
    interval_timestamps = stage1_result['timestamps'][start_frame:end_frame+1]

    # Extract phoneme alignment
    if pipeline.phoneme_aligner is None:
        from src.utils.hybrid_phoneme_aligner_v2 import HybridPhonemeAligner
        pipeline.phoneme_aligner = HybridPhonemeAligner(
            whisper_model="base",
            device="cuda",
            compute_type="float16"
        )

    alignment = pipeline.phoneme_aligner.align_video(video_path)
    phoneme_intervals = [
        {'phoneme': p, 'start': s, 'end': e}
        for p, (s, e) in zip(alignment['phonemes'], alignment['intervals'])
    ]

    # Match phonemes to frames
    from src.data.preprocessing import match_phoneme_to_frames
    phoneme_labels = match_phoneme_to_frames(phoneme_intervals, interval_timestamps)

    print(f"\n[Phoneme Extraction]")
    print(f"  Total phonemes before filtering: {len(alignment['phonemes'])}")
    print(f"  Unique phonemes: {set(alignment['phonemes'])}")

    # Get interval phoneme dict
    from src.xai.hybrid_utils import get_interval_phoneme_dict, resample_frames_to_pia_format

    phonemes = get_interval_phoneme_dict(
        phoneme_labels=phoneme_labels,
        timestamps=interval_timestamps
    )

    # Resample to 14×5
    resampled_frames, matched_phonemes = resample_frames_to_pia_format(
        frames=interval_frames,
        timestamps=interval_timestamps,
        phonemes=phonemes,
        target_phonemes=14,
        frames_per_phoneme=5
    )

    print(f"\n[Resampling to 14×5]")
    print(f"  Resampled shape: {resampled_frames.shape}")
    print(f"  Matched phonemes: {matched_phonemes}")

    # Count non-zero frames per phoneme
    for pi, phoneme in enumerate(matched_phonemes):
        non_zero_count = np.count_nonzero(resampled_frames[pi].sum(axis=(1, 2, 3)))
        print(f"    [{pi:2d}] {phoneme:4s}: {non_zero_count}/5 frames have data")

    # Create dummy geometry and arcface (just for testing)
    print(f"\n[Creating dummy MAR and ArcFace features...]")
    geometry = np.random.rand(14, 5, 1).astype(np.float32) * 0.5 + 0.3  # MAR range [0.3, 0.8]
    arcface = np.random.rand(14, 5, 512).astype(np.float32)  # Random embeddings

    # Convert to tensors
    geoms_tensor = torch.from_numpy(geometry).float().unsqueeze(0)  # (1, 14, 5, 1)
    imgs_tensor = torch.from_numpy(resampled_frames).float().unsqueeze(0)  # (1, 14, 5, H, W, 3)
    imgs_tensor = imgs_tensor.permute(0, 1, 2, 5, 3, 4)  # (1, 14, 5, 3, H, W)
    arcs_tensor = torch.from_numpy(arcface).float().unsqueeze(0)  # (1, 14, 5, 512)
    mask_tensor = torch.ones(1, 14, 5).bool()

    # Run PIA model
    print(f"\n[Running PIA Model...]")
    print(f"  geoms: {geoms_tensor.shape}")
    print(f"  imgs: {imgs_tensor.shape}")
    print(f"  arcs: {arcs_tensor.shape}")
    print(f"  mask: {mask_tensor.shape}")

    xai_result = pipeline.pia_explainer.explain(
        geoms=geoms_tensor,
        imgs=imgs_tensor,
        arcs=arcs_tensor,
        mask=mask_tensor,
        video_id="debug_test",
        confidence_threshold=0.5
    )

    # Print attention scores
    print(f"\n[PIA XAI Result Keys]")
    print(f"  Available keys: {list(xai_result.keys())}")

    if 'phoneme_analysis' in xai_result:
        phoneme_analysis = xai_result['phoneme_analysis']
        print(f"\n[Phoneme Analysis Structure]")
        print(f"  Keys: {list(phoneme_analysis.keys())}")

        if 'phoneme_scores' in phoneme_analysis:
            phoneme_scores = phoneme_analysis['phoneme_scores']
            print(f"  Number of phoneme scores: {len(phoneme_scores)}")
            print(f"  First item keys: {list(phoneme_scores[0].keys()) if phoneme_scores else 'N/A'}")

            print(f"\n[PIA Attention Scores]")
            print(f"  {'Idx':<4} {'Phoneme':<8} {'MFA':<4} {'Score':>8} {'Frames':>8} {'Importance':<10}")
            print(f"  {'-'*4} {'-'*8} {'-'*4} {'-'*8} {'-'*8} {'-'*10}")

            for pi, item in enumerate(phoneme_scores):
                # Correct key names from PIAExplainer._analyze_phoneme_attention()
                phoneme_korean = item['phoneme']  # Already Korean from phoneme_to_korean map
                phoneme_mfa = item['phoneme_mfa']
                score = item['score']
                non_zero = np.count_nonzero(resampled_frames[pi].sum(axis=(1, 2, 3)))
                importance = item['importance_level']
                print(f"  {pi:<4} {phoneme_korean:<8} {phoneme_mfa:<4} {score:>8.4f} {non_zero}/5     {importance:<10}")

    # Check temporal analysis structure
    if 'temporal_analysis' in xai_result:
        temporal = xai_result['temporal_analysis']
        print(f"\n[Temporal Analysis Structure]")
        print(f"  Keys: {list(temporal.keys())}")

        if 'heatmap' in temporal:
            heatmap = temporal['heatmap']
            print(f"\n[Temporal Heatmap]")
            print(f"  Shape: {len(heatmap)} phonemes × {len(heatmap[0]) if heatmap else 0} frames")
            for pi, phoneme_row in enumerate(heatmap):
                phoneme = matched_phonemes[pi]
                avg_attn = np.mean(phoneme_row)
                max_attn = np.max(phoneme_row)
                print(f"    [{pi:2d}] {phoneme:4s}: avg={avg_attn:.4f}, max={max_attn:.4f}")

    print("\n" + "=" * 80)
    print("Debug Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()