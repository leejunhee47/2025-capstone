"""
Batch Temporal Visualization
Generate temporal visualizations for multiple test samples
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualize_temporal_prediction import visualize_temporal_prediction


def main():
    # Configuration
    original_video_root = "E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터"
    model_path = "E:/capstone/mobile_deepfake_detector/models/checkpoints/mmms-ba_best.pth"
    config_path = "E:/capstone/mobile_deepfake_detector/configs/train_teacher_korean.yaml"

    # Test samples: 3 Fake, 2 Real
    test_samples = [
        # Fake samples
        {
            "npz_path": "E:/capstone/preprocessed_data_phoneme/test/00000.npz",
            "label": "FAKE",
            "output": "E:/capstone/mobile_deepfake_detector/outputs/temporal_viz/test_00000_fake.png"
        },
        {
            "npz_path": "E:/capstone/preprocessed_data_phoneme/test/00001.npz",
            "label": "FAKE",
            "output": "E:/capstone/mobile_deepfake_detector/outputs/temporal_viz/test_00001_fake.png"
        },
        {
            "npz_path": "E:/capstone/preprocessed_data_phoneme/test/00004.npz",
            "label": "FAKE",
            "output": "E:/capstone/mobile_deepfake_detector/outputs/temporal_viz/test_00004_fake.png"
        },
        # Real samples
        {
            "npz_path": "E:/capstone/preprocessed_data_phoneme/test/00002.npz",
            "label": "REAL",
            "output": "E:/capstone/mobile_deepfake_detector/outputs/temporal_viz/test_00002_real.png"
        },
        {
            "npz_path": "E:/capstone/preprocessed_data_phoneme/test/00003.npz",
            "label": "REAL",
            "output": "E:/capstone/mobile_deepfake_detector/outputs/temporal_viz/test_00003_real.png"
        }
    ]

    # Create output directory
    output_dir = Path("E:/capstone/mobile_deepfake_detector/outputs/temporal_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("BATCH TEMPORAL VISUALIZATION")
    print("="*80)
    print(f"Processing {len(test_samples)} samples (3 Fake, 2 Real)")
    print("="*80)
    print()

    # Process each sample
    for i, sample in enumerate(test_samples, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(test_samples)}] Processing {sample['label']} sample: {Path(sample['npz_path']).name}")
        print(f"{'='*80}\n")

        visualize_temporal_prediction(
            npz_path=sample['npz_path'],
            original_video_root=original_video_root,
            model_path=model_path if Path(model_path).exists() else None,
            config_path=config_path,
            output_path=sample['output'],
            num_sample_frames=6
        )

    print("\n" + "="*80)
    print("BATCH VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll {len(test_samples)} visualizations saved to:")
    print(f"  {output_dir}")
    print("\nGenerated files:")
    for sample in test_samples:
        print(f"  - {Path(sample['output']).name} ({sample['label']})")
    print()


if __name__ == "__main__":
    main()