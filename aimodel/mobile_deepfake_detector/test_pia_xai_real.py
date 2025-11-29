"""
Test PIAExplainer with REAL preprocessed video data

Tests the complete pipeline:
1. Load real preprocessed data from preprocessed_data_phoneme
2. Run PIA model inference
3. Generate XAI explanations
4. Display results in Korean

Usage:
    python test_pia_xai_real.py --fake  # Test with FAKE sample
    python test_pia_xai_real.py --real  # Test with REAL sample
"""

import torch
import sys
from pathlib import Path
import json
import numpy as np
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.pia_model import create_pia_model
from src.xai.pia_explainer import PIAExplainer
from src.utils.korean_phoneme_config import get_phoneme_vocab
from src.data.phoneme_dataset import KoreanPhonemeDataset


def test_pia_xai_with_real_data(target_label: int, case_name: str, output_filename: str):
    """
    Test PIAExplainer with real preprocessed video data.

    Args:
        target_label: 0 for REAL, 1 for FAKE
        case_name: 'REAL' or 'FAKE' (for display)
        output_filename: Output JSON filename
    """
    print("=" * 80)
    print(f"Testing PIA Model + XAI with {case_name} Video Data")
    print("=" * 80)

    # ===== 1. Load Dataset =====
    print("\n[1/6] Loading PhonemeDataset...")

    dataset_path = Path("../preprocessed_data_phoneme")

    try:
        dataset = KoreanPhonemeDataset(
            data_dir=str(dataset_path),
            split='test',
            phoneme_vocab=None,  # Use default KEEP_PHONEMES_KOREAN
            transform=None
        )
        print(f"   Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"   [ERROR] Failed to load dataset: {e}")
        print(f"   Make sure preprocessed_data_phoneme/test exists and contains .npz files")
        return

    if len(dataset) == 0:
        print("   [ERROR] Dataset is empty!")
        return

    # Find best sample with target label (highest valid data ratio)
    print(f"\n[2/6] Finding best {case_name} sample...")
    best_idx = None
    best_valid_ratio = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample['label'].item() == target_label:
            valid_ratio = sample['mask'].sum().item() / sample['mask'].numel()
            if valid_ratio > best_valid_ratio:
                best_valid_ratio = valid_ratio
                best_idx = idx

    if best_idx is None:
        print(f"   [ERROR] No {case_name} sample found in dataset!")
        sys.exit(1)

    sample = dataset[best_idx]
    print(f"   Using Sample {best_idx}: {sample['video_id']}")
    print(f"   Label: {case_name}")
    print(f"   Valid data ratio: {best_valid_ratio:.2%}")

    print(f"   Sample keys: {sample.keys()}")
    print(f"   geometry: {sample['geometry'].shape}")
    print(f"   images: {sample['images'].shape}")
    print(f"   arcface: {sample['arcface'].shape}")
    print(f"   mask: {sample['mask'].shape}")
    print(f"   label: {sample['label'].item()}")
    print(f"   video_id: {sample['video_id']}")

    # Add batch dimension
    geoms = sample['geometry'].unsqueeze(0)  # (1, P, F, 1)
    imgs = sample['images'].unsqueeze(0)     # (1, P, F, 3, H, W)
    arcs = sample['arcface'].unsqueeze(0)    # (1, P, F, 512)
    mask = sample['mask'].unsqueeze(0)       # (1, P, F)
    label = sample['label'].item()
    video_id = sample['video_id']

    print(f"\n   After adding batch dimension:")
    print(f"   geoms: {geoms.shape}")
    print(f"   imgs: {imgs.shape}")
    print(f"   arcs: {arcs.shape}")

    # Check MAR statistics
    mar_values = geoms.squeeze().numpy()
    mar_nonzero = mar_values[mar_values != 0]  # Exclude padding (0 values)

    print(f"\n   MAR Statistics (Excluding Padding):")
    print(f"   - Mean: {np.mean(mar_nonzero):.4f} (non-zero only)")
    print(f"   - Std: {np.std(mar_nonzero):.4f}")
    print(f"   - Min: {np.min(mar_nonzero):.4f}")
    print(f"   - Max: {np.max(mar_nonzero):.4f}")
    print(f"   - Non-zero ratio: {np.sum(mar_values != 0) / mar_values.size:.2%}")
    print(f"   - Total slots: {mar_values.size}, Valid data: {len(mar_nonzero)}")

    # ===== 3. Create PIA Model =====
    print(f"\n[3/6] Creating PIA model...")
    model = create_pia_model(
        num_phonemes=14,
        frames_per_phoneme=5,
        use_temporal_loss=False
    )

    # Load trained checkpoint
    checkpoint_path = Path("outputs/pia_aug50/checkpoints/best.pth")
    if checkpoint_path.exists():
        print(f"   Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   [OK] Checkpoint loaded (epoch {checkpoint.get('epoch', '?')})")
    else:
        print(f"   [WARNING] Checkpoint not found: {checkpoint_path}")
        print(f"   Using randomly initialized model!")

    model.eval()
    print(f"   [OK] Model ready")

    # ===== 4. Create PIAExplainer =====
    print(f"\n[4/6] Creating PIAExplainer...")
    phoneme_vocab = get_phoneme_vocab()
    explainer = PIAExplainer(
        model=model,
        phoneme_vocab=phoneme_vocab,
        device='cpu'
    )
    print(f"   [OK] PIAExplainer created")

    # ===== 5. Run Inference + XAI =====
    print(f"\n[5/6] Running PIA inference + XAI explanation...")
    print(f"   Video ID: {video_id}")
    print(f"   True Label: {'FAKE' if label == 1 else 'REAL'}")

    result = explainer.explain(
        geoms=geoms,
        imgs=imgs,
        arcs=arcs,
        mask=mask,
        video_id=video_id,
        confidence_threshold=0.5
    )
    print(f"   [OK] XAI explanation generated")

    # ===== 6. Display Results =====
    print("\n" + "=" * 80)
    print("REAL VIDEO XAI RESULTS")
    print("=" * 80)

    # Video Info
    print(f"\n[Video Information]")
    print(f"  Video ID: {result['video_info']['video_id']}")
    print(f"  Num Phonemes: {result['video_info']['num_phonemes']}")
    print(f"  True Label: {'FAKE' if label == 1 else 'REAL'}")

    # Detection
    detection = result['detection']
    print(f"\n[Detection Result]")
    print(f"  Prediction: {detection['prediction_label']}")
    print(f"  Confidence: {detection['confidence']:.2%}")
    print(f"  Correct: {'YES' if detection['is_fake'] == (label == 1) else 'NO'}")

    # Korean Summary
    summary = result['summary']
    print(f"\n[Korean Summary]")
    try:
        print(f"  {summary['overall']}")
        print(f"\n  {summary['reasoning']}")
        print(f"\n  주요 발견사항:")
        for line in summary['key_findings'].split('\n'):
            # Remove bullet points for Windows console
            line_clean = line.replace('•', '-')
            print(f"    {line_clean}")
    except UnicodeEncodeError:
        print("  [Korean text encoding issue on Windows console]")
        print(f"  See JSON file for full Korean summary")

    # Branch Contributions
    branches = result['model_info']['branch_contributions']
    print(f"\n[Branch Contributions]")
    print(f"  Visual (입 모양): {branches['visual']:.2%}")
    print(f"  Geometry (MAR): {branches['geometry']:.2%}")
    print(f"  Identity (얼굴): {branches['identity']:.2%}")

    # Top Phonemes
    phoneme_scores = result['phoneme_analysis']['phoneme_scores']
    print(f"\n[Top 5 Phonemes by Attention]")
    for i, phoneme in enumerate(phoneme_scores[:5], 1):
        print(f"  {i}. {phoneme['phoneme']} ({phoneme['phoneme_mfa']}): "
              f"{phoneme['score']:.4f} [{phoneme['importance_level']}]")

    # Geometry Analysis
    geometry = result['geometry_analysis']
    print(f"\n[Geometry Analysis (MAR - Real Data)]")
    print(f"  Mean MAR: {geometry['mean_mar']:.4f}")
    print(f"  Std MAR: {geometry['std_mar']:.4f}")
    print(f"  Abnormal Phonemes: {geometry['num_abnormal']}")

    if geometry['abnormal_phonemes']:
        print(f"\n  Detected MAR Abnormalities (Top 3):")
        for abn in geometry['abnormal_phonemes'][:3]:
            print(f"    - {abn['phoneme']} ({abn['phoneme_mfa']}): "
                  f"Measured={abn['measured_mar']:.4f}, "
                  f"Expected={abn['expected_range']}, "
                  f"Deviation={abn['deviation']:.4f}")

    # Save Result
    print(f"\n[6/6] Saving result to JSON...")
    output_file = f"mobile_deepfake_detector/{output_filename}"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"   [OK] Result saved to: {output_file}")

    # Final Summary
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"  Video: {video_id}")
    print(f"  True Label: {'FAKE' if label == 1 else 'REAL'}")
    print(f"  Predicted: {detection['prediction_label']} ({detection['confidence']:.1%})")
    print(f"  Correct: {'YES' if detection['is_fake'] == (label == 1) else 'NO'}")
    print(f"  MAR Range: [{np.min(mar_values):.4f}, {np.max(mar_values):.4f}]")
    print("=" * 80)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test PIA XAI with REAL or FAKE samples')
    parser.add_argument('--fake', action='store_true', help='Test with FAKE sample')
    parser.add_argument('--real', action='store_true', help='Test with REAL sample')
    args = parser.parse_args()

    # Determine target label and output filename
    if args.fake:
        target_label = 1  # FAKE
        case_name = "FAKE"
        output_filename = "test_pia_xai_fake_result.json"
    elif args.real:
        target_label = 0  # REAL
        case_name = "REAL"
        output_filename = "test_pia_xai_real_result.json"
    else:
        print("Error: Please specify --fake or --real")
        print("Usage:")
        print("  python test_pia_xai_real.py --fake  # Test with FAKE sample")
        print("  python test_pia_xai_real.py --real  # Test with REAL sample")
        sys.exit(1)

    test_pia_xai_with_real_data(target_label, case_name, output_filename)
