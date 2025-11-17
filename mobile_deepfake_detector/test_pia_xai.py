"""
Test script for PIAExplainer (XAI)

Tests the explainability module with dummy data to ensure:
1. Hooks capture activations correctly
2. All 5 analysis functions work
3. Korean summary is generated
4. Final result matches DeepfakeXAIResult interface
"""

import torch
import sys
from pathlib import Path
import json

# Add paths
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(src_path / 'models'))
sys.path.insert(0, str(src_path / 'xai'))
sys.path.insert(0, str(src_path / 'utils'))

from pia_model import create_pia_model
from pia_explainer import PIAExplainer
from korean_phoneme_config import get_phoneme_vocab


def test_pia_explainer():
    """Test PIAExplainer with dummy data."""
    print("=" * 60)
    print("Testing PIAExplainer (XAI)")
    print("=" * 60)

    # ===== 1. Create PIA Model =====
    print("\n[1] Creating PIA model...")
    model = create_pia_model(
        num_phonemes=14,
        frames_per_phoneme=5,
        use_temporal_loss=False
    )
    model.eval()
    print("[OK] Model created")

    # ===== 2. Create PIAExplainer =====
    print("\n[2] Creating PIAExplainer...")
    phoneme_vocab = get_phoneme_vocab()
    print(f"   Phoneme vocab ({len(phoneme_vocab)}): {phoneme_vocab}")

    explainer = PIAExplainer(
        model=model,
        phoneme_vocab=phoneme_vocab,
        device='cpu'
    )
    print("[OK] PIAExplainer created")

    # ===== 3. Create Dummy Input =====
    print("\n[3] Creating dummy input...")
    B = 1  # Batch size (XAI requires B=1)
    P = 14  # Phonemes
    F = 5  # Frames per phoneme
    H, W = 112, 112  # Image size

    geoms = torch.randn(B, P, F, 1)  # Vertical MAR only
    imgs = torch.randn(B, P, F, 3, H, W)
    arcs = torch.randn(B, P, F, 512)

    print(f"   geoms: {geoms.shape}")
    print(f"   imgs: {imgs.shape}")
    print(f"   arcs: {arcs.shape}")
    print("[OK] Dummy input created")

    # ===== 4. Run XAI Explanation =====
    print("\n[4] Running XAI explanation...")
    result = explainer.explain(
        geoms=geoms,
        imgs=imgs,
        arcs=arcs,
        video_id="test_video_001",
        confidence_threshold=0.5
    )
    print("[OK] Explanation generated")

    # ===== 5. Verify Result Structure =====
    print("\n[5] Verifying result structure...")
    required_keys = [
        'metadata', 'video_info', 'detection', 'summary',
        'phoneme_analysis', 'temporal_analysis', 'geometry_analysis', 'model_info'
    ]

    for key in required_keys:
        if key in result:
            print(f"   [OK] {key}")
        else:
            print(f"   [FAIL] {key} MISSING!")

    # ===== 6. Print Results =====
    print("\n" + "=" * 60)
    print("XAI RESULTS")
    print("=" * 60)

    # Detection
    detection = result['detection']
    print(f"\n[Detection]")
    print(f"  Prediction: {detection['prediction_label']}")
    print(f"  Confidence: {detection['confidence']:.2%}")
    print(f"  Is Fake: {detection['is_fake']}")

    # Korean Summary
    summary = result['summary']
    print(f"\n[Korean Summary]")
    print(f"  Overall: {summary['overall']}")
    print(f"  Reasoning: {summary['reasoning']}")
    print(f"\n  Key Findings:")
    for line in summary['key_findings'].split('\n'):
        print(f"    {line}")

    # Branch Contributions
    branches = result['model_info']['branch_contributions']
    print(f"\n[Branch Contributions]")
    print(f"  Visual: {branches['visual']:.2%}")
    print(f"  Geometry: {branches['geometry']:.2%}")
    print(f"  Identity: {branches['identity']:.2%}")

    # Phoneme Analysis
    phoneme_scores = result['phoneme_analysis']['phoneme_scores']
    print(f"\n[Top 5 Phonemes by Attention]")
    for i, phoneme in enumerate(phoneme_scores[:5], 1):
        print(f"  {i}. {phoneme['phoneme']} ({phoneme['phoneme_mfa']}): "
              f"{phoneme['score']:.4f} [{phoneme['importance_level']}]")

    # Geometry Analysis
    geometry = result['geometry_analysis']
    print(f"\n[Geometry Analysis (MAR)]")
    print(f"  Mean MAR: {geometry['mean_mar']:.4f}")
    print(f"  Std MAR: {geometry['std_mar']:.4f}")
    print(f"  Abnormal Phonemes: {geometry['num_abnormal']}")

    if geometry['abnormal_phonemes']:
        print(f"\n  Detected Abnormalities:")
        for abn in geometry['abnormal_phonemes'][:3]:  # Show top 3
            print(f"    - {abn['phoneme']} ({abn['phoneme_mfa']}): "
                  f"MAR={abn['measured_mar']:.4f}, "
                  f"Expected={abn['expected_range']}, "
                  f"Deviation={abn['deviation']:.4f}")

    # ===== 7. Save Result =====
    print(f"\n[7] Saving result to JSON...")
    output_file = "mobile_deepfake_detector/test_pia_xai_result.json"

    # Convert numpy arrays to lists for JSON serialization
    result_json = result.copy()
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] Result saved to: {output_file}")

    # ===== 8. Final Verification =====
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)

    checks = [
        ("Hooks registered", len(explainer.hook_handles) > 0),
        ("Visual activations captured", explainer.activations['visual'] is not None),
        ("Geometry activations captured", explainer.activations['geometry'] is not None),
        ("Identity activations captured", explainer.activations['identity'] is not None),
        ("Attention weights captured", explainer.activations['attention_weights'] is not None),
        ("Korean summary generated", len(result['summary']['overall']) > 0),
        ("Branch contributions sum to 1", abs(sum(branches.values()) - 1.0) < 0.01),
        ("Phoneme scores sum to 1", abs(sum(p['score'] for p in phoneme_scores) - 1.0) < 0.01),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print("[WARNING]  SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    test_pia_explainer()
