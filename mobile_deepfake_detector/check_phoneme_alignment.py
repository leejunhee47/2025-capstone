"""
Quick test to check phoneme-frame alignment quality
"""
import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from mobile_deepfake_detector.src.xai.stage2_analyzer import Stage2Analyzer

# Initialize Stage2
stage2 = Stage2Analyzer(
    pia_model_path="F:/preprocessed_data_pia_optimized/checkpoints/best.pth",
    pia_config_path="configs/train_pia.yaml",
    device="cuda"
)

# Test interval - 변조 영상 (실제 음성이 있는 구간)
test_video = Path("E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터/train_탐지방해/변조영상/01_변조영상/01_변조영상/0e23d546a5f952542a00_e37c8c26b0c1c0714c74_4_1680.mp4")

interval = {
    'interval_id': 0,
    'start_frame': 0,      # 상대 인덱스 (stage1_result['frames'] 내에서)
    'end_frame': 60,       # 61개 프레임 (0~60)
    'start_time': 5.0,     # 중간 구간 (음성이 있을 가능성 높음)
    'end_time': 7.0,
    'duration': 2.0,
    'frame_count': 61
}

# Stage1 result는 30fps로 재추출할 것이므로 빈 프레임으로 전달
# SLOW PATH로 가도록 precomputed features를 제외
stage1_result = {
    'video_path': str(test_video),
    'fps': 5.0,  # Stage1의 원래 fps (SLOW PATH 트리거용)
    'frames': np.zeros((61, 224, 224, 3), dtype=np.float32),  # 더미 프레임
    'timestamps': np.array([5.0 + i * (2.0/60) for i in range(61)])  # 5-7초 균등 분포
}

print("\n" + "="*80)
print("Checking phoneme-frame alignment quality...")
print("="*80 + "\n")

# Run analysis
result = stage2.run_stage2_interval_xai(
    stage1_result=stage1_result,
    interval=interval,
    output_dir="outputs/test_30fps"
)

print(f"\n[OK] Analysis complete!")

# DEBUG: Check actual phoneme_labels before resampling
print(f"\n" + "="*80)
print("DEBUG: Checking raw phoneme_labels (before resampling)")
print(f"="*80)

# Manually count non-<sil> in phoneme_labels
# We need to access interval_features from result
# But result doesn't expose it, so we need to check indirectly

print(f"\nResult structure:")
print(f"  Top-level keys: {list(result.keys())}")

# ===== TEST 1: Verify Stage2 Prediction (FAKE Video) =====
print("\n" + "="*80)
print("TEST 1: Stage2 Prediction Verification (CRITICAL)")
print("="*80)

# Check for prediction in pia_xai.detection structure
if 'pia_xai' in result and 'detection' in result['pia_xai']:
    detection = result['pia_xai']['detection']

    # DEBUG: Print detection structure
    print(f"\n[DEBUG] Detection structure:")
    print(f"  Keys: {list(detection.keys())}")
    print(f"  Full detection: {detection}")

    # Extract prediction (use prediction_label or is_fake)
    prediction_label = detection.get('prediction_label', '')
    is_fake = detection.get('is_fake', None)
    confidence = detection.get('confidence', 0.0)

    # Determine verdict from available fields
    if prediction_label:
        verdict = prediction_label.lower()
    elif is_fake is not None:
        verdict = 'fake' if is_fake else 'real'
    else:
        verdict = 'unknown'

    print(f"\n[Stage2 Prediction]")
    print(f"  Verdict: {verdict.upper()}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  is_fake: {is_fake}")

    # This is a FAKE video (0e23d546a5f952542a00_e37c8c26b0c1c0714c74_4_1680.mp4)
    # Stage2 MUST identify it as FAKE (not REAL)
    if verdict == 'fake' or is_fake is True:
        print(f"\n[PASS] Stage2 correctly identified FAKE video as FAKE")
        print(f"  -> 30fps optimization successfully resolved the misclassification issue!")

        # Check for overconfidence (previous issue was 99.4% confidence)
        if confidence > 0.95:
            print(f"\n[WARNING] Very high confidence ({confidence:.1%})")
            print(f"  -> Consider calibration if this becomes a pattern")
        else:
            print(f"\n[OK] Confidence is reasonable ({confidence:.1%})")
    elif verdict == 'real' or is_fake is False:
        print(f"\n[FAIL] Stage2 misclassified FAKE video as REAL")
        print(f"  -> This is the CRITICAL BUG that 30fps optimization should have fixed!")
        print(f"  -> Confidence: {confidence:.1%}")
        print(f"\n  Possible causes:")
        print(f"    1. Phoneme matching still poor (check TEST 2 below)")
        print(f"    2. 30fps re-extraction not triggered")
        print(f"    3. Model calibration issue")
    else:
        print(f"\n[UNKNOWN] Could not determine verdict")
        print(f"  prediction_label: {prediction_label}")
        print(f"  is_fake: {is_fake}")
else:
    print(f"\n[ERROR] 'detection' not found in result['pia_xai']!")
    if 'pia_xai' in result:
        print(f"  Available pia_xai keys: {list(result['pia_xai'].keys())}")
    else:
        print(f"  Available result keys: {list(result.keys())}")

# ===== TEST 2: Phoneme-Frame Alignment Quality =====
print("\n" + "="*80)
print("TEST 2: Phoneme-Frame Alignment Quality")
print("="*80)

if 'pia_xai' in result:
    print(f"  pia_xai keys: {list(result['pia_xai'].keys())}")

    # Check phoneme_analysis structure
    if 'phoneme_analysis' in result['pia_xai']:
        phoneme_analysis = result['pia_xai']['phoneme_analysis']
        print(f"\n[OK] Found phoneme_analysis!")
        print(f"  Type: {type(phoneme_analysis)}")
        print(f"  Keys: {list(phoneme_analysis.keys()) if isinstance(phoneme_analysis, dict) else 'N/A'}")

        # Extract matching rate from phoneme_analysis
        if isinstance(phoneme_analysis, dict):
            if 'matching_rate' in phoneme_analysis:
                matching_rate = phoneme_analysis['matching_rate']
                total_frames = phoneme_analysis.get('total_frames', 'N/A')
                matched_frames = phoneme_analysis.get('matched_frames', 'N/A')

                print(f"\n=== Phoneme-Frame Alignment Quality ===")
                print(f"  Total frames: {total_frames}")
                print(f"  Matched frames: {matched_frames}")
                print(f"  Matching rate: {matching_rate:.1f}% (target: 90%)")

                if matching_rate >= 90:
                    print(f"\n[EXCELLENT] {matching_rate:.1f}% >= 90% (목표 달성!)")
                elif matching_rate >= 70:
                    print(f"\n[GOOD] {matching_rate:.1f}% (70-90%)")
                else:
                    print(f"\n[NEEDS IMPROVEMENT] {matching_rate:.1f}% < 70%")
            else:
                print(f"\n  Available keys in phoneme_analysis:")
                for key in phoneme_analysis.keys():
                    print(f"    - {key}: {phoneme_analysis[key]}")
    else:
        print(f"\n[ERROR] No 'phoneme_analysis' in pia_xai")
        print(f"  Available keys: {list(result['pia_xai'].keys())}")
else:
    print(f"\n[ERROR] No 'pia_xai' in result")

# ===== FINAL SUMMARY =====
print("\n" + "="*80)
print("FINAL TEST SUMMARY")
print("="*80)

test_results = []

# Test 1: Prediction accuracy
if 'pia_xai' in result and 'detection' in result['pia_xai']:
    detection = result['pia_xai']['detection']
    prediction_label = detection.get('prediction_label', '').lower()
    is_fake = detection.get('is_fake', None)
    confidence = detection.get('confidence', 0.0)

    # Determine verdict
    verdict = prediction_label if prediction_label else ('fake' if is_fake else 'real')

    if verdict == 'fake' or is_fake is True:
        if confidence > 0.95:
            test_results.append(("[PASS]", f"Stage2 correctly identified FAKE video (confidence: {confidence:.1%} - may need calibration)"))
        else:
            test_results.append(("[PASS]", f"Stage2 correctly identified FAKE video (confidence: {confidence:.1%})"))
    elif verdict == 'real' or is_fake is False:
        test_results.append(("[FAIL]", f"Stage2 misclassified FAKE as REAL (CRITICAL BUG, confidence: {confidence:.1%})"))
    else:
        test_results.append(("[FAIL]", f"Unknown verdict: {verdict}"))
else:
    test_results.append(("[FAIL]", "Prediction not found in result['pia_xai']['detection']"))

# Test 2: Phoneme matching
if 'pia_xai' in result and 'phoneme_analysis' in result['pia_xai']:
    phoneme_analysis = result['pia_xai']['phoneme_analysis']
    if 'matching_rate' in phoneme_analysis:
        matching_rate = phoneme_analysis['matching_rate']
        if matching_rate >= 90:
            test_results.append(("[PASS]", f"Phoneme matching: {matching_rate:.1f}% (excellent)"))
        elif matching_rate >= 70:
            test_results.append(("[PASS]", f"Phoneme matching: {matching_rate:.1f}% (good)"))
        else:
            test_results.append(("[FAIL]", f"Phoneme matching: {matching_rate:.1f}% (poor)"))
    else:
        test_results.append(("[INFO]", "Phoneme matching rate not found in this test mode"))
else:
    test_results.append(("[FAIL]", "Phoneme analysis not found"))

print("\n")
for status, message in test_results:
    print(f"  {status}: {message}")

# Overall result
all_passed = all(status == "[PASS]" for status, _ in test_results)
if all_passed:
    print(f"\n{'='*80}")
    print(f"[SUCCESS] ALL TESTS PASSED! 30fps optimization is working correctly!")
    print(f"{'='*80}")
else:
    print(f"\n{'='*80}")
    print(f"[WARNING] SOME TESTS FAILED - Review results above")
    print(f"{'='*80}")

print("\n" + "="*80 + "\n")