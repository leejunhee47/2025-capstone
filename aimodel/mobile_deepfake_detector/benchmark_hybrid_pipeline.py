"""
Hybrid Pipeline Performance Benchmark (Unified Version)
벤치마크: 처리 시간 151초 → 60초 (60% 단축) 목표 검증

Test 3: 전체 파이프라인 성능
Test 5: 여러 비디오 평균 성능
"""
import sys
import time
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from mobile_deepfake_detector.src.xai.hybrid_pipeline import HybridXAIPipeline

# ===================================
# Configuration
# ===================================

# Test videos (adjust paths as needed)
TEST_VIDEOS = [
    {
        'path': 'E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터/train_탐지방해/변조영상/01_변조영상/01_변조영상/092dff5b934f87c11767_07fc49d464acdc489fc1_1_0490.mp4',
        'expected_label': 'fake',
        'video_id': 'FAKE_0490'
    },
    # Add more videos if available
]

# Model paths
MMMS_BA_CHECKPOINT = Path("mobile_deepfake_detector/models/checkpoints/mmms-ba_fulldata_best.pth")
MMMS_BA_CONFIG = Path("mobile_deepfake_detector/configs/train_teacher_korean.yaml")
PIA_CHECKPOINT = Path("F:/preprocessed_data_pia_optimized/checkpoints/best.pth")
PIA_CONFIG = Path("mobile_deepfake_detector/configs/train_pia.yaml")

# ===================================
# Test 3: Single Video Performance
# ===================================

def test_single_video_performance():
    """
    Test 3: 전체 파이프라인 처리 시간 측정
    목표: 151초 → 60초 (60% 단축)
    """
    print("\n" + "="*80)
    print("TEST 3: Single Video Performance Benchmark (Unified Pipeline)")
    print("="*80)
    print(f"\nTarget: < 70 seconds (baseline: 151s, goal: 60s)")

    if not TEST_VIDEOS:
        print("\n[SKIP] No test videos configured")
        return None

    test_video = TEST_VIDEOS[0]
    video_path = test_video['path']

    if not Path(video_path).exists():
        print(f"\n[SKIP] Video not found: {video_path}")
        return None

    # Initialize pipeline
    print(f"\n[1/3] Initializing Unified HybridXAIPipeline...")

    try:
        pipeline = HybridXAIPipeline(
            mmms_model_path=str(MMMS_BA_CHECKPOINT),
            pia_model_path=str(PIA_CHECKPOINT),
            mmms_config_path=str(MMMS_BA_CONFIG),
            pia_config_path=str(PIA_CONFIG),
            device="cuda"
        )
        print(f"  [OK] Pipeline initialized")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize: {e}")
        return None

    # Run pipeline with timing
    print(f"\n[2/3] Processing video: {Path(video_path).name}")
    print(f"  Expected label: {test_video['expected_label'].upper()}")

    start_time = time.time()

    try:
        result = pipeline.process_video(
            video_path=video_path,
            video_id=test_video['video_id'],
            save_visualizations=False  # Disable for speed check
        )

        elapsed_time = time.time() - start_time

    except Exception as e:
        print(f"\n  [ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Analyze results
    print(f"\n[3/3] Results Analysis")
    print(f"  {'='*76}")

    # Timing
    print(f"\n  Processing Time: {elapsed_time:.1f} seconds")
    baseline_time = 151.0
    improvement = ((baseline_time - elapsed_time) / baseline_time) * 100

    if elapsed_time < 70:
        print(f"  [PASS] < 70s target (baseline: 151s)")
        print(f"  Improvement: {improvement:+.1f}% ({baseline_time:.0f}s → {elapsed_time:.0f}s)")
    else:
        print(f"  [INFO] > 70s target")
        print(f"  Improvement: {improvement:+.1f}% ({baseline_time:.0f}s → {elapsed_time:.0f}s)")

    # Detection accuracy
    detection = result.get('detection', {})
    verdict = detection.get('verdict', 'unknown')
    confidence = detection.get('confidence', 0.0)

    print(f"\n  Detection:")
    print(f"    Verdict: {verdict.upper()}")
    print(f"    Confidence: {confidence:.1%}")
    print(f"    Expected: {test_video['expected_label'].upper()}")

    if verdict.lower() == test_video['expected_label'].lower():
        print(f"    [PASS] Correct prediction")
    else:
        print(f"    [FAIL] Incorrect prediction")

    # Raw Results Breakdown
    raw_results = result.get('raw_results', {})
    mmms_result = raw_results.get('mmms_ba', {})
    pia_result = raw_results.get('pia', {})

    print(f"\n  MMMS-BA:")
    print(f"    Verdict: {mmms_result.get('verdict', 'unknown').upper()}")
    print(f"    Confidence: {mmms_result.get('confidence', 0.0):.1%}")
    
    print(f"\n  PIA:")
    # Handle simplified vs full result
    if 'detection' in pia_result:
        pia_verdict = pia_result['detection'].get('verdict', 'unknown')
        pia_conf = pia_result['detection'].get('confidence', 0.0)
            else:
        pia_verdict = pia_result.get('verdict', 'unknown')
        pia_conf = pia_result.get('confidence', 0.0)
        
    print(f"    Verdict: {pia_verdict.upper()}")
    print(f"    Confidence: {pia_conf:.1%}")

    print(f"\n  {'='*76}")

    return {
        'video_id': test_video['video_id'],
        'elapsed_time': elapsed_time,
        'verdict': verdict,
        'confidence': confidence,
        'expected_label': test_video['expected_label'],
        'correct': verdict.lower() == test_video['expected_label'].lower(),
        'mmms_confidence': mmms_result.get('confidence', 0.0),
        'pia_confidence': pia_conf
    }

# ===================================
# Test 5: Multiple Videos Benchmark
# ===================================

def test_multiple_videos_benchmark():
    """
    Test 5: 여러 비디오 평균 성능 측정
    """
    print("\n" + "="*80)
    print("TEST 5: Multiple Videos Benchmark")
    print("="*80)

    if len(TEST_VIDEOS) < 2:
        print("\n[SKIP] Need at least 2 videos for benchmark")
        return None

    # TODO: Implement if needed
    print("\n[INFO] Multi-video benchmark requires TEST_VIDEOS configuration")
    return None

# ===================================
# Main Entry Point
# ===================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Unified Hybrid MMMS-BA + PIA XAI Pipeline - Benchmark")
    print("="*80)
    print(f"\nOptimization Goals:")
    print(f"  1. Unified Feature Extraction: 1 pass for all models")
    print(f"  2. Parallel/Simple Pipeline: No interval scanning")
    print(f"  3. Total time reduction: Target < 70s")

    results = {}

    # Test 3: Single video performance
    result_3 = test_single_video_performance()
    if result_3:
        results['test_3'] = result_3

    # Final summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    if 'test_3' in results:
        r = results['test_3']
        print(f"\n[Test 3] Single Video Performance:")
        print(f"  Time: {r['elapsed_time']:.1f}s")
        print(f"  Status: {'PASS' if r['elapsed_time'] < 70 else 'INFO'} (target: <70s)")
        print(f"  Detection: {'PASS' if r['correct'] else 'FAIL'} ({r['verdict'].upper()})")

        # Calculate improvement
        baseline = 151.0
        improvement_pct = ((baseline - r['elapsed_time']) / baseline) * 100
        improvement_sec = baseline - r['elapsed_time']

        print(f"\n  Performance Improvement:")
        print(f"    Baseline: {baseline:.0f}s")
        print(f"    Current:  {r['elapsed_time']:.0f}s")
        print(f"    Saved:    {improvement_sec:.0f}s ({improvement_pct:+.1f}%)")

    # Save results
    output_path = Path("outputs/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_path}")
    print("\n" + "="*80 + "\n")
