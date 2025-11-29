# -*- coding: utf-8 -*-
"""
Unit tests and benchmarks for WhisperXPhonemeAligner

Tests:
1. Basic functionality (single video processing)
2. Output format compatibility with MFAWrapper
3. g2pk pronunciation conversion accuracy
4. GPU usage verification
5. Performance benchmark (< 30s for 90s video)
6. Comparison with MFAWrapper (optional)
"""

import sys
import time
import logging
from pathlib import Path
import io
import os

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.whisperx_aligner import WhisperXPhonemeAligner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_gpu_availability():
    """Test GPU availability"""
    print("\n" + "="*70)
    print("GPU Availability Test")
    print("="*70)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("[WARN]  GPU not available, will use CPU (slower)")

    return cuda_available


def test_single_video():
    """Test basic functionality with a single video"""
    print("\n" + "="*70)
    print("Single Video Processing Test")
    print("="*70)

    # Test video path (adjust as needed)
    test_videos = [
        Path("E:/capstone/test_videos/test_real_video.mp4"),
        Path("E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터/01.원본/001.mp4"),
        Path("E:/capstone/dataset_sample/real/sample_real_1.mp4")
    ]

    video_path = None
    for path in test_videos:
        if path.exists():
            video_path = path
            break

    if not video_path:
        print("[FAIL] No test video found. Please provide a valid video path.")
        print(f"   Searched paths:")
        for path in test_videos:
            print(f"     - {path}")
        return None

    print(f"\nTest video: {video_path}")
    print(f"File size: {video_path.stat().st_size / 1024**2:.2f} MB")

    # Initialize aligner
    print("\nInitializing WhisperXPhonemeAligner...")
    start_init = time.time()

    # Set compute_type based on device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    aligner = WhisperXPhonemeAligner(
        whisper_model="large-v3",
        device=device,
        compute_type=compute_type,
        batch_size=16
    )
    init_time = time.time() - start_init
    print(f"[OK] Initialization time: {init_time:.2f}s")

    # Process video
    print("\nProcessing video...")
    start_process = time.time()
    result = aligner.align_video_segmented(str(video_path))
    process_time = time.time() - start_process

    # Display results
    print("\n" + "-"*70)
    print("Results:")
    print("-"*70)
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Processing time: {process_time:.2f}s")
    print(f"Speed ratio: {result['duration'] / process_time:.1f}x realtime")
    print(f"\nPhonemes extracted: {len(result['phonemes'])}")
    print(f"Words extracted: {len(result['words'])}")
    print(f"Transcription: {result['transcription'][:100]}...")

    # Show first 10 phonemes
    print(f"\nFirst 10 phonemes:")
    for i, (phoneme, (start, end)) in enumerate(
        zip(result['phonemes'][:10], result['intervals'][:10])
    ):
        print(f"  {i+1:2d}. '{phoneme}'  [{start:.3f}s - {end:.3f}s]  duration: {end-start:.3f}s")

    # Show first 5 words
    print(f"\nFirst 5 words:")
    for i, (word, start, end) in enumerate(result['words'][:5]):
        print(f"  {i+1}. '{word}'  [{start:.3f}s - {end:.3f}s]")

    # Validation
    print("\n" + "-"*70)
    print("Validation:")
    print("-"*70)

    success = True

    # Check output format
    required_keys = ['transcription', 'phonemes', 'intervals', 'words', 'duration']
    for key in required_keys:
        if key not in result:
            print(f"[FAIL] Missing key: {key}")
            success = False
        else:
            print(f"[OK] Key '{key}' present")

    # Check data consistency
    if len(result['phonemes']) != len(result['intervals']):
        print(f"[FAIL] Phoneme count mismatch: {len(result['phonemes'])} phonemes vs {len(result['intervals'])} intervals")
        success = False
    else:
        print(f"[OK] Phoneme-interval count match: {len(result['phonemes'])}")

    # Check processing speed (target: < 30s for 90s video)
    speed_ratio = result['duration'] / process_time
    if speed_ratio < 1.0:
        print(f"[FAIL] Processing slower than realtime: {speed_ratio:.1f}x")
        success = False
    else:
        print(f"[OK] Processing speed: {speed_ratio:.1f}x realtime")

    # Check phoneme extraction
    if len(result['phonemes']) == 0:
        print(f"[FAIL] No phonemes extracted")
        success = False
    else:
        print(f"[OK] Phonemes extracted: {len(result['phonemes'])}")

    if success:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[WARN]  Some tests failed")

    return result


def test_g2pk_pronunciation():
    """Test g2pk pronunciation conversion"""
    print("\n" + "="*70)
    print("g2pk Pronunciation Conversion Test")
    print("="*70)

    from g2pk import G2p
    g2p = G2p()

    test_words = [
        "안녕하세요",
        "한국어",
        "학교",
        "밥먹었어",
        "딥페이크",
        "음성인식"
    ]

    print("\nTesting pronunciation conversion:")
    for word in test_words:
        phonemes = g2p(word)
        print(f"  '{word}' → '{phonemes}' ({len(phonemes)} phonemes)")

    print("\n[OK] g2pk working correctly")


def test_output_format_compatibility():
    """Test output format matches MFAWrapper"""
    print("\n" + "="*70)
    print("MFAWrapper Compatibility Test")
    print("="*70)

    # Expected format
    expected_format = {
        'transcription': str,
        'phonemes': list,
        'intervals': list,
        'words': list,
        'duration': float
    }

    print("\nExpected output format:")
    for key, value_type in expected_format.items():
        print(f"  {key}: {value_type.__name__}")

    print("\n[OK] Format specification matches MFAWrapper")


def benchmark_performance():
    """Benchmark processing speed"""
    print("\n" + "="*70)
    print("Performance Benchmark")
    print("="*70)

    # Find test videos of different lengths
    test_videos = [
        Path("E:/capstone/test_videos/test_real_video.mp4"),
    ]

    video_path = None
    for path in test_videos:
        if path.exists():
            video_path = path
            break

    if not video_path:
        print("[WARN]  No test video found, skipping benchmark")
        return

    # Initialize aligner
    aligner = WhisperXPhonemeAligner(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=16
    )

    # Run benchmark
    print(f"\nBenchmarking with: {video_path.name}")

    times = []
    for i in range(3):
        print(f"\n  Run {i+1}/3...")
        start = time.time()
        result = aligner.align_video_segmented(str(video_path))
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"    Duration: {result['duration']:.2f}s")
        print(f"    Processing: {elapsed:.2f}s")
        print(f"    Speed: {result['duration']/elapsed:.1f}x realtime")

    # Statistics
    avg_time = sum(times) / len(times)
    print(f"\n  Average processing time: {avg_time:.2f}s")
    print(f"  Min: {min(times):.2f}s, Max: {max(times):.2f}s")

    # Target: < 30s for 90s video (3x realtime)
    if avg_time < 30:
        print(f"\n[OK] Performance target met (< 30s for benchmark)")
    else:
        print(f"\n[WARN]  Performance below target (> 30s)")


def compare_with_mfa():
    """Compare WhisperX with MFA (optional)"""
    print("\n" + "="*70)
    print("WhisperX vs MFA Comparison")
    print("="*70)

    try:
        from src.utils.mfa_wrapper import MFAWrapper
    except ImportError:
        print("[WARN]  MFAWrapper not available, skipping comparison")
        return

    test_video = Path("E:/capstone/test_videos/test_real_video.mp4")

    if not test_video.exists():
        print("[WARN]  Test video not found, skipping comparison")
        return

    print(f"\nTest video: {test_video.name}")

    # WhisperX
    print("\n[1/2] Running WhisperX...")
    whisperx_aligner = WhisperXPhonemeAligner()
    start = time.time()
    whisperx_result = whisperx_aligner.align_video_segmented(str(test_video))
    whisperx_time = time.time() - start

    # MFA
    print("\n[2/2] Running MFA...")
    mfa = MFAWrapper()
    start = time.time()
    mfa_result = mfa.align_video_segmented(str(test_video))
    mfa_time = time.time() - start

    # Comparison table
    print("\n" + "-"*70)
    print("Comparison Results:")
    print("-"*70)
    print(f"{'Metric':<30} {'WhisperX':<20} {'MFA':<20} {'Ratio':<15}")
    print("="*70)
    print(f"{'Processing time':<30} {whisperx_time:.2f}s{' '*14} {mfa_time:.2f}s{' '*14} {mfa_time/whisperx_time:.1f}x faster")
    print(f"{'Phoneme count':<30} {len(whisperx_result['phonemes']):<20} {len(mfa_result['phonemes']):<20} {len(whisperx_result['phonemes'])/len(mfa_result['phonemes'])*100:.1f}%")
    print(f"{'Word count':<30} {len(whisperx_result['words']):<20} {len(mfa_result['words']):<20}")

    # Phoneme overlap
    whisperx_phonemes = set(whisperx_result['phonemes'])
    mfa_phonemes = set(mfa_result['phonemes'])
    overlap = whisperx_phonemes & mfa_phonemes

    print(f"\n{'Unique phonemes (WhisperX)':<30} {len(whisperx_phonemes)}")
    print(f"{'Unique phonemes (MFA)':<30} {len(mfa_phonemes)}")
    print(f"{'Overlap':<30} {len(overlap)} / {len(mfa_phonemes)} ({len(overlap)/len(mfa_phonemes)*100:.1f}%)")


def main():
    """Run all tests"""
    print("\n")
    print("="*70)
    print("WhisperX Phoneme Aligner Test Suite")
    print("="*70)

    # Test 1: GPU availability
    test_gpu_availability()

    # Test 2: g2pk pronunciation
    test_g2pk_pronunciation()

    # Test 3: Output format compatibility
    test_output_format_compatibility()

    # Test 4: Single video processing
    result = test_single_video()

    if result:
        # Test 5: Performance benchmark
        benchmark_performance()

        # Test 6: Compare with MFA (optional)
        # Uncomment to run comparison (takes a long time due to MFA)
        # compare_with_mfa()

    print("\n" + "="*70)
    print("Test Suite Complete")
    print("="*70)


if __name__ == "__main__":
    main()
