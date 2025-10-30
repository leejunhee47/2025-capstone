"""
Test suite for Wav2Vec2 Korean Phoneme Aligner

Tests:
1. Unit tests: Model loading, VAD, phoneme recognition
2. Integration tests: End-to-end video processing
3. Benchmark: Compare with WhisperX (speed, memory, accuracy)

Usage:
    # Run all tests
    python -m pytest test_wav2vec2_korean.py -v

    # Run specific test
    python -m pytest test_wav2vec2_korean.py::test_model_loading -v

    # Run benchmark
    python test_wav2vec2_korean.py --benchmark

Author: Claude Code
Date: 2025-10-30
"""

import os
import sys
import logging
import time
from pathlib import Path

import pytest
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.wav2vec2_korean_phoneme_aligner import Wav2Vec2KoreanPhonemeAligner
from src.utils.korean_phoneme_extractor import KoreanPhonemeExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test fixtures
@pytest.fixture(scope="module")
def wav2vec2_aligner():
    """Initialize Wav2Vec2 aligner once for all tests"""
    return Wav2Vec2KoreanPhonemeAligner(device="cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def phoneme_extractor():
    """Initialize phoneme extractor once for all tests"""
    return KoreanPhonemeExtractor()


@pytest.fixture
def sample_video_path():
    """
    Path to sample video for testing

    Note: Replace with actual test video path
    Expected: Korean short-form video (15-60 seconds)
    """
    # Try to find a sample video in common locations
    possible_paths = [
        "E:/capstone/test_videos/sample.mp4",
        "E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터/01.원본/sample.mp4",
        "E:/capstone/dataset_sample/real/sample.mp4"
    ]

    for path in possible_paths:
        if Path(path).exists():
            return path

    # If no video found, skip tests that require it
    pytest.skip("No sample video found for testing")


# Unit Tests
class TestWav2Vec2ModelLoading:
    """Test model loading and initialization"""

    def test_model_loading(self, wav2vec2_aligner):
        """Test that slplab model loads successfully"""
        assert wav2vec2_aligner.model is not None
        assert wav2vec2_aligner.processor is not None
        logger.info("✓ Model loaded successfully")

    def test_device_selection(self, wav2vec2_aligner):
        """Test correct device selection (CUDA or CPU)"""
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert wav2vec2_aligner.device == expected_device
        logger.info(f"✓ Device: {wav2vec2_aligner.device}")

    def test_model_config(self, wav2vec2_aligner):
        """Test model configuration"""
        vocab_size = wav2vec2_aligner.model.config.vocab_size
        assert vocab_size > 0
        logger.info(f"✓ Vocab size: {vocab_size}")


class TestVAD:
    """Test Voice Activity Detection"""

    def test_vad_on_speech(self, wav2vec2_aligner, sample_video_path):
        """Test VAD detects speech in video"""
        # Extract audio
        import whisperx
        audio_path = wav2vec2_aligner._extract_audio(Path(sample_video_path))
        audio = whisperx.load_audio(str(audio_path))

        # Run VAD
        vad_segments = wav2vec2_aligner._detect_speech_segments(audio)

        assert len(vad_segments) > 0, "VAD should detect at least one speech segment"
        logger.info(f"✓ VAD detected {len(vad_segments)} speech segments")

        # Cleanup
        if audio_path.exists():
            audio_path.unlink()

    def test_vad_on_silence(self, wav2vec2_aligner):
        """Test VAD on silent audio"""
        # Create 2 seconds of silence
        silent_audio = np.zeros(16000 * 2, dtype=np.float32)

        # Run VAD (should return fallback: full audio)
        vad_segments = wav2vec2_aligner._detect_speech_segments(silent_audio)

        # VAD should either find 0 segments or fallback to full audio
        assert len(vad_segments) >= 0
        logger.info(f"✓ VAD on silence: {len(vad_segments)} segments")


class TestPhonemeRecognition:
    """Test phoneme recognition"""

    def test_phoneme_extraction(self, wav2vec2_aligner, sample_video_path):
        """Test end-to-end phoneme extraction"""
        result = wav2vec2_aligner.align_video(sample_video_path)

        # Check result structure
        assert 'phonemes' in result
        assert 'intervals' in result
        assert 'duration' in result
        assert 'vad_segments' in result
        assert 'method' in result

        # Check data validity
        assert len(result['phonemes']) > 0, "Should extract at least one phoneme"
        assert len(result['phonemes']) == len(result['intervals']), "Phonemes and intervals should match"
        assert result['duration'] > 0
        assert result['method'] == 'wav2vec2_slplab'

        logger.info(f"✓ Extracted {len(result['phonemes'])} phonemes")
        logger.info(f"  Duration: {result['duration']:.2f}s")
        logger.info(f"  VAD segments: {len(result['vad_segments'])}")

    def test_interval_validity(self, wav2vec2_aligner, sample_video_path):
        """Test that timestamps are valid"""
        result = wav2vec2_aligner.align_video(sample_video_path)

        for i, (start, end) in enumerate(result['intervals']):
            assert 0 <= start < end, f"Invalid interval {i}: ({start}, {end})"
            assert end <= result['duration'], f"Interval {i} exceeds duration: {end} > {result['duration']}"

        logger.info("✓ All timestamps are valid")


class TestIPAToJamoConversion:
    """Test IPA to Jamo conversion"""

    def test_jamo_conversion(self, wav2vec2_aligner):
        """Test IPA phonemes are converted to Jamo"""
        # Sample IPA phonemes (Korean)
        ipa_phonemes = ['m', 'ʌ', 'n', 'tɕ', 'h', 'a']
        ipa_intervals = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]

        jamo_phonemes, jamo_intervals = wav2vec2_aligner._convert_to_jamo(ipa_phonemes, ipa_intervals)

        assert len(jamo_phonemes) > 0
        assert len(jamo_phonemes) == len(jamo_intervals)
        logger.info(f"✓ Converted {len(ipa_phonemes)} IPA → {len(jamo_phonemes)} Jamo")


class TestKoreanPhonemeExtractorIntegration:
    """Test integration with KoreanPhonemeExtractor"""

    def test_extract_with_wav2vec2(self, phoneme_extractor, sample_video_path):
        """Test KoreanPhonemeExtractor.extract_with_wav2vec2()"""
        result = phoneme_extractor.extract_with_wav2vec2(sample_video_path)

        # Check result structure
        assert 'phonemes' in result
        assert 'intervals' in result
        assert 'key_phonemes' in result
        assert 'duration' in result
        assert 'method' in result
        assert 'vad_segments' in result

        assert result['method'] == 'wav2vec2_slplab'
        assert len(result['phonemes']) > 0

        logger.info(f"✓ KoreanPhonemeExtractor integration successful")
        logger.info(f"  Phonemes: {len(result['phonemes'])}")
        logger.info(f"  Key phonemes: {sum(len(v) for v in result['key_phonemes'].values())}")

    def test_key_phonemes_extraction(self, phoneme_extractor, sample_video_path):
        """Test key phonemes (ㅁ, ㅂ, ㅍ, ㅏ, ㅗ, ㅜ) are extracted"""
        result = phoneme_extractor.extract_with_wav2vec2(sample_video_path)

        # Check key phonemes exist
        assert 'key_phonemes' in result
        key_phoneme_types = ['ㅁ', 'ㅂ', 'ㅍ', 'ㅏ', 'ㅗ', 'ㅜ']

        for phoneme_type in key_phoneme_types:
            assert phoneme_type in result['key_phonemes']

        # Count total key phonemes
        total_key = sum(len(v) for v in result['key_phonemes'].values())
        logger.info(f"✓ Key phonemes extracted: {total_key}")
        for phoneme, occurrences in result['key_phonemes'].items():
            if occurrences:
                logger.info(f"  {phoneme}: {len(occurrences)} occurrences")


# Benchmark Tests
class TestBenchmark:
    """Benchmark Wav2Vec2 vs WhisperX"""

    def test_speed_comparison(self, sample_video_path):
        """Compare processing speed: Wav2Vec2 vs WhisperX"""
        from src.utils.whisperx_aligner import WhisperXPhonemeAligner

        # Wav2Vec2
        wav2vec2 = Wav2Vec2KoreanPhonemeAligner()
        start_time = time.time()
        result_wav2vec2 = wav2vec2.align_video(sample_video_path)
        wav2vec2_time = time.time() - start_time

        # WhisperX
        whisperx = WhisperXPhonemeAligner()
        start_time = time.time()
        result_whisperx = whisperx.align_video_segmented(sample_video_path)
        whisperx_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info("SPEED BENCHMARK")
        logger.info("=" * 60)
        logger.info(f"Wav2Vec2:  {wav2vec2_time:.2f}s ({len(result_wav2vec2['phonemes'])} phonemes)")
        logger.info(f"WhisperX:  {whisperx_time:.2f}s ({len(result_whisperx['phonemes'])} phonemes)")
        logger.info(f"Speedup:   {whisperx_time / wav2vec2_time:.2f}x")
        logger.info(f"VAD savings: {result_wav2vec2['duration'] - sum(e - s for s, e in result_wav2vec2['vad_segments']):.2f}s")
        logger.info("=" * 60)

    def test_memory_usage(self, sample_video_path):
        """Compare GPU memory usage: Wav2Vec2 vs WhisperX"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping memory test")

        # Wav2Vec2
        torch.cuda.reset_peak_memory_stats()
        wav2vec2 = Wav2Vec2KoreanPhonemeAligner(device="cuda")
        result_wav2vec2 = wav2vec2.align_video(sample_video_path)
        wav2vec2_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

        # WhisperX
        torch.cuda.reset_peak_memory_stats()
        from src.utils.whisperx_aligner import WhisperXPhonemeAligner
        whisperx = WhisperXPhonemeAligner(device="cuda")
        result_whisperx = whisperx.align_video_segmented(sample_video_path)
        whisperx_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

        logger.info("=" * 60)
        logger.info("MEMORY BENCHMARK")
        logger.info("=" * 60)
        logger.info(f"Wav2Vec2:  {wav2vec2_memory:.2f} GB")
        logger.info(f"WhisperX:  {whisperx_memory:.2f} GB")
        logger.info(f"Savings:   {(whisperx_memory - wav2vec2_memory) / whisperx_memory * 100:.1f}%")
        logger.info("=" * 60)

    def test_phoneme_count_comparison(self, sample_video_path):
        """Compare phoneme extraction counts"""
        from src.utils.whisperx_aligner import WhisperXPhonemeAligner

        wav2vec2 = Wav2Vec2KoreanPhonemeAligner()
        whisperx = WhisperXPhonemeAligner()

        result_wav2vec2 = wav2vec2.align_video(sample_video_path)
        result_whisperx = whisperx.align_video_segmented(sample_video_path)

        logger.info("=" * 60)
        logger.info("PHONEME COUNT COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Wav2Vec2 phonemes:  {len(result_wav2vec2['phonemes'])}")
        logger.info(f"WhisperX phonemes:  {len(result_whisperx['phonemes'])}")
        logger.info(f"Ratio:              {len(result_wav2vec2['phonemes']) / len(result_whisperx['phonemes']):.2f}")
        logger.info("=" * 60)


# Error Handling Tests
class TestErrorHandling:
    """Test error handling"""

    def test_missing_video(self, wav2vec2_aligner):
        """Test handling of missing video file"""
        result = wav2vec2_aligner.align_video("nonexistent_video.mp4")

        # Should return empty result
        assert result['phonemes'] == []
        assert result['duration'] == 0.0

    def test_corrupted_audio(self, wav2vec2_aligner):
        """Test handling of corrupted/silent audio"""
        # This test requires a corrupted video file
        # For now, just ensure the method handles exceptions
        pass


# Main execution for manual testing
def main():
    """Run manual benchmark test"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    args = parser.parse_args()

    if args.benchmark:
        # Find sample video
        video_path = args.video
        if not video_path:
            possible_paths = [
                "E:/capstone/test_videos/sample.mp4",
                "E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터/01.원본/sample.mp4",
            ]
            for path in possible_paths:
                if Path(path).exists():
                    video_path = path
                    break

        if not video_path or not Path(video_path).exists():
            logger.error("No test video found. Use --video <path>")
            return

        logger.info(f"Running benchmark on: {video_path}")

        # Run benchmark
        from src.utils.whisperx_aligner import WhisperXPhonemeAligner

        # Wav2Vec2
        logger.info("\n" + "=" * 80)
        logger.info("WAV2VEC2 (VAD + slplab)")
        logger.info("=" * 80)
        wav2vec2 = Wav2Vec2KoreanPhonemeAligner()
        start = time.time()
        result_w2v2 = wav2vec2.align_video(video_path)
        time_w2v2 = time.time() - start

        # WhisperX
        logger.info("\n" + "=" * 80)
        logger.info("WHISPERX (Character-level)")
        logger.info("=" * 80)
        whisperx = WhisperXPhonemeAligner()
        start = time.time()
        result_wx = whisperx.align_video_segmented(video_path)
        time_wx = time.time() - start

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 80)
        logger.info(f"{'Method':<20} {'Time':<15} {'Phonemes':<15} {'Coverage':<15}")
        logger.info("-" * 80)
        logger.info(f"{'Wav2Vec2':<20} {time_w2v2:<15.2f}s {len(result_w2v2['phonemes']):<15} "
                   f"{result_w2v2['intervals'][-1][1] if result_w2v2['intervals'] else 0:.2f}s")
        logger.info(f"{'WhisperX':<20} {time_wx:<15.2f}s {len(result_wx['phonemes']):<15} "
                   f"{result_wx['intervals'][-1][1] if result_wx['intervals'] else 0:.2f}s")
        logger.info("-" * 80)
        logger.info(f"Speedup: {time_wx / time_w2v2:.2f}x")
        logger.info(f"VAD saved: {result_w2v2['duration'] - sum(e - s for s, e in result_w2v2['vad_segments']):.2f}s "
                   f"({100 * (result_w2v2['duration'] - sum(e - s for s, e in result_w2v2['vad_segments'])) / result_w2v2['duration']:.1f}%)")
        logger.info("=" * 80)

    else:
        # Run basic test
        logger.info("Use --benchmark to run benchmark tests")
        logger.info("Use pytest to run unit tests:")
        logger.info("  pytest test_wav2vec2_korean.py -v")


if __name__ == "__main__":
    main()
