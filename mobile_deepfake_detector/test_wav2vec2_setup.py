"""
Quick setup verification for Wav2Vec2 Korean Phoneme Aligner

Checks:
1. All dependencies installed
2. slplab model downloads successfully
3. VAD model loads
4. Basic functionality works

Usage:
    conda activate whisperx_env
    cd mobile_deepfake_detector
    python test_wav2vec2_setup.py
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check all required packages are installed"""
    logger.info("=" * 60)
    logger.info("STEP 1: Checking dependencies...")
    logger.info("=" * 60)

    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'whisperx': 'WhisperX',
        'numpy': 'NumPy',
        'scipy': 'SciPy (for WAV processing)'
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} - NOT INSTALLED")
            missing.append(package)

    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False

    logger.info("\n✓ All dependencies installed\n")
    return True


def check_model_download():
    """Check slplab model downloads successfully"""
    logger.info("=" * 60)
    logger.info("STEP 2: Downloading slplab/wav2vec2-xls-r-300m_phone-mfa_korean...")
    logger.info("=" * 60)

    try:
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        import torch

        model_id = "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"

        logger.info(f"Loading processor from {model_id}...")
        processor = AutoProcessor.from_pretrained(model_id)
        logger.info("✓ Processor loaded")

        logger.info(f"Loading model from {model_id}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Wav2Vec2ForCTC.from_pretrained(
            model_id,
            use_safetensors=True  # Use safetensors to avoid PyTorch 2.6 requirement
        ).to(device)
        model.eval()
        logger.info(f"✓ Model loaded on {device}")
        logger.info(f"  - Layers: {model.config.num_hidden_layers}")
        logger.info(f"  - Vocab size: {model.config.vocab_size}")
        logger.info(f"  - Hidden size: {model.config.hidden_size}")

        logger.info("\n✓ slplab model ready\n")
        return True

    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False


def check_vad_model():
    """Check VAD model loads (optional, may require HF token)"""
    logger.info("=" * 60)
    logger.info("STEP 3: Checking VAD model (pyannote)...")
    logger.info("=" * 60)

    try:
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection

        logger.info("Loading pyannote segmentation model...")
        model = Model.from_pretrained("pyannote/segmentation-3.0")
        logger.info("✓ VAD model loaded")

        vad_pipeline = VoiceActivityDetection(segmentation=model)
        logger.info("✓ VAD pipeline created")

        logger.info("\n✓ VAD ready\n")
        return True

    except Exception as e:
        logger.warning(f"⚠ VAD model loading failed: {e}")
        logger.warning("  This may require Hugging Face token for pyannote models")
        logger.warning("  VAD will use fallback (full audio processing)")
        logger.warning("  To enable VAD:")
        logger.warning("    1. Get token from https://huggingface.co/settings/tokens")
        logger.warning("    2. huggingface-cli login")
        logger.warning("    3. Accept pyannote license: https://huggingface.co/pyannote/segmentation-3.0")
        return False


def check_ipa_to_jamo():
    """Check IPA to Jamo conversion module"""
    logger.info("=" * 60)
    logger.info("STEP 4: Checking IPA to Jamo conversion...")
    logger.info("=" * 60)

    try:
        from src.utils.ipa_to_jamo import convert_ipa_to_jamo, convert_ipa_list_to_jamo

        # Test conversion
        test_ipa = ['m', 'ʌ', 'n', 'tɕ', 'h', 'a']
        jamo_list, unmapped = convert_ipa_list_to_jamo(test_ipa)

        logger.info(f"Test: {test_ipa}")
        logger.info(f"Result: {jamo_list}")
        logger.info(f"Unmapped: {unmapped if unmapped else 'None'}")

        logger.info("\n✓ IPA to Jamo conversion working\n")
        return True

    except Exception as e:
        logger.error(f"✗ IPA to Jamo conversion failed: {e}")
        return False


def test_basic_functionality():
    """Test basic Wav2Vec2 aligner functionality"""
    logger.info("=" * 60)
    logger.info("STEP 5: Testing basic functionality...")
    logger.info("=" * 60)

    try:
        from src.utils.wav2vec2_korean_phoneme_aligner import Wav2Vec2KoreanPhonemeAligner
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing aligner on {device}...")

        aligner = Wav2Vec2KoreanPhonemeAligner(device=device)
        logger.info("✓ Aligner initialized successfully")

        # Test empty result
        empty = aligner._empty_result()
        assert 'phonemes' in empty
        assert 'method' in empty
        logger.info("✓ Empty result generation works")

        # Test uniform distribution
        intervals = aligner._uniform_distribution(1.0, 10)
        assert len(intervals) == 10
        logger.info("✓ Uniform distribution works")

        logger.info("\n✓ Basic functionality OK\n")
        return True

    except Exception as e:
        logger.error(f"✗ Basic functionality test failed: {e}", exc_info=True)
        return False


def main():
    """Run all checks"""
    logger.info("\n" + "=" * 60)
    logger.info("WAV2VEC2 KOREAN PHONEME ALIGNER - SETUP VERIFICATION")
    logger.info("=" * 60 + "\n")

    results = []

    # Check 1: Dependencies
    results.append(("Dependencies", check_dependencies()))

    if not results[0][1]:
        logger.error("\n❌ Cannot proceed without dependencies. Install missing packages first.\n")
        sys.exit(1)

    # Check 2: Model download
    results.append(("Model Download", check_model_download()))

    # Check 3: VAD model (optional)
    results.append(("VAD Model", check_vad_model()))

    # Check 4: IPA to Jamo
    results.append(("IPA to Jamo", check_ipa_to_jamo()))

    # Check 5: Basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{name:<25} {status}")

    logger.info("=" * 60)

    all_critical_passed = all(
        passed for name, passed in results
        if name != "VAD Model"  # VAD is optional
    )

    if all_critical_passed:
        logger.info("\n✅ Setup verification PASSED!")
        logger.info("\nNext steps:")
        logger.info("  1. Test on real video:")
        logger.info("     python test_wav2vec2_setup.py --test-video <path>")
        logger.info("  2. Run benchmark:")
        logger.info("     python tests/test_wav2vec2_korean.py --benchmark --video <path>")
        logger.info("  3. Run full test suite:")
        logger.info("     pytest tests/test_wav2vec2_korean.py -v")
    else:
        logger.error("\n❌ Setup verification FAILED. Fix errors above.\n")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-video', type=str, help='Test on specific video file')
    args = parser.parse_args()

    if args.test_video:
        # Test on actual video
        from src.utils.wav2vec2_korean_phoneme_aligner import Wav2Vec2KoreanPhonemeAligner

        logger.info(f"\nTesting on video: {args.test_video}")
        aligner = Wav2Vec2KoreanPhonemeAligner()

        import time
        start = time.time()
        result = aligner.align_video(args.test_video)
        elapsed = time.time() - start

        logger.info(f"\n{'='*60}")
        logger.info("TEST RESULT")
        logger.info(f"{'='*60}")
        logger.info(f"Duration:      {result['duration']:.2f}s")
        logger.info(f"Processing:    {elapsed:.2f}s")
        logger.info(f"Phonemes:      {len(result['phonemes'])}")
        logger.info(f"VAD segments:  {len(result['vad_segments'])}")
        if result['vad_segments']:
            speech_duration = sum(e - s for s, e in result['vad_segments'])
            logger.info(f"Speech time:   {speech_duration:.2f}s ({100 * speech_duration / result['duration']:.1f}%)")
        logger.info(f"Method:        {result['method']}")
        logger.info(f"{'='*60}")

        # Show segment-by-segment results
        if result['vad_segments'] and result['phonemes']:
            logger.info("\n" + "="*60)
            logger.info("SEGMENT-BY-SEGMENT RESULTS")
            logger.info("="*60)

            for seg_idx, (seg_start, seg_end) in enumerate(result['vad_segments'], 1):
                # Get segment text if available
                segment_text = ""
                if 'segment_texts' in result and seg_idx-1 < len(result['segment_texts']):
                    segment_text = result['segment_texts'][seg_idx-1]

                logger.info(f"\nSegment #{seg_idx} [{seg_start:.3f}s - {seg_end:.3f}s]")
                if segment_text:
                    logger.info(f"  Text: \"{segment_text}\"")
                logger.info(f"  Phonemes:")

                # Find all phonemes in this segment
                phoneme_num = 1
                for i, (p_start, p_end) in enumerate(result['intervals']):
                    if seg_start <= p_start <= seg_end:
                        phoneme = result['phonemes'][i]
                        logger.info(f"    {phoneme_num}. '{phoneme}' [{p_start:.3f}s - {p_end:.3f}s]")
                        phoneme_num += 1

    else:
        main()
