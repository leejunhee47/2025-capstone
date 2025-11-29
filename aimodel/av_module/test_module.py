"""
Test script for audio and video extraction module
Tests the module with sample videos from dataset
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules directly from current directory
from video_processor import VideoProcessor
from preprocessing_pipeline import PreprocessingPipeline
from config import Config
from utils import setup_logging, get_dataset_videos


def save_audio_preview(audio, video_id, output_dir="test_output/preview/audio"):
    """
    Save audio as WAV file for preview

    Args:
        audio: Audio numpy array
        video_id: Video identifier
        output_dir: Output directory for WAV files
    """
    import soundfile as sf
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_path = output_path / f"{video_id}.wav"
    sf.write(str(wav_path), audio, 16000)  # 16kHz sampling rate
    print(f"  Audio preview saved: {wav_path}")


def save_frames_preview(frames, video_id, output_dir="test_output/preview/frames", max_frames=5):
    """
    Save frames as JPG images for preview

    Args:
        frames: Frames numpy array (N, H, W, C)
        video_id: Video identifier
        output_dir: Output directory for images
        max_frames: Maximum number of frames to save
    """
    import cv2
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_frames = min(len(frames), max_frames)
    for i in range(num_frames):
        frame = frames[i]

        # Convert normalized [0,1] to [0,255]
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        # RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img_path = output_path / f"{video_id}_frame_{i:03d}.jpg"
        cv2.imwrite(str(img_path), frame_bgr)

    print(f"  Frames preview saved: {output_path / video_id}_frame_*.jpg ({num_frames} frames)")


def test_single_video():
    """Test processing a single video"""
    print("\n" + "="*60)
    print("TEST 1: Single Video Processing")
    print("="*60)

    # Find a sample video
    dataset_root = Path("dataset_sample/원천데이터/train_변조")
    videos = list(dataset_root.rglob("*.mp4"))

    if len(videos) == 0:
        print("No videos found in dataset!")
        return False

    video_path = videos[0]
    print(f"Testing with: {video_path}")

    # Create processor
    processor = VideoProcessor()

    # Process video
    try:
        result = processor.process_video(
            video_path,
            extract_audio=True,
            extract_frames=True,
            max_frames=50  # Limit frames for testing
        )

        print("\n[PASS] Processing successful!")
        print(f"  Audio shape: {result['audio'].shape if result['audio'] is not None else 'None'}")
        print(f"  Frames shape: {result['frames'].shape if result['frames'] is not None else 'None'}")
        print(f"  Processing time: {result['metadata']['processing_time']:.2f}s")

        return True

    except Exception as e:
        print(f"\n[FAIL] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_validation():
    """Test video validation"""
    print("\n" + "="*60)
    print("TEST 2: Video Validation")
    print("="*60)

    dataset_root = Path("dataset_sample/원천데이터/train_변조")
    videos = list(dataset_root.rglob("*.mp4"))[:3]  # Test 3 videos

    processor = VideoProcessor()

    for video_path in videos:
        print(f"\nValidating: {video_path.name}")
        validation = processor.validate_video(video_path)

        print(f"  Exists: {validation['exists']}")
        print(f"  Size: {validation['size_bytes'] / (1024*1024):.2f} MB")
        print(f"  Can open: {validation['can_open']}")
        print(f"  Has video: {validation['has_video']}")
        print(f"  Has audio: {validation['has_audio']}")

        if validation['video_info']:
            info = validation['video_info']
            print(f"  Video info: {info['width']}x{info['height']}, "
                  f"{info['fps']:.2f}fps, {info['frame_count']} frames")

    return True


def test_batch_processing():
    """Test batch processing with pipeline"""
    print("\n" + "="*60)
    print("TEST 3: Batch Processing with Pipeline")
    print("="*60)

    # Find sample videos
    dataset_root = Path("dataset_sample/원천데이터/train_변조")
    videos = list(dataset_root.rglob("*.mp4"))[:5]  # Test 5 videos

    if len(videos) == 0:
        print("No videos found!")
        return False

    print(f"Testing with {len(videos)} videos")

    # Create pipeline
    output_dir = Path("test_output")
    pipeline = PreprocessingPipeline(
        output_dir=output_dir,
        num_workers=2  # Use 2 workers for testing
    )

    # Process dataset
    try:
        results = pipeline.preprocess_dataset(
            videos,
            extract_audio=True,
            extract_frames=True,
            max_frames=30,  # Limit frames for testing
            use_multiprocessing=True,
            save_results=True
        )

        successful = sum(1 for r in results if r['metadata']['success'])
        print(f"\n[PASS] Batch processing completed: {successful}/{len(videos)} successful")

        # Check output files
        print("\nOutput files created:")
        for subdir in ['audio', 'frames', 'metadata']:
            files = list((output_dir / subdir).glob('*'))
            print(f"  {subdir}/: {len(files)} files")

        return True

    except Exception as e:
        print(f"\n[FAIL] Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_extraction():
    """Test audio extraction specifically"""
    print("\n" + "="*60)
    print("TEST 4: Audio Extraction")
    print("="*60)

    dataset_root = Path("dataset_sample/원천데이터/train_변조")
    videos = list(dataset_root.rglob("*.mp4"))

    if len(videos) == 0:
        print("No videos found!")
        return False

    video_path = videos[0]
    print(f"Testing with: {video_path.name}")

    processor = VideoProcessor()

    try:
        # Extract raw audio
        result = processor.process_video(
            video_path,
            extract_audio=True,
            extract_frames=False,
            audio_feature_type='raw'
        )

        if result['audio'] is not None:
            audio = result['audio']
            print(f"\n[PASS] Audio extracted successfully!")
            print(f"  Shape: {audio.shape}")
            print(f"  Duration: {len(audio) / 16000:.2f}s")
            print(f"  Sample rate: 16kHz")
            print(f"  Min/Max values: {audio.min():.4f} / {audio.max():.4f}")

            # Save audio preview as WAV
            video_id = video_path.stem
            save_audio_preview(audio, video_id)

            return True
        else:
            print("\n[FAIL] Audio extraction failed!")
            return False

    except Exception as e:
        print(f"\n[FAIL] Audio extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_extraction():
    """Test frame extraction specifically"""
    print("\n" + "="*60)
    print("TEST 5: Frame Extraction")
    print("="*60)

    dataset_root = Path("dataset_sample/원천데이터/train_변조")
    videos = list(dataset_root.rglob("*.mp4"))

    if len(videos) == 0:
        print("No videos found!")
        return False

    video_path = videos[0]
    print(f"Testing with: {video_path.name}")

    processor = VideoProcessor()

    try:
        # Extract frames
        result = processor.process_video(
            video_path,
            extract_audio=False,
            extract_frames=True,
            max_frames=50
        )

        if result['frames'] is not None:
            frames = result['frames']
            print(f"\n[PASS] Frames extracted successfully!")
            print(f"  Shape: {frames.shape}")
            print(f"  Number of frames: {len(frames)}")
            print(f"  Frame size: {frames.shape[1]}x{frames.shape[2]}")
            print(f"  Channels: {frames.shape[3]}")
            print(f"  Value range: [{frames.min():.4f}, {frames.max():.4f}]")
            print(f"  Normalized: {'Yes' if frames.max() <= 1.0 else 'No'}")

            # Save frames preview as JPG images
            video_id = video_path.stem
            save_frames_preview(frames, video_id, max_frames=5)

            return True
        else:
            print("\n[FAIL] Frame extraction failed!")
            return False

    except Exception as e:
        print(f"\n[FAIL] Frame extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" AUDIO AND VIDEO EXTRACTION MODULE TEST SUITE")
    print("="*70)

    # Setup logging
    setup_logging(level=20)  # INFO level

    tests = [
        ("Single Video Processing", test_single_video),
        ("Video Validation", test_video_validation),
        ("Audio Extraction", test_audio_extraction),
        ("Frame Extraction", test_frame_extraction),
        ("Batch Processing", test_batch_processing),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Print summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for r in results.values() if r)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70 + "\n")

    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
