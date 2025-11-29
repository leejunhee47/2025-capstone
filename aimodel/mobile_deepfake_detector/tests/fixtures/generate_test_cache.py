"""
Generate NPZ cache files for test fixtures using FeatureCache.

Usage:
    python tests/fixtures/generate_test_cache.py

This generates pre-extracted features for real/fake test videos
to speed up E2E tests from ~90s to ~1s per video.

The FeatureCache system will:
1. Check if video is already cached (by video_id + path hash)
2. If cached, skip extraction
3. If not cached, extract and save to cache

Note: UnifiedFeatureExtractor automatically trims videos longer than 60s.
"""

import sys
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.xai.unified_feature_extractor import UnifiedFeatureExtractor
from src.utils.feature_cache import FeatureCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test video paths
TEST_VIDEOS = {
    'real': Path("E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터/train_탐지방해/원본영상/01_원본영상/01_원본영상/0e105f8ec5146f9737d0_132.mp4"),
    'fake': Path("E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터/train_탐지방해/변조영상/01_변조영상/01_변조영상/0e105f8ec5146f9737d0_026f9b9514f28f37a3fd_2_0035.mp4"),
}


def main():
    print("=" * 60)
    print("Generating Test Feature Cache")
    print("=" * 60)

    # Initialize cache
    cache = FeatureCache()
    print(f"\nCache directory: {cache.cache_dir}")
    print(f"Currently cached: {len(cache.index)} videos")

    # Check which videos need processing
    videos_to_process = []
    for label, video_path in TEST_VIDEOS.items():
        if not video_path.exists():
            print(f"\n[SKIP] {label.upper()} video not found: {video_path}")
            continue

        if cache.exists(str(video_path)):
            print(f"\n[CACHED] {label.upper()}: {video_path.name}")
        else:
            print(f"\n[NEED] {label.upper()}: {video_path.name}")
            videos_to_process.append((label, video_path))

    if not videos_to_process:
        print("\n" + "=" * 60)
        print("All test videos are already cached!")
        print("=" * 60)
        return

    # Initialize extractor with max_duration=60.0 (built-in trimming)
    print(f"\n[1/2] Initializing UnifiedFeatureExtractor (max_duration=60s)...")
    extractor = UnifiedFeatureExtractor(device="cuda", max_duration=60.0)

    # Process videos
    for i, (label, video_path) in enumerate(videos_to_process, 1):
        print(f"\n[{i}/{len(videos_to_process)}] Processing {label.upper()} video...")
        print(f"  Input: {video_path}")

        # get_or_extract will:
        # 1. Check cache first
        # 2. If not cached, extract (with auto-trimming if >60s)
        # 3. Save to cache
        features = cache.get_or_extract(str(video_path), extractor)

        # Print summary
        print(f"  Features extracted:")
        for key, value in features.items():
            if hasattr(value, 'shape'):
                print(f"    {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"    {key}: list[{len(value)}]")
            else:
                print(f"    {key}: {type(value).__name__}")

    print("\n" + "=" * 60)
    print("Done! Cache updated.")
    print(f"Total cached videos: {len(cache.index)}")
    print("=" * 60)


if __name__ == "__main__":
    main()