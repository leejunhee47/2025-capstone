"""
Feature Cache System for Hybrid XAI Pipeline

video_id 기반으로 전처리된 features를 NPZ 파일로 캐싱하여
테스트 및 추론 시 feature extraction 시간을 절약합니다.

Usage:
    cache = FeatureCache(cache_dir="path/to/cache")

    # 캐시에서 로드 시도, 없으면 추출 후 저장
    features = cache.get_or_extract(video_path, extractor)

    # 캐시만 확인
    features = cache.load(video_path)  # None if not cached

    # 강제 저장
    cache.save(video_path, features)
"""

import json
import hashlib
import logging
import platform
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _get_default_cache_dir() -> Path:
    """Get platform-specific default cache directory."""
    if platform.system() == "Windows":
        return Path("E:/capstone/mobile_deepfake_detector/cache/unified_features")
    else:
        # Linux (AWS EC2)
        return Path("/home/ec2-user/2025-capstone/aimodel/mobile_deepfake_detector/cache/unified_features")


class FeatureCache:
    """Video feature cache with NPZ storage."""

    # Default cache directory (platform-specific)
    DEFAULT_CACHE_DIR = _get_default_cache_dir()

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize feature cache.

        Args:
            cache_dir: Directory to store cached NPZ files.
                      If None, uses DEFAULT_CACHE_DIR.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index file for video_id -> npz mapping
        self.index_path = self.cache_dir / "cache_index.json"
        self.index = self._load_index()

        logger.info(f"FeatureCache initialized: {self.cache_dir}")
        logger.info(f"  Cached videos: {len(self.index)}")

    def _load_index(self) -> Dict[str, str]:
        """Load cache index from JSON file."""
        if self.index_path.exists():
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save cache index to JSON file."""
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    @staticmethod
    def get_video_id(video_path: str) -> str:
        """
        Extract video_id from video path.

        Uses filename without extension as video_id.
        Example: "/path/to/video_123.mp4" -> "video_123"
        """
        return Path(video_path).stem

    @staticmethod
    def get_video_hash(video_path: str) -> str:
        """
        Generate short hash from video path for unique identification.

        Useful when video_ids might collide across different directories.
        """
        return hashlib.md5(str(video_path).encode()).hexdigest()[:8]

    def _get_cache_key(self, video_path: str) -> str:
        """Generate unique cache key for video."""
        video_id = self.get_video_id(video_path)
        # Include path hash to avoid collisions
        path_hash = self.get_video_hash(video_path)
        return f"{video_id}_{path_hash}"

    def _get_npz_path(self, cache_key: str) -> Path:
        """Get NPZ file path for cache key."""
        return self.cache_dir / f"{cache_key}.npz"

    def exists(self, video_path: str) -> bool:
        """Check if video features are cached."""
        cache_key = self._get_cache_key(video_path)
        return cache_key in self.index and self._get_npz_path(cache_key).exists()

    def load(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Load cached features for video.

        Args:
            video_path: Path to video file

        Returns:
            Features dict if cached, None otherwise
        """
        cache_key = self._get_cache_key(video_path)

        if cache_key not in self.index:
            logger.debug(f"Cache miss (not in index): {video_path}")
            return None

        npz_path = self._get_npz_path(cache_key)
        if not npz_path.exists():
            logger.warning(f"Cache index entry exists but NPZ missing: {npz_path}")
            # Remove stale index entry
            del self.index[cache_key]
            self._save_index()
            return None

        logger.info(f"Cache hit: {cache_key}")
        return self._load_npz(npz_path)

    def save(self, video_path: str, features: Dict[str, Any]) -> Path:
        """
        Save features to cache.

        Args:
            video_path: Path to video file
            features: Features dict from UnifiedFeatureExtractor

        Returns:
            Path to saved NPZ file
        """
        cache_key = self._get_cache_key(video_path)
        npz_path = self._get_npz_path(cache_key)

        # Save features to NPZ
        self._save_npz(npz_path, features)

        # Update index
        self.index[cache_key] = {
            'video_path': str(video_path),
            'video_id': self.get_video_id(video_path),
            'npz_path': str(npz_path.name),
            'keys': list(features.keys()),
        }
        self._save_index()

        logger.info(f"Cached: {cache_key} -> {npz_path}")
        return npz_path

    def get_or_extract(
        self,
        video_path: str,
        extractor: Any,
        force_extract: bool = False
    ) -> Dict[str, Any]:
        """
        Get features from cache or extract and cache.

        Args:
            video_path: Path to video file
            extractor: UnifiedFeatureExtractor instance
            force_extract: If True, always extract (ignore cache)

        Returns:
            Features dict
        """
        if not force_extract:
            cached = self.load(video_path)
            if cached is not None:
                return cached

        # Extract features
        logger.info(f"Extracting features: {video_path}")
        features = extractor.extract_all(video_path)

        # Save to cache
        self.save(video_path, features)

        return features

    def _save_npz(self, npz_path: Path, features: Dict[str, Any]):
        """Save features dict to NPZ file."""
        npz_data = {}

        for key, value in features.items():
            if isinstance(value, np.ndarray):
                npz_data[key] = value
            elif isinstance(value, (int, float)):
                npz_data[key] = np.array(value)
            elif isinstance(value, list):
                # Handle phoneme_intervals (list of dicts)
                if key == 'phoneme_intervals':
                    npz_data[f'{key}_json'] = np.array(json.dumps(value))
                else:
                    try:
                        npz_data[key] = np.array(value)
                    except:
                        logger.debug(f"Skipping non-serializable list: {key}")
            elif key == 'whisper_result':
                # Store whisper_result as JSON for potential reuse
                try:
                    npz_data[f'{key}_json'] = np.array(json.dumps(value, default=str))
                except:
                    logger.debug("Skipping whisper_result (serialization failed)")
            else:
                logger.debug(f"Skipping: {key} ({type(value)})")

        np.savez_compressed(npz_path, **npz_data)

    def _load_npz(self, npz_path: Path) -> Dict[str, Any]:
        """Load features dict from NPZ file."""
        data = np.load(npz_path, allow_pickle=True)
        features = {}

        for key in data.files:
            if key.endswith('_json'):
                # Restore JSON-serialized data
                original_key = key.replace('_json', '')
                try:
                    features[original_key] = json.loads(str(data[key]))
                except:
                    features[original_key] = data[key]
            elif data[key].ndim == 0:
                # Scalar value
                features[key] = data[key].item()
            else:
                features[key] = data[key]

        return features

    def list_cached(self) -> list:
        """List all cached video entries."""
        return list(self.index.values())

    def clear(self):
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self._save_index()
        logger.info("Cache cleared")


# Singleton instance for convenience
_default_cache: Optional[FeatureCache] = None

def get_default_cache() -> FeatureCache:
    """Get or create default FeatureCache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = FeatureCache()
    return _default_cache