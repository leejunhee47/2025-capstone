"""
Phoneme-Organized Dataset for Korean Deepfake Detection (PIA-style)

This module implements a PyTorch Dataset that organizes frames by phoneme class
following the PIA (Phoneme-Temporal and Identity-Dynamic Analysis) approach.

Key differences from temporal datasets:
- Traditional: [frame1, frame2, frame3, ...] (time order)
- PIA: {phoneme1: [frames], phoneme2: [frames], ...} (phoneme class order)

Tensor organization:
- geometry: (batch, P, F, geo_dim) where P=14 phonemes, F=5 frames per phoneme, geo_dim=1 (MAR) or 4 (full)
- images: (batch, P, F, 3, H, W)
- arcface: (batch, P, F, 512)
- mask: (batch, P, F) - indicates which slots have valid data
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from torch.utils.data import Dataset
import logging
import cv2
import random

# Import Korean phoneme configuration
from ..utils.korean_phoneme_config import (
    KEEP_PHONEMES_KOREAN,
    IGNORED_PHONEMES_KOREAN,
    IMAGES_PER_PHON,
    IMAGE_SIZE,
    get_phoneme_vocab,
    is_kept_phoneme,
    is_ignored_phoneme
)


# ============================================================================
# Individual Augmentation Functions (YouTube Compression Simulation)
# ============================================================================

def apply_jpeg_compression(images: np.ndarray,
                          quality_range: Tuple[int, int] = (30, 70)) -> np.ndarray:
    """
    Apply JPEG compression to simulate YouTube compression artifacts.

    Args:
        images: (T, H, W, 3) frames in range [0, 1]
        quality_range: JPEG quality range (lower = more compression)
            - YouTube typical range: 30-70
            - Original range: 70-90

    Returns:
        Compressed images (T, H, W, 3) in range [0, 1]

    Reference:
        arXiv:2508.08765 - YouTube uses JPEG quality 30-70 for 1080p videos
    """
    T, H, W, C = images.shape
    compressed = images.copy()

    # Random quality for this batch
    quality = np.random.randint(*quality_range)

    # Apply JPEG compression to each frame
    for t in range(T):
        frame = (compressed[t] * 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        compressed[t] = decoded.astype(np.float32) / 255.0

    return compressed


def apply_gaussian_blur(images: np.ndarray,
                       kernel_sizes: List[int] = [3, 5, 7],
                       apply_prob: float = 0.5) -> np.ndarray:
    """
    Apply Gaussian blur to simulate compression artifacts.

    Args:
        images: (T, H, W, 3) frames in range [0, 1]
        kernel_sizes: List of possible kernel sizes (must be odd)
        apply_prob: Probability of applying blur

    Returns:
        Blurred images (T, H, W, 3) in range [0, 1]
    """
    if np.random.random() > apply_prob:
        return images

    T = images.shape[0]
    blurred = images.copy()
    kernel_size = np.random.choice(kernel_sizes)

    for t in range(T):
        frame = (blurred[t] * 255).astype(np.uint8)
        blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        blurred[t] = blurred_frame.astype(np.float32) / 255.0

    return blurred


def apply_gaussian_noise(images: np.ndarray,
                        std_range: Tuple[float, float] = (0.005, 0.02),
                        apply_prob: float = 0.5) -> np.ndarray:
    """
    Apply Gaussian noise to simulate transmission errors and sensor noise.

    Args:
        images: (T, H, W, 3) frames in range [0, 1]
        std_range: Range for noise standard deviation
        apply_prob: Probability of applying noise

    Returns:
        Noisy images (T, H, W, 3) in range [0, 1]
    """
    if np.random.random() > apply_prob:
        return images

    noise_std = np.random.uniform(*std_range)
    noise = np.random.normal(0, noise_std, images.shape)
    noisy = np.clip(images + noise, 0.0, 1.0)

    return noisy.astype(np.float32)


def apply_brightness_contrast(images: np.ndarray,
                              alpha_range: Tuple[float, float] = (0.8, 1.2),
                              beta_range: Tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
    """
    Apply random brightness and contrast adjustment.

    Simulates re-encoding effects and camera auto-adjustment.

    Args:
        images: (T, H, W, 3) frames in range [0, 1]
        alpha_range: Contrast multiplier range (1.0 = no change)
        beta_range: Brightness offset range (0.0 = no change)

    Returns:
        Adjusted images (T, H, W, 3) in range [0, 1]

    Formula:
        output = alpha * input + beta
    """
    alpha = np.random.uniform(*alpha_range)  # Contrast
    beta = np.random.uniform(*beta_range)     # Brightness

    adjusted = np.clip(images * alpha + beta, 0.0, 1.0)
    return adjusted.astype(np.float32)


def apply_geometry_jitter(geometry: np.ndarray,
                         jitter_range: Tuple[float, float] = (-0.0005, 0.0005)) -> np.ndarray:
    """
    Apply random jitter to geometry features.

    Simulates MAR (Mouth Aspect Ratio) detection errors due to compression.

    Args:
        geometry: (T, geo_dim) geometry features (e.g., MAR values)
        jitter_range: Range for uniform random jitter

    Returns:
        Jittered geometry (T, geo_dim)
    """
    jitter = np.random.uniform(*jitter_range, geometry.shape)
    jittered = geometry + jitter

    return jittered.astype(np.float32)


# ============================================================================
# Integrated Augmentation Function
# ============================================================================

def apply_youtube_compression(images: np.ndarray,
                              geometry: np.ndarray,
                              arcface: np.ndarray,
                              jpeg_quality_range: Tuple[int, int] = (30, 70),
                              apply_prob: float = 0.6) -> tuple:
    """
    Apply YouTube compression augmentation to both Real and Fake samples.

    Simulates platform compression and low-quality recording conditions to improve
    generalization to real-world YouTube videos (horizontal + Shorts).

    This function integrates multiple augmentation techniques:
    1. JPEG compression (quality 30-70) - YouTube typical range
    2. Gaussian blur (kernel 3, 5, 7) - Compression artifacts
    3. Gaussian noise (std 0.005-0.02) - Transmission errors
    4. Brightness/contrast adjustment - Re-encoding effects
    5. Geometry jitter - MAR detection error simulation

    Args:
        images: (T, H, W, 3) lip crops in range [0, 1]
        geometry: (T, geo_dim) geometry features (e.g., MAR)
        arcface: (T, 512) identity embeddings
        jpeg_quality_range: JPEG quality range (default: 30-70 for YouTube)
        apply_prob: Probability of applying augmentation (default: 0.6)

    Returns:
        Augmented (images, geometry, arcface)

    Reference:
        arXiv:2508.08765 "Bridging the Gap: Social Network Compression Emulation"
    """
    # Skip augmentation with probability (1 - apply_prob)
    if np.random.random() > apply_prob:
        return images, geometry, arcface

    # Apply individual augmentation functions
    aug_images = images.copy()
    aug_geometry = geometry.copy()

    # 1. JPEG compression (YouTube quality 30-70)
    aug_images = apply_jpeg_compression(aug_images, jpeg_quality_range)

    # 2. Gaussian blur (compression artifacts)
    aug_images = apply_gaussian_blur(aug_images, kernel_sizes=[3, 5, 7], apply_prob=0.5)

    # 3. Gaussian noise (transmission errors)
    aug_images = apply_gaussian_noise(aug_images, std_range=(0.005, 0.02), apply_prob=0.5)

    # 4. Brightness/contrast adjustment (re-encoding)
    aug_images = apply_brightness_contrast(aug_images, alpha_range=(0.8, 1.2), beta_range=(-0.1, 0.1))

    # 5. Geometry jitter (MAR detection errors)
    aug_geometry = apply_geometry_jitter(aug_geometry, jitter_range=(-0.0005, 0.0005))

    # ArcFace embeddings remain unchanged (identity must be preserved)
    return aug_images, aug_geometry, arcface


class KoreanPhonemeDataset(Dataset):
    """
    PIA-style phoneme-organized dataset for Korean deepfake detection.

    Organizes frames by phoneme class rather than temporal sequence.
    Each phoneme class gets up to IMAGES_PER_PHON (5) representative frames.

    Example usage:
        >>> dataset = KoreanPhonemeDataset('preprocessed_data_real/', 'train')
        >>> batch = dataset[0]
        >>> print(batch['geometry'].shape)  # (14, 5, geo_dim) - auto-detected from data
        >>> print(batch['images'].shape)    # (14, 5, 3, 112, 112)
        >>> print(batch['arcface'].shape)   # (14, 5, 512)
        >>> print(batch['mask'].shape)      # (14, 5)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        phoneme_vocab: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        augment_real: bool = False,
        augment_ratio: float = 1.0
    ):
        """
        Initialize phoneme-organized dataset.

        Args:
            data_dir: Directory containing preprocessed .npz files
            split: Dataset split ('train', 'val', 'test')
            phoneme_vocab: List of phonemes to use (default: KEEP_PHONEMES_KOREAN)
            transform: Optional transform for lip crop images
            augment_real: Apply augmentation to Real samples (only for train split)
            augment_ratio: Ratio of augmented Real samples (1.0 = same as original)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.logger = logging.getLogger(__name__)

        # YouTube compression augmentation (for both Real and Fake in train split)
        self.augment_real = augment_real and (split == 'train')  # Keep variable name for compatibility
        self.augment_ratio = augment_ratio

        # Phoneme vocabulary (14 Korean phonemes)
        if phoneme_vocab is None:
            self.phoneme_vocab = get_phoneme_vocab()  # Sorted order (A, B, BB, CHh, E, EU, I, M, O, Ph, U, iA, iO, iU)
        else:
            self.phoneme_vocab = sorted(phoneme_vocab)  # Sort to match preprocessing order

        self.P = len(self.phoneme_vocab)  # 14 phonemes
        self.F = IMAGES_PER_PHON           # 5 frames per phoneme
        self.H, self.W = IMAGE_SIZE        # 112 x 112

        self.logger.info(f"Phoneme vocabulary ({self.P}): {self.phoneme_vocab}")

        # Load file list
        self.npz_files = self._load_file_list()
        self.logger.info(f"Loaded {len(self.npz_files)} {split} samples from {self.data_dir}")

        # Auto-detect geometry dimension from first file
        self.geo_dim = self._detect_geo_dim()
        self.logger.info(f"Detected geometry dimension: {self.geo_dim}")

    def _load_file_list(self) -> List[Tuple[Path, bool]]:
        """
        Load list of .npz files for this split.
        If augment_real is True, add augmented samples (both Real and Fake).

        Returns:
            List of tuples (npz_path, should_augment)
        """
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        npz_files = sorted(list(split_dir.glob("*.npz")))

        if len(npz_files) == 0:
            raise ValueError(f"No .npz files found in {split_dir}")

        # Apply YouTube compression augmentation (both Real and Fake)
        if self.augment_real:
            # Select samples for augmentation (both Real and Fake)
            num_total = len(npz_files)
            num_augment = int(num_total * self.augment_ratio)

            # Random sample for augmentation
            random.seed(42)  # Reproducibility
            augmented_files = random.sample(npz_files, min(num_augment, num_total))

            # Create file list: (path, should_augment)
            original_list = [(f, False) for f in npz_files]
            augmented_list = [(f, True) for f in augmented_files]

            all_files = original_list + augmented_list
            random.shuffle(all_files)

            # Count Real/Fake for statistics
            num_real = sum(1 for f in npz_files if int(np.load(f, allow_pickle=True)['label']) == 0)
            num_fake = num_total - num_real

            # Log augmentation statistics
            self.logger.info(f"YouTube compression augmentation: {num_total} original → +{len(augmented_files)} augmented = {len(all_files)} total samples")
            self.logger.info(f"  Original: Real={num_real}, Fake={num_fake}")
            self.logger.info(f"  Augmented samples: {len(augmented_files)} (both Real and Fake)")
            self.logger.info(f"  Total dataset size: {len(all_files)}")

            return all_files

        # No augmentation
        return [(f, False) for f in npz_files]

    def _detect_geo_dim(self) -> int:
        """
        Auto-detect geometry feature dimension from first .npz file.

        Returns:
            Geometry dimension (e.g., 1 for MAR only, 4 for full geometry)
        """
        if len(self.npz_files) == 0:
            raise ValueError("No files available to detect geometry dimension")

        # Load first file to check geometry shape (extract path from tuple)
        first_file, _ = self.npz_files[0]
        data = np.load(first_file, allow_pickle=True)
        geometry = data['geometry']  # (T, geo_dim)

        if len(geometry.shape) == 1:
            # Single dimension case
            return 1
        else:
            # Multi-dimensional case
            return geometry.shape[1]

    def __len__(self) -> int:
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one sample organized by phoneme class.

        Args:
            idx: Sample index

        Returns:
            Dictionary with PIA-format tensors:
                - geometry: (P, F, geo_dim) - Lip geometry per phoneme
                - images: (P, F, 3, H, W) - Lip crops per phoneme
                - arcface: (P, F, 512) - ArcFace embeddings per phoneme
                - mask: (P, F) - Valid data mask
                - label: Scalar (0=Real, 1=Fake)
        """
        # Load .npz file (extract path and augmentation flag)
        npz_path, should_augment = self.npz_files[idx]
        data = np.load(npz_path, allow_pickle=True)

        # Extract data
        frames = data['frames']              # (T, 224, 224, 3)
        lip = data['lip']                    # (T, 96, 96, 3) or (T, 112, 112, 3)
        arcface = data['arcface']            # (T, 512)
        geometry = data['geometry']          # (T, geo_dim)
        phoneme_labels = data['phoneme_labels']  # (T,) - String labels
        timestamps = data['timestamps']      # (T,) - Frame timestamps
        label = int(data['label'])           # 0 or 1

        # Apply YouTube compression augmentation (both Real and Fake)
        if should_augment:  # Real 조건 제거 - 모든 클래스에 적용
            lip, geometry, arcface = apply_youtube_compression(
                lip, geometry, arcface,
                jpeg_quality_range=(30, 70),
                apply_prob=0.6
            )

        T = len(frames)  # Number of frames

        # Group frames by phoneme class
        by_phoneme = self._group_by_phoneme(
            phoneme_labels, lip, arcface, geometry, T
        )

        # Build PIA-format tensors
        geoms_tensor, imgs_tensor, arcs_tensor, mask_tensor = self._build_tensors(by_phoneme)

        # Check and replace NaN/Inf values
        if torch.isnan(geoms_tensor).any() or torch.isinf(geoms_tensor).any():
            self.logger.warning(f"NaN/Inf detected in geometry for {npz_path.name}, replacing with 0")
            geoms_tensor = torch.nan_to_num(geoms_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if torch.isnan(imgs_tensor).any() or torch.isinf(imgs_tensor).any():
            self.logger.warning(f"NaN/Inf detected in images for {npz_path.name}, replacing with 0")
            imgs_tensor = torch.nan_to_num(imgs_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if torch.isnan(arcs_tensor).any() or torch.isinf(arcs_tensor).any():
            self.logger.warning(f"NaN/Inf detected in arcface for {npz_path.name}, replacing with 0")
            arcs_tensor = torch.nan_to_num(arcs_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            'geometry': geoms_tensor,   # (P, F, 4)
            'images': imgs_tensor,      # (P, F, 3, H, W)
            'arcface': arcs_tensor,     # (P, F, 512)
            'mask': mask_tensor,        # (P, F)
            'label': torch.tensor(label, dtype=torch.long),
            'video_id': str(data.get('video_id', npz_path.stem))
        }

    def _group_by_phoneme(
        self,
        phoneme_labels: np.ndarray,
        lip: np.ndarray,
        arcface: np.ndarray,
        geometry: np.ndarray,
        T: int
    ) -> Dict[str, List[Dict]]:
        """
        Group frames by phoneme class.

        Args:
            phoneme_labels: (T,) array of phoneme strings
            lip: (T, H, W, 3) lip crop frames
            arcface: (T, 512) embeddings
            geometry: (T, 4) geometry features
            T: Number of frames

        Returns:
            Dictionary mapping phoneme -> list of frame dictionaries
        """
        by_phoneme = {p: [] for p in self.phoneme_vocab}

        for frame_idx in range(T):
            phoneme = str(phoneme_labels[frame_idx]).strip()

            # Skip ignored phonemes (silence, padding, etc.)
            if is_ignored_phoneme(phoneme):
                continue

            # Only keep selected phonemes
            if not is_kept_phoneme(phoneme):
                continue

            # Add frame data to this phoneme's list
            by_phoneme[phoneme].append({
                'lip': lip[frame_idx],           # (H, W, 3)
                'arcface': arcface[frame_idx],   # (512,)
                'geometry': geometry[frame_idx]  # (4,)
            })

        return by_phoneme

    def _build_tensors(
        self,
        by_phoneme: Dict[str, List[Dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build PIA-format tensors from grouped phoneme data.

        Following PIA's strategy: Take only first F frames per phoneme.

        Args:
            by_phoneme: Dictionary mapping phoneme -> list of frame dicts

        Returns:
            (geometry, images, arcface, mask) tensors
        """
        # Initialize tensors (use auto-detected geo_dim instead of hardcoded 4)
        geoms = torch.zeros(self.P, self.F, self.geo_dim, dtype=torch.float32)
        imgs = torch.zeros(self.P, self.F, 3, self.H, self.W, dtype=torch.float32)
        arcs = torch.zeros(self.P, self.F, 512, dtype=torch.float32)
        mask = torch.zeros(self.P, self.F, dtype=torch.bool)

        # Fill tensors for each phoneme
        for pi, phoneme in enumerate(self.phoneme_vocab):
            # Get frames for this phoneme (PIA strategy: first F frames only)
            frames_list = by_phoneme[phoneme][:self.F]

            # Fill available frames
            for fi, frame_data in enumerate(frames_list):
                # Geometry features
                geoms[pi, fi] = torch.tensor(frame_data['geometry'], dtype=torch.float32)

                # Lip crop image
                lip_crop = frame_data['lip']  # (H, W, 3) in [0, 255]

                # Resize if needed
                if lip_crop.shape[0] != self.H or lip_crop.shape[1] != self.W:
                    import cv2
                    lip_crop = cv2.resize(lip_crop, (self.W, self.H))

                # Convert to tensor and normalize
                lip_tensor = torch.tensor(lip_crop, dtype=torch.float32) / 255.0  # [0, 1]
                lip_tensor = lip_tensor.permute(2, 0, 1)  # (3, H, W)

                # Apply optional transform
                if self.transform is not None:
                    lip_tensor = self.transform(lip_tensor)

                imgs[pi, fi] = lip_tensor

                # ArcFace embedding
                arcs[pi, fi] = torch.tensor(frame_data['arcface'], dtype=torch.float32)

                # Mark as valid
                mask[pi, fi] = True

        return geoms, imgs, arcs, mask

    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with statistics
        """
        total_frames_per_phoneme = defaultdict(int)
        total_samples_per_class = {0: 0, 1: 0}  # Real vs Fake

        for idx in range(len(self)):
            data = np.load(self.npz_files[idx], allow_pickle=True)
            phoneme_labels = data['phoneme_labels']
            label = int(data['label'])

            # Count label
            total_samples_per_class[label] += 1

            # Count phonemes
            for phoneme in phoneme_labels:
                phoneme = str(phoneme).strip()
                if is_kept_phoneme(phoneme):
                    total_frames_per_phoneme[phoneme] += 1

        return {
            'total_samples': len(self),
            'real_samples': total_samples_per_class[0],
            'fake_samples': total_samples_per_class[1],
            'phoneme_frame_counts': dict(total_frames_per_phoneme),
            'phoneme_vocab_size': self.P,
            'frames_per_phoneme': self.F
        }


# ============================================================================
# Helper Functions
# ============================================================================

def collate_phoneme_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for phoneme-organized batches.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with stacked tensors
    """
    geometry = torch.stack([sample['geometry'] for sample in batch])  # (B, P, F, 4)
    images = torch.stack([sample['images'] for sample in batch])      # (B, P, F, 3, H, W)
    arcface = torch.stack([sample['arcface'] for sample in batch])    # (B, P, F, 512)
    mask = torch.stack([sample['mask'] for sample in batch])          # (B, P, F)
    labels = torch.stack([sample['label'] for sample in batch])       # (B,)

    return {
        'geometry': geometry,
        'images': images,
        'arcface': arcface,
        'mask': mask,
        'label': labels,
        'video_ids': [sample['video_id'] for sample in batch]
    }


def print_batch_shapes(batch: Dict[str, torch.Tensor]):
    """
    Print shapes of batch tensors for debugging.

    Args:
        batch: Batched dictionary from DataLoader
    """
    print("Batch shapes:")
    print(f"  geometry: {batch['geometry'].shape}")  # Expected: (B, 14, 5, 4)
    print(f"  images:   {batch['images'].shape}")    # Expected: (B, 14, 5, 3, 112, 112)
    print(f"  arcface:  {batch['arcface'].shape}")   # Expected: (B, 14, 5, 512)
    print(f"  mask:     {batch['mask'].shape}")      # Expected: (B, 14, 5)
    print(f"  labels:   {batch['label'].shape}")     # Expected: (B,)
    print(f"  Batch size: {batch['geometry'].shape[0]}")
