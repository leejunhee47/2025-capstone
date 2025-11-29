"""
Data augmentation for deepfake detection
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from typing import Dict, Optional
import random


class AudioAugmentation:
    """
    Audio augmentation for MFCC features
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = True

        # Augmentation params
        self.noise_prob = 0.2
        self.noise_std = 0.01
        self.time_mask_prob = 0.2
        self.time_mask_param = 10

    def add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to audio"""
        if random.random() > self.noise_prob:
            return audio

        noise = torch.randn_like(audio) * self.noise_std
        return audio + noise

    def time_mask(self, audio: torch.Tensor) -> torch.Tensor:
        """Mask random time frames (SpecAugment style)"""
        if random.random() > self.time_mask_prob:
            return audio

        T = audio.size(0)
        if T <= self.time_mask_param:
            return audio

        t = random.randint(1, min(self.time_mask_param, T))
        t0 = random.randint(0, T - t)

        audio_aug = audio.clone()
        audio_aug[t0:t0+t, :] = 0
        return audio_aug

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation"""
        if not self.enabled:
            return audio

        audio = self.add_noise(audio)
        audio = self.time_mask(audio)

        return audio


class VisualAugmentation:
    """
    Visual augmentation for video frames
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = True

        # Color jitter
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        self.color_jitter_prob = 0.5

        # Gaussian noise
        self.noise_prob = 0.3
        self.noise_std = 0.02

        # Gaussian blur
        self.blur_prob = 0.3

    def add_noise(self, frames: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to frames"""
        if random.random() > self.noise_prob:
            return frames

        noise = torch.randn_like(frames) * self.noise_std
        return torch.clamp(frames + noise, 0, 1)

    def apply_color_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply color jitter to all frames"""
        if random.random() > self.color_jitter_prob:
            return frames

        # Apply same color jitter to all frames
        N, C, H, W = frames.shape
        frames_aug = []

        for i in range(N):
            frame = frames[i]  # (C, H, W)
            frame_aug = self.color_jitter(frame)
            frames_aug.append(frame_aug)

        return torch.stack(frames_aug, dim=0)

    def apply_blur(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur"""
        if random.random() > self.blur_prob:
            return frames

        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.1, 2.0)

        N, C, H, W = frames.shape
        frames_aug = []

        for i in range(N):
            frame = frames[i]
            frame_aug = F.gaussian_blur(frame, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
            frames_aug.append(frame_aug)

        return torch.stack(frames_aug, dim=0)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply visual augmentation"""
        if not self.enabled:
            return frames

        frames = self.apply_color_jitter(frames)
        frames = self.add_noise(frames)
        frames = self.apply_blur(frames)

        return frames


class LipAugmentation:
    """
    Lip augmentation (similar to visual but lighter)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = True

        # Lighter augmentation for lip region
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        )
        self.color_jitter_prob = 0.3

        self.noise_prob = 0.2
        self.noise_std = 0.01

    def add_noise(self, lip: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise"""
        if random.random() > self.noise_prob:
            return lip

        noise = torch.randn_like(lip) * self.noise_std
        return torch.clamp(lip + noise, 0, 1)

    def apply_color_jitter(self, lip: torch.Tensor) -> torch.Tensor:
        """Apply color jitter"""
        if random.random() > self.color_jitter_prob:
            return lip

        N, C, H, W = lip.shape
        lip_aug = []

        for i in range(N):
            frame = lip[i]
            frame_aug = self.color_jitter(frame)
            lip_aug.append(frame_aug)

        return torch.stack(lip_aug, dim=0)

    def __call__(self, lip: torch.Tensor) -> torch.Tensor:
        """Apply lip augmentation"""
        if not self.enabled:
            return lip

        lip = self.apply_color_jitter(lip)
        lip = self.add_noise(lip)

        return lip


class MultiModalAugmentation:
    """
    Combined augmentation for all modalities
    """

    def __init__(self, config: Optional[Dict] = None, train: bool = True):
        self.config = config or {}
        self.train = train
        self.enabled = self.config.get('enabled', True) and train

        if self.enabled:
            self.audio_aug = AudioAugmentation(config)
            self.visual_aug = VisualAugmentation(config)
            self.lip_aug = LipAugmentation(config)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation to all modalities

        Args:
            sample: Dictionary with 'audio', 'frames', 'lip', 'label'

        Returns:
            Augmented sample
        """
        if not self.enabled:
            return sample

        # Apply augmentation
        sample['audio'] = self.audio_aug(sample['audio'])
        sample['frames'] = self.visual_aug(sample['frames'])
        sample['lip'] = self.lip_aug(sample['lip'])

        return sample
