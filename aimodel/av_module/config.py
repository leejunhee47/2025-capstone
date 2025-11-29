"""
Configuration settings for audio and video extraction module
"""

import os
from pathlib import Path


class Config:
    """Configuration class for audio and video processing"""

    # Audio extraction settings
    AUDIO_SAMPLE_RATE = 16000  # 16kHz
    AUDIO_CHANNELS = 1  # Mono
    AUDIO_FORMAT = 'wav'
    AUDIO_CODEC = 'pcm_s16le'

    # Frame extraction settings
    TARGET_FPS = 30  # 30 frames per second
    FRAME_WIDTH = 224
    FRAME_HEIGHT = 224
    FRAME_CHANNELS = 3  # RGB

    # Normalization settings
    NORMALIZE_FRAMES = True
    PIXEL_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    PIXEL_STD = [0.229, 0.224, 0.225]   # ImageNet std

    # Processing settings
    MAX_FRAMES_PER_VIDEO = None  # None = extract all frames
    UNIFORM_FRAME_SAMPLING = True  # Sample frames uniformly

    # Temporary file settings
    TEMP_DIR_PREFIX = 'av_extraction_'
    AUTO_CLEANUP = True

    # Performance settings
    NUM_WORKERS = 4  # For multiprocessing
    BATCH_SIZE = 8

    # Output settings
    OUTPUT_AUDIO_FORMAT = 'npy'  # 'npy' or 'wav'
    OUTPUT_FRAME_FORMAT = 'npy'  # 'npy' or 'jpg'

    # FFmpeg settings
    FFMPEG_LOGLEVEL = 'quiet'  # 'quiet', 'error', 'warning', 'info'
    FFMPEG_THREADS = 1

    # Error handling
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    @classmethod
    def get_audio_params(cls):
        """Get audio extraction parameters as dict"""
        return {
            'sample_rate': cls.AUDIO_SAMPLE_RATE,
            'channels': cls.AUDIO_CHANNELS,
            'format': cls.AUDIO_FORMAT,
            'codec': cls.AUDIO_CODEC
        }

    @classmethod
    def get_frame_params(cls):
        """Get frame extraction parameters as dict"""
        return {
            'fps': cls.TARGET_FPS,
            'width': cls.FRAME_WIDTH,
            'height': cls.FRAME_HEIGHT,
            'normalize': cls.NORMALIZE_FRAMES,
            'max_frames': cls.MAX_FRAMES_PER_VIDEO
        }

    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        assert cls.AUDIO_SAMPLE_RATE > 0, "Sample rate must be positive"
        assert cls.AUDIO_CHANNELS in [1, 2], "Channels must be 1 (mono) or 2 (stereo)"
        assert cls.TARGET_FPS > 0, "FPS must be positive"
        assert cls.FRAME_WIDTH > 0 and cls.FRAME_HEIGHT > 0, "Frame dimensions must be positive"
        assert cls.NUM_WORKERS > 0, "Number of workers must be positive"
        return True


# Validate configuration on import
Config.validate()
