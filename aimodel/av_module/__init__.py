"""
Audio and Video Extraction Module
Provides tools for extracting and preprocessing audio and video features from video files
"""

__version__ = '1.0.0'
__author__ = 'Audio-Visual Deepfake Detection Team'

# Core components
from config import Config
from temp_file_manager import TempFileManager
from audio_extractor import AudioExtractor
from frame_extractor import FrameExtractor
from video_processor import VideoProcessor
from preprocessing_pipeline import PreprocessingPipeline

# Utility functions
from utils import (
    save_array_to_npy,
    load_array_from_npy,
    save_metadata_to_json,
    load_metadata_from_json,
    get_video_id_from_path,
    create_output_structure,
    get_dataset_videos,
    setup_logging
)

__all__ = [
    # Core classes
    'Config',
    'TempFileManager',
    'AudioExtractor',
    'FrameExtractor',
    'VideoProcessor',
    'PreprocessingPipeline',

    # Utility functions
    'save_array_to_npy',
    'load_array_from_npy',
    'save_metadata_to_json',
    'load_metadata_from_json',
    'get_video_id_from_path',
    'create_output_structure',
    'get_dataset_videos',
    'setup_logging',
]
