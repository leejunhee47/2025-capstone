"""
Video processor - orchestrates audio and frame extraction
Main coordinator for processing video files
"""

import time
from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging
import numpy as np

from config import Config
from temp_file_manager import TempFileManager
from audio_extractor import AudioExtractor
from frame_extractor import FrameExtractor

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Main video processor that orchestrates audio and frame extraction
    Acts as coordinator - delegates extraction to specialized modules
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize VideoProcessor

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.audio_extractor = AudioExtractor(self.config)
        self.frame_extractor = FrameExtractor(self.config)

    def process_video(
        self,
        video_path: Union[str, Path],
        extract_audio: bool = True,
        extract_frames: bool = True,
        temp_manager: Optional[TempFileManager] = None,
        audio_feature_type: str = 'raw',
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process video file - extract audio and frames

        Args:
            video_path: Path to video file
            extract_audio: Extract audio
            extract_frames: Extract frames
            temp_manager: TempFileManager (creates new if None)
            audio_feature_type: Type of audio features ('raw', 'mfcc', 'mel')
            max_frames: Maximum frames to extract

        Returns:
            Dict with processing results:
            {
                'audio': audio_array or None,
                'frames': frames_array or None,
                'metadata': {...}
            }
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Processing video: {video_path}")
        start_time = time.time()

        # Use provided temp_manager or create new one
        own_temp_manager = temp_manager is None
        if own_temp_manager:
            temp_manager = TempFileManager()
            temp_manager.__enter__()

        try:
            result = {
                'audio': None,
                'frames': None,
                'metadata': {
                    'video_path': str(video_path),
                    'video_name': video_path.name,
                    'success': False,
                    'error': None
                }
            }

            # Extract audio
            if extract_audio:
                try:
                    logger.info(f"Extracting audio from {video_path.name}")
                    audio = self.audio_extractor.extract_audio_features(
                        video_path,
                        temp_manager,
                        feature_type=audio_feature_type
                    )
                    result['audio'] = audio
                    result['metadata']['audio_shape'] = audio.shape
                    result['metadata']['audio_duration'] = len(audio) / self.config.AUDIO_SAMPLE_RATE
                    logger.info(f"Audio extracted: shape={audio.shape}")
                except Exception as e:
                    logger.error(f"Failed to extract audio: {e}")
                    result['metadata']['audio_error'] = str(e)

            # Extract frames
            if extract_frames:
                try:
                    logger.info(f"Extracting frames from {video_path.name}")
                    frames = self.frame_extractor.extract_frames(
                        video_path,
                        max_frames=max_frames,
                        uniform_sampling=self.config.UNIFORM_FRAME_SAMPLING,
                        preprocess=True
                    )
                    result['frames'] = frames
                    result['metadata']['frames_shape'] = frames.shape
                    result['metadata']['num_frames'] = len(frames)
                    logger.info(f"Frames extracted: shape={frames.shape}")
                except Exception as e:
                    logger.error(f"Failed to extract frames: {e}")
                    result['metadata']['frames_error'] = str(e)

            # Get video info
            try:
                video_info = self.frame_extractor.get_video_info(video_path)
                result['metadata']['video_info'] = video_info
            except Exception as e:
                logger.warning(f"Could not get video info: {e}")

            # Mark success if at least one extraction succeeded
            result['metadata']['success'] = (
                result['audio'] is not None or result['frames'] is not None
            )

            # Processing time
            processing_time = time.time() - start_time
            result['metadata']['processing_time'] = processing_time
            logger.info(f"Processing completed in {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise

        finally:
            # Cleanup temp manager if we created it
            if own_temp_manager:
                temp_manager.__exit__(None, None, None)

    def process_video_batch(
        self,
        video_paths: list,
        extract_audio: bool = True,
        extract_frames: bool = True,
        audio_feature_type: str = 'raw',
        max_frames: Optional[int] = None
    ) -> list:
        """
        Process multiple videos sequentially

        Args:
            video_paths: List of video paths
            extract_audio: Extract audio
            extract_frames: Extract frames
            audio_feature_type: Type of audio features
            max_frames: Maximum frames per video

        Returns:
            List of processing results
        """
        results = []

        logger.info(f"Processing batch of {len(video_paths)} videos")

        for video_path in video_paths:
            try:
                result = self.process_video(
                    video_path,
                    extract_audio=extract_audio,
                    extract_frames=extract_frames,
                    audio_feature_type=audio_feature_type,
                    max_frames=max_frames
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results.append({
                    'audio': None,
                    'frames': None,
                    'metadata': {
                        'video_path': str(video_path),
                        'success': False,
                        'error': str(e)
                    }
                })

        successful = sum(1 for r in results if r['metadata']['success'])
        logger.info(f"Batch processing completed: {successful}/{len(video_paths)} successful")

        return results

    def validate_video(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate video file without full processing

        Args:
            video_path: Path to video file

        Returns:
            Dict with validation results
        """
        video_path = Path(video_path)

        validation = {
            'exists': video_path.exists(),
            'is_file': video_path.is_file() if video_path.exists() else False,
            'size_bytes': video_path.stat().st_size if video_path.exists() else 0,
            'can_open': False,
            'has_audio': False,
            'has_video': False,
            'video_info': None,
            'audio_info': None
        }

        if not validation['exists']:
            return validation

        # Try to get video info
        try:
            video_info = self.frame_extractor.get_video_info(video_path)
            validation['video_info'] = video_info
            validation['can_open'] = True
            validation['has_video'] = video_info['frame_count'] > 0
        except Exception as e:
            logger.warning(f"Could not validate video stream: {e}")

        # Try to get audio info
        try:
            audio_info = self.audio_extractor.get_audio_info(video_path)
            validation['audio_info'] = audio_info
            validation['has_audio'] = 'error' not in audio_info
        except Exception as e:
            logger.warning(f"Could not validate audio stream: {e}")

        return validation

    def estimate_processing_time(
        self,
        video_path: Union[str, Path],
        extract_audio: bool = True,
        extract_frames: bool = True
    ) -> float:
        """
        Estimate processing time for a video

        Args:
            video_path: Path to video file
            extract_audio: Will extract audio
            extract_frames: Will extract frames

        Returns:
            Estimated time in seconds
        """
        video_info = self.frame_extractor.get_video_info(video_path)
        duration = video_info.get('duration', 0)
        frame_count = video_info.get('frame_count', 0)

        # Rough estimates (can be calibrated)
        audio_time = duration * 0.1 if extract_audio else 0  # ~10% of video duration
        frame_time = frame_count * 0.01 if extract_frames else 0  # ~0.01s per frame

        return audio_time + frame_time

    def get_extraction_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics from extraction result

        Args:
            result: Result from process_video

        Returns:
            Dict with statistics
        """
        stats = {
            'success': result['metadata']['success'],
            'has_audio': result['audio'] is not None,
            'has_frames': result['frames'] is not None,
            'processing_time': result['metadata'].get('processing_time', 0)
        }

        if result['audio'] is not None:
            stats['audio_samples'] = len(result['audio'])
            stats['audio_duration'] = result['metadata'].get('audio_duration', 0)

        if result['frames'] is not None:
            stats['num_frames'] = result['metadata'].get('num_frames', 0)
            stats['frames_shape'] = result['metadata'].get('frames_shape', None)

        return stats
