"""
Frame extraction and preprocessing module using OpenCV
Handles frame extraction from video files with preprocessing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, List
import logging

from config import Config

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extracts and preprocesses frames from video files using OpenCV
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize FrameExtractor

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()

    def extract_frames(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        uniform_sampling: bool = True,
        preprocess: bool = True
    ) -> np.ndarray:
        """
        Extract frames from video file

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (None = all)
            uniform_sampling: Sample frames uniformly across video
            preprocess: Apply preprocessing (resize, normalize)

        Returns:
            Frames as numpy array (N, H, W, C)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"Video info - {video_path.name}: "
                f"frames={total_frames}, fps={fps:.2f}, size={width}x{height}"
            )

            if total_frames == 0:
                raise RuntimeError(f"Video has no frames: {video_path}")

            # Determine frame indices to extract
            frame_indices = self._get_frame_indices(
                total_frames,
                max_frames or self.config.MAX_FRAMES_PER_VIDEO,
                uniform_sampling
            )

            # Extract frames
            frames = []
            for idx in frame_indices:
                frame = self._read_frame_at_index(cap, idx)
                if frame is not None:
                    if preprocess:
                        frame = self._preprocess_frame(frame)
                    frames.append(frame)

            cap.release()

            if len(frames) == 0:
                raise RuntimeError(f"No frames extracted from {video_path}")

            frames_array = np.array(frames)
            logger.info(
                f"Extracted {len(frames)} frames from {video_path.name}: "
                f"shape={frames_array.shape}"
            )

            return frames_array

        except Exception as e:
            cap.release()
            logger.error(f"Error extracting frames from {video_path}: {e}")
            raise

    def _get_frame_indices(
        self,
        total_frames: int,
        max_frames: Optional[int],
        uniform: bool
    ) -> List[int]:
        """
        Get frame indices to extract

        Args:
            total_frames: Total number of frames in video
            max_frames: Maximum frames to extract
            uniform: Sample uniformly

        Returns:
            List of frame indices
        """
        if max_frames is None or total_frames <= max_frames:
            # Extract all frames
            return list(range(total_frames))

        if uniform:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            return indices.tolist()
        else:
            # Sequential sampling from start
            return list(range(max_frames))

    def _read_frame_at_index(self, cap: cv2.VideoCapture, index: int) -> Optional[np.ndarray]:
        """
        Read frame at specific index

        Args:
            cap: OpenCV VideoCapture object
            index: Frame index

        Returns:
            Frame as numpy array or None if failed
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()

        if not ret or frame is None:
            logger.warning(f"Failed to read frame at index {index}")
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame (resize and normalize)

        Args:
            frame: Input frame (H, W, C)

        Returns:
            Preprocessed frame
        """
        # Resize to target dimensions
        frame = cv2.resize(
            frame,
            (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1]
        if self.config.NORMALIZE_FRAMES:
            frame = frame.astype(np.float32) / 255.0

        return frame

    def preprocess_frames(
        self,
        frames: np.ndarray,
        resize: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess already extracted frames

        Args:
            frames: Input frames (N, H, W, C)
            resize: Resize frames
            normalize: Normalize frames

        Returns:
            Preprocessed frames
        """
        processed_frames = []

        for frame in frames:
            if resize:
                frame = cv2.resize(
                    frame,
                    (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT),
                    interpolation=cv2.INTER_LINEAR
                )

            if normalize:
                if frame.dtype == np.uint8:
                    frame = frame.astype(np.float32) / 255.0

            processed_frames.append(frame)

        return np.array(processed_frames)

    def extract_frames_at_fps(
        self,
        video_path: Union[str, Path],
        target_fps: Optional[float] = None,
        preprocess: bool = True
    ) -> np.ndarray:
        """
        Extract frames at specific FPS

        Args:
            video_path: Path to video file
            target_fps: Target FPS (uses config if None)
            preprocess: Apply preprocessing

        Returns:
            Frames array (N, H, W, C)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        target_fps = target_fps or self.config.TARGET_FPS

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if original_fps == 0:
                raise RuntimeError(f"Could not determine FPS for {video_path}")

            # Calculate frame interval
            frame_interval = max(1, int(original_fps / target_fps))

            frames = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if preprocess:
                        frame = self._preprocess_frame(frame)

                    frames.append(frame)

                frame_idx += 1

            cap.release()

            if len(frames) == 0:
                raise RuntimeError(f"No frames extracted from {video_path}")

            frames_array = np.array(frames)
            logger.info(
                f"Extracted {len(frames)} frames at {target_fps}fps from {video_path.name}: "
                f"shape={frames_array.shape}"
            )

            return frames_array

        except Exception as e:
            cap.release()
            logger.error(f"Error extracting frames at FPS from {video_path}: {e}")
            raise

    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get video information

        Args:
            video_path: Path to video file

        Returns:
            Dict with video information
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }

        cap.release()
        return info

    def extract_single_frame(
        self,
        video_path: Union[str, Path],
        frame_index: int = 0,
        preprocess: bool = True
    ) -> np.ndarray:
        """
        Extract a single frame from video

        Args:
            video_path: Path to video file
            frame_index: Index of frame to extract
            preprocess: Apply preprocessing

        Returns:
            Frame as numpy array (H, W, C)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            frame = self._read_frame_at_index(cap, frame_index)
            cap.release()

            if frame is None:
                raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

            if preprocess:
                frame = self._preprocess_frame(frame)

            return frame

        except Exception as e:
            cap.release()
            raise
