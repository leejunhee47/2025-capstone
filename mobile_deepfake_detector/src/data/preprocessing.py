"""
Data preprocessing for short-form videos (Shorts)
"""

import sys
import cv2
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import from existing av_module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "av_module"))
try:
    from video_processor import VideoProcessor
except ImportError:
    logging.warning("av_module not found. Using built-in implementation.")
    VideoProcessor = None


class VideoPreprocessor:
    """
    Video preprocessing for shorts (15-60 seconds)
    """

    def __init__(
        self,
        target_fps: int = 30,
        max_frames: int = 50,
        frame_size: Tuple[int, int] = (224, 224),
        max_duration: int = 60,
        min_duration: int = 15
    ):
        """
        Initialize video preprocessor

        Args:
            target_fps: Target frames per second
            max_frames: Maximum number of frames to extract
            frame_size: Target frame size (H, W)
            max_duration: Maximum video duration (seconds)
            min_duration: Minimum video duration (seconds)
        """
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.max_duration = max_duration
        self.min_duration = min_duration

        self.logger = logging.getLogger(__name__)

    def extract_frames(
        self,
        video_path: str,
        sampling: str = "uniform"
    ) -> np.ndarray:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            sampling: Sampling method ('uniform', 'keyframe', 'scene_change')

        Returns:
            frames: Array of shape (N, H, W, 3)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        self.logger.info(f"Video: {Path(video_path).name}")
        self.logger.info(f"  Duration: {duration:.1f}s, FPS: {fps:.1f}, Frames: {total_frames}")

        # Check duration
        if duration < self.min_duration:
            self.logger.warning(f"Video too short: {duration:.1f}s < {self.min_duration}s")
        elif duration > self.max_duration:
            self.logger.warning(f"Video too long: {duration:.1f}s > {self.max_duration}s")

        # Sample frame indices
        if sampling == "uniform":
            indices = self._uniform_sampling(total_frames, self.max_frames)
        elif sampling == "keyframe":
            indices = self._keyframe_sampling(cap, total_frames, self.max_frames)
        else:
            indices = self._uniform_sampling(total_frames, self.max_frames)

        # Extract frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize
                frame = cv2.resize(frame, self.frame_size)

                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0

                frames.append(frame)

        cap.release()

        frames = np.array(frames)  # (N, H, W, 3)
        self.logger.info(f"  Extracted {len(frames)} frames")

        return frames

    def _uniform_sampling(self, total_frames: int, max_frames: int) -> List[int]:
        """Uniform frame sampling"""
        if total_frames <= max_frames:
            return list(range(total_frames))

        # Sample evenly
        step = total_frames / max_frames
        indices = [int(i * step) for i in range(max_frames)]

        return indices

    def _keyframe_sampling(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        max_frames: int
    ) -> List[int]:
        """Keyframe-based sampling (simplified)"""
        # For now, use uniform sampling
        # TODO: Implement actual keyframe detection
        return self._uniform_sampling(total_frames, max_frames)

    def detect_face_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face region in frame

        Args:
            frame: Input frame (H, W, 3)

        Returns:
            face_coords: [x, y, w, h] or None
        """
        # Convert to uint8 if needed
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)

        # Load cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Return largest face
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            return faces[largest_idx]

        return None

    def extract_lip_region(
        self,
        frames: np.ndarray,
        lip_size: Tuple[int, int] = (96, 96)
    ) -> np.ndarray:
        """
        Extract lip region from frames

        Args:
            frames: Input frames (N, H, W, 3)
            lip_size: Target lip region size

        Returns:
            lip_frames: Lip regions (N, lip_H, lip_W, 3)
        """
        lip_frames = []

        for frame in frames:
            face_coords = self.detect_face_region(frame)

            if face_coords is not None:
                x, y, w, h = face_coords

                # Estimate lip region (lower third of face)
                lip_y = y + int(h * 0.6)
                lip_h = int(h * 0.4)
                lip_x = x + int(w * 0.2)
                lip_w = int(w * 0.6)

                # Convert to uint8 for cropping
                if frame.max() <= 1.0:
                    frame_uint8 = (frame * 255).astype(np.uint8)
                else:
                    frame_uint8 = frame.astype(np.uint8)

                # Crop lip region
                lip_region = frame_uint8[lip_y:lip_y+lip_h, lip_x:lip_x+lip_w]

                # Resize
                if lip_region.size > 0:
                    lip_region = cv2.resize(lip_region, lip_size)
                    lip_region = lip_region.astype(np.float32) / 255.0
                else:
                    # Use blank frame if detection failed
                    lip_region = np.zeros((*lip_size, 3), dtype=np.float32)
            else:
                # Use blank frame if no face detected
                lip_region = np.zeros((*lip_size, 3), dtype=np.float32)

            lip_frames.append(lip_region)

        return np.array(lip_frames)


class AudioPreprocessor:
    """
    Audio preprocessing for deepfake detection
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        hop_length: int = 512,
        n_fft: int = 2048
    ):
        """
        Initialize audio preprocessor

        Args:
            sample_rate: Target sample rate
            n_mfcc: Number of MFCC coefficients
            hop_length: Hop length for STFT
            n_fft: FFT window size
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.logger = logging.getLogger(__name__)

    def extract_audio(
        self,
        video_path: str,
        feature_type: str = "mfcc"
    ) -> Union[np.ndarray, None]:
        """
        Extract audio from video

        Args:
            video_path: Path to video file
            feature_type: Feature type ('raw', 'mfcc', 'mel')

        Returns:
            audio_features: Audio features or None if failed
        """
        import subprocess
        import tempfile

        # Extract audio using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                tmp_path
            ]

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            # Load audio
            audio, sr = librosa.load(tmp_path, sr=self.sample_rate)

            self.logger.info(f"Audio: duration={len(audio)/sr:.1f}s, sr={sr}Hz")

            # Extract features
            if feature_type == "raw":
                return audio

            elif feature_type == "mfcc":
                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=self.n_mfcc,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                )
                # (n_mfcc, time) -> (time, n_mfcc)
                return mfcc.T

            elif feature_type == "mel":
                mel = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                )
                # Convert to log scale
                mel_db = librosa.power_to_db(mel, ref=np.max)
                return mel_db.T

            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

        except subprocess.CalledProcessError:
            self.logger.error(f"Failed to extract audio from: {video_path}")
            return None

        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return None

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)


class ShortsPreprocessor:
    """
    Complete preprocessing pipeline for shorts videos
    """

    def __init__(self, config: Dict):
        """
        Initialize shorts preprocessor

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize sub-processors
        self.video_processor = VideoPreprocessor(
            target_fps=config.get('target_fps', 30),
            max_frames=config.get('max_frames', 50),
            frame_size=tuple(config.get('frame_size', [224, 224])),
            max_duration=config.get('max_duration', 60),
            min_duration=config.get('min_duration', 15)
        )

        self.audio_processor = AudioPreprocessor(
            sample_rate=config.get('sample_rate', 16000),
            n_mfcc=config.get('n_mfcc', 40),
            hop_length=config.get('hop_length', 512),
            n_fft=config.get('n_fft', 2048)
        )

        self.logger = logging.getLogger(__name__)

    def process_video(
        self,
        video_path: str,
        extract_audio: bool = True,
        extract_lip: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process single video

        Args:
            video_path: Path to video file
            extract_audio: Whether to extract audio
            extract_lip: Whether to extract lip region

        Returns:
            result: Dictionary with processed data
                - frames: (N, H, W, 3)
                - audio: (T, n_mfcc) or None
                - lip: (N, lip_H, lip_W, 3) or None
        """
        self.logger.info(f"\nProcessing: {Path(video_path).name}")

        result = {}

        # Extract frames
        frames = self.video_processor.extract_frames(video_path)
        result['frames'] = frames

        # Extract audio
        if extract_audio:
            audio = self.audio_processor.extract_audio(video_path, feature_type='mfcc')
            result['audio'] = audio
        else:
            result['audio'] = None

        # Extract lip region
        if extract_lip:
            lip_size = tuple(self.config.get('lip_size', [96, 96]))
            lip_frames = self.video_processor.extract_lip_region(frames, lip_size)
            result['lip'] = lip_frames
        else:
            result['lip'] = None

        self.logger.info(f"✓ Processing complete")
        self.logger.info(f"  Frames: {frames.shape if frames is not None else None}")
        self.logger.info(f"  Audio: {audio.shape if result['audio'] is not None else None}")
        self.logger.info(f"  Lip: {lip_frames.shape if result['lip'] is not None else None}")

        return result
