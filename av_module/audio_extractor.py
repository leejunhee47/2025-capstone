"""
Audio extraction module using FFmpeg
Handles audio extraction from video files
"""

import os
import subprocess
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Union, Tuple
import logging

from config import Config
from temp_file_manager import TempFileManager

logger = logging.getLogger(__name__)


class AudioExtractor:
    """
    Extracts audio from video files using FFmpeg
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize AudioExtractor

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        """Verify FFmpeg is installed"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg is installed but returned error")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg: "
                "https://ffmpeg.org/download.html"
            )
        except Exception as e:
            logger.warning(f"Could not verify FFmpeg: {e}")

    def extract_audio(
        self,
        video_path: Union[str, Path],
        temp_manager: TempFileManager,
        return_array: bool = True
    ) -> Union[np.ndarray, Path]:
        """
        Extract audio from video file

        Args:
            video_path: Path to video file
            temp_manager: TempFileManager instance
            return_array: If True, return numpy array; if False, return path to WAV file

        Returns:
            Audio as numpy array (if return_array=True) or Path to WAV file
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create temporary WAV file
        temp_audio_path = temp_manager.create_temp_file(suffix='.wav', prefix='audio_')

        try:
            # Extract audio using FFmpeg
            self._extract_audio_ffmpeg(video_path, temp_audio_path)

            # Validate audio file
            if not temp_audio_path.exists() or temp_audio_path.stat().st_size == 0:
                raise RuntimeError(f"Failed to extract audio from {video_path}")

            # Load and return audio
            if return_array:
                audio_array = self._load_audio(temp_audio_path)
                logger.info(
                    f"Extracted audio from {video_path.name}: "
                    f"shape={audio_array.shape}, duration={len(audio_array)/self.config.AUDIO_SAMPLE_RATE:.2f}s"
                )
                return audio_array
            else:
                logger.info(f"Extracted audio to {temp_audio_path}")
                return temp_audio_path

        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {e}")
            raise

    def _extract_audio_ffmpeg(self, video_path: Path, output_path: Path):
        """
        Extract audio using FFmpeg command

        Args:
            video_path: Input video file
            output_path: Output audio file
        """
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', self.config.AUDIO_CODEC,
            '-ar', str(self.config.AUDIO_SAMPLE_RATE),
            '-ac', str(self.config.AUDIO_CHANNELS),
            '-threads', str(self.config.FFMPEG_THREADS),
            '-loglevel', self.config.FFMPEG_LOGLEVEL,
            str(output_path),
            '-y'  # Overwrite output file
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,  # 5 minutes timeout
                check=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg error: {error_msg}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FFmpeg timeout while processing {video_path}")

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """
        Load audio file as numpy array

        Args:
            audio_path: Path to audio file

        Returns:
            Audio as numpy array
        """
        try:
            audio, sr = librosa.load(
                str(audio_path),
                sr=self.config.AUDIO_SAMPLE_RATE,
                mono=(self.config.AUDIO_CHANNELS == 1)
            )
            return audio
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {e}")

    def extract_audio_features(
        self,
        video_path: Union[str, Path],
        temp_manager: TempFileManager,
        feature_type: str = 'raw'
    ) -> np.ndarray:
        """
        Extract audio features from video

        Args:
            video_path: Path to video file
            temp_manager: TempFileManager instance
            feature_type: Type of features ('raw', 'mfcc', 'mel')

        Returns:
            Audio features as numpy array
        """
        audio = self.extract_audio(video_path, temp_manager, return_array=True)

        if feature_type == 'raw':
            return audio
        elif feature_type == 'mfcc':
            return self._extract_mfcc(audio)
        elif feature_type == 'mel':
            return self._extract_mel(audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def _extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract MFCC features

        Args:
            audio: Audio array
            n_mfcc: Number of MFCC coefficients

        Returns:
            MFCC features (time_steps, n_mfcc)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.config.AUDIO_SAMPLE_RATE,
            n_mfcc=n_mfcc
        )
        return mfcc.T  # Transpose to (time_steps, n_mfcc)

    def _extract_mel(self, audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """
        Extract mel spectrogram features

        Args:
            audio: Audio array
            n_mels: Number of mel bands

        Returns:
            Mel spectrogram (time_steps, n_mels)
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.AUDIO_SAMPLE_RATE,
            n_mels=n_mels
        )
        # Convert to log scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.T  # Transpose to (time_steps, n_mels)

    def get_audio_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get audio information from video file without extracting

        Args:
            video_path: Path to video file

        Returns:
            Dict with audio information
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 'a:0',
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=True
            )
            import json
            info = json.loads(result.stdout.decode('utf-8'))

            if 'streams' in info and len(info['streams']) > 0:
                stream = info['streams'][0]
                return {
                    'codec': stream.get('codec_name', 'unknown'),
                    'sample_rate': int(stream.get('sample_rate', 0)),
                    'channels': int(stream.get('channels', 0)),
                    'duration': float(stream.get('duration', 0))
                }
            else:
                return {'error': 'No audio stream found'}
        except Exception as e:
            logger.warning(f"Could not get audio info for {video_path}: {e}")
            return {'error': str(e)}
