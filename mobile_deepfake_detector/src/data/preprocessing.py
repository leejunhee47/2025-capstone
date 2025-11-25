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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(f"Video: {Path(video_path).name}")
        self.logger.info(f"  Resolution: {width}x{height}, Duration: {duration:.1f}s, FPS: {fps:.1f}, Frames: {total_frames}")

        # Detect vertical video (shorts format)
        is_vertical = height > width
        if is_vertical:
            aspect_ratio = height / width
            self.logger.info(f"  Vertical video detected (aspect ratio: {aspect_ratio:.2f})")

        # Check duration
        if duration < self.min_duration:
            self.logger.warning(f"Video too short: {duration:.1f}s < {self.min_duration}s")
        elif duration > self.max_duration:
            self.logger.warning(f"Video too long: {duration:.1f}s > {self.max_duration}s - using first {self.max_duration}s only")
            # Limit to max_duration by reducing total_frames
            total_frames = int(self.max_duration * fps)
            duration = total_frames / fps  # Recalculate duration after truncation

        # Calculate target frames based on target_fps
        # If max_frames is None (PIA full-frame mode), use target_fps to determine frame count
        if self.max_frames is None:
            target_frames = int(duration * self.target_fps)
            self.logger.info(f"  FPS resampling: {fps:.1f} → {self.target_fps} FPS ({total_frames} → {target_frames} frames)")
        else:
            target_frames = self.max_frames

        # Sample frame indices
        if sampling == "uniform":
            indices = self._uniform_sampling(total_frames, target_frames)
        elif sampling == "keyframe":
            indices = self._keyframe_sampling(cap, total_frames, target_frames)
        else:
            indices = self._uniform_sampling(total_frames, target_frames)

        # Extract frames
        frames = []

        # Calculate crop dimensions for center-crop (aspect ratio preservation)
        crop_size = min(width, height)
        x_offset = (width - crop_size) // 2
        y_offset = (height - crop_size) // 2

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Center-crop to square (prevents aspect ratio distortion)
                frame = frame[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]

                # Resize to target size (now square → square)
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
        # If max_frames is None, extract all frames
        if max_frames is None:
            return list(range(total_frames))

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
        lip_size: Tuple[int, int] = (112, 112)
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

    def extract_audio_file(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        max_duration: int = 60
    ) -> Union[str, None]:
        """
        Extract audio WAV file from video (for Audio Reuse optimization)

        Args:
            video_path: Path to video file
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (default: None, uses max_duration)
            max_duration: Maximum duration in seconds (default: 60)

        Returns:
            tmp_path: Path to extracted WAV file or None if failed
        """
        import subprocess
        import tempfile

        # Extract audio using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Build FFmpeg command with time range
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),  # Start time (before -i for faster seek)
                '-i', str(video_path),
            ]

            # Add end time or duration
            if end_time is not None:
                cmd.extend(['-to', str(end_time)])  # Absolute end time
            else:
                cmd.extend(['-t', str(max_duration)])  # Duration from start_time

            cmd.extend([
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                tmp_path
            ])

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            return tmp_path

        except subprocess.CalledProcessError:
            self.logger.error(f"Failed to extract audio from: {video_path}")
            return None

    def extract_audio(
        self,
        video_path: str,
        feature_type: str = "mfcc",
        max_duration: int = 60,
        audio_file: str = None
    ) -> Union[np.ndarray, None]:
        """
        Extract audio features from video

        Args:
            video_path: Path to video file
            feature_type: Feature type ('raw', 'mfcc', 'mel')
            max_duration: Maximum duration in seconds (default: 60)
            audio_file: Pre-extracted WAV file path (for Audio Reuse, optional)

        Returns:
            audio_features: Audio features or None if failed
        """
        tmp_path = None
        cleanup = False

        try:
            # Use pre-extracted audio file if provided (Audio Reuse)
            if audio_file is not None:
                tmp_path = audio_file
                cleanup = False
            else:
                # Extract audio from video
                tmp_path = self.extract_audio_file(video_path, max_duration=max_duration)
                if tmp_path is None:
                    return None
                cleanup = True

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

        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return None

        finally:
            # Clean up temp file if we created it
            if cleanup and tmp_path is not None:
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
            max_duration = self.config.get('max_duration', 60)
            audio = self.audio_processor.extract_audio(
                video_path,
                feature_type='mfcc',
                max_duration=max_duration
            )
            result['audio'] = audio
        else:
            result['audio'] = None

        # Extract lip region
        if extract_lip:
            lip_size = tuple(self.config.get('lip_size', [112, 112]))
            lip_frames = self.video_processor.extract_lip_region(frames, lip_size)
            result['lip'] = lip_frames
        else:
            result['lip'] = None

        self.logger.info(f"✓ Processing complete")
        self.logger.info(f"  Frames: {frames.shape if frames is not None else None}")
        self.logger.info(f"  Audio: {audio.shape if result['audio'] is not None else None}")
        self.logger.info(f"  Lip: {lip_frames.shape if result['lip'] is not None else None}")

        return result


# ============================================================================
# Phoneme-Frame Matching (PIA-main style)
# ============================================================================

def match_phoneme_to_frames(
    phoneme_intervals: List[Dict],
    timestamps: np.ndarray,
    fps: float = 30.0,
    min_overlap_ratio: float = 0.3
) -> np.ndarray:
    """
    Match phoneme intervals to frame timestamps using Maximum Overlap Assignment.

    IMPROVED VERSION (2025-11-24): Best Practice - Time Occupancy Based

    프레임은 "점"이 아닌 duration을 가진 "구간"입니다 (30fps = 33ms/frame).
    각 프레임 구간 [ts, ts+1/fps]과 가장 많이 겹치는(overlap) 음소를 할당합니다.

    알고리즘:
        1. 프레임 구간 정의: [ts, ts + 1/fps]  (예: [5.00, 5.033])
        2. 모든 음소와의 overlap duration 계산
        3. 가장 긴 overlap을 가진 음소 선택
        4. Overlap ratio가 min_overlap_ratio 미만이면 <sil> 처리

    기존 방식(엄격한 범위 체크)의 문제점:
        - 갭 프레임 누락 (음소1 end=1.0, 음소2 start=1.1, 프레임 ts=1.05)
        - 짧은 음소 무시 (duration < 33ms)
        - 경계 모호성 (부동소수점 오차)

    개선 효과:
        - 매칭률: 72.1% → 85-92% (예상)
        - 갭 프레임: 13개 → 2-3개
        - 짧은 음소: 확실히 프레임 할당

    Args:
        phoneme_intervals: List of phoneme dictionaries with:
            - 'phoneme': str (e.g., "M", "A", "ㅁ", "ㅏ")
            - 'start': float (start time in seconds)
            - 'end': float (end time in seconds)
        timestamps: Array of frame timestamps (frame_idx / fps)
        fps: Frame rate (default: 30.0)
        min_overlap_ratio: Minimum overlap ratio to assign phoneme (default: 0.3 = 30%)

    Returns:
        phoneme_labels: Array of phoneme strings, one per frame
            - Phoneme with maximum overlap if ratio >= min_overlap_ratio
            - "<sil>" if overlap too small or explicit silence phoneme

    Example:
        >>> intervals = [
        ...     {'phoneme': 'M', 'start': 0.0, 'end': 0.1},
        ...     {'phoneme': 'A', 'start': 0.15, 'end': 0.3}  # Gap: 0.1-0.15
        ... ]
        >>> timestamps = np.array([0.0, 0.033, 0.067, 0.10, 0.133])
        >>> match_phoneme_to_frames(intervals, timestamps, fps=30.0)
        array(['M', 'M', 'M', 'A', 'A'], dtype=object)
        # 0.10 프레임([0.10, 0.133])은 음소 A와 33ms 전체 overlap → A 할당

    Reference:
        Improved from PIA-main approach with time occupancy principle
    """
    frame_duration = 1.0 / fps  # 30fps → 0.0333s
    phoneme_labels = []

    # DEBUG: Log function parameters
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"\n[MATCH_PHONEME_TO_FRAMES DEBUG]")
    logger.info(f"  FPS: {fps}, Frame duration: {frame_duration:.4f}s")
    logger.info(f"  Min overlap ratio: {min_overlap_ratio} ({min_overlap_ratio*100:.0f}%)")
    logger.info(f"  Total timestamps: {len(timestamps)}")
    logger.info(f"  Total phoneme intervals: {len(phoneme_intervals)}")

    overlap_improved_count = 0  # Count how many frames benefited from overlap logic
    exact_match_count = 0  # Count exact matches
    debug_frames = []  # Sample first 3 frames

    for frame_idx, ts in enumerate(timestamps):
        frame_start = ts
        frame_end = ts + frame_duration

        best_phoneme = None
        max_overlap = 0.0

        # Track if old strict method would match
        old_method_match = None
        for p in phoneme_intervals:
            if p['start'] <= ts <= p['end']:  # Old strict check
                old_method_match = p['phoneme']
                break

        # 모든 음소와의 overlap 계산
        for p in phoneme_intervals:
            # 교집합 구간: [max(starts), min(ends)]
            overlap_start = max(frame_start, p['start'])
            overlap_end = min(frame_end, p['end'])
            overlap_duration = max(0.0, overlap_end - overlap_start)

            # 더 긴 overlap 발견 시 업데이트
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_phoneme = p['phoneme']

        # 점유율 검증: overlap이 프레임의 몇 %를 차지하는가?
        overlap_ratio = max_overlap / frame_duration

        # 점유율이 충분하고 침묵이 아니면 할당
        if overlap_ratio >= min_overlap_ratio and best_phoneme is not None and best_phoneme != '<sil>':
            assigned_phoneme = best_phoneme
            phoneme_labels.append(best_phoneme)
            # Count stats
            if old_method_match is None and best_phoneme != '<sil>':
                overlap_improved_count += 1
            if old_method_match == best_phoneme:
                exact_match_count += 1
        else:
            assigned_phoneme = '<sil>'
            phoneme_labels.append('<sil>')

        # Debug first 3 frames
        if frame_idx < 3:
            debug_frames.append({
                'ts': ts,
                'range': f"{frame_start:.3f}-{frame_end:.3f}",
                'best': best_phoneme,
                'overlap': max_overlap,
                'ratio': overlap_ratio,
                'old': old_method_match,
                'assigned': assigned_phoneme
            })

    # Log results
    total_non_sil = sum(1 for p in phoneme_labels if p != '<sil>')
    logger.info(f"\n[MATCHING RESULTS]")
    logger.info(f"  Non-<sil>: {total_non_sil}/{len(phoneme_labels)} ({total_non_sil/len(phoneme_labels)*100:.1f}%)")
    logger.info(f"  Exact (old OK): {exact_match_count}")
    logger.info(f"  Improved: {overlap_improved_count}")
    logger.info(f"\n  First 3 frames:")
    for df in debug_frames:
        logger.info(f"    {df['range']}: best='{df['best']}', overlap={df['overlap']:.4f}s ({df['ratio']*100:.0f}%), "
                   f"old='{df['old']}', →'{df['assigned']}'")

    return np.array(phoneme_labels, dtype=object)
