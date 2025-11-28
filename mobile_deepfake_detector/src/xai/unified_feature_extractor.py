import cv2
import numpy as np
import logging
import tempfile
import subprocess
import os
import time
import librosa
from typing import Dict, Any, List, Tuple, Optional, Union, cast
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from ..utils.hybrid_phoneme_aligner_v2 import HybridPhonemeAligner
from ..utils.enhanced_mar_extractor import EnhancedMARExtractor
from ..utils.arcface_extractor import ArcFaceExtractor
from ..data.preprocessing import AudioPreprocessor, VideoPreprocessor

logger = logging.getLogger(__name__)

class UnifiedFeatureExtractor:
    def __init__(self, device="cuda", max_duration: Optional[float] = 60.0):
        self.device = device
        self.max_duration = max_duration
        logger.info(f"Initializing UnifiedFeatureExtractor on {device}...")
        
        # Initialize extractors
        self.phoneme_aligner = HybridPhonemeAligner(device=device)
        self.mar_extractor = EnhancedMARExtractor()
        self.arcface_extractor = ArcFaceExtractor(device=device)
        
        # Audio processor for MFCC
        self.audio_processor = AudioPreprocessor(
            sample_rate=16000,
            n_mfcc=40,
            hop_length=512,
            n_fft=2048
        )
        
        # Video processor for lip cropping (helper)
        self.video_processor = VideoPreprocessor(
            target_fps=30,
            max_frames=99999, 
            frame_size=(224, 224)
        )
        
        logger.info("UnifiedFeatureExtractor initialized.")

    def extract_all(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract all features from video at 30fps (and downsample for 10fps).

        Optimized with:
        - Frame Reuse: Single video decode → BGR + RGB versions
        - Audio Reuse: Single WAV extraction → MFCC + WhisperX
        - ThreadPool: Parallel CPU tasks (MAR + MFCC + Lip)
        - GPU Sequential: WhisperX → ArcFace (avoid VRAM conflicts)
        """
        video_path_obj = Path(video_path)
        logger.info(f"Extracting all features for: {video_path_obj.name}")

        timings: Dict[str, float] = {}
        total_start = time.time()

        working_video_path, temp_clip = self._prepare_video_path(video_path_obj)
        cleanup_paths: List[Path] = []
        if temp_clip is not None:
            cleanup_paths.append(temp_clip)

        audio_wav_path: Optional[Path] = None

        try:
            # ============================================================
            # STEP 1: Frame Reuse - Single video decode, dual output
            # ============================================================
            t0 = time.time()
            frames_rgb_224, frames_bgr_384 = self._extract_frames_dual(
                str(working_video_path), target_fps=30.0
            )
            timings['frame_extraction'] = time.time() - t0

            num_frames = len(frames_rgb_224)
            timestamps_30fps = np.arange(num_frames) / 30.0
            logger.info(f"  ✓ Frames: {num_frames} @ 30fps ({timings['frame_extraction']:.2f}s)")

            # ============================================================
            # STEP 2: Audio Reuse - Single WAV extraction
            # ============================================================
            t0 = time.time()
            audio_wav_path = self._extract_audio_wav(str(working_video_path))
            cleanup_paths.append(audio_wav_path)
            timings['audio_wav_extraction'] = time.time() - t0
            logger.info(f"  ✓ WAV extracted ({timings['audio_wav_extraction']:.2f}s)")

            # ============================================================
            # STEP 3: Parallel CPU tasks (MAR + MFCC + Lip)
            # ============================================================
            t0 = time.time()

            # Prepare 10fps frames for lip extraction
            frames_rgb_10fps = frames_rgb_224[::3]

            def extract_mar():
                """Extract MAR from BGR frames"""
                result = self.mar_extractor.extract_from_frames(frames_bgr_384, fps=30.0)
                mar = np.array(result['mar_vertical'])
                mar = self._interpolate_nans(mar)
                return mar.reshape(-1, 1)

            def extract_mfcc():
                """Extract MFCC from WAV file"""
                return self._extract_mfcc_from_wav(str(audio_wav_path))

            def extract_lip():
                """Extract lip regions at 10fps"""
                return self.video_processor.extract_lip_region(
                    np.array(frames_rgb_10fps),
                    lip_size=(112, 112)
                )

            with ThreadPoolExecutor(max_workers=3) as executor:
                future_mar = executor.submit(extract_mar)
                future_mfcc = executor.submit(extract_mfcc)
                future_lip = executor.submit(extract_lip)

                mar_features_30fps = future_mar.result()
                audio_mfcc = future_mfcc.result()
                lip_10fps = future_lip.result()

            timings['parallel_cpu'] = time.time() - t0
            logger.info(f"  ✓ Parallel CPU tasks ({timings['parallel_cpu']:.2f}s)")

            # Fallback for MFCC
            if audio_mfcc is None:
                audio_mfcc = np.zeros((100, 40), dtype=np.float32)

            # ============================================================
            # STEP 4: GPU Sequential (WhisperX → ArcFace)
            # ============================================================
            # 4-1. WhisperX Transcription
            t0 = time.time()
            whisper_result = self.phoneme_aligner.transcribe_only(
                str(working_video_path),
                audio_path=str(audio_wav_path)  # Reuse WAV
            )
            timings['whisperx'] = time.time() - t0
            logger.info(f"  ✓ WhisperX ({timings['whisperx']:.2f}s)")

            # 4-2. Phoneme Alignment
            t0 = time.time()
            alignment_result = self.phoneme_aligner.align_from_transcription(
                whisper_result,
                timestamps=timestamps_30fps
            )
            timings['phoneme_alignment'] = time.time() - t0

            # Build phoneme intervals
            phoneme_intervals = []
            if 'phonemes' in alignment_result and 'intervals' in alignment_result:
                phonemes = cast(List[str], alignment_result['phonemes'])
                intervals = cast(List[Tuple[float, float]], alignment_result['intervals'])
                for phoneme_value, interval_value in zip(phonemes, intervals):
                    start = float(interval_value[0])
                    end = float(interval_value[1])
                    phoneme_intervals.append({
                        'phoneme': phoneme_value,
                        'start': start,
                        'end': end
                    })

            # Generate string phoneme labels
            phoneme_labels_str = np.full(len(timestamps_30fps), '', dtype='<U10')
            for ph_dict in phoneme_intervals:
                start = float(ph_dict.get('start', 0.0))
                end = float(ph_dict.get('end', 0.0))
                phoneme = str(ph_dict.get('phoneme', ''))
                mask = (timestamps_30fps >= start) & (timestamps_30fps <= end)
                phoneme_labels_str[mask] = phoneme

            # 4-3. ArcFace Extraction (GPU)
            t0 = time.time()
            arcface_features_30fps = self.arcface_extractor.extract_from_frames(frames_bgr_384)
            timings['arcface'] = time.time() - t0
            logger.info(f"  ✓ ArcFace ({timings['arcface']:.2f}s)")

            # ============================================================
            # STEP 5: Build output
            # ============================================================
            # 30fps → 10fps for MMMS-BA
            frames_10fps = frames_rgb_224[::3]

            timings['total'] = time.time() - total_start
            logger.info(f"  ✓ Total extraction time: {timings['total']:.2f}s")
            logger.info(f"  Timing breakdown: {timings}")

            features = {
                'frames_30fps': frames_rgb_224,
                'timestamps_30fps': timestamps_30fps,
                'audio': audio_mfcc,
                'phoneme_labels_30fps': phoneme_labels_str,
                'phoneme_intervals': phoneme_intervals,
                'mar_30fps': mar_features_30fps,
                'arcface_30fps': arcface_features_30fps,

                # Derived 10fps features
                'frames_10fps': frames_10fps,
                'lip_10fps': lip_10fps,

                'duration': len(frames_rgb_224) / 30.0,
                'fps': 30.0,
                'whisper_result': whisper_result,
                'timings': timings  # Include for debugging
            }

            return features
        finally:
            for path in cleanup_paths:
                try:
                    if path is not None:
                        Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

    def _extract_frames_sequential(self, video_path: str, target_fps: float = 30.0) -> Tuple[np.ndarray, float]:
        """
        Extract frames at target FPS using sequential read with skip.

        Optimized: Uses sequential read instead of seek operations.
        OpenCV seek (cap.set) is slow for non-keyframes.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate which frames to keep
        frame_interval = original_fps / target_fps
        frames_to_extract = set()
        idx = 0.0
        while int(idx) < total_frames:
            frames_to_extract.add(int(idx))
            idx += frame_interval

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames we need
            if frame_idx in frames_to_extract:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {total_frames} total (target {target_fps}fps)")
        return np.array(frames), target_fps

    def _prepare_video_path(self, video_path: Path) -> Tuple[Path, Optional[Path]]:
        """
        Clamp video duration by creating a temporary clipped file when necessary.
        Returns working video path and optional temporary clip path (to clean up).
        """
        if self.max_duration is None or self.max_duration <= 0:
            return video_path, None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Failed to open video for duration check: {video_path}")
            return video_path, None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        if duration <= self.max_duration + 1e-3:
            return video_path, None

        logger.info(f"Clipping video {video_path.name} from {duration:.2f}s to {self.max_duration:.2f}s")
        fd, tmp_path = tempfile.mkstemp(suffix=video_path.suffix or ".mp4")
        os.close(fd)
        clipped_path = Path(tmp_path)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-t", f"{self.max_duration}",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            str(clipped_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode != 0:
            logger.warning(f"Failed to clip video. Using original file: {video_path}")
            clipped_path.unlink(missing_ok=True)
            return video_path, None

        return clipped_path, clipped_path

    def _extract_frames_dual(
        self, video_path: str, target_fps: float = 30.0
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Extract frames at target FPS with dual output (Frame Reuse).

        Returns:
            frames_rgb_224: (N, 224, 224, 3) float32 [0,1] - for model input
            frames_bgr_384: List of (384, 384, 3) uint8 BGR - for MAR/ArcFace
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate which frames to keep
        frame_interval = original_fps / target_fps
        frames_to_extract = set()
        idx = 0.0
        while int(idx) < total_frames:
            frames_to_extract.add(int(idx))
            idx += frame_interval

        frames_rgb_224 = []
        frames_bgr_384 = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frames_to_extract:
                # BGR 384x384 for MAR/ArcFace (keep as uint8)
                frame_bgr_384 = cv2.resize(frame, (384, 384))
                frames_bgr_384.append(frame_bgr_384)

                # RGB 224x224 normalized for model input
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb_224 = cv2.resize(frame_rgb, (224, 224))
                frame_rgb_224 = frame_rgb_224.astype(np.float32) / 255.0
                frames_rgb_224.append(frame_rgb_224)

            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(frames_rgb_224)} dual frames (target {target_fps}fps)")
        return np.array(frames_rgb_224), frames_bgr_384

    def _extract_audio_wav(self, video_path: str) -> Path:
        """
        Extract audio from video to a temporary WAV file (Audio Reuse).

        Returns:
            Path to temporary WAV file (caller must cleanup)
        """
        fd, tmp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)

        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.audio_processor.sample_rate),
            '-ac', '1',
            '-y',
            tmp_path
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return Path(tmp_path)
        except subprocess.CalledProcessError as e:
            Path(tmp_path).unlink(missing_ok=True)
            raise RuntimeError(f"Failed to extract audio from {video_path}") from e

    def _extract_mfcc_from_wav(self, wav_path: str) -> Optional[np.ndarray]:
        """
        Extract MFCC features from WAV file.

        Returns:
            MFCC features (T, n_mfcc) or None if failed
        """
        try:
            audio, sr = librosa.load(wav_path, sr=self.audio_processor.sample_rate)

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.audio_processor.n_mfcc,
                hop_length=self.audio_processor.hop_length,
                n_fft=self.audio_processor.n_fft
            )
            return mfcc.T  # (n_mfcc, T) -> (T, n_mfcc)
        except Exception as e:
            logger.error(f"Failed to extract MFCC from {wav_path}: {e}")
            return None

    def _interpolate_nans(self, data: np.ndarray) -> np.ndarray:
        """Linear interpolation for NaN values."""
        nans = np.isnan(data)
        if not nans.any():
            return data

        def _indices(mask: np.ndarray) -> np.ndarray:
            return mask.nonzero()[0]

        data[nans] = np.interp(_indices(nans), _indices(~nans), data[~nans])
        return data

