import cv2
import numpy as np
import logging
import tempfile
import subprocess
import os
from typing import Dict, Any, List, Tuple, Optional, Union, cast
from pathlib import Path

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
        """
        video_path_obj = Path(video_path)
        logger.info(f"Extracting all features for: {video_path_obj.name}")

        working_video_path, temp_clip = self._prepare_video_path(video_path_obj)
        cleanup_paths: List[Path] = []
        if temp_clip is not None:
            cleanup_paths.append(temp_clip)
        
        try:
            # 1. Extract Frames at 30fps (Full Video or clipped)
            frames_30fps, fps = self._extract_frames_sequential(str(working_video_path), target_fps=30.0)
            num_frames = len(frames_30fps)
            timestamps_30fps = np.arange(num_frames) / 30.0

            # 2. WhisperX Transcription & Phoneme Alignment
            whisper_result = self.phoneme_aligner.transcribe_only(str(working_video_path))
            alignment_result = self.phoneme_aligner.align_from_transcription(
                whisper_result,
                timestamps=timestamps_30fps
            )

            # [NEW] Preserve exact phoneme intervals for PIA
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

            # [FIX] Generate string phoneme labels directly from intervals
            # The aligner returns integer indices which can be ambiguous with 0-initialization.
            # We rebuild exact string labels here for consistency with training logic.
            phoneme_labels_str = np.full(len(timestamps_30fps), '', dtype='<U10')
            for ph_dict in phoneme_intervals:
                start_value = cast(float, ph_dict.get('start', 0.0))
                end_value = cast(float, ph_dict.get('end', 0.0))
                start = float(start_value)
                end = float(end_value)
                phoneme = str(ph_dict.get('phoneme', ''))
                # Find frames in this interval
                mask = (timestamps_30fps >= start) & (timestamps_30fps <= end)
                phoneme_labels_str[mask] = phoneme

            phoneme_labels_30fps = phoneme_labels_str

            # 3. Audio MFCC (Full audio)
            audio_mfcc = self.audio_processor.extract_audio(
                str(working_video_path),
                feature_type='mfcc'
            )
            if audio_mfcc is None:
                audio_mfcc = np.zeros((100, 40), dtype=np.float32)  # Fallback

            # 4. MAR Extraction (30fps)
            # Prepare BGR frames for extractors
            frames_bgr = [cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for f in frames_30fps]

            mar_result = self.mar_extractor.extract_from_frames(frames_bgr, fps=30.0)
            # We typically use mar_vertical for PIA
            mar_features_30fps = np.array(mar_result['mar_vertical'])

            # Handle NaNs in MAR (interpolation)
            mar_features_30fps = self._interpolate_nans(mar_features_30fps)
            mar_features_30fps = mar_features_30fps.reshape(-1, 1)  # (T, 1)

            # 5. ArcFace Extraction (30fps)
            arcface_features_30fps = self.arcface_extractor.extract_from_frames(frames_bgr)

            # 6. Downsample for MMMS-BA (10fps)
            # 30fps -> 10fps means taking every 3rd frame
            frames_10fps = frames_30fps[::3]

            # 7. Lip features for MMMS-BA (10fps)
            # [FIX] MMMS-BA uses 112x112 crop (matched with training config)
            lip_10fps = self.video_processor.extract_lip_region(
                np.array(frames_10fps),
                lip_size=(112, 112)
            )

            features = {
                'frames_30fps': frames_30fps,
                'timestamps_30fps': timestamps_30fps,
                'audio': audio_mfcc,
                'phoneme_labels_30fps': phoneme_labels_30fps,
                'phoneme_intervals': phoneme_intervals,  # [NEW] Exact intervals
                'mar_30fps': mar_features_30fps,  # (T, 1)
                'arcface_30fps': arcface_features_30fps,  # (T, 512)

                # Derived 10fps features
                'frames_10fps': frames_10fps,
                'lip_10fps': lip_10fps,

                'duration': len(frames_30fps) / 30.0,
                'fps': 30.0,
                'whisper_result': whisper_result
            }

            return features
        finally:
            for path in cleanup_paths:
                try:
                    path.unlink(missing_ok=True)
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

    def _interpolate_nans(self, data: np.ndarray) -> np.ndarray:
        """Linear interpolation for NaN values."""
        nans = np.isnan(data)
        if not nans.any():
            return data

        def _indices(mask: np.ndarray) -> np.ndarray:
            return mask.nonzero()[0]

        data[nans] = np.interp(_indices(nans), _indices(~nans), data[~nans])
        return data

