"""
Korean Phoneme Extraction for Deepfake Detection

This module extracts Korean phonemes from audio using:
1. OpenAI Whisper (fine-tuned for Korean) for speech-to-text
2. g2pk for grapheme-to-phoneme conversion
3. Alignment for timestamp extraction

Based on PIA paper's phoneme-temporal analysis.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Korean phoneme processing
try:
    from transformers import pipeline
    import torch
    from jamo import h2j, j2hcj
    from g2pk import G2p
except ImportError as e:
    raise ImportError(
        "Required libraries not installed. Run:\n"
        "pip install transformers jamo g2pk torch"
    ) from e


class KoreanPhonemeExtractor:
    """
    Extract Korean phonemes from audio/video files

    Key phonemes (PIA-based):
    - ㅁ, ㅂ, ㅍ: Bilabials (lips closed)
    - ㅏ: Open vowel (mouth open)
    - ㅗ, ㅜ: Rounded vowels (lips rounded)
    """

    # Key phonemes for lip-sync detection (based on PIA paper)
    KEY_PHONEMES = {
        'ㅁ': 'bilabial_m',      # /m/ - lips fully closed
        'ㅂ': 'bilabial_b_p',    # /b/, /p/ - lips closed
        'ㅍ': 'bilabial_p',      # /p'/ - aspirated bilabial
        'ㅏ': 'open_a',          # /a/ - mouth open
        'ㅗ': 'rounded_o',       # /o/ - lips rounded
        'ㅜ': 'rounded_u'        # /u/ - lips rounded
    }

    def __init__(self, model_name: str = "SungBeom/whisper-small-ko"):
        """
        Initialize Korean phoneme extractor

        Args:
            model_name: HuggingFace Whisper model for Korean ASR
                       Default: SungBeom/whisper-small-ko (fine-tuned on AI Hub 13,946hrs, WER: 9.48%)
                       Alternatives:
                         - openai/whisper-large-v3 (multilingual, larger)
                         - seastar105/whisper-medium-ko-zeroth (Zeroth dataset)
                         - p4b/whisper-small-ko-fl-v2 (FLEURS dataset)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading Korean Whisper ASR model: {model_name}")

        # Load Whisper model as pipeline (PyTorch only)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device}")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            framework="pt"  # Force PyTorch, avoid TensorFlow
        )

        # Load Korean grapheme-to-phoneme converter
        self.g2p = G2p()

        self.logger.info("Korean phoneme extractor initialized")

    def extract_from_video(self, video_path: str) -> Dict:
        """
        Extract phonemes from video file

        Args:
            video_path: Path to video file

        Returns:
            dict: {
                'transcription': "안녕하세요",
                'phonemes': ['ㅇ', 'ㅏ', 'ㄴ', ...],
                'intervals': [(0.0, 0.1), (0.1, 0.2), ...],
                'key_phonemes': {
                    'ㅁ': [(0.5, 0.65, 'bilabial_m'), ...],
                    ...
                },
                'duration': 12.3
            }
        """
        self.logger.info(f"Extracting audio from video: {Path(video_path).name}")

        # Extract audio from video
        audio, sr = self._extract_audio_from_video(video_path)

        # Run ASR
        transcription = self._speech_to_text(audio, sr)

        # Convert to phonemes with time intervals
        phonemes, intervals = self._text_to_phonemes(transcription, len(audio) / sr)

        # Filter key phonemes
        key_phonemes = self._filter_key_phonemes(phonemes, intervals)

        return {
            'transcription': transcription,
            'phonemes': phonemes,
            'intervals': intervals,
            'key_phonemes': key_phonemes,
            'duration': len(audio) / sr
        }

    def _extract_audio_from_video(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio from video file using librosa

        Args:
            video_path: Path to video file

        Returns:
            audio: Audio waveform (mono)
            sr: Sample rate (16000 Hz)
        """
        try:
            # librosa can read audio from video files directly
            audio, sr = librosa.load(video_path, sr=16000, mono=True)
            self.logger.info(f"Audio extracted: duration={len(audio)/sr:.2f}s, sr={sr}Hz")
            return audio, sr
        except Exception as e:
            self.logger.error(f"Failed to extract audio from {video_path}: {e}")
            raise

    def _speech_to_text(self, audio: np.ndarray, sr: int) -> str:
        """
        Convert speech to text using Whisper

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            transcription: Korean text
        """
        self.logger.info("Running Korean ASR (Whisper)...")

        # Whisper expects audio as numpy array
        # Specify language for better accuracy
        # return_timestamps=True required for audio > 30 seconds
        result = self.pipe(
            audio,
            return_timestamps=True,
            generate_kwargs={"language": "ko", "task": "transcribe"}
        )

        transcription = result["text"].strip()
        self.logger.info(f"Transcription: {transcription}")
        return transcription

    def _text_to_phonemes(
        self,
        text: str,
        duration: float
    ) -> Tuple[List[str], List[Tuple[float, float]]]:
        """
        Convert Korean text to phonemes using g2pk

        Args:
            text: Korean text
            duration: Audio duration (seconds)

        Returns:
            phonemes: List of phonemes
            intervals: List of (start_time, end_time) tuples for each phoneme
        """
        self.logger.info("Converting text to phonemes (g2pk)...")

        # Remove spaces for phoneme conversion
        text_no_space = text.replace(" ", "")

        # Convert to phonemes using g2pk
        phoneme_text = self.g2p(text_no_space)

        # Split into individual phonemes (Jamo decomposition)
        phonemes = []
        for char in phoneme_text:
            if '\u3131' <= char <= '\u318E':  # Hangul compatibility Jamo
                phonemes.append(char)
            elif '\uAC00' <= char <= '\uD7A3':  # Hangul syllables
                # Decompose syllable into Jamo
                jamo = j2hcj(h2j(char))
                phonemes.extend(list(jamo))
            else:
                # Keep other characters as-is
                if char.strip():
                    phonemes.append(char)

        # Estimate time intervals (uniform distribution)
        # Note: This is a simplified approach
        # For accurate timestamps, use forced alignment (e.g., Montreal Forced Aligner)
        num_phonemes = len(phonemes)
        phoneme_duration = duration / num_phonemes

        intervals = []
        for i in range(num_phonemes):
            start_time = i * phoneme_duration
            end_time = (i + 1) * phoneme_duration
            intervals.append((start_time, end_time))

        self.logger.info(f"Extracted {num_phonemes} phonemes")
        return phonemes, intervals

    def _filter_key_phonemes(
        self,
        phonemes: List[str],
        intervals: List[Tuple[float, float]]
    ) -> Dict[str, List[Tuple[float, float, str]]]:
        """
        Filter key phonemes for lip-sync analysis

        Args:
            phonemes: List of all phonemes
            intervals: List of (start_time, end_time) tuples

        Returns:
            key_phonemes: {
                'ㅁ': [(0.5, 0.65, 'bilabial_m'), (1.2, 1.35, 'bilabial_m')],
                ...
            }
            Each tuple is (start_time, end_time, phoneme_type)
        """
        key_phonemes = {k: [] for k in self.KEY_PHONEMES.keys()}

        for phoneme, (start, end) in zip(phonemes, intervals):
            if phoneme in self.KEY_PHONEMES:
                key_phonemes[phoneme].append(
                    (start, end, self.KEY_PHONEMES[phoneme])
                )

        return key_phonemes

    def extract_with_forced_alignment(self, video_path: str) -> Dict:
        """
        Extract phonemes from video using MFA forced alignment

        This method uses Montreal Forced Aligner to get accurate phoneme-level
        timestamps instead of the uniform distribution assumption.

        Args:
            video_path: Path to video file

        Returns:
            dict: {
                'transcription': "안녕하세요",
                'phonemes': ['ㅇ', 'ㅏ', 'ㄴ', ...],
                'intervals': [(0.0, 0.12), (0.12, 0.18), ...],  # MFA timestamps
                'key_phonemes': {
                    'ㅁ': [(0.5, 0.65, 'bilabial_m'), ...],
                    ...
                },
                'duration': 12.3,
                'method': 'mfa'  # Indicates forced alignment was used
            }
        """
        self.logger.info(f"Extracting phonemes using MFA: {Path(video_path).name}")

        # Import MFA wrapper (lazy import to avoid dependency if not used)
        try:
            from .mfa_wrapper import MFAWrapper
        except ImportError as e:
            raise ImportError(
                "MFA wrapper not available. Make sure mfa_wrapper.py is in the same directory."
            ) from e

        # Initialize MFA wrapper
        mfa = MFAWrapper()

        # Run forced alignment (segmented 방식 사용 - 긴 오디오 대응)
        mfa_result = mfa.align_video_segmented(video_path)

        # Filter key phonemes
        key_phonemes = self._filter_key_phonemes(
            mfa_result['phonemes'],
            mfa_result['intervals']
        )

        return {
            'transcription': mfa_result['transcription'],
            'phonemes': mfa_result['phonemes'],
            'intervals': mfa_result['intervals'],
            'key_phonemes': key_phonemes,
            'duration': mfa_result['duration'],
            'method': 'mfa',  # Indicate this used forced alignment
            'words': mfa_result.get('words', [])  # Include word-level info if available
        }

    def extract_with_wav2vec2(self, video_path: str) -> Dict:
        """
        Extract phonemes from video using Wav2Vec2 phoneme recognition (slplab model)

        This method uses:
        1. WhisperX VAD for speech segment detection (50% faster by skipping silence)
        2. slplab/wav2vec2-xls-r-300m_phone-mfa_korean for accurate phoneme timestamps
        3. IPA-to-Jamo conversion for Korean Jamo characters

        Advantages:
        - 50% faster than WhisperX (processes only speech segments)
        - 42% less GPU memory (VAD filters non-speech)
        - Phoneme-level accuracy (±20-50ms vs ±100-150ms for character-level)
        - No external dependencies (unlike MFA which requires conda setup)

        Args:
            video_path: Path to video file

        Returns:
            dict: {
                'transcription': '',  # Not available (no ASR, VAD+phoneme only)
                'phonemes': ['ㅎ', 'ㅏ', 'ㄱ', ...],  # Jamo characters
                'intervals': [(0.0, 0.015), (0.015, 0.030), ...],  # Wav2Vec2 CTC timestamps
                'key_phonemes': {
                    'ㅁ': [(0.5, 0.65, 'bilabial_m'), ...],
                    ...
                },
                'duration': 12.3,
                'method': 'wav2vec2_slplab',  # Indicates Wav2Vec2 phoneme recognition
                'vad_segments': [(0.5, 3.2), (4.1, 7.8), ...]  # Speech segments from VAD
            }
        """
        self.logger.info(f"Extracting phonemes using Wav2Vec2: {Path(video_path).name}")

        # Import Wav2Vec2 aligner (lazy import to avoid dependency if not used)
        try:
            from .wav2vec2_korean_phoneme_aligner import Wav2Vec2KoreanPhonemeAligner
        except ImportError as e:
            raise ImportError(
                "Wav2Vec2 Korean aligner not available. Make sure wav2vec2_korean_phoneme_aligner.py "
                "is in the same directory and dependencies are installed:\n"
                "pip install whisperx transformers torch"
            ) from e

        # Initialize Wav2Vec2 aligner
        wav2vec2 = Wav2Vec2KoreanPhonemeAligner()

        # Run VAD + phoneme recognition
        wav2vec2_result = wav2vec2.align_video(video_path)

        # Filter key phonemes
        key_phonemes = self._filter_key_phonemes(
            wav2vec2_result['phonemes'],
            wav2vec2_result['intervals']
        )

        return {
            'transcription': wav2vec2_result.get('transcription', ''),  # Empty for Wav2Vec2
            'phonemes': wav2vec2_result['phonemes'],
            'intervals': wav2vec2_result['intervals'],
            'key_phonemes': key_phonemes,
            'duration': wav2vec2_result['duration'],
            'method': 'wav2vec2_slplab',  # Indicate this used Wav2Vec2 phoneme recognition
            'vad_segments': wav2vec2_result.get('vad_segments', []),  # Include VAD info
            'words': wav2vec2_result.get('words', [])  # Empty for phoneme-level model
        }


def main():
    """Test function"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python korean_phoneme_extractor.py <video_path>")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Extract phonemes
    extractor = KoreanPhonemeExtractor()
    result = extractor.extract_from_video(sys.argv[1])

    # Print results
    print("\n=== Phoneme Extraction Results ===")
    print(f"원본 문장: {result['transcription']}")
    print(f"총 음소 개수: {len(result['phonemes'])}")
    print(f"Duration: {result['duration']:.2f}s")
    print("\n핵심 음소:")
    for phoneme, occurrences in result['key_phonemes'].items():
        if occurrences:
            times = [f"{t:.2f}s" for t, _ in occurrences]
            print(f"  {phoneme}: {len(occurrences)}회 at {times}")
        else:
            print(f"  {phoneme}: 0회")


if __name__ == "__main__":
    main()
