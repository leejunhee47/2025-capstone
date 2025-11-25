"""
WhisperX-based Korean Phoneme Aligner

This module provides a drop-in replacement for MFAWrapper using WhisperX
for 225x faster processing (75min → 15-20s for 90s video).

WhisperX integrates:
- VAD (Voice Activity Detection): pyannote
- ASR (Automatic Speech Recognition): Whisper large-v3
- Alignment: Wav2Vec2 Korean model

Key advantages:
- GPU acceleration for entire pipeline
- Automatic VAD-based segmentation
- Batch processing for parallel segment handling
- Native character-level timestamps

Requirements:
- pip install git+https://github.com/m-bain/whisperx.git
- pip install g2pk (for pronunciation-based Jamo conversion)
"""

import os
import logging
import warnings
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

warnings.filterwarnings('ignore')

# WhisperX
try:
    import whisperx
except ImportError as e:
    raise ImportError(
        "WhisperX가 설치되지 않았습니다. 실행:\n"
        "pip install git+https://github.com/m-bain/whisperx.git"
    ) from e

# g2pk (Grapheme-to-Phoneme Korean)
try:
    from g2pk import G2p
except ImportError as e:
    raise ImportError(
        "g2pk가 설치되지 않았습니다. 실행:\n"
        "pip install g2pk"
    ) from e

# jamo (Hangul Jamo decomposition)
try:
    from jamo import h2j, j2hcj
except ImportError as e:
    raise ImportError(
        "jamo가 설치되지 않았습니다. 실행:\n"
        "pip install jamo"
    ) from e


# ============================================================================
# Korean Vowel Duration Weights (Heuristic-based)
# ============================================================================
# These durations are based on Korean phonetics research and general patterns.
# Used for weighted timestamp interpolation when precise alignment is unavailable.
KOREAN_VOWEL_DURATIONS = {
    # 단모음 (Monophthongs): 90-120ms
    'ㅏ': 110,  # /a/ - most open vowel
    'ㅓ': 105,  # /ʌ/
    'ㅗ': 115,  # /o/
    'ㅜ': 110,  # /u/
    'ㅡ': 95,   # /ɨ/ - tense vowel, slightly shorter
    'ㅣ': 90,   # /i/ - tense vowel, shortest
    'ㅐ': 105,  # /ɛ/
    'ㅔ': 100,  # /e/
    'ㅚ': 120,  # /ø/ - rounded, slightly longer
    'ㅟ': 115,  # /y/ - rounded

    # 이중모음 (Diphthongs): 120-150ms (1.2-1.4x longer than monophthongs)
    'ㅑ': 130,  # /ja/
    'ㅕ': 125,  # /jʌ/
    'ㅛ': 135,  # /jo/
    'ㅠ': 130,  # /ju/
    'ㅒ': 125,  # /jɛ/
    'ㅖ': 120,  # /je/
    'ㅘ': 140,  # /wa/ - compound diphthong, longest
    'ㅙ': 135,  # /wɛ/
    'ㅝ': 135,  # /wʌ/
    'ㅞ': 130,  # /we/
    'ㅢ': 125,  # /ɨi/
}


def is_korean_vowel(jamo: str) -> bool:
    """
    Check if a jamo character is a vowel

    Args:
        jamo: Single jamo character

    Returns:
        bool: True if vowel, False otherwise
    """
    return jamo in KOREAN_VOWEL_DURATIONS


class WhisperXPhonemeAligner:
    """
    WhisperX-based Korean phoneme aligner with g2pk pronunciation conversion

    Provides MFAWrapper-compatible interface:
        align_video_segmented(video_path) -> Dict

    Output format (100% compatible):
        {
            'transcription': str,
            'phonemes': List[str],       # Jamo characters (pronunciation-based)
            'intervals': List[Tuple],    # (start_time, end_time) for each phoneme
            'words': List[Tuple],        # (word, start_time, end_time)
            'duration': float
        }
    """

    def __init__(
        self,
        whisper_model: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
        language: str = "ko"
    ):
        """
        Initialize WhisperX phoneme aligner

        Args:
            whisper_model: Whisper model size ('large-v3', 'medium', 'small', 'base')
            device: 'cuda' for GPU, 'cpu' for CPU
            compute_type: 'float16' (fast), 'int8' (faster), 'float32' (accurate)
            batch_size: Number of segments to process in parallel (16 recommended for 8GB GPU)
            language: Language code ('ko' for Korean)
        """
        self.logger = logging.getLogger(__name__)
        self.device = device if torch.cuda.is_available() else "cpu"

        if self.device == "cpu" and device == "cuda":
            self.logger.warning("CUDA not available, falling back to CPU")

        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = language

        self.logger.info(f"Initializing WhisperX on {self.device} with {compute_type}")

        # Load Whisper model (includes VAD)
        self.logger.info(f"Loading Whisper model: {whisper_model}")
        self.model = whisperx.load_model(
            whisper_model,
            device=self.device,
            compute_type=compute_type
        )

        # Load Wav2Vec2 alignment model for Korean
        self.logger.info("Loading Korean alignment model (Wav2Vec2)")
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device
        )

        # Initialize g2pk for pronunciation-based Jamo conversion
        self.logger.info("Initializing g2pk (Korean G2P)")
        self.g2p = G2p()

        self.logger.info(f"WhisperX initialized successfully")
        self.logger.info(f"  - Device: {self.device}")
        self.logger.info(f"  - Compute type: {compute_type}")
        self.logger.info(f"  - Batch size: {batch_size}")
        self.logger.info(f"  - Alignment model: {self.align_metadata.get('model_name', 'korean')}")

    def align_video_segmented(self, video_path: str) -> Dict:
        """
        Main interface for phoneme extraction (MFAWrapper compatible)

        Pipeline:
        1. Extract audio from video (ffmpeg)
        2. WhisperX transcription with VAD + ASR (GPU batch processing)
        3. Wav2Vec2 word-level alignment (GPU)
        4. g2pk pronunciation conversion (CPU)
        5. Distribute timestamps across phonemes

        Args:
            video_path: Path to video file

        Returns:
            dict: {
                'transcription': Full transcription text,
                'phonemes': List of Jamo phonemes (pronunciation-based),
                'intervals': List of (start, end) tuples for each phoneme,
                'words': List of (word, start, end) tuples,
                'duration': Audio duration in seconds
            }
        """
        video_path = Path(video_path)

        if not video_path.exists():
            self.logger.error(f"Video file not found: {video_path}")
            return self._empty_result()

        self.logger.info(f"Processing video: {video_path.name}")

        try:
            # Step 1: Extract audio
            self.logger.info("  [1/4] Extracting audio from video...")
            audio_path = self._extract_audio(video_path)

            # Step 2: Load audio with WhisperX
            self.logger.info("  [2/4] Loading audio...")
            audio = whisperx.load_audio(str(audio_path))
            duration = len(audio) / 16000.0  # WhisperX uses 16kHz
            self.logger.info(f"    Audio duration: {duration:.2f}s")

            # Step 3: WhisperX transcription (VAD + ASR with batch processing)
            self.logger.info(f"  [3/4] Running WhisperX transcription (batch_size={self.batch_size})...")
            result = self.model.transcribe(
                audio,
                batch_size=self.batch_size,
                language=self.language
            )

            transcription = result.get("text", "")
            num_segments = len(result.get("segments", []))
            self.logger.info(f"    Transcription: '{transcription[:100]}...'")
            self.logger.info(f"    Segments detected: {num_segments}")

            if not result.get("segments"):
                self.logger.warning("No segments detected (silent video or VAD failure)")
                return self._empty_result()

            # Step 4: Wav2Vec2 alignment (character-level timestamps)
            self.logger.info("  [4/4] Running Wav2Vec2 alignment...")
            aligned_result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                device=self.device,
                return_char_alignments=True  # Enable character-level for accurate jamo timestamps
            )

            # Step 5: Extract phonemes with g2pk + timestamp distribution
            phonemes, intervals, words = self._extract_phonemes_g2pk(aligned_result)

            # Cleanup
            if audio_path.exists():
                audio_path.unlink()

            # Final result
            final_result = {
                'transcription': transcription,
                'phonemes': phonemes,
                'intervals': intervals,
                'words': words,
                'duration': duration
            }

            self.logger.info(f"✓ WhisperX processing complete:")
            self.logger.info(f"    Phonemes extracted: {len(phonemes)}")
            self.logger.info(f"    Words extracted: {len(words)}")
            self.logger.info(f"    Coverage: {intervals[-1][1] if intervals else 0:.2f}s / {duration:.2f}s")

            return final_result

        except torch.cuda.OutOfMemoryError:
            self.logger.error("GPU out of memory, retrying with CPU...")
            self.device = "cpu"
            self.model = whisperx.load_model(
                "large-v3", device="cpu", compute_type="float32"
            )
            return self.align_video_segmented(str(video_path))

        except Exception as e:
            self.logger.error(f"WhisperX processing failed: {e}", exc_info=True)
            return self._empty_result()

    def _extract_audio(self, video_path: Path) -> Path:
        """
        Extract audio from video using ffmpeg

        Args:
            video_path: Path to video file

        Returns:
            Path: Temporary WAV file path
        """
        # Create temporary WAV file
        temp_dir = Path(tempfile.gettempdir())
        audio_path = temp_dir / f"{video_path.stem}_whisperx.wav"

        # ffmpeg command: extract audio as 16kHz mono WAV
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-i", str(video_path),
            "-ar", "16000",  # 16kHz (WhisperX default)
            "-ac", "1",      # Mono
            "-vn",           # No video
            "-f", "wav",
            str(audio_path)
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg audio extraction failed: {e}")

        return audio_path

    def _distribute_vowel_timestamps(self, word_start: float, word_end: float, jamos: List[str]) -> Tuple[List[str], List[Tuple]]:
        """
        Extract only vowels from jamos and distribute timestamps using duration weights

        This method improves over uniform distribution by:
        1. Filtering only vowels (consonants don't affect MAR measurements)
        2. Using phonetic duration weights (diphthongs longer than monophthongs)
        3. Proportional time allocation based on expected durations

        Args:
            word_start: Word start time in seconds
            word_end: Word end time in seconds
            jamos: List of all jamo characters from the word

        Returns:
            tuple: (vowel_phonemes, vowel_intervals)
                - vowel_phonemes: List of vowel jamos only
                - vowel_intervals: List of (start, end) tuples for each vowel

        Example:
            word "학교" (0.5 ~ 1.0s):
            jamos = ['ㅎ', 'ㅏ', 'ㄱ', 'ㄱ', 'ㅛ']

            Vowels: ['ㅏ', 'ㅛ']
            Weights: [110ms, 135ms]
            Ratios: [0.449, 0.551]

            Output:
            (['ㅏ', 'ㅛ'], [(0.5, 0.7245), (0.7245, 1.0)])
        """
        # 1. Filter only vowels
        vowels = [j for j in jamos if is_korean_vowel(j)]

        if not vowels:
            # No vowels in this word (rare, but handle gracefully)
            return [], []

        # 2. Get expected durations for each vowel
        expected_durations = [KOREAN_VOWEL_DURATIONS[v] for v in vowels]
        total_expected = sum(expected_durations)

        # 3. Calculate actual word duration
        word_duration = word_end - word_start

        # 4. Distribute time proportionally based on weights
        vowel_intervals = []
        current_time = word_start

        for vowel, expected_dur in zip(vowels, expected_durations):
            # Calculate weight ratio
            ratio = expected_dur / total_expected
            actual_dur = word_duration * ratio

            # Create interval
            vowel_intervals.append((round(current_time, 3), round(current_time + actual_dur, 3)))
            current_time += actual_dur

        return vowels, vowel_intervals

    def _extract_phonemes_g2pk(self, aligned_result: Dict) -> Tuple[List[str], List[Tuple], List[Tuple]]:
        """
        Extract phonemes from WhisperX character-level alignment with Jamo decomposition

        Args:
            aligned_result: WhisperX alignment output with character-level timestamps

        Returns:
            tuple: (phonemes, intervals, words)
                - phonemes: List of individual Jamo characters (자모 단위)
                - intervals: List of (start, end) tuples from Wav2Vec2
                - words: List of (word, start, end) tuples
        """
        phonemes = []
        intervals = []
        words = []

        for segment in aligned_result.get("segments", []):
            # Process each word in the segment
            for word_info in segment.get("words", []):
                word_text = word_info.get("word", "").strip()
                word_start = word_info.get("start", 0.0)
                word_end = word_info.get("end", 0.0)

                if not word_text or word_end <= word_start:
                    continue

                # Store word-level timestamp
                words.append((word_text, word_start, word_end))

                # Character-level alignment (Wav2Vec2가 제공)
                char_alignments = word_info.get("chars", [])

                if char_alignments:
                    # Use Wav2Vec2 character-level timestamps (더 정확함!)
                    for char_info in char_alignments:
                        char = char_info.get("char", "")
                        char_start = char_info.get("start", 0.0)
                        char_end = char_info.get("end", 0.0)

                        if not char.strip():
                            continue

                        # g2pk로 발음 변환 (단일 문자)
                        try:
                            phoneme_str = self.g2p(char)
                        except Exception as e:
                            phoneme_str = char

                        # 자모 분해
                        for syllable in phoneme_str:
                            if syllable.strip():
                                # 한글 음절인지 확인 (0xAC00-0xD7A3)
                                if 0xAC00 <= ord(syllable) <= 0xD7A3:
                                    # 한글 음절 → 자모 분해
                                    # h2j: 한글 → 자모, j2hcj: 호환 자모 → 개별 자모
                                    try:
                                        jamo_chars = j2hcj(h2j(syllable))
                                        jamo_list = list(jamo_chars)

                                        # 문자 시간을 자모 개수만큼 분배
                                        # Wav2Vec2가 이미 문자별로 분석했으므로 더 정확
                                        char_duration = char_end - char_start
                                        jamo_duration = char_duration / len(jamo_list) if jamo_list else 0

                                        for i, jamo in enumerate(jamo_list):
                                            jamo_start = char_start + i * jamo_duration
                                            jamo_end = char_start + (i + 1) * jamo_duration
                                            phonemes.append(jamo)
                                            intervals.append((round(jamo_start, 3), round(jamo_end, 3)))
                                    except Exception as e:
                                        # Jamo decomposition 실패 시 음절 그대로
                                        phonemes.append(syllable)
                                        intervals.append((round(char_start, 3), round(char_end, 3)))
                                else:
                                    # 비한글 문자는 그대로
                                    phonemes.append(syllable)
                                    intervals.append((round(char_start, 3), round(char_end, 3)))
                else:
                    # Fallback: char alignment 없으면 가중치 기반 모음 분배
                    self.logger.warning(f"No character alignment for word '{word_text}', using weighted vowel distribution")

                    try:
                        phoneme_str = self.g2p(word_text)
                    except Exception as e:
                        phoneme_str = word_text

                    # 자모 분해 (전체 자모 추출)
                    word_jamos = []
                    for syllable in phoneme_str:
                        if syllable.strip():
                            if 0xAC00 <= ord(syllable) <= 0xD7A3:
                                try:
                                    jamo_chars = j2hcj(h2j(syllable))
                                    word_jamos.extend(list(jamo_chars))
                                except:
                                    word_jamos.append(syllable)
                            else:
                                word_jamos.append(syllable)

                    # 모음만 추출 + 가중치 기반 타임스탬프 분배
                    vowels, vowel_intervals = self._distribute_vowel_timestamps(
                        word_start, word_end, word_jamos
                    )

                    # 모음만 phonemes/intervals에 추가
                    phonemes.extend(vowels)
                    intervals.extend(vowel_intervals)

        return phonemes, intervals, words

    def _empty_result(self) -> Dict:
        """Return empty result for failed processing"""
        return {
            'transcription': '',
            'phonemes': [],
            'intervals': [],
            'words': [],
            'duration': 0.0
        }
