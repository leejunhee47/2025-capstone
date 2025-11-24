"""
하이브리드 음소 정렬기 (Hybrid Phoneme Aligner)
WhisperX + Wav2Vec2 강제 정렬(Forced Alignment) 사용

PIA 논문 구현 - 한국어 음소(phoneme) 추출

파이프라인:
1. WhisperX 전사(transcription) + 세그먼트 분할 (VAD 포함)
2. 텍스트 → 음소 변환 (G2PK + 자모 분해)
3. Wav2Vec2 강제 정렬 (세그먼트별)
"""

# pyright: reportMissingTypeStubs=false

import logging
from typing import Dict, List, Tuple, Union, Optional, TypedDict
from pathlib import Path
import numpy as np
import torch
import whisperx  # type: ignore[import]
from g2pk import G2p  # type: ignore[import]
from jamo import h2j, j2hcj  # type: ignore[import]
from .jamo_to_mfa import jamo_to_mfa
from .korean_phoneme_config import KEEP_PHONEMES_KOREAN, is_kept_phoneme
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torchaudio.functional import forced_align  # type: ignore[import]

# 로깅 설정
logging.basicConfig(level=logging.INFO)


# ===== Type Definitions for Optimization (30fps) =====

class WhisperTranscriptionResult(TypedDict):
    """
    WhisperX transcription 결과 구조체

    STEP 1.5에서 1회만 transcribe하여 Stage1/Stage2가 공유
    """
    segments: List[dict]      # WhisperX segments with timestamps
    transcription: str        # Full text transcription
    audio: np.ndarray        # Audio samples (for reuse, 16kHz)
    audio_path: str          # Path to WAV file (for cleanup)


class PhonemeAlignmentResult(TypedDict):
    """
    Phoneme alignment 결과 구조체

    align_from_transcription()이 반환하는 타입
    """
    phonemes: List[str]                      # List of phoneme symbols (14 key phonemes)
    intervals: List[Tuple[float, float]]     # Time intervals [(start, end), ...]
    phoneme_labels: np.ndarray              # Frame-matched phoneme indices
    transcription: str                       # Original transcription text
    duration: float                          # Total audio duration
    method: str                              # Alignment method name


class HybridPhonemeAligner:
    """
    PIA 스타일 음소 정렬기 (WhisperX + Wav2Vec2)

    WhisperX로 전사(transcription)와 세그먼트 분할을 수행한 후,
    세그먼트별로 Wav2Vec2 강제 정렬(forced alignment)을 적용하여
    정확한 음소 레벨 타임스탬프를 생성합니다.
    """

    def __init__(
        self,
        whisper_model: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        """
        하이브리드 정렬기 초기화

        Args:
            whisper_model: WhisperX 모델 크기
            device: 사용할 디바이스 (cuda/cpu)
            compute_type: WhisperX 계산 타입
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initializing HybridPhonemeAligner on {self.device}")

        # 전사(transcription)를 위한 WhisperX 로드
        self.logger.info(f"Loading WhisperX model: {whisper_model}")
        self.whisper_model = whisperx.load_model(
            whisper_model,
            self.device,
            compute_type=compute_type,
            language="ko"
        )

        # 강제 정렬(forced alignment)을 위한 한국어 Wav2Vec2 모델 로드
        model_name = "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"
        self.logger.info(f"Loading Wav2Vec2 model: {model_name}")

        try:
            # safetensors로 먼저 로드 시도
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                use_safetensors=True
            )
        except Exception as e:
            self.logger.warning(f"Failed to load with safetensors: {e}")
            # trust_remote_code로 폴백
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )

        self.wav2vec2_model = self.wav2vec2_model.to(self.device)
        self.wav2vec2_model.eval()

        # WhisperX 한국어 정렬 모델 로드 (PIA 스타일 정제)
        self.logger.info("Loading Korean align model for WhisperX")
        try:
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code="ko",  # kresnik/wav2vec2-large-xlsr-korean 사용
                device=self.device
            )
            self.logger.info("  Align model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load align model: {e}")
            self.logger.warning("  Will use uniform distribution only")
            self.align_model = None
            self.align_metadata = None

        # 한국어 발음 변환을 위한 G2PK 로드
        self.logger.info("Loading G2PK for pronunciation conversion")
        self.g2p = G2p()

        # 음소 vocabulary 매핑 구축
        self._build_phoneme_vocab()

        self.logger.info("HybridPhonemeAligner initialized successfully")

    def _build_phoneme_vocab(self):
        """MFA 음소(phoneme)와 모델 vocabulary 간 매핑 구축"""
        # 모델 vocabulary 가져오기
        vocab = self.processor.tokenizer.get_vocab()

        # MFA 음소 기호들 (vocabulary 확인 결과)
        mfa_phonemes = [
            'A', 'B', 'BB', 'CHh', 'D', 'DD', 'E', 'EO', 'EU',
            'G', 'GG', 'H', 'I', 'J', 'JJ', 'Kh', 'L', 'M', 'N',
            'NG', 'O', 'Ph', 'R', 'S', 'SS', 'Th', 'U',
            'euI', 'iA', 'iE', 'iEO', 'iO', 'iU', 'k', 'oA',
            'oE', 'p', 't', 'uEO', 'uI'
        ]

        # 양방향 매핑 생성
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}

        for phoneme in mfa_phonemes:
            if phoneme in vocab:
                token_id = vocab[phoneme]
                self.phoneme_to_id[phoneme] = token_id
                self.id_to_phoneme[token_id] = phoneme

        # 특수 토큰 추가
        if "[PAD]" in vocab:
            self.pad_token_id = vocab["[PAD]"]
        elif "<pad>" in vocab:
            self.pad_token_id = vocab["<pad>"]
        else:
            self.pad_token_id = 0

        self.logger.info(f"Built phoneme vocabulary with {len(self.phoneme_to_id)} entries")

    def align_video(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path, None] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> Dict:
        """
        PIA 파이프라인을 사용한 비디오 음소 정렬

        **리팩토링**: transcribe_only() + align_from_transcription() wrapper
        하위 호환성을 위해 유지됩니다.

        Args:
            video_path: 비디오 파일 경로
            audio_path: 사전 추출된 오디오 파일 경로 (Audio Reuse 최적화, 선택 사항)
            start_time: 시작 시간 (초, 기본값: 0.0)
            end_time: 종료 시간 (초, 기본값: None - 전체 비디오)

        Returns:
            음소(phonemes), 구간(intervals), 전사(transcription)를 포함한 딕셔너리
        """
        try:
            self.logger.info(f"[ALIGN_VIDEO] Processing: {video_path}")

            # Step 1: Transcription (WhisperX)
            transcription_result = self.transcribe_only(
                video_path=video_path,
                audio_path=audio_path,
                start_time=start_time,
                end_time=end_time
            )

            # Step 2: Alignment (Phoneme matching)
            alignment_result = self.align_from_transcription(
                transcription_result=transcription_result,
                timestamps=None,  # No frame matching in standalone mode
                start_time=0.0    # No offset for full video
            )

            # Cleanup audio file if we extracted it
            audio_file = Path(transcription_result['audio_path'])
            if audio_file.exists() and audio_path is None:
                audio_file.unlink()
                self.logger.info(f"  ✓ Cleaned up temporary audio: {audio_file.name}")

            # Return result in original format (backward compatibility)
            return {
                'phonemes': alignment_result['phonemes'],
                'intervals': alignment_result['intervals'],
                'transcription': alignment_result['transcription'],
                'duration': alignment_result['duration'],
                'method': alignment_result['method']
            }

        except Exception as e:
            self.logger.error(f"Error processing {video_path}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._empty_result()

    def transcribe_only(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path, None] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> WhisperTranscriptionResult:
        """
        WhisperX transcription만 수행 (alignment는 하지 않음)

        30fps 최적화: STEP 1.5에서 1회만 호출하여 Stage1/Stage2가 공유

        Args:
            video_path: 비디오 파일 경로
            audio_path: 사전 추출된 오디오 파일 경로 (선택 사항)
            start_time: 시작 시간 (초, 기본값: 0.0)
            end_time: 종료 시간 (초, 기본값: None - 전체 비디오)

        Returns:
            WhisperTranscriptionResult: segments, transcription, audio, audio_path
        """
        if not isinstance(video_path, Path):
            video_path = Path(video_path)

        if not video_path.exists():
            self.logger.error(f"Video not found: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")

        cleanup_audio = False

        try:
            # 단계 1: 비디오에서 오디오 추출 (또는 사전 추출된 파일 사용)
            self.logger.info(f"[TRANSCRIBE_ONLY] Processing: {video_path.name}")
            if start_time > 0.0 or end_time is not None:
                self.logger.info(f"  Time range: {start_time:.2f}s - {end_time if end_time else 'end'}s")

            if audio_path is not None:
                # Audio Reuse: 사전 추출된 오디오 사용
                audio_path = Path(audio_path)
                cleanup_audio = False
                self.logger.info(f"  Using pre-extracted audio: {audio_path.name}")
            else:
                # 기존 방식: 비디오에서 오디오 추출
                audio_path = self._extract_audio(video_path, start_time, end_time)
                cleanup_audio = True

            # WhisperX를 위한 오디오 로드
            audio = whisperx.load_audio(str(audio_path))

            # 단계 2: WhisperX 전사 + 세그먼트 분할
            self.logger.info("  WhisperX transcription...")
            result = self.whisper_model.transcribe(
                audio,
                batch_size=4,
                language="ko"
            )

            segments = result.get("segments", [])
            if not segments:
                self.logger.warning("No segments found")
                segments = []

            transcription = result.get("text", "")
            self.logger.info(f"  ✓ Transcription: {transcription[:100]}...")
            self.logger.info(f"  ✓ Found {len(segments)} segments")

            # Return WhisperTranscriptionResult
            return {
                'segments': segments,
                'transcription': transcription,
                'audio': audio,
                'audio_path': str(audio_path)
            }

        except Exception as e:
            self.logger.error(f"Error in transcribe_only: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def align_from_transcription(
        self,
        transcription_result: WhisperTranscriptionResult,
        timestamps: Optional[np.ndarray] = None,
        start_time: float = 0.0
    ) -> PhonemeAlignmentResult:
        """
        Pre-computed transcription으로부터 phoneme alignment만 수행

        30fps 최적화: Stage2가 STEP 1.5의 shared transcription을 사용

        Args:
            transcription_result: transcribe_only()의 결과
            timestamps: Frame timestamps for phoneme-frame matching (optional)
            start_time: Interval offset (초, Stage2 interval의 시작 시간)

        Returns:
            PhonemeAlignmentResult: phonemes, intervals, phoneme_labels
        """
        try:
            segments = transcription_result['segments']
            audio = transcription_result['audio']
            transcription = transcription_result['transcription']

            if not segments:
                self.logger.warning("No segments in transcription_result")
                return self._empty_alignment_result()

            self.logger.info(f"[ALIGN_FROM_TRANSCRIPTION] Processing {len(segments)} segments")
            if start_time > 0.0:
                self.logger.info(f"  Interval offset: {start_time:.2f}s")

            # 단계 1: 각 세그먼트 처리
            all_phonemes = []
            all_intervals = []

            for seg_idx, segment in enumerate(segments):
                self.logger.info(f"  Processing segment {seg_idx+1}/{len(segments)}")

                # 세그먼트 시간 범위
                seg_start = segment.get("start", 0)
                seg_end = segment.get("end", 0)

                # Interval offset 적용 (Stage2에서 호출 시)
                # 예: video의 10.0s~15.0s 구간이면, segment timestamps를 10.0s 기준으로 조정
                if start_time > 0.0:
                    # Skip segments outside the interval
                    if seg_end < start_time:
                        continue
                    if seg_start > start_time + 30.0:  # Assume max 30s interval
                        continue

                # 빈 세그먼트 건너뛰기
                if seg_start >= seg_end:
                    continue

                # 세그먼트 오디오 샘플 가져오기
                start_sample = int(seg_start * 16000)
                end_sample = int(seg_end * 16000)
                segment_audio = audio[start_sample:end_sample]

                # 너무 짧은 세그먼트 건너뛰기
                if len(segment_audio) < 400:  # 최소 25ms
                    continue

                # 텍스트를 음소로 변환
                segment_text = segment.get("text", "").strip()
                if not segment_text:
                    continue

                phonemes = self._text_to_phonemes(segment_text)

                if not phonemes:
                    self.logger.warning(f"No phonemes for segment {seg_idx}: {segment_text}")
                    continue

                self.logger.info(f"    Text: {segment_text}")
                self.logger.info(f"    Phonemes ({len(phonemes)}): {''.join(phonemes[:20])}...")

                # 단계 2: 이 세그먼트에 PIA 스타일 정렬 적용
                phoneme_intervals = self._align_segment_pia_style(
                    segment_audio,
                    phonemes,
                    seg_start,  # 이 세그먼트의 시간 오프셋 (absolute time)
                    segment_text
                )

                if phoneme_intervals:
                    all_phonemes.extend(phonemes)
                    all_intervals.extend(phoneme_intervals)
                    self.logger.info(f"    ✓ Aligned {len(phoneme_intervals)} phonemes")

            self.logger.info(f"Total: {len(all_phonemes)} phonemes aligned (before filtering)")

            # 단계 3: 14개 핵심 음소로 필터링
            filtered_phonemes = []
            filtered_intervals = []

            for phoneme, interval in zip(all_phonemes, all_intervals):
                if is_kept_phoneme(phoneme):
                    filtered_phonemes.append(phoneme)
                    filtered_intervals.append(interval)

            self.logger.info(f"Filtered: {len(filtered_phonemes)} phonemes kept (from {len(all_phonemes)})")
            self.logger.info(f"Unique phonemes: {set(filtered_phonemes)}")

            # 단계 4: Phoneme-to-frame matching (if timestamps provided)
            phoneme_labels = np.array([], dtype=np.int32)
            if timestamps is not None and len(timestamps) > 0:
                phoneme_labels = self._match_phonemes_to_frames(
                    filtered_intervals,
                    timestamps,
                    start_time  # Apply offset for interval matching
                )
                self.logger.info(f"  ✓ Matched {len(phoneme_labels)} frames to phonemes")

            # Return PhonemeAlignmentResult
            return {
                'phonemes': filtered_phonemes,
                'intervals': filtered_intervals,
                'phoneme_labels': phoneme_labels,
                'transcription': transcription,
                'duration': float(len(audio) / 16000),
                'method': 'pia_style_from_shared_transcription'
            }

        except Exception as e:
            self.logger.error(f"Error in align_from_transcription: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _match_phonemes_to_frames(
        self,
        phoneme_intervals: List[Tuple[float, float]],
        timestamps: np.ndarray,
        offset: float = 0.0
    ) -> np.ndarray:
        """
        Match phoneme intervals to frame timestamps

        Args:
            phoneme_intervals: List of (start, end) tuples (absolute time)
            timestamps: Frame timestamps (relative time from offset)
            offset: Time offset to convert relative → absolute

        Returns:
            phoneme_labels: Array of phoneme indices per frame
        """
        phoneme_labels = np.zeros(len(timestamps), dtype=np.int32)

        for idx, rel_timestamp in enumerate(timestamps):
            # Convert relative timestamp to absolute time
            abs_timestamp = rel_timestamp + offset

            # Find matching phoneme
            for phoneme_idx, (phn_start, phn_end) in enumerate(phoneme_intervals):
                if phn_start <= abs_timestamp <= phn_end:
                    phoneme_labels[idx] = phoneme_idx
                    break

        return phoneme_labels

    def _empty_alignment_result(self) -> PhonemeAlignmentResult:
        """Return empty alignment result"""
        return {
            'phonemes': [],
            'intervals': [],
            'phoneme_labels': np.array([], dtype=np.int32),
            'transcription': '',
            'duration': 0.0,
            'method': 'empty'
        }

    def _apply_phoneme_padding(
        self,
        intervals: List[Tuple[float, float]],
        time_offset: float,
        segment_duration: float,
        padding_ms: float = 0.080
    ) -> List[Tuple[float, float]]:
        """
        음소 구간에 패딩을 적용하여 프레임 개수를 늘립니다.

        5프레임(166.7ms @ 30fps) 확보를 위해 각 음소 구간을 앞뒤로 확장합니다.

        Args:
            intervals: 원본 음소 구간 리스트 [(start, end), ...]
            time_offset: 세그먼트 시작 시간
            segment_duration: 세그먼트 전체 길이
            padding_ms: 앞뒤로 추가할 패딩 (초 단위, 기본 80ms)

        Returns:
            패딩이 적용된 음소 구간 리스트
        """
        padded = []
        segment_end = time_offset + segment_duration

        # DEBUG: 첫 3개 음소의 패딩 전후 비교
        if len(intervals) > 0:
            self.logger.info(f"[PADDING] Applying {padding_ms*1000:.0f}ms padding to {len(intervals)} phonemes")
            for i, (start, end) in enumerate(intervals[:min(3, len(intervals))]):
                orig_duration = (end - start) * 1000  # ms
                start_padded = max(time_offset, start - padding_ms)
                end_padded = min(segment_end, end + padding_ms)
                new_duration = (end_padded - start_padded) * 1000  # ms
                self.logger.info(f"  [{i}] {orig_duration:.1f}ms → {new_duration:.1f}ms (+{new_duration-orig_duration:.1f}ms)")

        for start, end in intervals:
            # 앞뒤로 패딩 추가 (경계 체크)
            start_padded = max(time_offset, start - padding_ms)
            end_padded = min(segment_end, end + padding_ms)

            padded.append((round(start_padded, 3), round(end_padded, 3)))

        return padded

    def _align_segment_pia_style(
        self,
        audio_segment: np.ndarray,
        phonemes: List[str],
        time_offset: float,
        segment_text: str
    ) -> List[Tuple[float, float]]:
        """
        PIA 스타일 3단계 정렬:
        1. 균등 분배 (초기 추정)
        2. WhisperX align()으로 한국어 텍스트 처리 (음절 레벨 타임스탬프)
        3. 음절 → 자모 분배 (문자 레벨 정제)

        Args:
            audio_segment: 세그먼트의 오디오 샘플
            phonemes: 정렬할 MFA 음소 리스트
            time_offset: 전체 오디오에서 세그먼트의 시작 시간
            segment_text: 이 세그먼트의 원본 한국어 텍스트

        Returns:
            각 음소의 (start, end) 튜플 리스트
        """
        segment_duration = len(audio_segment) / 16000

        # 음소 구간 확장 파라미터 (5프레임 확보를 위해)
        # 30fps 기준: 5프레임 = 166.7ms 필요
        # 앞뒤 각 80ms 확장 → 총 160ms 추가
        PHONEME_PADDING_MS = 0.080  # 80ms in seconds

        # 단계 1: 균등 분배 (초기 추정)
        dur = segment_duration / len(phonemes) if phonemes else 0
        uniform_intervals = []

        for i in range(len(phonemes)):
            start = time_offset + i * dur
            end = time_offset + (i + 1) * dur
            uniform_intervals.append((start, end))

        # 균등 분배에 패딩 적용
        uniform_intervals = self._apply_phoneme_padding(
            uniform_intervals, time_offset, segment_duration, PHONEME_PADDING_MS
        )

        # 단계 2-3: WhisperX align + 자모 분배 (모델이 있는 경우)
        if self.align_model is not None and self.align_metadata is not None:
            try:
                self.logger.debug("Attempting WhisperX syllable alignment + jamo distribution...")

                # 한국어 텍스트로 WhisperX align 호출 (MFA 음소가 아님!)
                transcript = [{
                    "start": 0.0,
                    "end": segment_duration,
                    "text": segment_text.strip()
                }]

                result = whisperx.align(
                    transcript,
                    self.align_model,
                    self.align_metadata,
                    audio_segment,
                    device=self.device,
                    return_char_alignments=True
                )

                # DEBUG: WhisperX align 결과 구조 출력
                self.logger.debug(f"WhisperX result keys: {result.keys()}")
                self.logger.debug(f"Number of segments: {len(result.get('segments', []))}")

                if result.get('segments'):
                    seg = result['segments'][0]
                    self.logger.debug(f"First segment keys: {seg.keys()}")
                    self.logger.debug(f"First segment text: {seg.get('text', 'N/A')}")
                    self.logger.debug(f"Number of words: {len(seg.get('words', []))}")

                    # 세그먼트 레벨 chars 확인
                    if 'chars' in seg:
                        self.logger.debug(f"Segment has chars! Count: {len(seg['chars'])}")
                        if seg['chars']:
                            self.logger.debug(f"First 3 chars: {seg['chars'][:3]}")
                    else:
                        self.logger.debug("No 'chars' key in segment!")

                    if seg.get('words'):
                        word = seg['words'][0]
                        self.logger.debug(f"First word keys: {word.keys()}")
                        self.logger.debug(f"First word: {word}")

                        # 단어 레벨에 chars가 있는지 확인
                        if 'chars' in word:
                            self.logger.debug(f"Word has chars! Count: {len(word['chars'])}")
                            if word['chars']:
                                self.logger.debug(f"First char: {word['chars'][0]}")
                        else:
                            self.logger.debug("No 'chars' key in word (chars are at segment level)")

                # whisperx_aligner 로직을 사용하여 자모 레벨 타임스탬프 추출
                jamo_intervals = self._extract_and_distribute_chars(result, time_offset)

                if jamo_intervals and len(jamo_intervals) >= len(phonemes):
                    self.logger.debug(f"WhisperX refinement successful: {len(jamo_intervals)} jamos extracted")
                    # 음소 개수와 일치하는 첫 N개 구간 반환 (패딩 적용)
                    selected_intervals = jamo_intervals[:len(phonemes)]
                    padded_intervals = self._apply_phoneme_padding(selected_intervals, time_offset, segment_duration)
                    return padded_intervals
                else:
                    self.logger.debug(f"WhisperX refinement incomplete ({len(jamo_intervals)} jamos vs {len(phonemes)} phonemes), using uniform distribution")
                    # uniform_intervals는 이미 패딩이 적용되어 있음
                    return uniform_intervals

            except Exception as e:
                self.logger.warning(f"WhisperX align failed: {e}, using uniform distribution")
                return uniform_intervals
        else:
            # align 모델이 없으면 균등 분배 사용
            self.logger.debug("No align model available, using uniform distribution")
            return uniform_intervals

    def _extract_and_distribute_chars(
        self,
        align_result: Dict,
        time_offset: float
    ) -> List[Tuple[float, float]]:
        """
        WhisperX align 결과에서 문자 레벨 타임스탬프를 추출하고
        음절 지속시간을 자모에 분배합니다.

        WhisperX는 SEGMENT 레벨에서 chars를 반환합니다 (단어 레벨이 아님!)
        구조: result['segments'][i]['chars'] = [{'char': '멋', 'start': 0.0, 'end': 0.1}, ...]

        Args:
            align_result: return_char_alignments=True로 호출한 WhisperX align() 출력
            time_offset: 모든 타임스탬프에 추가할 시작 시간 오프셋

        Returns:
            각 자모의 (start, end) 튜플 리스트
        """
        intervals = []

        # WhisperX 결과에서 문자 레벨 정렬 추출
        for segment in align_result.get("segments", []):
            # WhisperX는 SEGMENT 레벨에서 chars를 반환합니다 (단어 레벨이 아님!)
            char_alignments = segment.get("chars", [])

            if char_alignments:
                # 이전 문자의 캡핑된 끝점 추적 (타임스탬프 공백 방지)
                prev_char_end = 0.0

                # Wav2Vec2 문자 레벨 타임스탬프 사용
                for char_info in char_alignments:
                    char = char_info.get("char", "")
                    char_start = char_info.get("start", 0.0)
                    char_end = char_info.get("end", 0.0)

                    if not char.strip():
                        continue

                    # 이전 캡핑으로 인한 공백 제거: 이전 문자의 끝에서 바로 시작
                    adjusted_start = prev_char_end

                    # 단일 문자에 G2PK 발음 적용
                    try:
                        phoneme_str = self.g2p(char)
                    except Exception:
                        phoneme_str = char

                    # 원본 문자 지속시간 계산 (WhisperX 타이밍 기준)
                    orig_char_duration = char_end - char_start
                    max_char_duration = 0.5  # 문자당 최대 500ms

                    # 원본 길이를 캡핑 (adjusted_start와 무관하게 원본 길이 유지)
                    if orig_char_duration > max_char_duration:
                        self.logger.info(
                            f"Capped long char '{char}' from {orig_char_duration:.3f}s to {max_char_duration:.3f}s "
                            f"(original=[{char_start:.3f}, {char_end:.3f}], adjusted_start={adjusted_start:.3f})"
                        )
                        capped_char_duration = max_char_duration
                    else:
                        capped_char_duration = orig_char_duration

                    # 캡핑된 끝점 계산 (adjusted_start + 캡핑된 길이)
                    adjusted_end = adjusted_start + capped_char_duration

                    # 음절을 자모로 분해
                    for syllable in phoneme_str:
                        if syllable.strip():
                            # 한국어 음절인지 확인 (0xAC00-0xD7A3)
                            if 0xAC00 <= ord(syllable) <= 0xD7A3:
                                try:
                                    # 분해: h2j (한글 → 자모), j2hcj (호환 자모 → 개별 자모)
                                    jamo_chars = j2hcj(h2j(syllable))
                                    jamo_list = list(jamo_chars)

                                    # 캡핑된 지속시간을 자모에 분배
                                    jamo_duration = capped_char_duration / len(jamo_list) if jamo_list else 0

                                    for i, jamo in enumerate(jamo_list):
                                        # adjusted_start 기준으로 계산
                                        jamo_start = adjusted_start + i * jamo_duration
                                        jamo_end = adjusted_start + (i + 1) * jamo_duration

                                        # 시간 오프셋 추가 및 반올림
                                        intervals.append((
                                            round(time_offset + jamo_start, 3),
                                            round(time_offset + jamo_end, 3)
                                        ))
                                except Exception:
                                    # 자모 분해 실패, 음절을 그대로 사용
                                    intervals.append((
                                        round(time_offset + adjusted_start, 3),
                                        round(time_offset + adjusted_end, 3)
                                    ))
                            else:
                                # 한국어가 아닌 문자, 그대로 사용
                                intervals.append((
                                    round(time_offset + adjusted_start, 3),
                                    round(time_offset + adjusted_end, 3)
                                ))

                    # 다음 문자를 위해 캡핑된 끝점 업데이트
                    prev_char_end = adjusted_end

        return intervals

    def _uniform_distribution(
        self,
        phonemes: List[str],
        duration: float,
        time_offset: float
    ) -> List[Tuple[float, float]]:
        """
        폴백(Fallback): 지속시간에 걸쳐 음소를 균등하게 분배

        Args:
            phonemes: 음소 리스트
            duration: 세그먼트 지속시간 (초 단위)
            time_offset: 시작 시간 오프셋

        Returns:
            (start, end) 튜플 리스트
        """
        if not phonemes:
            return []

        phoneme_duration = duration / len(phonemes)
        intervals = []

        for i in range(len(phonemes)):
            start = time_offset + i * phoneme_duration
            end = time_offset + (i + 1) * phoneme_duration
            intervals.append((round(start, 3), round(end, 3)))

        return intervals

    def _text_to_phonemes(self, text: str) -> List[str]:
        """
        G2PK + 자모를 사용하여 한국어 텍스트를 MFA 음소 시퀀스로 변환

        Args:
            text: 한국어 텍스트

        Returns:
            MFA 음소 기호 리스트
        """
        # G2PK 발음 규칙 적용
        try:
            pronounced_text = self.g2p(text)
        except Exception as e:
            self.logger.warning(f"G2PK failed: {e}, using original text")
            pronounced_text = text

        # 먼저 자모로 분해
        jamo_list = []

        for char in pronounced_text:
            if not char.strip():
                continue

            # 한국어 음절 (0xAC00-0xD7A3)
            if 0xAC00 <= ord(char) <= 0xD7A3:
                try:
                    jamo_chars = j2hcj(h2j(char))
                    jamo_list.extend(list(jamo_chars))
                except Exception:
                    # 분해 실패 시 건너뛰기
                    pass
            # 이미 자모이거나 다른 문자 - 건너뛰기

        # 자모를 MFA 음소로 변환
        mfa_phonemes = jamo_to_mfa(jamo_list)

        return mfa_phonemes

    def _extract_audio(
        self,
        video_path: Path,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> Path:
        """
        ffmpeg를 사용하여 비디오에서 오디오 추출

        Args:
            video_path: 비디오 파일 경로
            start_time: 시작 시간 (초, 기본값: 0.0)
            end_time: 종료 시간 (초, 기본값: None - 전체 비디오)

        Returns:
            추출된 오디오 파일 경로 (WAV, 16kHz)
        """
        import subprocess
        import tempfile

        # 임시 오디오 파일 생성
        audio_path = Path(tempfile.mktemp(suffix=".wav"))

        # ffmpeg로 오디오 추출 (시간 범위 지원)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),  # 시작 시간
            "-i", str(video_path),
        ]

        # 종료 시간 또는 전체 비디오
        if end_time is not None:
            cmd.extend(["-to", str(end_time)])  # 절대 종료 시간

        cmd.extend([
            "-vn",  # 비디오 없음
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16kHz 샘플레이트
            "-ac", "1",  # 모노
            str(audio_path)
        ])

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        return audio_path

    def _empty_result(self) -> Dict:
        """빈 결과 구조 반환"""
        return {
            'transcription': '',
            'phonemes': [],
            'intervals': [],
            'duration': 0.0,
            'method': 'pia_style_whisperx_align'
        }