import json
import time
import os
import uuid
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
import logging

from .unified_feature_extractor import UnifiedFeatureExtractor
from .stage1_scanner import Stage1Scanner
from .stage2_analyzer import Stage2Analyzer
from .result_aggregator import ResultAggregator
from .hybrid_utils import convert_for_json
from .pia_visualizer import PIAVisualizer
from ..utils.feature_cache import FeatureCache
from ..utils.jamo_to_mfa import mfa_to_korean
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridXAIPipeline:
    """
    Unified Hybrid XAI Pipeline (MMMS-BA + PIA)
    
    Optimized Architecture:
    1. Unified Feature Extraction (30fps, Single Pass)
    2. Parallel Prediction (MMMS-BA & PIA)
    3. Ensemble Decision
    """

    def __init__(
        self,
        mmms_model_path: str,
        pia_model_path: str,
        mmms_config_path: str = "configs/train_teacher_korean.yaml",
        pia_config_path: str = "configs/train_pia.yaml",
        device: str = "cuda",
        use_cache: bool = True
    ):
        logger.info("=" * 80)
        logger.info("Initializing Unified Hybrid Pipeline...")
        logger.info("=" * 80)

        self.device = device
        self.mmms_model_path = mmms_model_path
        self.pia_model_path = pia_model_path
        self.use_cache = use_cache

        # 1. Unified Feature Extractor
        self.feature_extractor = UnifiedFeatureExtractor(device=device)

        # 1.5 Feature Cache (optional)
        self.feature_cache = FeatureCache() if use_cache else None
        if use_cache:
            logger.info(f"Feature cache enabled: {self.feature_cache.cache_dir}")

        # 2. Models
        self.stage1 = Stage1Scanner(
            model_path=mmms_model_path,
            config_path=mmms_config_path,
            device=device
        )

        self.stage2 = Stage2Analyzer(
            pia_model_path=pia_model_path,
            pia_config_path=pia_config_path,
            device=device
        )

        # 3. Aggregator
        self.aggregator = ResultAggregator()

        logger.info("Unified Hybrid Pipeline Initialized.")

    def _download_from_url(self, url: str, video_id: str = None) -> str:
        """
        YouTube/TikTok/Reels URL에서 비디오 다운로드.

        Args:
            url: 비디오 URL (YouTube Shorts, TikTok, Instagram Reels 등)
            video_id: 저장할 파일명 (None이면 자동 생성)

        Returns:
            다운로드된 비디오 파일 경로
        """
        try:
            import yt_dlp
        except ImportError:
            raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

        if video_id is None:
            video_id = f"video_{uuid.uuid4().hex[:8]}"

        # 다운로드 디렉토리 설정
        download_dir = Path(__file__).parent.parent.parent / "cache" / "downloads"
        download_dir.mkdir(parents=True, exist_ok=True)

        output_template = str(download_dir / f"{video_id}.%(ext)s")

        logger.info(f"Downloading video from URL: {url[:50]}...")

        ydl_opts = {
            'format': 'best[ext=mp4]/bestvideo+bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4',
            'socket_timeout': 30,
            'retries': 3,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # 다운로드된 파일 찾기
            downloaded_files = list(download_dir.glob(f"{video_id}.*"))
            if downloaded_files:
                video_path = str(downloaded_files[0])
                logger.info(f"Downloaded: {Path(video_path).name}")
                return video_path
            else:
                raise RuntimeError(f"Download completed but file not found: {video_id}")

        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise RuntimeError(f"Failed to download video from URL: {e}")

    def process_video(
        self,
        video_path: str,
        video_id: str = None,
        output_dir: str = None,
        threshold: float = 0.6, # Unused but kept for API compatibility
        min_interval_frames: int = 14, # Unused
        save_visualizations: bool = True,
        use_preprocessed: bool = None # Unused
    ) -> Dict[str, Any]:

        start_time = time.time()

        # URL 감지 및 다운로드
        cleanup_downloaded_file = False
        if video_path.startswith(('http://', 'https://')):
            logger.info(f"URL detected: {video_path[:60]}...")
            local_video_path = self._download_from_url(video_path, video_id)
            cleanup_downloaded_file = True
        else:
            local_video_path = video_path

        if video_id is None:
            video_id = Path(local_video_path).stem

        if output_dir is None:
            output_dir = f"outputs/xai/hybrid/{video_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing: {local_video_path}")

        # STEP 1: Unified Feature Extraction (with cache)
        logger.info("STEP 1: Unified Feature Extraction...")
        try:
            if self.feature_cache is not None:
                features = self.feature_cache.get_or_extract(local_video_path, self.feature_extractor)
            else:
                features = self.feature_extractor.extract_all(local_video_path)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._error_result("Feature Extraction Failed", str(e))

        # Post-process: Re-compute phoneme_labels_30fps using CENTER-BASED SELECTION
        # This ensures cached data uses the correct algorithm even if cache was created
        # with the old last-write-wins logic.
        features = self._recompute_phoneme_labels_center_based(features)

        # STEP 2: MMMS-BA Prediction
        logger.info("STEP 2: MMMS-BA Prediction...")
        try:
            mmms_result = self.stage1.predict_full_video(features)
        except Exception as e:
            logger.error(f"MMMS-BA prediction failed: {e}")
            mmms_result = {'verdict': 'unknown', 'confidence': 0.0, 'probabilities': {'fake': 0.0, 'real': 0.0}}

        # STEP 3: Find suspicious intervals
        logger.info("STEP 3: Finding suspicious intervals...")
        try:
            suspicious_intervals = self.stage1.find_suspicious_intervals(
                frame_probs=mmms_result.get('frame_probs'),
                timestamps=features.get('timestamps_30fps')[::3],  # 10fps timestamps
                min_candidates=5,
                max_candidates=10
            )
        except Exception as e:
            logger.error(f"Finding suspicious intervals failed: {e}")
            suspicious_intervals = []

        # STEP 3.5: Filter intervals by phoneme coverage (Pre-filtering)
        # [MODIFIED] 모든 의심 구간 보존, phoneme_valid 플래그로 구분
        logger.info("STEP 3.5: Filtering intervals by phoneme coverage...")
        all_suspicious_intervals = []  # 시각화용: 모든 구간 (음소 유무 무관)

        if len(suspicious_intervals) > 0:
            from ..utils.korean_phoneme_config import KEEP_PHONEMES_KOREAN

            # Get global phoneme set (14 fixed phonemes from training)
            global_phoneme_set = KEEP_PHONEMES_KOREAN

            # Get 30fps features for coverage calculation
            timestamps_30fps = features.get('timestamps_30fps')
            phoneme_labels_30fps = features.get('phoneme_labels_30fps')

            if timestamps_30fps is not None and phoneme_labels_30fps is not None:
                valid_intervals = []
                skipped_count = 0

                for interval in suspicious_intervals:
                    # Extract interval phonemes (30fps)
                    mask = (timestamps_30fps >= interval['start']) & (timestamps_30fps <= interval['end'])
                    interval_phoneme_labels = phoneme_labels_30fps[mask]

                    # Remove padding and get unique phonemes
                    interval_phonemes = set(interval_phoneme_labels)
                    interval_phonemes.discard('<pad>')
                    interval_phonemes.discard('')

                    # Calculate coverage (intersection with global set)
                    coverage = len(interval_phonemes & global_phoneme_set)

                    # [NEW] 모든 구간에 phoneme_valid 플래그 추가
                    COVERAGE_THRESHOLD = 3
                    interval_with_flag = interval.copy()
                    interval_with_flag['phoneme_coverage'] = coverage
                    interval_with_flag['phoneme_valid'] = coverage >= COVERAGE_THRESHOLD
                    all_suspicious_intervals.append(interval_with_flag)

                    # Filter by threshold (3 phonemes, relaxed for short intervals)
                    if coverage >= COVERAGE_THRESHOLD:
                        valid_intervals.append(interval)
                        logger.info(
                            f"  ✓ {interval['interval_id']}: {coverage}/{len(global_phoneme_set)} phonemes matched "
                            f"({interval['start']:.1f}~{interval['end']:.1f}s, dur={interval['duration']:.1f}s)"
                        )
                    else:
                        skipped_count += 1
                        logger.warning(
                            f"  ✗ {interval['interval_id']}: {coverage}/{len(global_phoneme_set)} phonemes, SKIPPED "
                            f"(below threshold {COVERAGE_THRESHOLD})"
                        )

                # Update suspicious_intervals with filtered results (PIA용)
                if len(valid_intervals) == 0:
                    logger.warning(
                        f"All {len(suspicious_intervals)} intervals filtered out due to low phoneme coverage. "
                        f"Will analyze full video instead."
                    )
                    suspicious_intervals = []  # Trigger full video fallback
                else:
                    logger.info(
                        f"Phoneme coverage filtering: {len(valid_intervals)}/{len(suspicious_intervals)} intervals passed "
                        f"({skipped_count} skipped)"
                    )
                    suspicious_intervals = valid_intervals
            else:
                logger.warning("Phoneme labels not available, skipping coverage filtering")
                # 음소 정보 없으면 모든 구간을 valid로 표시
                for interval in suspicious_intervals:
                    interval_with_flag = interval.copy()
                    interval_with_flag['phoneme_valid'] = True
                    interval_with_flag['phoneme_coverage'] = -1  # Unknown
                    all_suspicious_intervals.append(interval_with_flag)

        # all_suspicious_intervals가 비어있으면 원본 사용
        if not all_suspicious_intervals and suspicious_intervals:
            for interval in suspicious_intervals:
                interval_with_flag = interval.copy()
                interval_with_flag['phoneme_valid'] = True
                interval_with_flag['phoneme_coverage'] = -1
                all_suspicious_intervals.append(interval_with_flag)

        # STEP 4: PIA Full Video Analysis (verdict + XAI 동시 획득)
        # [OPTIMIZED] 기존: interval 분석 → full 분석 (2번 실행)
        #             변경: full 분석 1회로 통합 (~15초 절감)
        logger.info("STEP 4: PIA analysis (full video)...")
        try:
            pia_result = self.stage2.predict_full_video(features)

            # suspicious_intervals 정보를 pia_result에 첨부 (시각화용)
            if len(suspicious_intervals) > 0:
                best_interval = suspicious_intervals[0]  # 가장 의심스러운 구간
                pia_result['best_interval'] = {
                    'start': best_interval['start'],
                    'end': best_interval['end'],
                    'mmms_confidence': best_interval.get('confidence', 0)
                }
                pia_result['all_intervals'] = suspicious_intervals
                logger.info(f"  Best suspicious interval: {best_interval['start']:.1f}~{best_interval['end']:.1f}s")

        except Exception as e:
            logger.error(f"PIA prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            pia_result = {'verdict': 'unknown', 'confidence': 0.0, 'probabilities': {'fake': 0.0, 'real': 0.0}}

        # STEP 5: Ensemble & Summary
        logger.info("STEP 5: Ensemble & Summary...")
        detection = self.aggregator.ensemble(mmms_result, pia_result)

        video_info = self.aggregator.extract_video_info(local_video_path)
        summary = self.aggregator.generate_korean_summary(detection, pia_result, video_info)
        
        # Visualizations
        stage1_viz_path = None
        pia_viz_path = None
        interval_viz_paths: List[str] = []
        
        if save_visualizations:
            # [NEW] MMMS-BA Timeline 시각화 (모든 의심 구간 표시)
            try:
                stage1_viz_path = self._visualize_stage1_timeline(
                    mmms_result=mmms_result,
                    suspicious_intervals=suspicious_intervals,
                    all_suspicious_intervals=all_suspicious_intervals,  # 음소 필터링 전 모든 구간
                    features=features,
                    video_path=local_video_path,
                    output_dir=output_dir
                )
                logger.info(f"Generated Stage1 timeline: {stage1_viz_path}")
            except Exception as e:
                logger.warning(f"Stage1 timeline visualization failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # [NEW] PIA XAI 시각화 (interval 분석 결과가 있는 경우)
            try:
                if 'all_intervals' in pia_result and len(pia_result['all_intervals']) > 0:
                    # Interval별 XAI 결과가 있는 경우
                    for idx, interval_result in enumerate(pia_result['all_intervals'][:3]):  # 최대 3개
                        # XAI 정보가 있는지 확인 (predict_interval은 XAI 정보 없음)
                        # predict_full_video 결과만 XAI 정보 포함
                        pass  # TODO: predict_interval 결과에 XAI 정보 추가 필요
                
                # [NEW] Ensure all_phonemes is available for visualization (extract from features)
                if 'all_phonemes' not in pia_result or not pia_result.get('all_phonemes'):
                    phoneme_labels = features.get('phoneme_labels_30fps')
                    if phoneme_labels is not None:
                        seen = set()
                        all_phonemes_list = []
                        skip_tokens = {'<pad>', '<PAD>', '<unk>', '<UNK>', '', 'sil', 'sp', 'spn'}
                        for p in phoneme_labels:
                            if p not in seen and p not in skip_tokens:
                                all_phonemes_list.append(p)
                                seen.add(p)
                        pia_result['all_phonemes'] = all_phonemes_list
                        logger.info(f"[Transcription] Extracted {len(all_phonemes_list)} unique phonemes from features")

                # Full video PIA 결과에 XAI 정보가 있는 경우
                if 'geometry_analysis' in pia_result or 'phoneme_analysis' in pia_result:
                    # [FIX] 의심 구간 한글 transcription 추출 (단어 레벨 필터링)
                    korean_transcription = None
                    whisper_result = features.get('whisper_result', {})

                    # 1순위: best_interval (PIA 결과에서)
                    if 'best_interval' in pia_result:
                        interval = pia_result['best_interval']
                        korean_transcription = self._get_transcription_for_interval(
                            whisper_result,
                            interval.get('start'),
                            interval.get('end')
                        )
                    # 2순위: suspicious_intervals의 첫 번째 구간
                    elif suspicious_intervals and len(suspicious_intervals) > 0:
                        first_interval = suspicious_intervals[0]
                        korean_transcription = self._get_transcription_for_interval(
                            whisper_result,
                            first_interval.get('start'),
                            first_interval.get('end')
                        )
                        logger.info(f"[Transcription] Using first suspicious interval: {first_interval.get('start'):.1f}s - {first_interval.get('end'):.1f}s")
                    # 3순위: 전체 transcription
                    else:
                        korean_transcription = self._get_transcription_for_interval(whisper_result)

                    pia_viz_path = self._visualize_pia_result(
                        pia_result=pia_result,
                        video_path=local_video_path,
                        output_dir=output_dir,
                        final_detection=detection,  # 앙상블 결과 전달
                        korean_transcription=korean_transcription,  # 한글 전사 전달
                        features=features,  # [NEW] 30fps MAR 데이터용
                        suspicious_intervals=suspicious_intervals  # [NEW] 의심 구간 정보
                    )
                    logger.info(f"Generated PIA XAI visualization: {pia_viz_path}")
            except Exception as e:
                logger.warning(f"PIA visualization generation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        processing_time_ms = (time.time() - start_time) * 1000
        
        # Final Result
        result = self.aggregator.build_final_result(
            video_path=local_video_path,
            video_id=video_id,
            detection=detection,
            summary=summary,
            video_info=video_info,
            processing_time_ms=processing_time_ms,
            suspicious_intervals=suspicious_intervals
        )
        
        # Save JSON
        json_path = Path(output_dir) / "result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Use converter to handle numpy types
            safe_result = convert_for_json(result)
            json.dump(safe_result, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\n[Pipeline Complete]")
        logger.info(f"  Processing time: {processing_time_ms/1000:.1f}s")
        logger.info(f"  Result saved to: {json_path}")
        logger.info(f"Verdict: {detection['verdict'].upper()} ({detection['confidence']:.1%})")
        logger.info("=" * 80 + "\n")

        # 다운로드한 임시 파일 정리
        if cleanup_downloaded_file and Path(local_video_path).exists():
            try:
                os.remove(local_video_path)
                logger.info(f"Cleaned up downloaded file: {local_video_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup downloaded file: {e}")

        return result

    def _recompute_phoneme_labels_center_based(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-compute phoneme_labels_30fps from phoneme_intervals using CENTER-BASED SELECTION.

        This ensures cached data uses the correct algorithm even if the cache was created
        with the old last-write-wins logic.

        Args:
            features: Feature dictionary with phoneme_intervals and timestamps_30fps

        Returns:
            Updated features with recomputed phoneme_labels_30fps
        """
        phoneme_intervals = features.get('phoneme_intervals', [])
        timestamps_30fps = features.get('timestamps_30fps')

        if timestamps_30fps is None or len(phoneme_intervals) == 0:
            return features

        # Pre-compute interval data with centers
        interval_data = []
        for ph_dict in phoneme_intervals:
            start = float(ph_dict.get('start', 0.0))
            end = float(ph_dict.get('end', 0.0))
            phoneme = str(ph_dict.get('phoneme', ''))
            center = (start + end) / 2.0
            interval_data.append({
                'start': start,
                'end': end,
                'phoneme': phoneme,
                'center': center
            })

        # Re-compute labels using center-based selection
        phoneme_labels_str = np.full(len(timestamps_30fps), '', dtype='<U10')

        for idx, ts in enumerate(timestamps_30fps):
            matching = []
            for iv in interval_data:
                if iv['start'] <= ts <= iv['end']:
                    distance = abs(ts - iv['center'])
                    matching.append((iv['phoneme'], distance))

            if matching:
                best_phoneme = min(matching, key=lambda x: x[1])[0]
                phoneme_labels_str[idx] = best_phoneme

        # Update features with recomputed labels
        features['phoneme_labels_30fps'] = phoneme_labels_str

        # Log unique phonemes for debugging
        unique_phonemes = sorted(set(p for p in phoneme_labels_str if p))
        logger.info(f"  [CENTER-BASED] Recomputed phoneme labels: {len(unique_phonemes)} unique phonemes")

        return features

    def _error_result(self, error_title, error_msg):
        return {
            'metadata': {'error': True},
            'detection': {'verdict': 'error', 'confidence': 0.0},
            'summary': {'title': error_title, 'detailed_explanation': error_msg}
        }

    def _get_transcription_for_interval(
        self,
        whisper_result: Dict[str, Any],
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None
    ) -> str:
        """
        의심 구간에 해당하는 한글 단어만 추출 (단어 레벨 필터링).

        Args:
            whisper_result: WhisperX 결과 (segments, transcription 포함)
            start_sec: 시작 시간 (None이면 전체)
            end_sec: 종료 시간 (None이면 전체)

        Returns:
            필터링된 한글 텍스트
        """
        segments = whisper_result.get('segments', [])

        # Helper: 세그먼트 텍스트에서 전체 transcription 추출
        def get_full_transcription():
            # 1. transcription 필드 확인
            trans = whisper_result.get('transcription', '')
            if trans:
                return trans.strip()
            # 2. segments의 text 필드 결합
            segment_texts = [seg.get('text', '') for seg in segments if seg.get('text')]
            return ' '.join(segment_texts).strip()

        # Fallback: 의심 구간이 없으면 전체 transcription 반환
        if start_sec is None or end_sec is None:
            return get_full_transcription()

        # 1차 시도: 단어 레벨 필터링
        interval_words = []
        for seg in segments:
            words = seg.get('words', [])
            for word_info in words:
                word_start = word_info.get('start', 0)
                word_end = word_info.get('end', 0)

                # 단어가 의심 구간과 겹치는지 확인
                if word_end >= start_sec and word_start <= end_sec:
                    interval_words.append(word_info.get('word', ''))

        if interval_words:
            return ' '.join(interval_words).strip()

        # 2차 시도: 세그먼트 레벨 필터링 (단어 타임스탬프가 없는 경우)
        interval_segments = []
        for seg in segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            # 세그먼트가 의심 구간과 겹치는지 확인
            if seg_end >= start_sec and seg_start <= end_sec:
                seg_text = seg.get('text', '')
                if seg_text:
                    interval_segments.append(seg_text.strip())

        if interval_segments:
            return ' '.join(interval_segments).strip()

        # 최종 fallback: 전체 transcription
        return get_full_transcription()

    def _aggregate_interval_results(self, interval_results: List[Dict]) -> Dict[str, Any]:
        """
        Multi-factor Weighted Voting (Epsilon 바닥값 추가).

        Factors:
        - Confidence: 모델 신뢰도
        - Duration: 구간 길이 (정규화: / 3.0초)
        - Frame count: 유효 프레임 수 (정규화: / 60 frames)
        - Epsilon: 0 방지 바닥값

        Args:
            interval_results: List of interval prediction results

        Returns:
            Aggregated result with best interval info
        """
        EPSILON = 1e-6  # [보강] 0 방지

        weights = []
        for r in interval_results:
            conf_w = max(r['confidence'], EPSILON)
            dur_w = max(min(r['interval']['duration'] / 3.0, 1.0), EPSILON)
            frame_w = max(min(r['valid_frame_count'] / 60, 1.0), EPSILON)

            combined = conf_w * dur_w * frame_w
            weights.append(max(combined, EPSILON))

        total_weight = sum(weights)

        # [보강] Total weight = 0 극단적 경우
        if total_weight < EPSILON:
            logger.error("Total weight near zero, using uniform weights")
            weights = [1.0 / len(interval_results)] * len(interval_results)
            total_weight = 1.0

        weighted_fake = sum(
            r['probabilities']['fake'] * w
            for r, w in zip(interval_results, weights)
        ) / total_weight

        verdict = 'fake' if weighted_fake > 0.5 else 'real'
        best_idx = np.argmax(weights)
        best_interval = interval_results[best_idx]

        logger.info(f"  Aggregated {len(interval_results)} intervals: "
                   f"{verdict.upper()} ({max(weighted_fake, 1-weighted_fake):.2%})")

        return {
            'verdict': verdict,
            'confidence': max(weighted_fake, 1 - weighted_fake),
            'probabilities': {'real': 1 - weighted_fake, 'fake': weighted_fake},
            'best_interval': best_interval['interval'],
            'matched_phonemes': best_interval.get('matched_phonemes', []),
            'num_intervals_analyzed': len(interval_results),
            'all_intervals': interval_results
        }
    
    def _visualize_stage1_timeline(
        self,
        mmms_result: Dict[str, Any],
        suspicious_intervals: List[Dict],
        features: Dict[str, Any],
        video_path: str,
        output_dir: str,
        all_suspicious_intervals: List[Dict] = None
    ) -> str:
        """
        MMMS-BA Timeline 시각화 생성.

        Args:
            mmms_result: Stage1 예측 결과
            suspicious_intervals: 음소 필터링된 의심 구간 리스트 (PIA 분석용)
            features: 추출된 특징 (frames_10fps 포함)
            video_path: 비디오 경로
            output_dir: 출력 디렉토리
            all_suspicious_intervals: 필터링 전 모든 의심 구간 (phoneme_valid 플래그 포함)

        Returns:
            시각화 파일 경로
        """
        video_name = Path(video_path).stem
        output_path = Path(output_dir) / f"{video_name}_stage1_timeline.png"

        # 데이터 추출
        frame_probs = mmms_result.get('frame_probs', [])
        if len(frame_probs) == 0:
            logger.warning("No frame probabilities for timeline visualization")
            return str(output_path)

        timestamps_10fps = features.get('timestamps_30fps', [])[::3]  # 10fps
        if len(timestamps_10fps) != len(frame_probs):
            # 길이 맞추기
            min_len = min(len(timestamps_10fps), len(frame_probs))
            timestamps_10fps = timestamps_10fps[:min_len]
            frame_probs = frame_probs[:min_len]

        timestamps = np.array(timestamps_10fps)
        fake_probs = np.array(frame_probs)
        threshold = mmms_result.get('threshold')
        if threshold is None:
            threshold = max(float(np.mean(fake_probs)) * 0.8, 0.5)

        # 프레임 가져오기 (10fps)
        frames_10fps = features.get('frames_10fps', [])
        if len(frames_10fps) == 0:
            logger.warning("No frames for timeline visualization")
            return str(output_path)

        # 시각화 생성
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1.5, 0.5], hspace=0.3)

        # Row 1: Sample Frames
        ax_frames = fig.add_subplot(gs[0])
        num_sample = 6
        frame_indices = np.linspace(0, len(frames_10fps)-1, num_sample, dtype=int)

        montage_frames = []
        for idx in frame_indices:
            frame = frames_10fps[idx]  # (224, 224, 3), [0, 1]
            frame_uint8 = (frame * 255).astype(np.uint8)
            # 프레임 번호 추가
            cv2.putText(frame_uint8, f"#{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_uint8, f"{timestamps[idx]:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            prob = fake_probs[idx]
            color = (255, 0, 0) if prob > threshold else (0, 255, 0)
            cv2.putText(frame_uint8, f"P(Fake)={prob:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            montage_frames.append(frame_uint8)

        montage = np.concatenate(montage_frames, axis=1)
        ax_frames.imshow(montage)
        ax_frames.axis('off')
        ax_frames.set_title(f"Video: {video_name} | Duration: {timestamps[-1]:.2f}s ({len(frames_10fps)} frames)",
                           fontsize=14, fontweight='bold')

        # Row 2: Temporal Probability Graph
        ax_graph = fig.add_subplot(gs[1])
        ax_graph.plot(timestamps, fake_probs * 100, 'b-', linewidth=2, label='Fake Probability')
        ax_graph.axhline(y=threshold * 100, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold*100:.0f}%)')

        # [MODIFIED] 의심 구간 하이라이트 (모든 구간 표시, 음소 유무에 따라 색상 구분)
        # all_suspicious_intervals가 있으면 사용, 없으면 suspicious_intervals 사용
        intervals_to_display = all_suspicious_intervals if all_suspicious_intervals else suspicious_intervals

        valid_shown = False
        invalid_shown = False
        for interval in intervals_to_display:
            start = interval['start']
            end = interval['end']
            is_valid = interval.get('phoneme_valid', True)  # 플래그 없으면 valid로 간주

            if is_valid:
                # 빨강: 음소 충분 (PIA 분석 대상)
                color = 'red'
                alpha = 0.3
                label = 'Suspicious (PIA)' if not valid_shown else ''
                valid_shown = True
            else:
                # 주황: 음소 부족 (PIA 분석 제외)
                color = 'orange'
                alpha = 0.2
                label = 'Suspicious (No Phoneme)' if not invalid_shown else ''
                invalid_shown = True

            ax_graph.axvspan(start, end, alpha=alpha, color=color, label=label)

        ax_graph.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax_graph.set_ylabel('Fake Probability (%)', fontsize=12, fontweight='bold')
        ax_graph.set_title(f"MMMS-BA Frame-Level Fake Probability | Video: {video_name}",
                          fontsize=14, fontweight='bold')
        ax_graph.set_ylim(0, 100)
        ax_graph.grid(True, alpha=0.3)
        ax_graph.legend(loc='upper right')

        # Row 3: Statistics
        ax_stats = fig.add_subplot(gs[2])
        ax_stats.axis('off')

        # [MODIFIED] 통계에 음소 커버리지 정보 추가
        total_intervals = len(intervals_to_display) if intervals_to_display else 0
        valid_intervals = len([i for i in intervals_to_display if i.get('phoneme_valid', True)]) if intervals_to_display else 0

        stats_text = (
            f"Suspicious Intervals: {total_intervals} (PIA: {valid_intervals}, No Phoneme: {total_intervals - valid_intervals}) | "
            f"Mean P(Fake): {np.mean(fake_probs)*100:.1f}% | "
            f"Max P(Fake): {np.max(fake_probs)*100:.1f}% | "
            f"Suspicious Frames: {np.sum(fake_probs > threshold)}/{len(fake_probs)}"
        )
        ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_path)
    
    def _visualize_pia_result(
        self,
        pia_result: Dict[str, Any],
        video_path: str,
        output_dir: str,
        final_detection: Optional[Dict[str, Any]] = None,
        korean_transcription: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
        suspicious_intervals: Optional[List[Dict]] = None
    ) -> str:
        """
        PIA XAI 결과 시각화 생성.

        Args:
            pia_result: PIA XAI 결과 (predict_full_video 결과)
            video_path: 비디오 경로
            output_dir: 출력 디렉토리
            final_detection: 최종 앙상블 결과 (Detection Summary용)
            korean_transcription: WhisperX 한글 전사 결과 (단어 레벨 필터링됨)
            features: 30fps 전체 특징 데이터 (mar_30fps, phoneme_labels_30fps 등)
            suspicious_intervals: 의심 구간 리스트

        Returns:
            시각화 파일 경로
        """
        video_name = Path(video_path).stem
        output_path = Path(output_dir) / f"{video_name}_pia_xai.png"

        visualizer = PIAVisualizer(dpi=150)

        # [FIX 1] 한글 transcription 사용 (단어 레벨 필터링된 결과)
        transcription = korean_transcription or ""

        # Get time range from interval info
        time_range = ""
        if 'best_interval' in pia_result:
            interval = pia_result['best_interval']
            start = interval.get('start', 0)
            end = interval.get('end', 0)
            time_range = f"[{start:.1f}s - {end:.1f}s]"

        # 2x2 패널 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 제목 위치 명시적 지정 (y=0.99, 최상단)
        fig.suptitle(f"PIA XAI Analysis | Video: {video_name}", fontsize=14, fontweight='bold', y=0.99)

        # Transcription 자막 위치 조정 (제목 아래 여유 확보)
        if transcription:
            subtitle = f"의심 구간 {time_range} 음성: '{transcription}'" if time_range else f"음성: '{transcription}'"
            fig.text(0.5, 0.95, subtitle, ha='center', fontsize=11, style='italic', color='#333333')

        # Panel 1: Geometry Analysis (30fps MAR 데이터 사용)
        # [NEW] features가 있으면 30fps MAR 기반 시각화, 없으면 기존 방식
        if features is not None and 'mar_30fps' in features:
            visualizer.plot_geometry_analysis_30fps(
                features=features,
                suspicious_intervals=suspicious_intervals,
                geometry_analysis=pia_result.get('geometry_analysis', {}),
                ax=axes[0, 0],
                title="MAR Deviation Analysis"
            )
        elif 'geometry_analysis' in pia_result:
            visualizer.plot_geometry_analysis(pia_result, ax=axes[0, 0], title="MAR Deviation Analysis")

        # Panel 2: Phoneme Attention (의심 구간 기반)
        # [NEW] features가 있으면 의심 구간 기반 어텐션 시각화
        if features is not None and 'phoneme_labels_30fps' in features:
            visualizer.plot_phoneme_attention_30fps(
                features=features,
                suspicious_intervals=suspicious_intervals,
                phoneme_analysis=pia_result.get('phoneme_analysis', {}),
                ax=axes[0, 1],
                title="Phoneme Attention"
            )
        elif 'phoneme_analysis' in pia_result:
            visualizer.plot_phoneme_attention(pia_result, ax=axes[0, 1], title="Phoneme Attention")

        # Panel 3: Branch Contributions
        if 'model_info' in pia_result and 'branch_contributions' in pia_result['model_info']:
            visualizer.plot_branch_contributions(pia_result, ax=axes[1, 0], title="Branch Contributions")

        # [FIX 2] Panel 4: Detection Summary - 최종 앙상블 결과 사용
        # final_detection이 있으면 앙상블 결과를, 없으면 PIA 결과를 표시
        # Note: plot_detection_summary()는 result['detection']으로 접근하므로 래핑 필요
        if final_detection:
            detection_result = {
                'detection': final_detection,
                'video_info': pia_result.get('video_info', {}),
                'summary': {}  # Detection Summary 패널에서는 사용 안함
            }
        else:
            detection_result = pia_result
        visualizer.plot_detection_summary(detection_result, ax=axes[1, 1], title="Detection Summary")

        # tight_layout에 상단 여백 확보 (제목 + 자막 공간)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_path)


def main():
    """Example usage of the hybrid pipeline for inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Unified Hybrid MMMS-BA + PIA XAI Pipeline")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--mmms-model', type=str,
                        default='models/checkpoints/mmms-ba_fulldata_best.pth',
                        help='MMMS-BA model checkpoint')
    parser.add_argument('--pia-model', type=str,
                        default='models/checkpoints/pia-best.pth',
                        help='PIA model checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/hybrid_xai', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = HybridXAIPipeline(
        mmms_model_path=args.mmms_model,
        pia_model_path=args.pia_model,
        device=args.device
    )

    # Run pipeline on video
    result = pipeline.process_video(
        video_path=args.video,
        output_dir=args.output_dir
    )

    print(f"\nPipeline complete! Results saved to {args.output_dir}")
    print(f"Overall verdict: {result['detection']['verdict'].upper()}")
    print(f"Confidence: {result['detection']['confidence']:.2%}")


if __name__ == "__main__":
    main()
