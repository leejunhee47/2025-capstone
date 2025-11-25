import json
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
import logging

from .unified_feature_extractor import UnifiedFeatureExtractor
from .stage1_scanner import Stage1Scanner
from .stage2_analyzer import Stage2Analyzer
from .result_aggregator import ResultAggregator
from .hybrid_utils import convert_for_json
from .thumbnail_generator import generate_detection_card
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
        
        if video_id is None:
            video_id = Path(video_path).stem
            
        if output_dir is None:
            output_dir = f"outputs/xai/hybrid/{video_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing: {video_path}")

        # STEP 1: Unified Feature Extraction (with cache)
        logger.info("STEP 1: Unified Feature Extraction...")
        try:
            if self.feature_cache is not None:
                features = self.feature_cache.get_or_extract(video_path, self.feature_extractor)
            else:
                features = self.feature_extractor.extract_all(video_path)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._error_result("Feature Extraction Failed", str(e))
            
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
        logger.info("STEP 3.5: Filtering intervals by phoneme coverage...")
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

                    # Filter by threshold (3 phonemes, relaxed for short intervals)
                    COVERAGE_THRESHOLD = 3
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

                # Update suspicious_intervals with filtered results
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

        # STEP 4: PIA on intervals
        logger.info("STEP 4: PIA analysis on suspicious intervals...")
        try:
            if len(suspicious_intervals) == 0:
                # Fallback Tier 2: Full video
                logger.warning("No suspicious intervals found, analyzing full video")
                pia_result = self.stage2.predict_full_video(features)
            else:
                interval_results = []
                unknown_count = 0

                # Max 5개 + Early stop with reset
                for i, interval in enumerate(suspicious_intervals[:5]):
                    logger.info(f"Analyzing interval {i+1}/{min(5, len(suspicious_intervals))}: "
                               f"{interval['start']:.1f}~{interval['end']:.1f}s")

                    result = self.stage2.predict_interval(
                        features,
                        interval['start'],
                        interval['end'],
                        filter_silence=True
                    )

                    if result['verdict'] == 'unknown':
                        # [보강] Early stop reset: Low frame count는 pass
                        # [OPTIMIZED] 30 → 15 (stage2_analyzer와 일치)
                        if result.get('valid_frame_count', 0) < 15:
                            logger.info(f"Interval {i+1} unknown due to low frame count (<15), "
                                       f"not counting toward early stop")
                            continue  # unknown count 증가 안함

                        unknown_count += 1
                        logger.warning(f"Interval {i+1} returned unknown, consecutive count: {unknown_count}")

                        if unknown_count >= 2:
                            logger.warning("Early stop: 2 consecutive unknowns")
                            break
                    else:
                        unknown_count = 0  # Reset on success
                        interval_results.append(result)

                # Fallback Tier 1: Top 2 재시도 (침묵 포함)
                if len(interval_results) == 0:
                    logger.warning("All intervals failed, retrying top 2 without silence filtering")
                    for interval in suspicious_intervals[:2]:
                        result = self.stage2.predict_interval(
                            features,
                            interval['start'],
                            interval['end'],
                            filter_silence=False  # 침묵 포함
                        )
                        if result['verdict'] != 'unknown':
                            interval_results.append(result)

                    # Fallback Tier 2: Full video
                    if len(interval_results) == 0:
                        logger.error("All fallback strategies failed, analyzing full video")
                        pia_result = self.stage2.predict_full_video(features)
                    else:
                        pia_result = self._aggregate_interval_results(interval_results)
                else:
                    # Weighted voting
                    pia_result = self._aggregate_interval_results(interval_results)

            # [NEW] STEP 4.5: Fetch XAI info via predict_full_video() for visualizations
            # Only if pia_result doesn't already have XAI info (from interval analysis)
            if 'geometry_analysis' not in pia_result and 'phoneme_analysis' not in pia_result:
                logger.info("STEP 4.5: Fetching XAI info from full video analysis...")
                try:
                    xai_full_result = self.stage2.predict_full_video(features)
                    # Merge XAI info only (keep interval analysis verdict/confidence)
                    if 'geometry_analysis' in xai_full_result:
                        pia_result['geometry_analysis'] = xai_full_result['geometry_analysis']
                    if 'phoneme_analysis' in xai_full_result:
                        pia_result['phoneme_analysis'] = xai_full_result['phoneme_analysis']
                    if 'model_info' in xai_full_result:
                        pia_result['model_info'] = xai_full_result['model_info']
                    # [NEW] Merge phoneme transcription info for Korean display
                    if 'all_phonemes' in xai_full_result:
                        pia_result['all_phonemes'] = xai_full_result['all_phonemes']
                    if 'matched_phonemes' in xai_full_result and 'matched_phonemes' not in pia_result:
                        pia_result['matched_phonemes'] = xai_full_result['matched_phonemes']
                    logger.info("XAI info merged successfully")
                except Exception as e:
                    logger.warning(f"Failed to fetch XAI info: {e}")

        except Exception as e:
            logger.error(f"PIA prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            pia_result = {'verdict': 'unknown', 'confidence': 0.0, 'probabilities': {'fake': 0.0, 'real': 0.0}}

        # STEP 5: Ensemble & Summary
        logger.info("STEP 5: Ensemble & Summary...")
        detection = self.aggregator.ensemble(mmms_result, pia_result)
        
        video_info = self.aggregator.extract_video_info(video_path)
        summary = self.aggregator.generate_korean_summary(detection, pia_result, video_info)
        
        # Thumbnail & Visualizations
        thumbnail_path = None
        stage1_viz_path = None
        pia_viz_path = None
        interval_viz_paths: List[str] = []
        
        if save_visualizations:
            # Thumbnail 생성
            thumbnail_path = str(Path(output_dir) / 'thumbnail.png')
            try:
                generate_detection_card(
                    video_path=video_path,
                    detection=detection,
                    output_path=thumbnail_path
                )
                logger.info(f"Generated detection card: {thumbnail_path}")
            except Exception as e:
                logger.warning(f"Thumbnail generation failed: {e}")
                thumbnail_path = None
            
            # [NEW] MMMS-BA Timeline 시각화
            try:
                stage1_viz_path = self._visualize_stage1_timeline(
                    mmms_result=mmms_result,
                    suspicious_intervals=suspicious_intervals,
                    features=features,
                    video_path=video_path,
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
                    pia_viz_path = self._visualize_pia_result(
                        pia_result=pia_result,
                        video_path=video_path,
                        output_dir=output_dir
                    )
                    logger.info(f"Generated PIA XAI visualization: {pia_viz_path}")
            except Exception as e:
                logger.warning(f"PIA visualization generation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        processing_time_ms = (time.time() - start_time) * 1000
        
        # Final Result
        result = self.aggregator.build_final_result(
            video_path=video_path,
            video_id=video_id,
            output_dir=output_dir,
            detection=detection,
            summary=summary,
            video_info=video_info,
            processing_time_ms=processing_time_ms,
            mmms_result=mmms_result,
            pia_result=pia_result,
            thumbnail_path=thumbnail_path,
            stage1_timeline_path=stage1_viz_path,
            pia_xai_path=pia_viz_path,
            interval_xai_paths=interval_viz_paths,
            suspicious_intervals=suspicious_intervals  # [NEW] Pass suspicious intervals for interface-compliant JSON
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
        
        return result

    def _error_result(self, error_title, error_msg):
        return {
            'metadata': {'error': True},
            'detection': {'verdict': 'error', 'confidence': 0.0},
            'summary': {'title': error_title, 'detailed_explanation': error_msg}
        }

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
        output_dir: str
    ) -> str:
        """
        MMMS-BA Timeline 시각화 생성.
        
        Args:
            mmms_result: Stage1 예측 결과
            suspicious_intervals: 의심 구간 리스트
            features: 추출된 특징 (frames_10fps 포함)
            video_path: 비디오 경로
            output_dir: 출력 디렉토리
            
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
        
        # 의심 구간 하이라이트
        for interval in suspicious_intervals:
            start = interval['start']
            end = interval['end']
            ax_graph.axvspan(start, end, alpha=0.3, color='red', label='Suspicious Interval' if suspicious_intervals.index(interval) == 0 else '')
        
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
        stats_text = (
            f"Suspicious Intervals: {len(suspicious_intervals)} | "
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
        output_dir: str
    ) -> str:
        """
        PIA XAI 결과 시각화 생성.
        
        Args:
            pia_result: PIA XAI 결과 (predict_full_video 결과)
            video_path: 비디오 경로
            output_dir: 출력 디렉토리
            
        Returns:
            시각화 파일 경로
        """
        video_name = Path(video_path).stem
        output_path = Path(output_dir) / f"{video_name}_pia_xai.png"

        visualizer = PIAVisualizer(dpi=150)

        # Extract transcription - prefer all_phonemes (full sequence), fallback to matched_phonemes
        transcription = ""
        all_phonemes = pia_result.get('all_phonemes', [])
        if all_phonemes:
            transcription = mfa_to_korean(all_phonemes)
        else:
            # Fallback to matched_phonemes (14 representative phonemes)
            matched_phonemes = pia_result.get('matched_phonemes', [])
            if matched_phonemes:
                transcription = mfa_to_korean(matched_phonemes)

        # Get time range from interval info
        time_range = ""
        if 'best_interval' in pia_result:
            interval = pia_result['best_interval']
            start = interval.get('start', 0)
            end = interval.get('end', 0)
            time_range = f"[{start:.1f}s - {end:.1f}s]"

        # 2x2 패널 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"PIA XAI Analysis | Video: {video_name}", fontsize=14, fontweight='bold')

        # Add transcription subtitle if available
        if transcription:
            subtitle = f"의심 구간 {time_range} 음성: '{transcription}'" if time_range else f"음성: '{transcription}'"
            fig.text(0.5, 0.96, subtitle, ha='center', fontsize=11, style='italic', color='#333333')
        
        # Panel 1: Geometry Analysis (가장 중요)
        if 'geometry_analysis' in pia_result:
            visualizer.plot_geometry_analysis(pia_result, ax=axes[0, 0], title="MAR Deviation Analysis")
        
        # Panel 2: Phoneme Attention
        if 'phoneme_analysis' in pia_result:
            visualizer.plot_phoneme_attention(pia_result, ax=axes[0, 1], title="Phoneme Attention")
        
        # Panel 3: Branch Contributions
        if 'model_info' in pia_result and 'branch_contributions' in pia_result['model_info']:
            visualizer.plot_branch_contributions(pia_result, ax=axes[1, 0], title="Branch Contributions")
        
        # Panel 4: Detection Summary
        if 'detection' in pia_result:
            visualizer.plot_detection_summary(pia_result, ax=axes[1, 1], title="Detection Summary")
        
        plt.tight_layout()
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
                        default='F:/preprocessed_data_pia_optimized/checkpoints/best.pth',
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
