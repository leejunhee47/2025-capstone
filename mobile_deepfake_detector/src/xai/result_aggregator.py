"""
Result Aggregator Module

Aggregates insights from multiple intervals, computes overall detection,
and generates Korean summaries for final results.

Author: Claude
Date: 2025-11-17
"""

import numpy as np
import cv2
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Setup logger
logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Result Aggregator for combining insights and generating final results.

    Aggregates insights across all Stage2 intervals, computes overall
    detection verdict, and generates user-friendly Korean summaries.
    """

    def __init__(self):
        """Initialize ResultAggregator."""
        logger.info("Initializing ResultAggregator...")

    def aggregate_insights(self, stage2_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate insights across all Stage2 intervals.

        Args:
            stage2_results: List of Stage2 interval analysis results

        Returns:
            aggregated_insights: Dict with top_suspicious_phonemes, branch_trends, mar_summary
        """
        if len(stage2_results) == 0:
            return {
                'top_suspicious_phonemes': [],
                'branch_trends': {
                    'visual_avg': 0.0,
                    'geometry_avg': 0.0,
                    'identity_avg': 0.0,
                    'most_dominant': 'visual'
                },
                'mar_summary': {
                    'intervals_with_anomalies': 0,
                    'total_anomalous_frames': 0,
                    'avg_deviation_percent': 0.0
                }
            }

        # ========================================
        # 1. Top Suspicious Phonemes (across all intervals)
        # ========================================
        phoneme_aggregation = {}  # {phoneme: {'attentions': [], 'intervals': []}}

        for idx, result in enumerate(stage2_results):
            interval_id = result['interval_id']
            for phoneme_score in result['phoneme_analysis']['phoneme_scores']:
                phoneme = phoneme_score['phoneme_korean']
                attention = phoneme_score['attention_weight']

                if phoneme not in phoneme_aggregation:
                    phoneme_aggregation[phoneme] = {'attentions': [], 'intervals': []}

                phoneme_aggregation[phoneme]['attentions'].append(attention)
                if interval_id not in phoneme_aggregation[phoneme]['intervals']:
                    phoneme_aggregation[phoneme]['intervals'].append(interval_id)

        # Calculate average attention and sort
        top_suspicious_phonemes = []
        for phoneme, data in phoneme_aggregation.items():
            avg_attention = np.mean(data['attentions'])
            appearance_count = len(data['intervals'])
            top_suspicious_phonemes.append({
                'phoneme': phoneme,
                'avg_attention': float(avg_attention),
                'appearance_count': appearance_count,
                'intervals': data['intervals']
            })

        # Sort by average attention (descending)
        top_suspicious_phonemes.sort(key=lambda x: x['avg_attention'], reverse=True)

        # ========================================
        # 2. Branch Contribution Trends
        # ========================================
        visual_contributions = []
        geometry_contributions = []
        identity_contributions = []

        for result in stage2_results:
            bc = result['branch_contributions']
            visual_contributions.append(bc['visual']['contribution_percent'])
            geometry_contributions.append(bc['geometry']['contribution_percent'])
            identity_contributions.append(bc['identity']['contribution_percent'])

        visual_avg = float(np.mean(visual_contributions))
        geometry_avg = float(np.mean(geometry_contributions))
        identity_avg = float(np.mean(identity_contributions))

        # Determine most dominant branch
        avg_contributions = {
            'visual': visual_avg,
            'geometry': geometry_avg,
            'identity': identity_avg
        }
        most_dominant = max(avg_contributions, key=avg_contributions.get)

        branch_trends = {
            'visual_avg': visual_avg,
            'geometry_avg': geometry_avg,
            'identity_avg': identity_avg,
            'most_dominant': most_dominant
        }

        # ========================================
        # 3. MAR Anomaly Summary
        # ========================================
        intervals_with_anomalies = 0
        total_anomalous_frames = 0
        all_deviations = []

        for result in stage2_results:
            anomalous_frames = result['geometry_analysis']['anomalous_frames']

            if len(anomalous_frames) > 0:
                intervals_with_anomalies += 1
                total_anomalous_frames += len(anomalous_frames)

                for frame in anomalous_frames:
                    all_deviations.append(frame['deviation_percent'])

        avg_deviation_percent = float(np.mean(all_deviations)) if all_deviations else 0.0

        mar_summary = {
            'intervals_with_anomalies': intervals_with_anomalies,
            'total_anomalous_frames': total_anomalous_frames,
            'avg_deviation_percent': avg_deviation_percent
        }

        return {
            'top_suspicious_phonemes': top_suspicious_phonemes,
            'branch_trends': branch_trends,
            'mar_summary': mar_summary
        }

    def compute_overall_detection(
        self,
        stage1_timeline: Dict[str, Any],
        stage2_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute overall detection verdict combining Stage1 and Stage2 evidence.

        Args:
            stage1_timeline: Stage1 timeline result
            stage2_results: List of Stage2 interval results

        Returns:
            detection: Dict with verdict, confidence, probabilities, frame counts
        """
        # Method 1: Stage1 confidence from suspicious intervals
        # Get from statistics (frame_probabilities removed for JSON size reduction)
        stats = stage1_timeline['statistics']
        total_frames = stats['total_frames']
        suspicious_frame_count = stats['suspicious_frame_count']
        suspicious_frame_ratio = stats['suspicious_frame_ratio']

        # Use max interval mean_fake_prob as Stage1 confidence (more reliable than frame count ratio)
        # This better reflects the actual model confidence rather than just counting frames above threshold
        suspicious_intervals = stage1_timeline.get('suspicious_intervals', [])
        if len(suspicious_intervals) > 0:
            # Use highest confidence interval as Stage1 score
            stage1_score = max([interval['mean_fake_prob'] for interval in suspicious_intervals])
        else:
            # No intervals detected - use overall mean fake probability
            stage1_score = stage1_timeline['statistics']['mean_fake_prob']

        # Method 2: Stage2 PIA confidence average (if available)
        if len(stage2_results) > 0:
            pia_fake_confidences = [r['prediction']['probabilities']['fake'] for r in stage2_results]
            pia_avg_confidence = float(np.mean(pia_fake_confidences))
        else:
            # No intervals analyzed - use Stage1 score
            pia_avg_confidence = stage1_score

        # Combined score (weighted combination)
        # 80% weight on Stage1 max interval confidence, 20% on Stage2 PIA confidence
        # Stage1 score now uses mean_fake_prob (actual confidence) instead of frame count ratio
        # Increased Stage1 weight to 80% due to Stage2 false negatives from limited phoneme coverage
        # This better handles cases where PIA sees only partial context (e.g., 30 valid frames from 193)
        combined_score = 0.8 * stage1_score + 0.2 * pia_avg_confidence

        # Final verdict (threshold: 0.5)
        verdict = 'fake' if combined_score > 0.5 else 'real'

        # Probabilities
        probabilities = {
            'real': float(1.0 - combined_score),
            'fake': float(combined_score)
        }

        return {
            'verdict': verdict,
            'confidence': float(combined_score),
            'probabilities': probabilities,
            'suspicious_frame_count': suspicious_frame_count,
            'suspicious_frame_ratio': float(suspicious_frame_ratio)
        }

    def generate_korean_summary(
        self,
        detection: Dict[str, Any],
        aggregated: Dict[str, Any],
        stage1_timeline: Dict[str, Any],
        stage2_interval_analysis: List[Dict[str, Any]] = None,
        video_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate user-friendly Korean summary.

        Args:
            detection: Overall detection result
            aggregated: Aggregated insights
            stage1_timeline: Stage1 timeline
            stage2_interval_analysis: List of Stage2 interval results (optional)
            video_info: Video metadata (optional)

        Returns:
            summary: Dict with title, risk_level, primary_reason, detail_view, etc.
        """
        if stage2_interval_analysis is None:
            stage2_interval_analysis = []

        confidence = detection['confidence']
        verdict = detection['verdict']

        # Risk level mapping (verdict-aware)
        risk_level = self._compute_risk_level(confidence, verdict)

        # Title
        if verdict == 'fake':
            title = f"⚠️ 딥페이크 영상 의심됨 (신뢰도: {confidence*100:.1f}%)"
        else:
            title = f"✅ 진짜 영상으로 판정 (신뢰도: {(1-confidence)*100:.1f}%)"

        # Primary reason
        suspicious_intervals = stage1_timeline.get('suspicious_intervals', [])
        top_phonemes = aggregated.get('top_suspicious_phonemes', [])

        if verdict == 'fake' and len(suspicious_intervals) > 0 and len(top_phonemes) > 0:
            # Get first interval time range
            first_interval = suspicious_intervals[0]
            start_time = first_interval['start_time_sec']
            end_time = first_interval['end_time_sec']

            # Get top phoneme
            top_phoneme = top_phonemes[0]['phoneme']

            primary_reason = f"{start_time:.1f}초~{end_time:.1f}초 구간에서 '{top_phoneme}' 발음 시 입 움직임이 부자연스럽습니다"
        elif verdict == 'fake' and len(suspicious_intervals) > 0:
            primary_reason = f"영상 전체에서 {len(suspicious_intervals)}개 구간에 부자연스러운 음성-입모양 불일치가 감지되었습니다"
        elif verdict == 'real':
            primary_reason = "전체 영상에서 부자연스러운 음성-입모양 불일치가 감지되지 않았습니다"
        else:
            primary_reason = "분석 결과를 기반으로 진짜 영상으로 판정되었습니다"

        # Detailed explanation
        detailed_explanation = self._build_detailed_explanation(
            verdict=verdict,
            detection=detection,
            aggregated=aggregated,
            suspicious_intervals=suspicious_intervals
        )

        # NEW: Add detail_view structure for mobile app
        detail_view = self._build_detail_view(
            detection=detection,
            stage1_timeline=stage1_timeline,
            stage2_interval_analysis=stage2_interval_analysis,
            video_info=video_info
        )

        return {
            'title': title,
            'risk_level': risk_level,
            'primary_reason': primary_reason,
            'suspicious_interval_count': len(suspicious_intervals),
            'top_suspicious_phonemes': [p['phoneme'] for p in top_phonemes[:3]],
            'detailed_explanation': detailed_explanation,
            'detail_view': detail_view  # NEW
        }

    def _compute_risk_level(self, confidence: float, verdict: str) -> str:
        """
        Compute risk level from confidence score and verdict.

        Args:
            confidence: Combined confidence score (0.0-1.0)
            verdict: Detection verdict ('fake' or 'real')

        Returns:
            risk_level: 'low' | 'medium' | 'high' | 'critical'
        """
        if verdict == 'fake':
            # Fake 판정인 경우: confidence가 높을수록 risk 높음
            if confidence > 0.85:
                return 'critical'
            elif confidence > 0.7:
                return 'high'
            elif confidence > 0.5:  # Detection threshold와 일치
                return 'medium'
            else:
                return 'low'  # Should not happen (fake verdict requires >0.5 confidence)
        else:
            # Real 판정인 경우: fake score가 높을수록 (real confidence 낮을수록) risk 높음
            fake_score = 1.0 - confidence
            if fake_score > 0.4:  # Real confidence < 0.6 (애매한 경우)
                return 'medium'
            else:
                return 'low'

    def _build_detailed_explanation(
        self,
        verdict: str,
        detection: Dict[str, Any],
        aggregated: Dict[str, Any],
        suspicious_intervals: List[Dict[str, Any]]
    ) -> str:
        """Build detailed Korean explanation (under 200 characters)."""
        if verdict == 'fake':
            branch_trends = aggregated['branch_trends']
            mar_summary = aggregated['mar_summary']

            # Build explanation
            parts = []
            parts.append(f"전체 영상 중 {detection['suspicious_frame_ratio']:.1f}%가 의심스러운 것으로 판정되었습니다.")
            parts.append(f"{branch_trends['most_dominant']} 특징이 가장 두드러지게 나타났습니다.")

            if mar_summary['intervals_with_anomalies'] > 0:
                parts.append(f"{mar_summary['intervals_with_anomalies']}개 구간에서 입 움직임 비정상 패턴이 발견되었습니다.")

            explanation = " ".join(parts)
        else:
            explanation = f"전체 {len(suspicious_intervals)}개 구간을 분석한 결과, 부자연스러운 음성-입모양 불일치 패턴이 감지되지 않았습니다. 신뢰도 {(1-detection['confidence'])*100:.1f}%로 진짜 영상으로 판정되었습니다."

        # Truncate to 200 characters if needed
        if len(explanation) > 200:
            explanation = explanation[:197] + "..."

        return explanation

    def _build_detail_view(
        self,
        detection: Dict[str, Any],
        stage1_timeline: Dict[str, Any],
        stage2_interval_analysis: List[Dict[str, Any]],
        video_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Build detail_view structure for mobile app display.

        Args:
            detection: Overall detection result
            stage1_timeline: Stage1 timeline
            stage2_interval_analysis: List of Stage2 interval results
            video_info: Video metadata

        Returns:
            detail_view: Structured data for mobile UI
        """
        key_findings = []

        # Extract key findings from stage2_interval_analysis
        for interval_analysis in stage2_interval_analysis:
            # Phoneme anomaly
            phoneme_analysis = interval_analysis.get('phoneme_analysis', {})
            top_phoneme = phoneme_analysis.get('top_phoneme', '')

            if top_phoneme and top_phoneme != '<pad>':
                phoneme_scores = phoneme_analysis.get('phoneme_scores', [])
                if phoneme_scores:
                    top_score = phoneme_scores[0]
                    key_findings.append({
                        'type': 'phoneme_anomaly',
                        'icon': '🗣️',
                        'title': '발음-입모양 불일치',
                        'description': f"'{top_phoneme}' 발음 시 입 움직임 부자연스러움",
                        'severity': 'high',
                        'intervals': [interval_analysis['interval_id']]
                    })

            # Geometry anomaly (MAR)
            geometry_analysis = interval_analysis.get('geometry_analysis', {})
            statistics = geometry_analysis.get('statistics', {})
            mean_mar = statistics.get('mean_mar', 0)

            if mean_mar > 0:
                key_findings.append({
                    'type': 'geometry_anomaly',
                    'icon': '👄',
                    'title': '입 움직임 이상',
                    'description': f"MAR 값 평균: {mean_mar:.3f}",
                    'severity': 'medium',
                    'intervals': [interval_analysis['interval_id']]
                })

            # Visual branch dominance
            branch_contributions = interval_analysis.get('branch_contributions', {})
            top_branch = branch_contributions.get('top_branch', '')

            if top_branch == 'visual':
                visual_contrib = branch_contributions.get('visual', {})
                if isinstance(visual_contrib, dict):
                    contrib_pct = visual_contrib.get('contribution_percent', 0)
                else:
                    contrib_pct = visual_contrib

                key_findings.append({
                    'type': 'visual_artifact',
                    'icon': '👁️',
                    'title': '시각적 단서 감지',
                    'description': f"Visual 브랜치 {contrib_pct:.1f}% 기여",
                    'severity': 'high',
                    'intervals': [interval_analysis['interval_id']]
                })

        # Calculate timeline summary
        suspicious_intervals = stage1_timeline.get('suspicious_intervals', [])
        total_duration_sec = video_info.get('duration_sec', 0) if video_info else 0
        suspicious_duration_sec = sum(
            interval.get('duration_sec', 0)
            for interval in suspicious_intervals
        )
        suspicious_percentage = stage1_timeline.get('statistics', {}).get('mean_fake_prob', 0) * 100

        timeline_summary = {
            'total_duration_sec': float(total_duration_sec),
            'suspicious_duration_sec': float(suspicious_duration_sec),
            'suspicious_percentage': float(suspicious_percentage)
        }

        # Calculate confidence breakdown
        stage1_confidence = stage1_timeline.get('statistics', {}).get('mean_fake_prob', 0)
        stage2_confidence = (
            np.mean([interval.get('prediction', {}).get('confidence', 0)
                    for interval in stage2_interval_analysis])
            if stage2_interval_analysis else 0.0
        )

        confidence_breakdown = {
            'stage1_confidence': float(stage1_confidence),
            'stage2_confidence': float(stage2_confidence),
            'combined_confidence': float(detection['confidence'])
        }

        return {
            'key_findings': key_findings,
            'timeline_summary': timeline_summary,
            'confidence_breakdown': confidence_breakdown
        }

    def extract_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        cap.release()

        return {
            'duration_sec': float(duration_sec),
            'total_frames': total_frames,
            'fps': float(fps),
            'resolution': f"{width}x{height}",
            'original_path': str(video_path)
        }

    def build_final_result(
        self,
        video_path: str,
        video_id: str,
        output_dir: str,
        stage1_timeline: Dict[str, Any],
        stage2_interval_analysis: List[Dict[str, Any]],
        aggregated_insights: Dict[str, Any],
        detection: Dict[str, Any],
        summary: Dict[str, Any],
        video_info: Dict[str, Any],
        processing_time_ms: float,
        mmms_model_path: str,
        pia_model_path: str,
        thumbnail_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assemble final HybridDeepfakeXAIResult matching hybrid_xai_interface.ts.

        Args:
            video_path: Path to video file
            video_id: Video identifier
            output_dir: Output directory
            stage1_timeline: Stage1 timeline result
            stage2_interval_analysis: List of Stage2 interval results
            aggregated_insights: Aggregated insights
            detection: Overall detection result
            summary: Korean summary
            video_info: Video metadata
            processing_time_ms: Processing time in milliseconds
            mmms_model_path: Path to MMMS-BA model
            pia_model_path: Path to PIA model
            thumbnail_path: Path to generated thumbnail PNG (optional)

        Returns:
            Complete result with all 9 sections
        """
        # Generate request ID
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        # Metadata
        metadata = {
            'video_id': video_id,
            'request_id': request_id,
            'processed_at': datetime.utcnow().isoformat() + 'Z',
            'processing_time_ms': processing_time_ms,
            'pipeline_version': 'hybrid_v1.0'
        }

        # SIMPLIFIED: model_version (removed checkpoint paths and excess metadata)
        model_version = {
            'pipeline': 'hybrid_v1.0',
            'mmms_ba': 'v1.0',
            'pia': 'v1.0'
        }

        # Output file paths
        outputs = {
            'stage1_timeline': str(Path(output_dir) / "stage1_timeline.png"),
            'stage2_intervals': [
                str(Path(output_dir) / f"interval_{i}_xai.png")
                for i in range(len(stage2_interval_analysis))
            ],
            'combined_json': str(Path(output_dir) / "result.json"),
            'combined_summary': str(Path(output_dir) / "summary.png")
        }

        # Add thumbnails section
        thumbnails = {}
        if thumbnail_path:
            thumbnails['detection_card'] = thumbnail_path

        # NEW: Build report structure for mobile app display
        stage1_png = outputs.get('stage1_timeline', '')
        stage2_pngs = outputs.get('stage2_intervals', [])

        report = {
            'format': 'png_pages',
            'page_count': 2,
            'pages': [
                stage1_png,  # Page 1: Timeline
                stage2_pngs[0] if stage2_pngs else ''  # Page 2: XAI (first interval)
            ]
        }

        # Assemble complete result
        return {
            'metadata': metadata,
            'video_info': video_info,
            'detection': detection,
            'summary': summary,
            'stage1_timeline': stage1_timeline,
            'stage2_interval_analysis': stage2_interval_analysis,
            'aggregated_insights': aggregated_insights,
            'model_version': model_version,  # CHANGED from model_info
            'thumbnails': thumbnails,  # NEW
            'report': report,  # NEW
            'outputs': outputs  # Keep for backward compatibility
        }

    def convert_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(item) for item in obj]
        else:
            return obj

