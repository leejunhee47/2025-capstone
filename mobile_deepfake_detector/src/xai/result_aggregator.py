import uuid
from datetime import datetime
from typing import Dict, Any
import logging

# Setup logger
logger = logging.getLogger(__name__)

class ResultAggregator:
    """
    Result Aggregator for combining insights and generating final results.
    
    Simplified for Unified Hybrid Pipeline:
    - Ensembles MMMS-BA and PIA results
    - Generates Korean summaries based on overall analysis
    """

    def __init__(self):
        logger.info("Initializing ResultAggregator...")

    def ensemble(
        self,
        mmms_result: Dict[str, Any],
        pia_result: Dict[str, Any],
        weights: Dict[str, float] = {'mmms': 0.7, 'pia': 0.3}
    ) -> Dict[str, Any]:
        """
        Ensemble MMMS-BA and PIA results.
        
        Args:
            mmms_result: Output from Stage1Scanner.predict_full_video
            pia_result: Output from Stage2Analyzer.predict_full_video
            weights: Weights for ensemble (default: 0.4 for MMMS-BA, 0.6 for PIA)
            
        Returns:
            final_result: Combined detection result
        """
        # Extract probabilities (fake prob)
        mmms_prob = mmms_result.get('probabilities', {}).get('fake', 0.0)
        
        # PIA result structure (from explainer)
        if 'detection' in pia_result:
            pia_prob = pia_result['detection'].get('probabilities', {}).get('fake', 0.0)
        else:
            # Fallback or if PIA failed
            pia_prob = pia_result.get('probabilities', {}).get('fake', 0.0)
            if pia_prob == 0.0 and pia_result.get('verdict') == 'unknown':
                # If PIA failed, rely on MMMS-BA
                weights = {'mmms': 1.0, 'pia': 0.0}
                logger.warning("PIA result unknown, falling back to MMMS-BA only.")
            
        # Weighted ensemble
        w_mmms = weights['mmms']
        w_pia = weights['pia']
        
        # [NEW] Conservative Ensemble Logic: If any model is highly confident (>0.9) about FAKE,
        # prioritize that signal. This prevents one model's failure from masking a strong detection.
        STRONG_THRESHOLD = 0.90

        if mmms_prob > STRONG_THRESHOLD:
            logger.info(f"Strong FAKE signal from MMMS-BA ({mmms_prob:.4f}). Prioritizing MMMS-BA.")
            final_prob = mmms_prob
            w_mmms, w_pia = 1.0, 0.0

        elif pia_prob > STRONG_THRESHOLD:
            logger.info(f"Strong FAKE signal from PIA ({pia_prob:.4f}). Prioritizing PIA.")
            final_prob = pia_prob
            w_mmms, w_pia = 0.0, 1.0

        else:
            # Standard weighted average for ambiguous cases
            total_w = w_mmms + w_pia
            if total_w > 0:
                w_mmms /= total_w
                w_pia /= total_w

            final_prob = w_mmms * mmms_prob + w_pia * pia_prob
        
        verdict = 'fake' if final_prob > 0.5 else 'real'
        confidence = final_prob if verdict == 'fake' else 1.0 - final_prob

        # 개별 모델 판정 및 신뢰도 계산
        mmms_verdict = 'fake' if mmms_prob > 0.5 else 'real'
        mmms_confidence = mmms_prob if mmms_verdict == 'fake' else 1.0 - mmms_prob

        pia_verdict = 'fake' if pia_prob > 0.5 else 'real'
        pia_confidence = pia_prob if pia_verdict == 'fake' else 1.0 - pia_prob

        return {
            'verdict': verdict,
            'confidence': float(confidence),
            'probabilities': {
                'real': 1.0 - final_prob,
                'fake': float(final_prob)
            },
            'details': {
                # MMMS-BA 개별 결과
                'mmms_verdict': mmms_verdict,
                'mmms_confidence': float(mmms_confidence),
                'mmms_fake_prob': float(mmms_prob),
                # PIA 개별 결과
                'pia_verdict': pia_verdict,
                'pia_confidence': float(pia_confidence),
                'pia_fake_prob': float(pia_prob),
                # 앙상블 가중치
                'weights': {'mmms': w_mmms, 'pia': w_pia}
            }
        }

    def generate_korean_summary(
        self,
        detection: Dict[str, Any],
        pia_result: Dict[str, Any],
        video_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate user-friendly Korean summary based on ensemble result.
        """
        confidence = detection['confidence']
        verdict = detection['verdict']
        
        # Risk level
        risk_level = self._compute_risk_level(confidence, verdict)
        
        # Title
        if verdict == 'fake':
            title = f"⚠️ 딥페이크 영상 의심됨 (신뢰도: {confidence*100:.1f}%)"
        else:
            title = f"✅ 진짜 영상으로 판정 (신뢰도: {confidence*100:.1f}%)"
            
        # Primary Reason with MAR Deviation (우선순위 1) 또는 interval info
        primary_reason = ""
        if verdict == 'fake':
            # [우선순위 1] MAR Deviation 기반 설명 (가장 해석 가능)
            geometry_analysis = pia_result.get('geometry_analysis', {})
            abnormal_phonemes = geometry_analysis.get('abnormal_phonemes', [])
            
            if abnormal_phonemes:
                # 가장 심각한 이상 phoneme 선택 (z-score 기준)
                worst = max(abnormal_phonemes, key=lambda x: abs(x.get('z_score', 0)))
                
                # 편차 퍼센트 계산
                expected_mean = worst.get('expected_mean', 0.3)
                if expected_mean > 0:
                    deviation_pct = abs(worst['deviation'] / expected_mean * 100)
                else:
                    deviation_pct = abs(worst['deviation']) * 100
                
                # 방향 결정
                direction = "더 크게" if worst['deviation'] > 0 else "더 작게"
                
                # 구간 정보가 있으면 포함
                if 'best_interval' in pia_result:
                    interval = pia_result['best_interval']
                    primary_reason = (
                        f"{interval['start']:.1f}~{interval['end']:.1f}초 구간에서 "
                        f"'{worst['phoneme']}' 발음 시 입을 {deviation_pct:.0f}% {direction} 벌려 "
                        f"부자연스럽습니다."
                    )
                else:
                    primary_reason = (
                        f"'{worst['phoneme']}' 발음 시 입을 {deviation_pct:.0f}% {direction} 벌려 "
                        f"부자연스럽습니다."
                    )
            
            # [우선순위 2] Interval info (MAR deviation 없을 때)
            elif 'best_interval' in pia_result:
                interval = pia_result['best_interval']
                # Filter out <pad> tokens before selecting top phonemes
                all_phonemes = pia_result.get('matched_phonemes', [])
                valid_phonemes = [p for p in all_phonemes if p and p != '<pad>'][:3]

                # [HYBRID] Convert MFA codes to Korean characters for UX
                from ..utils.korean_phoneme_config import phoneme_to_korean
                korean_phonemes = [phoneme_to_korean(p) for p in valid_phonemes]
                phoneme_str = ', '.join(f"/{p}/" for p in korean_phonemes) if korean_phonemes else "여러 발음"

                primary_reason = (
                    f"{interval['start']:.1f}~{interval['end']:.1f}초 구간에서 "
                    f"{phoneme_str} 등의 발음 시 입모양 불일치가 감지되었습니다."
                )
            # [우선순위 3] Original PIA insights (Attention 기반)
            elif 'phoneme_analysis' in pia_result:
                top_phonemes = pia_result['phoneme_analysis'].get('top_suspicious_phonemes', [])
                if top_phonemes:
                    primary_reason = f"'{top_phonemes[0]['phoneme']}' 발음 시 입모양 부자연스러움이 감지되었습니다."
                else:
                    primary_reason = "영상 전반에 걸쳐 부자연스러운 입 움직임이 감지되었습니다."
            else:
                primary_reason = "딥페이크 탐지 모델이 영상 조작 흔적을 발견했습니다."
        else:
            primary_reason = "분석 결과, 딥페이크로 의심되는 특징이 발견되지 않았습니다."
            
        # Detailed Explanation
        detailed_explanation = self._build_detailed_explanation(detection, pia_result)
        
        # Detail View (Mobile)
        detail_view = self._build_detail_view(detection, pia_result, video_info)
        
        return {
            'title': title,
            'risk_level': risk_level,
            'primary_reason': primary_reason,
            'detailed_explanation': detailed_explanation,
            'detail_view': detail_view
        }

    def _compute_risk_level(self, confidence: float, verdict: str) -> str:
        if verdict == 'fake':
            if confidence > 0.85: return 'critical'
            elif confidence > 0.7: return 'high'
            elif confidence > 0.5: return 'medium'
            else: return 'low'
        else:
            # For real verdict, high confidence means low risk
            if confidence > 0.8: return 'low'
            elif confidence > 0.6: return 'medium'
            else: return 'high' # Uncertain real

    def _build_detailed_explanation(self, detection: Dict, pia_result: Dict) -> str:
        verdict = detection['verdict']
        details = detection['details']

        if verdict == 'fake':
            parts = [f"종합 분석 결과 {detection['confidence']*100:.1f}% 확률로 딥페이크입니다."]

            # [우선순위 1] MAR Deviation 상세 설명
            geometry_analysis = pia_result.get('geometry_analysis', {})
            abnormal_phonemes = geometry_analysis.get('abnormal_phonemes', [])
            
            if abnormal_phonemes:
                # 가장 심각한 이상 phoneme들 설명
                sorted_abnormal = sorted(
                    abnormal_phonemes,
                    key=lambda x: abs(x.get('z_score', 0)),
                    reverse=True
                )
                
                top_abnormal = sorted_abnormal[:2]  # Top 2
                phoneme_descriptions = []
                
                for abnormal in top_abnormal:
                    phoneme = abnormal['phoneme']
                    deviation = abnormal.get('deviation', 0)
                    expected_mean = abnormal.get('expected_mean', 0.3)
                    z_score = abnormal.get('z_score', 0)
                    
                    if expected_mean > 0:
                        deviation_pct = abs(deviation / expected_mean * 100)
                    else:
                        deviation_pct = abs(deviation) * 100
                    
                    direction = "더 크게" if deviation > 0 else "더 작게"
                    phoneme_descriptions.append(
                        f"'{phoneme}' 발음 시 입을 {deviation_pct:.0f}% {direction} 벌림"
                    )
                
                if phoneme_descriptions:
                    parts.append(f"입모양 분석 결과: {', '.join(phoneme_descriptions)}.")
                
                # 구간 정보 추가
                if 'best_interval' in pia_result:
                    interval = pia_result['best_interval']
                    parts.append(
                        f"이상 패턴은 {interval['start']:.1f}~{interval['end']:.1f}초 구간에서 "
                        f"특히 두드러집니다."
                    )

            # [우선순위 2] Interval info (MAR deviation 없을 때)
            elif 'best_interval' in pia_result:
                interval = pia_result['best_interval']
                parts.append(
                    f"PIA 모델이 {interval['start']:.1f}~{interval['end']:.1f}초 구간에서 "
                    f"입모양 불일치를 탐지했습니다."
                )

                # Multiple intervals
                if pia_result.get('num_intervals_analyzed', 0) > 1:
                    parts.append(
                        f"(총 {pia_result['num_intervals_analyzed']}개의 의심 구간 분석)"
                    )

            # [우선순위 3] Original branch contribution
            elif 'branch_contributions' in pia_result:
                bc = pia_result.get('branch_contributions', {})
                top_branch = bc.get('top_branch', 'unknown')
                if top_branch != 'unknown':
                    parts.append(f"특히 {top_branch} 특징에서 조작 흔적이 두드러집니다.")

            return " ".join(parts)
        else:
            return f"MMMS-BA와 PIA 모델 모두 정상 범주 내의 패턴을 보였습니다. (MMMS: {details['mmms_fake_prob']:.2f}, PIA: {details['pia_fake_prob']:.2f})"

    def _build_detail_view(self, detection: Dict, pia_result: Dict, video_info: Dict) -> Dict:
        """Build mobile app view data"""
        
        key_findings = []
        
        if detection['verdict'] == 'fake':
            # [우선순위 1] MAR Deviation 기반 발견사항 (가장 해석 가능)
            geometry_analysis = pia_result.get('geometry_analysis', {})
            abnormal_phonemes = geometry_analysis.get('abnormal_phonemes', [])
            
            if abnormal_phonemes:
                # 심각도 순으로 정렬 (z-score 기준)
                sorted_abnormal = sorted(
                    abnormal_phonemes, 
                    key=lambda x: abs(x.get('z_score', 0)), 
                    reverse=True
                )
                
                for abnormal in sorted_abnormal[:3]:  # Top 3
                    z_score = abs(abnormal.get('z_score', 0))
                    deviation = abnormal.get('deviation', 0)
                    expected_mean = abnormal.get('expected_mean', 0.3)
                    
                    # 편차 퍼센트 계산
                    if expected_mean > 0:
                        deviation_pct = abs(deviation / expected_mean * 100)
                    else:
                        deviation_pct = abs(deviation) * 100
                    
                    # 방향 결정
                    direction = "더 크게" if deviation > 0 else "더 작게"
                    
                    # 심각도 결정
                    if z_score > 3:
                        severity = 'critical'
                        icon = '🔴'
                    elif z_score > 2:
                        severity = 'high'
                        icon = '🟠'
                    else:
                        severity = 'medium'
                        icon = '🟡'
                    
                    key_findings.append({
                        'type': 'mar_deviation',
                        'icon': icon,
                        'title': f"'{abnormal['phoneme']}' 발음 이상",
                        'description': f"입을 {deviation_pct:.0f}% {direction} 벌림 (Z-score: {z_score:.1f})",
                        'severity': severity,
                        'phoneme': abnormal['phoneme'],
                        'measured_mar': abnormal.get('measured_mar', 0.0),
                        'expected_range': abnormal.get('expected_range', [0.0, 0.0]),
                        'z_score': z_score
                    })
            
            # [우선순위 2] Attention 기반 발견사항 (MAR deviation 없을 때만)
            if not abnormal_phonemes and 'phoneme_analysis' in pia_result:
                pa = pia_result['phoneme_analysis']
                for p_score in pa.get('phoneme_scores', [])[:2]: # Top 2
                    if p_score.get('is_suspicious'):
                        key_findings.append({
                            'type': 'phoneme_attention',
                            'icon': '🗣️',
                            'title': f"발음 '{p_score['phoneme_korean']}' 주목",
                            'description': f"모델이 이 음소에 집중함 (어텐션: {p_score['attention_weight']*100:.1f}%)",
                            'severity': 'medium',
                            'note': '주의: 어텐션이 높다고 의심스러운 것은 아닙니다'
                        })
                        
        return {
            'key_findings': key_findings,
            'model_results': {
                # MMMS-BA 개별 결과
                'mmms_verdict': detection['details']['mmms_verdict'],
                'mmms_confidence': detection['details']['mmms_confidence'],
                'mmms_fake_prob': detection['details']['mmms_fake_prob'],
                # PIA 개별 결과
                'pia_verdict': detection['details']['pia_verdict'],
                'pia_confidence': detection['details']['pia_confidence'],
                'pia_fake_prob': detection['details']['pia_fake_prob'],
                # 앙상블 결과
                'ensemble_verdict': detection['verdict'],
                'ensemble_confidence': detection['confidence']
            },
            'video_info': video_info or {}
        }

    def extract_video_info(self, video_path: str) -> Dict[str, Any]:
        import cv2
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
        detection: Dict[str, Any],
        summary: Dict[str, Any],
        video_info: Dict[str, Any],
        processing_time_ms: float,
        suspicious_intervals: list = None
    ) -> Dict[str, Any]:
        """
        최종 결과를 구성합니다.

        Args:
            video_path: 비디오 파일 경로
            video_id: 비디오 ID
            detection: 탐지 결과
            summary: 한국어 요약
            video_info: 비디오 정보
            processing_time_ms: 처리 시간 (밀리초)
            suspicious_intervals: 의심 구간 리스트

        Returns:
            최종 결과 딕셔너리
        """
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        # 의심 구간 프레임 개수 추가
        suspicious_frame_count = 0
        if suspicious_intervals and len(suspicious_intervals) > 0:
            suspicious_frame_count = suspicious_intervals[0].get('frame_count', 0)

        return {
            'metadata': {
                'video_id': video_id,
                'request_id': request_id,
                'processed_at': datetime.utcnow().isoformat() + 'Z',
                'processing_time_ms': processing_time_ms,
                'pipeline_version': 'unified_v2.0'
            },
            'video_info': video_info,
            'detection': detection,
            'summary': summary,
            'suspicious_frame_count': suspicious_frame_count
        }
