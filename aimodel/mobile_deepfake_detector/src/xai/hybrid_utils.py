"""
Hybrid MMMS-BA + PIA XAI Utility Functions

Core utilities for:
- Resampling frames to PIA 14×5 format
- Phoneme-frame timestamp matching
- Consecutive frame grouping
- Korean summary generation
- Severity calculation

This module reuses existing implementations from:
- hybrid_phoneme_aligner_v2.py: Phoneme extraction
- enhanced_mar_extractor.py: MAR geometry features
- arcface_extractor.py: Face identity embeddings
- phoneme_dataset.py: 14×5 grid sampling
- pia_explainer.py: XAI analysis
- mmms_ba_adapter.py: Full frame loading
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import os

# Import only necessary utility functions
try:
    from ..utils.korean_phoneme_config import get_phoneme_vocab, is_kept_phoneme
except ImportError:
    # For standalone testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.utils.korean_phoneme_config import get_phoneme_vocab, is_kept_phoneme


def group_consecutive_frames(
    suspicious_indices: np.ndarray,
    fps: float,
    min_interval_frames: int = 14,
    merge_gap_sec: float = 1.0,
    timestamps: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    연속된 의심 프레임들을 구간으로 그룹화

    Args:
        suspicious_indices: 의심 프레임 인덱스 배열 (e.g., [45, 46, 47, ..., 312])
        fps: 영상 FPS (timestamps=None일 때 fallback용)
        min_interval_frames: 최소 구간 프레임 수 (PIA는 최소 14 필요)
        merge_gap_sec: 이 시간 이내 구간은 병합 (초)
        timestamps: Optional (T,) 프레임의 실제 타임스탬프 배열.
                   제공되면 timestamps[frame_idx]로 정확한 시간 계산.
                   None이면 frame_idx/fps로 계산 (순차 프레임 가정).

    Returns:
        intervals: List of interval dictionaries
            [{
                'start_frame': 245,
                'end_frame': 312,
                'start_time': 8.2,
                'end_time': 10.4,
                'frame_indices': [245, 246, ..., 312]
            }, ...]
    """
    if len(suspicious_indices) == 0:
        return []

    # Sort indices
    sorted_indices = np.sort(suspicious_indices)

    # Find breaks (where consecutive sequence ends)
    breaks = np.where(np.diff(sorted_indices) > 1)[0]

    # Split into consecutive groups
    groups = []
    start_idx = 0

    for break_idx in breaks:
        group = sorted_indices[start_idx:break_idx + 1]
        if len(group) >= min_interval_frames:
            groups.append(group)
        start_idx = break_idx + 1

    # Last group
    last_group = sorted_indices[start_idx:]
    if len(last_group) >= min_interval_frames:
        groups.append(last_group)

    # Merge nearby groups (within merge_gap_sec)
    merge_gap_frames = int(merge_gap_sec * fps)
    merged_groups = []
    current_group = groups[0] if groups else np.array([])

    for i in range(1, len(groups)):
        next_group = groups[i]
        gap = next_group[0] - current_group[-1]

        if gap <= merge_gap_frames:
            # Merge: fill the gap
            fill_frames = np.arange(current_group[-1] + 1, next_group[0])
            current_group = np.concatenate([current_group, fill_frames, next_group])
        else:
            # Save current and start new
            merged_groups.append(current_group)
            current_group = next_group

    if len(current_group) > 0:
        merged_groups.append(current_group)

    # Build interval dictionaries
    intervals = []
    for interval_id, frame_indices in enumerate(merged_groups):
        start_frame = int(frame_indices[0])
        end_frame = int(frame_indices[-1])

        # Use actual timestamps if provided, otherwise fall back to frame_idx/fps
        if timestamps is not None:
            start_time = float(timestamps[start_frame])
            end_time = float(timestamps[end_frame])
        else:
            # Fallback: assume sequential frames
            start_time = start_frame / fps
            end_time = end_frame / fps

        intervals.append({
            'interval_id': interval_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'frame_count': len(frame_indices),
            'frame_indices': frame_indices.tolist()
        })

    return intervals


def resample_frames_to_pia_format(
    frames: np.ndarray,
    timestamps: np.ndarray,
    phoneme_labels: np.ndarray,  # [NEW] Frame-level phoneme labels
    target_phonemes: int = 14,
    frames_per_phoneme: int = 5,
    phonemes: Optional[List[Dict]] = None  # Keep for compatibility but unused
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Resample suspicious interval frames to PIA 14×5 format.

    Uses the same grouping logic as KoreanPhonemeDataset._group_by_phoneme.
    
    [UPDATED] Now uses frame-level phoneme labels directly to match training behavior.

    Args:
        frames: (N, H, W, 3) - Suspicious frames
        timestamps: (N,) - Frame timestamps (seconds)
        phoneme_labels: (N,) - Frame-level phoneme labels
        target_phonemes: 14 (PIA expects 14 phonemes)
        frames_per_phoneme: 5 (PIA expects 5 frames per phoneme)
        phonemes: Deprecated, kept for backward compatibility

    Returns:
        resampled_frames: (14, 5, H, W, 3)
        matched_phonemes: List[str] (length 14)
        valid_mask: (14,) bool array - True if phoneme has actual frames, False if padding
    """
    phoneme_vocab = get_phoneme_vocab()  # 14 sorted phonemes
    H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]

    # Group frames by phoneme (reuse dataset logic)
    by_phoneme = {p: [] for p in phoneme_vocab}

    # [NEW] Use frame-level labels directly (matches KoreanPhonemeDataset behavior)
    matched_count = 0
    for frame_idx in range(len(frames)):
        # Handle potential length mismatch if labels are shorter/longer
        if frame_idx >= len(phoneme_labels):
            break

        phoneme = str(phoneme_labels[frame_idx]).strip()

        if is_kept_phoneme(phoneme):
            by_phoneme[phoneme].append(frames[frame_idx])
            matched_count += 1
            
    # [DEBUG] Log matching statistics
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[DEBUG] resample_frames_to_pia_format: {matched_count}/{len(frames)} frames matched to kept phonemes")

    # Build 14×5 grid with valid mask tracking
    resampled = np.zeros((target_phonemes, frames_per_phoneme, H, W, C), dtype=frames.dtype)
    matched_phonemes = []
    valid_mask = np.zeros(target_phonemes, dtype=bool)

    for pi, phoneme in enumerate(phoneme_vocab):
        frames_list = by_phoneme[phoneme][:frames_per_phoneme]  # First F frames

        if len(frames_list) > 0:
            matched_phonemes.append(phoneme)
            valid_mask[pi] = True  # Mark as valid
        else:
            matched_phonemes.append('<pad>')  # Explicit padding marker
            valid_mask[pi] = False

        for fi, frame in enumerate(frames_list):
            resampled[pi, fi] = frame

    return resampled, matched_phonemes, valid_mask


# Removed helper functions (_pad_phonemes, _sample_phonemes, _interpolate_frames, _repeat_frames, match_phonemes_to_timestamps)
# These are now handled by the dataset logic or are unnecessary


def calculate_severity(score: float) -> str:
    """
    점수를 심각도 레벨로 변환

    Args:
        score: 0.0~1.0 (fake probability, attention weight, deviation %)

    Returns:
        severity: "low" | "medium" | "high" | "critical"
    """
    if score >= 0.9:
        return "critical"
    elif score >= 0.8:
        return "high"
    elif score >= 0.65:
        return "medium"
    else:
        return "low"


def calculate_risk_level(suspicious_ratio: float) -> str:
    """
    의심 프레임 비율을 위험 레벨로 변환

    Args:
        suspicious_ratio: 0~100 (percentage)

    Returns:
        risk_level: "low" | "medium" | "high" | "critical"
    """
    if suspicious_ratio > 50:
        return "critical"
    elif suspicious_ratio > 30:
        return "high"
    elif suspicious_ratio > 10:
        return "medium"
    else:
        return "low"


def _build_detailed_explanation_v2(
    verdict: str,
    confidence: float,
    suspicious_intervals: List[Dict],
    interval_xai_results: List[Dict]
) -> str:
    """
    Build detailed explanation using v2.0 style (MAR deviation priority).

    Priority order:
    1. MAR Deviation (most interpretable)
    2. Interval info (if MAR not available)
    3. Branch contribution (fallback)

    Args:
        verdict: "real" | "fake"
        confidence: 0.0~1.0
        suspicious_intervals: Stage 1 intervals
        interval_xai_results: Stage 2 XAI results (PIAExplainer format)

    Returns:
        detailed_explanation: Korean explanation string
    """
    if verdict == "fake" and interval_xai_results:
        parts = [f"종합 분석 결과 {confidence*100:.1f}% 확률로 딥페이크입니다."]

        # Collect all abnormal phonemes across intervals
        all_abnormal_phonemes = []
        best_interval = None

        for i, xai in enumerate(interval_xai_results):
            geometry_analysis = xai.get('geometry_analysis', {})
            abnormal_phonemes = geometry_analysis.get('abnormal_phonemes', [])

            if abnormal_phonemes:
                for abnormal in abnormal_phonemes:
                    all_abnormal_phonemes.append({
                        **abnormal,
                        'interval_idx': i,
                        'interval': suspicious_intervals[i] if i < len(suspicious_intervals) else None
                    })

                # Track first interval with abnormal phonemes
                if best_interval is None and i < len(suspicious_intervals):
                    best_interval = suspicious_intervals[i]

        # [Priority 1] MAR Deviation based explanation
        if all_abnormal_phonemes:
            # Sort by z-score (most severe first)
            sorted_abnormal = sorted(
                all_abnormal_phonemes,
                key=lambda x: abs(x.get('z_score', 0)),
                reverse=True
            )

            top_abnormal = sorted_abnormal[:2]  # Top 2
            phoneme_descriptions = []

            for abnormal in top_abnormal:
                phoneme = abnormal.get('phoneme', '?')
                deviation = abnormal.get('deviation', 0)
                expected_mean = abnormal.get('expected_mean', 0.3)

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

            # Add interval info if available
            if best_interval:
                parts.append(
                    f"이상 패턴은 {best_interval['start_time']:.1f}~{best_interval['end_time']:.1f}초 구간에서 "
                    f"특히 두드러집니다."
                )

        # [Priority 2] Interval info (if MAR deviation not available)
        elif suspicious_intervals:
            first_interval = suspicious_intervals[0]
            parts.append(
                f"PIA 모델이 {first_interval['start_time']:.1f}~{first_interval['end_time']:.1f}초 구간에서 "
                f"입모양 불일치를 탐지했습니다."
            )

            if len(suspicious_intervals) > 1:
                parts.append(f"(총 {len(suspicious_intervals)}개의 의심 구간 분석)")

        # [Priority 3] Branch contribution (fallback)
        elif interval_xai_results:
            first_xai = interval_xai_results[0]
            branch_contrib = first_xai.get('branch_contributions', {})

            if branch_contrib:
                # Find top branch
                top_branch = branch_contrib.get('top_branch', 'unknown')
                if top_branch != 'unknown':
                    parts.append(f"특히 {top_branch} 특징에서 조작 흔적이 두드러집니다.")

        return " ".join(parts)

    elif verdict == "real":
        # Real verdict - simpler explanation
        real_confidence = (1 - confidence) * 100
        return (
            f"전체 {len(suspicious_intervals)}개 구간을 분석한 결과, "
            f"부자연스러운 음성-입모양 불일치 패턴이 감지되지 않았습니다. "
            f"신뢰도 {real_confidence:.1f}%로 진짜 영상으로 판정되었습니다."
        )

    else:
        return "딥페이크 패턴이 감지되지 않았습니다."


def build_korean_summary(
    verdict: str,
    confidence: float,
    suspicious_intervals: List[Dict],
    interval_xai_results: List[Dict]
) -> Dict[str, any]:
    """
    Korean summary generation (simplified using PIAExplainer patterns).

    Reuses explanation logic from pia_explainer._generate_korean_explanation.

    Args:
        verdict: "real" | "fake"
        confidence: 0.0~1.0
        suspicious_intervals: Stage 1 intervals
        interval_xai_results: Stage 2 XAI results (PIAExplainer format)

    Returns:
        summary: {
            'title': str,
            'risk_level': str,
            'primary_reason': str,
            'suspicious_interval_count': int,
            'top_suspicious_phonemes': List[str],
            'detailed_explanation': str
        }
    """
    # Title
    if verdict == "fake":
        title = f"⚠️ 딥페이크 영상 의심됨 (신뢰도: {confidence*100:.1f}%)"
    else:
        title = f"✓ 실제 영상으로 판정 (신뢰도: {(1-confidence)*100:.1f}%)"

    # Risk level (keep existing logic - it's simple)
    suspicious_count = len(suspicious_intervals)
    if suspicious_count == 0:
        risk_level = "low"
    elif suspicious_count == 1:
        risk_level = "medium"
    elif suspicious_count == 2:
        risk_level = "high"
    else:
        risk_level = "critical"

    # Primary reason - align with PIAExplainer output structure
    if suspicious_intervals and interval_xai_results:
        first_interval = suspicious_intervals[0]
        first_xai = interval_xai_results[0]

        # Reuse phoneme attention analysis from PIAExplainer
        phoneme_analysis = first_xai.get('phoneme_analysis', {})
        if phoneme_analysis and 'phoneme_scores' in phoneme_analysis:
            top_phoneme_data = phoneme_analysis['phoneme_scores'][0]
            top_phoneme = top_phoneme_data.get('phoneme_korean', '알 수 없음')
            top_attention = top_phoneme_data.get('attention_weight', 0.0)
        else:
            # Fallback for older format
            top_phoneme = first_xai.get('top_phoneme', '알 수 없음')
            top_attention = first_xai.get('top_attention', 0.0)

        primary_reason = (
            f"{first_interval['start_time']:.1f}초~{first_interval['end_time']:.1f}초 구간에서 "
            f"'{top_phoneme}' 발음 시 비정상적인 패턴이 감지되었습니다 (어텐션: {top_attention*100:.1f}%)"
        )
    else:
        primary_reason = "의심 구간이 탐지되지 않았습니다."

    # Top suspicious phonemes (aggregate from all intervals)
    phoneme_counts = {}
    for xai in interval_xai_results:
        phoneme_analysis = xai.get('phoneme_analysis', {})
        for phoneme_data in phoneme_analysis.get('phoneme_scores', []):
            phoneme = phoneme_data.get('phoneme_korean')
            if phoneme:
                phoneme_counts[phoneme] = phoneme_counts.get(phoneme, 0) + 1

    top_phonemes = sorted(phoneme_counts.keys(), key=lambda p: phoneme_counts[p], reverse=True)[:3]

    # Detailed explanation (v2.0 style: MAR deviation priority)
    detailed = _build_detailed_explanation_v2(
        verdict=verdict,
        confidence=confidence,
        suspicious_intervals=suspicious_intervals,
        interval_xai_results=interval_xai_results
    )

    return {
        'title': title,
        'risk_level': risk_level,
        'primary_reason': primary_reason,
        'suspicious_interval_count': suspicious_count,
        'top_suspicious_phonemes': top_phonemes,
        'detailed_explanation': detailed
    }


def aggregate_interval_insights(interval_xai_results: List[Dict]) -> Dict:
    """
    Aggregate XAI results across intervals.

    Reuses PIAExplainer result structures for consistency.

    Args:
        interval_xai_results: List of Stage 2 XAI results (PIAExplainer format)

    Returns:
        insights: {
            'top_suspicious_phonemes': [...],
            'branch_trends': {...},
            'mar_summary': {...}
        }
    """
    # Top suspicious phonemes (reuse PIAExplainer phoneme analysis structure)
    phoneme_attentions = {}
    phoneme_intervals = {}

    for interval_id, xai in enumerate(interval_xai_results):
        phoneme_analysis = xai.get('phoneme_analysis', {})
        for phoneme_data in phoneme_analysis.get('phoneme_scores', []):
            phoneme = phoneme_data.get('phoneme_korean')
            attention = phoneme_data.get('attention_weight', 0.0)

            if phoneme:
                if phoneme not in phoneme_attentions:
                    phoneme_attentions[phoneme] = []
                    phoneme_intervals[phoneme] = []

                phoneme_attentions[phoneme].append(attention)
                phoneme_intervals[phoneme].append(interval_id)

    # Build top phonemes list
    top_phonemes = []
    for phoneme, attentions in phoneme_attentions.items():
        top_phonemes.append({
            'phoneme': phoneme,
            'avg_attention': np.mean(attentions),
            'appearance_count': len(attentions),
            'intervals': phoneme_intervals[phoneme]
        })

    top_phonemes = sorted(top_phonemes, key=lambda x: x['avg_attention'], reverse=True)[:5]

    # Branch trends (reuse branch_contributions structure from PIAExplainer)
    branch_contribs = {'visual': [], 'geometry': [], 'identity': []}

    for xai in interval_xai_results:
        branch_contributions = xai.get('branch_contributions', {})

        # Handle both new PIAExplainer format and legacy format
        for branch_name in ['visual', 'geometry', 'identity']:
            if branch_name in branch_contributions:
                if isinstance(branch_contributions[branch_name], dict):
                    contrib = branch_contributions[branch_name].get('contribution_percent', 0)
                else:
                    contrib = branch_contributions[branch_name]
            else:
                contrib = 0
            branch_contribs[branch_name].append(contrib)

    visual_avg = np.mean(branch_contribs['visual']) if branch_contribs['visual'] else 0
    geometry_avg = np.mean(branch_contribs['geometry']) if branch_contribs['geometry'] else 0
    identity_avg = np.mean(branch_contribs['identity']) if branch_contribs['identity'] else 0

    most_dominant = max(
        [('visual', visual_avg), ('geometry', geometry_avg), ('identity', identity_avg)],
        key=lambda x: x[1]
    )[0]

    # MAR summary (reuse geometry_analysis structure from PIAExplainer)
    total_anomalous = 0
    deviations = []

    for xai in interval_xai_results:
        geometry_analysis = xai.get('geometry_analysis', {})
        anomalous_frames = geometry_analysis.get('anomalous_frames', [])
        total_anomalous += len(anomalous_frames)

        for frame in anomalous_frames:
            deviation = frame.get('deviation_percent', 0)
            deviations.append(deviation)

    avg_deviation = np.mean(deviations) if deviations else 0

    return {
        'top_suspicious_phonemes': top_phonemes,
        'branch_trends': {
            'visual_avg': round(visual_avg, 1),
            'geometry_avg': round(geometry_avg, 1),
            'identity_avg': round(identity_avg, 1),
            'most_dominant': most_dominant
        },
        'mar_summary': {
            'intervals_with_anomalies': sum(
                1 for xai in interval_xai_results
                if len(xai.get('geometry_analysis', {}).get('anomalous_frames', [])) > 0
            ),
            'total_anomalous_frames': total_anomalous,
            'avg_deviation_percent': round(avg_deviation, 1)
        }
    }


# New helper functions - NPZ direct loading (no re-extraction needed!)
def extract_interval_features(
    features: Dict,
    start_frame: int,
    end_frame: int
) -> Dict[str, np.ndarray]:
    """
    Extract features for a specific interval from feature dictionary.

    Args:
        features: Full feature dictionary from VideoFeatureExtractor
        start_frame: Interval start frame index
        end_frame: Interval end frame index

    Returns:
        interval_features: Dictionary with sliced features
    """
    # Slice features to interval
    interval_features = {
        'geometry': features['geometry'][start_frame:end_frame+1],
        'arcface': features['arcface'][start_frame:end_frame+1],
        'phoneme_labels': features['phoneme_labels'][start_frame:end_frame+1],
        'timestamps': features['timestamps'][start_frame:end_frame+1],
        'frames': features['frames'][start_frame:end_frame+1]
    }

    return interval_features


def get_interval_phoneme_dict(
    phoneme_labels: np.ndarray,
    timestamps: np.ndarray
) -> List[Dict]:
    """
    Convert per-frame phoneme labels to phoneme interval format.

    Args:
        phoneme_labels: (N,) - Phoneme label per frame
        timestamps: (N,) - Timestamp per frame

    Returns:
        phoneme_intervals: List[{'phoneme': str, 'start': float, 'end': float}]
    """
    if len(phoneme_labels) == 0:
        return []

    intervals = []
    current_phoneme = phoneme_labels[0]
    start_time = timestamps[0]

    for i in range(1, len(phoneme_labels)):
        if phoneme_labels[i] != current_phoneme:
            # Phoneme changed - save interval
            intervals.append({
                'phoneme': current_phoneme,
                'start': float(start_time),
                'end': float(timestamps[i-1])
            })
            current_phoneme = phoneme_labels[i]
            start_time = timestamps[i]

    # Add last interval
    intervals.append({
        'phoneme': current_phoneme,
        'start': float(start_time),
        'end': float(timestamps[-1])
    })

    return intervals


def load_full_frames_from_npz(
    npz_path: str,
    use_mmms_adapter: bool = False,
    original_video_root: str = "E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load full frames from NPZ file.

    Args:
        npz_path: Path to NPZ file
        use_mmms_adapter: If True, use MMSBAdapter for FULL video frames (5396 frames)
                         If False, use NPZ frames (50 frames) - DEFAULT
        original_video_root: Root directory (only used if use_mmms_adapter=True)

    Returns:
        frames: (T, 224, 224, 3) - All frames
        timestamps: (T,) - Frame timestamps
    """
    if use_mmms_adapter:
        # MMMS-BA style: Load FULL video frames from original video
        adapter = MMSBAdapter(original_video_root=original_video_root)
        data = adapter.load_npz_with_full_frames(npz_path)
        return data['frames'], data['timestamps']
    else:
        # PIA style: Use NPZ frames directly (already preprocessed)
        data = np.load(npz_path)
        return data['frames'], data['timestamps']


# ========================================
# Format Conversion Utilities
# (Extracted from hybrid_mmms_pia_explainer.py to reduce duplication)
# ========================================

def convert_for_json(obj):
    """
    Convert numpy arrays and types to JSON-serializable format.

    Reused by:
    - HybridMMSBAPIA._convert_for_json()
    - HybridXAIPipeline result saving

    Args:
        obj: Any object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


def format_stage1_to_interface(
    stage1_result: Dict,
    suspicious_intervals_raw: List[Dict],
    threshold: float
) -> Dict[str, Any]:
    """
    Convert Stage1 internal format to TypeScript interface format.

    Extracted from Stage1Scanner._format_stage1_timeline() to reduce duplication.

    Args:
        stage1_result: Output from HybridMMSBAPIA.run_stage1_temporal_scan()
        suspicious_intervals_raw: Output from group_consecutive_frames()
        threshold: Detection threshold used

    Returns:
        stage1_timeline: Dict matching hybrid_xai_interface.ts Stage1Timeline
    """
    fake_probs = stage1_result['fake_probs']
    timestamps = stage1_result['timestamps']
    total_frames = stage1_result['total_frames']

    # Count suspicious frames (for statistics only - PNG provides visual details)
    suspicious_frame_count = int(np.sum(fake_probs > threshold))

    # Build suspicious_intervals array with severity and playback_info
    suspicious_intervals = []
    for interval in suspicious_intervals_raw:
        start_frame = interval['start_frame']
        end_frame = interval['end_frame']
        start_time_sec = float(interval['start_time'])
        end_time_sec = float(interval['end_time'])

        # Calculate mean and max fake prob for this interval
        interval_probs = fake_probs[start_frame:end_frame+1]
        mean_fake_prob = float(np.mean(interval_probs))
        max_fake_prob = float(np.max(interval_probs))

        # NEW: Add playback_info for mobile app video seeking
        playback_info = {
            'start_timestamp': f"{int(start_time_sec // 3600):02d}:{int((start_time_sec % 3600) // 60):02d}:{start_time_sec % 60:06.3f}",
            'end_timestamp': f"{int(end_time_sec // 3600):02d}:{int((end_time_sec % 3600) // 60):02d}:{end_time_sec % 60:06.3f}",
            'seek_url': f"#t={start_time_sec:.3f},{end_time_sec:.3f}",
            'thumbnail_frame': int((start_frame + end_frame) // 2)
        }

        suspicious_intervals.append({
            'interval_id': interval['interval_id'],
            'start_frame': int(start_frame),
            'end_frame': int(end_frame),
            'start_time_sec': start_time_sec,
            'end_time_sec': end_time_sec,
            'duration_sec': float(interval['duration']),
            'frame_count': int(interval['frame_count']),
            'mean_fake_prob': mean_fake_prob,
            'max_fake_prob': max_fake_prob,
            'severity': calculate_severity(max_fake_prob),
            'playback_info': playback_info  # NEW
        })

    # Build statistics (frame_probabilities removed - see PNG visualization)
    statistics = {
        'total_frames': int(total_frames),
        'suspicious_frame_count': suspicious_frame_count,
        'suspicious_frame_ratio': float(suspicious_frame_count / total_frames * 100) if total_frames > 0 else 0.0,
        'mean_fake_prob': float(np.mean(fake_probs)),
        'std_fake_prob': float(np.std(fake_probs)),
        'max_fake_prob': float(np.max(fake_probs)),
        'min_fake_prob': float(np.min(fake_probs)),
        'threshold_used': float(threshold)
    }

    return {
        # 'frame_probabilities' removed - use stage1_timeline.png for visual details
        'suspicious_intervals': suspicious_intervals,
        'statistics': statistics,
        'visualization': {}  # Populated by visualization function
    }


def format_stage2_to_interface(
    interval_xai: Dict,
    interval: Dict,
    output_dir: Optional[str]
) -> Dict[str, Any]:
    """
    Convert Stage2 PIA XAI result to TypeScript interface format.

    Extracted from Stage2Analyzer._format_stage2_result() to reduce duplication.

    Args:
        interval_xai: Result from HybridMMSBAPIA.run_stage2_interval_xai()
        interval: Original interval dict from Stage1
        output_dir: Output directory for visualization path

    Returns:
        Formatted result matching hybrid_xai_interface.ts IntervalXAI
    """
    from pathlib import Path

    pia_xai = interval_xai['pia_xai']

    # [FIX] 에러 발생 시 안전한 기본값(Fallback) 반환
    if 'error' in pia_xai:
        error_msg = pia_xai.get('error', 'Unknown error')
        return {
            'interval_id': interval['interval_id'],
            'time_range': f"{interval['start_time_sec']:.1f}s-{interval['end_time_sec']:.1f}s",
            'prediction': {
                'verdict': 'unknown',
                'confidence': 0.0,
                'probabilities': {'real': 0.5, 'fake': 0.5}
            },
            'branch_contributions': {
                'visual': {
                    'contribution_percent': 0,
                    'l2_norm': 0,
                    'rank': 0
                },
                'geometry': {
                    'contribution_percent': 0,
                    'l2_norm': 0,
                    'rank': 0
                },
                'identity': {
                    'contribution_percent': 0,
                    'l2_norm': 0,
                    'rank': 0
                },
                'top_branch': 'unknown',
                'explanation': f"Analysis failed: {error_msg}"
            },
            'phoneme_analysis': {
                'phoneme_scores': [],
                'top_phoneme': '',
                'total_phonemes': 0
            },
            # [DEPRECATED] temporal_analysis 제거 - Temporal Heatmap은 단순히 attention score를 5프레임에 복제한 것으로
            # 실제 시간 변화를 보여주지 않아 유용성이 낮음
            # 'temporal_analysis': {
            #     'heatmap_data': [],
            #     'high_risk_points': [],
            #     'statistics': {
            #         'mean_fake_prob': 0.0,
            #         'max_fake_prob': 0.0,
            #         'high_risk_count': 0
            #     }
            # },
            'geometry_analysis': {
                'statistics': {
                    'mean_mar': 0.0,
                    'std_mar': 0.0,
                    'min_mar': 0.0,
                    'max_mar': 0.0,
                    'expected_baseline': 0.39
                },
                'phoneme_mar': [],
                'anomalous_frames': []
            },
            'korean_explanation': {
                'summary': '분석 실패',
                'key_findings': [f'오류: {error_msg}'],
                'detailed_analysis': f'이 구간의 분석 중 오류가 발생했습니다: {error_msg}'
            },
            'visualization': {}
        }

    # 1. Extract prediction
    detection = pia_xai['detection']
    prediction = {
        'verdict': 'fake' if detection['is_fake'] else 'real',
        'confidence': detection['confidence'],
        'probabilities': detection['probabilities']
    }

    # 2. Extract branch contributions
    branch_contribs = pia_xai['model_info']['branch_contributions']

    # Calculate percentages and ranks
    visual_pct = branch_contribs['visual'] * 100
    geometry_pct = branch_contribs['geometry'] * 100
    identity_pct = branch_contribs['identity'] * 100

    # Sort to get ranks
    sorted_branches = sorted(
        [('visual', visual_pct), ('geometry', geometry_pct), ('identity', identity_pct)],
        key=lambda x: x[1],
        reverse=True
    )
    ranks = {branch: rank + 1 for rank, (branch, _) in enumerate(sorted_branches)}

    top_branch = sorted_branches[0][0]

    branch_contributions = {
        'visual': {
            'contribution_percent': visual_pct,
            'l2_norm': branch_contribs['visual'],
            'rank': ranks['visual']
        },
        'geometry': {
            'contribution_percent': geometry_pct,
            'l2_norm': branch_contribs['geometry'],
            'rank': ranks['geometry']
        },
        'identity': {
            'contribution_percent': identity_pct,
            'l2_norm': branch_contribs['identity'],
            'rank': ranks['identity']
        },
        'top_branch': top_branch,
        'explanation': f"{top_branch.capitalize()} branch dominated ({sorted_branches[0][1]:.1f}%)"
    }

    # 3. Extract phoneme analysis
    phoneme_scores_raw = pia_xai['phoneme_analysis']['phoneme_scores']

    # Convert to TypeScript format
    phoneme_scores = []
    for idx, ps in enumerate(phoneme_scores_raw):
        phoneme_scores.append({
            'phoneme_ipa': ps['phoneme_mfa'],
            'phoneme_korean': ps['phoneme'],
            'attention_weight': ps['score'],
            'rank': idx + 1,  # Will re-sort below
            'frame_count': 5,  # PIA uses 5 frames per phoneme
            'is_suspicious': ps['importance_level'] in ['high', 'medium'],
            'explanation': f"'{ps['phoneme']}' 발음 시 어텐션 {ps['score']*100:.1f}%"
        })

    # Re-sort by attention weight and update ranks
    phoneme_scores = sorted(phoneme_scores, key=lambda x: x['attention_weight'], reverse=True)
    for idx, ps in enumerate(phoneme_scores):
        ps['rank'] = idx + 1

    top_phonemes_list = pia_xai['phoneme_analysis']['top_suspicious_phonemes']
    top_phoneme = top_phonemes_list[0]['phoneme'] if top_phonemes_list else ''

    phoneme_analysis = {
        'phoneme_scores': phoneme_scores,
        'top_phoneme': top_phoneme,
        'total_phonemes': len(phoneme_scores)
    }

    # 4. [DEPRECATED] Temporal Analysis 제거
    # Temporal Heatmap은 단순히 attention score를 5프레임에 복제한 것으로
    # 실제 시간 변화를 보여주지 않아 유용성이 낮음
    # Geometry Analysis는 실제 MAR deviation과 Z-score 기반 이상 탐지를 제공
    
    # high_risk_points는 anomalous_frames에서 사용되므로 빈 리스트로 설정
    high_risk_points = []
    
    # temporal_analysis는 빈 값으로 설정 (하위 호환성 유지)
    temporal_analysis = {
        'heatmap_data': [],
        'high_risk_points': [],
        'statistics': {
            'mean_fake_prob': 0.0,
            'max_fake_prob': 0.0,
            'high_risk_count': 0
        }
    }

    # 5. Extract geometry (MAR) analysis
    geom_analysis = pia_xai['geometry_analysis']

    # Build phoneme-level MAR (if available)
    phoneme_mar = []
    if 'abnormal_phonemes' in geom_analysis:
        for abnormal in geom_analysis['abnormal_phonemes']:
            expected_range = abnormal['expected_range']
            expected_avg = (expected_range[0] + expected_range[1]) / 2
            deviation_pct = abs(abnormal['deviation'] / expected_avg * 100) if expected_avg > 0 else 0

            phoneme_mar.append({
                'phoneme': abnormal['phoneme'],
                'avg_mar': abnormal['measured_mar'],
                'expected_mar': expected_avg,
                'deviation_percent': deviation_pct,
                'is_anomalous': True,
                'explanation': f"'{abnormal['phoneme']}' 발음 시 입 벌림 이상 ({deviation_pct:.1f}% 편차)"
            })

    # Anomalous frames - geometry_analysis의 abnormal_phonemes에서 가져오기
    # (temporal_analysis가 제거되었으므로 geometry 기반으로만 구성)
    anomalous_frames = []
    if 'abnormal_phonemes' in geom_analysis:
        for abnormal in geom_analysis['abnormal_phonemes'][:5]:  # Top 5 abnormal phonemes
            anomalous_frames.append({
                'frame_index': 0,  # Frame-level 정보는 geometry_analysis에 없음
                'phoneme': abnormal.get('phoneme', ''),
                'mar_value': abnormal.get('measured_mar', 0.0),
                'expected_range': abnormal.get('expected_range', [0.0, 1.0]),
                'deviation_percent': abnormal.get('deviation_percent', 0.0),
                'severity': abnormal.get('severity', 'unknown')
            })

    geometry_analysis = {
        'statistics': {
            'mean_mar': geom_analysis.get('mean_mar', 0.0),
            'std_mar': geom_analysis.get('std_mar', 0.0),
            'min_mar': 0.0,  # Not available in current format
            'max_mar': 0.0,  # Not available in current format
            'expected_baseline': 0.39  # From Real videos baseline
        },
        'phoneme_mar': phoneme_mar,
        'anomalous_frames': anomalous_frames
    }

    # Include baseline_info if available (from statistical baseline)
    if 'baseline_info' in geom_analysis:
        geometry_analysis['baseline_info'] = geom_analysis['baseline_info']

    # Include original abnormal_phonemes for detailed analysis
    if 'abnormal_phonemes' in geom_analysis:
        geometry_analysis['abnormal_phonemes'] = geom_analysis['abnormal_phonemes']
    
    # 시각화 정보 추가
    if 'abnormal_phonemes' in geom_analysis and len(geom_analysis['abnormal_phonemes']) > 0:
        visualization_summary = []
        for phoneme_info in geom_analysis['abnormal_phonemes']:
            phoneme = phoneme_info.get('phoneme', '')
            deviation = phoneme_info.get('deviation', 0)
            deviation_percent = phoneme_info.get('deviation_percent', abs(deviation / phoneme_info.get('expected_mean', 1) * 100) if phoneme_info.get('expected_mean', 0) > 0 else 0)
            
            if deviation > 0:
                viz_desc = f"'{phoneme}' 발음 시 입을 {deviation_percent:.1f}% 더 크게 벌렸습니다"
            else:
                viz_desc = f"'{phoneme}' 발음 시 입을 {deviation_percent:.1f}% 더 작게 벌렸습니다"
            
            visualization_summary.append({
                'phoneme': phoneme,
                'description': viz_desc,
                'severity': phoneme_info.get('severity', 'unknown'),
                'deviation_percent': deviation_percent,
                'measured_mar': phoneme_info.get('measured_mar', 0),
                'expected_mar': phoneme_info.get('expected_mean', 0)
            })
        
        geometry_analysis['visualization'] = {
            'summary': visualization_summary,
            'plot_type': 'mar_deviation_bar_chart',
            'description': 'MAR Deviation 분석: 각 음소별 입 벌림 정도를 정상 범위와 비교하여 시각화'
        }

    # 6. Build Korean explanation
    korean_summary = pia_xai['summary']

    korean_explanation = {
        'summary': korean_summary['overall'],
        'key_findings': korean_summary['key_findings'].split('\n'),
        'detailed_analysis': korean_summary['reasoning']
    }

    # 7. Visualization path
    visualization = {}
    if output_dir:
        xai_plot_path = Path(output_dir) / f"interval_{interval['interval_id']}_xai.png"
        if xai_plot_path.exists():
            visualization['xai_plot_url'] = str(xai_plot_path)

    # Build final result matching TypeScript interface
    return {
        'interval_id': interval['interval_id'],
        'time_range': f"{interval['start_time_sec']:.1f}s-{interval['end_time_sec']:.1f}s",
        'prediction': prediction,
        'branch_contributions': branch_contributions,
        'phoneme_analysis': phoneme_analysis,
        'temporal_analysis': temporal_analysis,
        'geometry_analysis': geometry_analysis,
        'korean_explanation': korean_explanation,
        'visualization': visualization
    }


def resample_features_to_grid(
    features: np.ndarray,
    timestamps: np.ndarray,
    phoneme_labels: np.ndarray,  # [NEW] Frame-level labels
    target_phonemes: int = 14,
    frames_per_phoneme: int = 5,
    phonemes: Optional[List[Dict]] = None  # Deprecated
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Resample features to PIA 14×5 grid format.

    Shared utility used by both Stage 2 analyzer and preprocessing pipeline.
    Groups feature frames by phoneme and builds a fixed-size grid.
    
    [UPDATED] Now uses frame-level phoneme labels directly to match training behavior.

    Args:
        features: (N, D) - Feature array (e.g., MAR, ArcFace embeddings)
        timestamps: (N,) - Frame timestamps in seconds
        phoneme_labels: (N,) - Frame-level phoneme labels
        target_phonemes: Number of phonemes in grid (default: 14)
        frames_per_phoneme: Frames per phoneme slot (default: 5)
        phonemes: Deprecated, kept for backward compatibility

    Returns:
        resampled: (14, 5, D) - Feature grid aligned to phoneme vocabulary
        matched_phonemes: List[str] (length 14) - Phoneme labels or '<pad>'
        valid_mask: (14,) bool array - True if phoneme has actual features, False if padding
    """
    from ..utils.korean_phoneme_config import get_phoneme_vocab, is_kept_phoneme

    phoneme_vocab = get_phoneme_vocab()
    N, D = features.shape

    # Group features by phoneme
    by_phoneme = {p: [] for p in phoneme_vocab}

    # [NEW] Use frame-level labels directly (matches KoreanPhonemeDataset behavior)
    for frame_idx in range(len(features)):
        # Handle potential length mismatch
        if frame_idx >= len(phoneme_labels):
            break

        phoneme = str(phoneme_labels[frame_idx]).strip()

        if is_kept_phoneme(phoneme):
            by_phoneme[phoneme].append(features[frame_idx])

    # Build 14×5 grid with valid mask tracking
    resampled = np.zeros((target_phonemes, frames_per_phoneme, D), dtype=features.dtype)
    matched_phonemes = []
    valid_mask = np.zeros(target_phonemes, dtype=bool)

    for pi, phoneme in enumerate(phoneme_vocab):
        frames_list = by_phoneme[phoneme][:frames_per_phoneme]

        if len(frames_list) > 0:
            matched_phonemes.append(phoneme)
            valid_mask[pi] = True
        else:
            matched_phonemes.append('<pad>')
            valid_mask[pi] = False

        for fi, frame_feat in enumerate(frames_list):
            resampled[pi, fi] = frame_feat

    # Post-process: Handle NaN in features with phoneme-wise interpolation

    # Post-process: Handle NaN in features with phoneme-wise interpolation
    # This matches the training behavior in phoneme_dataset.py
    # For each phoneme, replace NaN with the mean of valid frames in that phoneme
    for pi in range(target_phonemes):
        for di in range(D):
            # Get values for this phoneme and dimension
            values = resampled[pi, :, di]

            # Find non-NaN values
            valid_mask_nan = ~np.isnan(values)

            if valid_mask_nan.any():
                # Compute mean of valid frames for this phoneme
                phoneme_mean = values[valid_mask_nan].mean()

                # Replace NaN with phoneme-specific mean
                nan_mask = np.isnan(values)
                if nan_mask.any():
                    values[nan_mask] = phoneme_mean
                    resampled[pi, :, di] = values
            else:
                # Entire phoneme has NaN → fill with 0 (will be masked out)
                resampled[pi, :, di] = 0.0

    return resampled, matched_phonemes, valid_mask
