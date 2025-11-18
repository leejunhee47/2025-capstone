"""
Hybrid MMMS-BA + PIA XAI Output Validator

Validates JSON output against the TypeScript interface defined in:
mobile_deepfake_detector/plans/hybrid_xai_interface.ts

This validator ensures that the Python pipeline outputs match the exact
schema expected by the TypeScript frontend/client.

Created: 2025-11-17
"""

import json
import sys
from pathlib import Path
from typing import TypedDict, Literal, List, Dict, Any, Optional, Tuple
from datetime import datetime


# ===================================
# TypedDict Schema (from TypeScript)
# ===================================

class Metadata(TypedDict):
    video_id: str
    request_id: str
    processed_at: str  # ISO 8601
    processing_time_ms: float
    pipeline_version: str


class VideoInfo(TypedDict):
    duration_sec: float
    total_frames: int
    fps: float
    resolution: str
    original_path: str


class Probabilities(TypedDict):
    real: float
    fake: float


class Detection(TypedDict):
    verdict: Literal["real", "fake"]
    confidence: float
    probabilities: Probabilities
    suspicious_frame_count: int
    suspicious_frame_ratio: float


class Summary(TypedDict):
    title: str
    risk_level: Literal["low", "medium", "high", "critical"]
    primary_reason: str
    suspicious_interval_count: int
    top_suspicious_phonemes: List[str]
    detailed_explanation: str


class FrameProbability(TypedDict):
    frame_index: int
    timestamp_sec: float
    fake_probability: float
    is_suspicious: bool


class SuspiciousInterval(TypedDict):
    interval_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    frame_count: int
    mean_fake_prob: float
    max_fake_prob: float
    severity: Literal["low", "medium", "high", "critical"]


class TimelineStatistics(TypedDict):
    mean_fake_prob: float
    std_fake_prob: float
    max_fake_prob: float
    min_fake_prob: float
    threshold_used: float


class TimelineVisualization(TypedDict, total=False):
    timeline_plot_url: Optional[str]


class Stage1Timeline(TypedDict):
    frame_probabilities: List[FrameProbability]
    suspicious_intervals: List[SuspiciousInterval]
    statistics: TimelineStatistics
    visualization: TimelineVisualization


class BranchContribution(TypedDict):
    contribution_percent: float
    l2_norm: float
    rank: int


class BranchContributions(TypedDict):
    visual: BranchContribution
    geometry: BranchContribution
    identity: BranchContribution
    top_branch: Literal["visual", "geometry", "identity"]
    explanation: str


class PhonemeScore(TypedDict):
    phoneme_ipa: str
    phoneme_korean: str
    attention_weight: float
    rank: int
    frame_count: int
    is_suspicious: bool
    explanation: str


class PhonemeAnalysis(TypedDict):
    phoneme_scores: List[PhonemeScore]
    top_phoneme: str
    total_phonemes: int


class HighRiskPoint(TypedDict):
    phoneme_index: int
    frame_index: int
    phoneme: str
    fake_probability: float


class TemporalStatistics(TypedDict):
    mean_fake_prob: float
    max_fake_prob: float
    high_risk_count: int


class TemporalAnalysis(TypedDict):
    heatmap_data: List[List[float]]  # 14×5 matrix
    high_risk_points: List[HighRiskPoint]
    statistics: TemporalStatistics


class GeometryStatistics(TypedDict):
    mean_mar: float
    std_mar: float
    min_mar: float
    max_mar: float
    expected_baseline: float


class PhonemeMAR(TypedDict):
    phoneme: str
    avg_mar: float
    expected_mar: float
    deviation_percent: float
    is_anomalous: bool
    explanation: str


class AnomalousFrame(TypedDict):
    frame_index: int
    phoneme: str
    mar_value: float
    expected_range: Tuple[float, float]
    deviation_percent: float
    severity: Literal["low", "medium", "high", "critical"]


class GeometryAnalysis(TypedDict):
    statistics: GeometryStatistics
    phoneme_mar: List[PhonemeMAR]
    anomalous_frames: List[AnomalousFrame]


class KoreanExplanation(TypedDict):
    summary: str
    key_findings: List[str]
    detailed_analysis: str


class IntervalVisualization(TypedDict, total=False):
    xai_plot_url: Optional[str]


class IntervalPrediction(TypedDict):
    verdict: Literal["real", "fake"]
    confidence: float
    probabilities: Probabilities


class Stage2IntervalAnalysis(TypedDict):
    interval_id: int
    time_range: str
    prediction: IntervalPrediction
    branch_contributions: BranchContributions
    phoneme_analysis: PhonemeAnalysis
    temporal_analysis: TemporalAnalysis
    geometry_analysis: GeometryAnalysis
    korean_explanation: KoreanExplanation
    visualization: IntervalVisualization


class TopSuspiciousPhoneme(TypedDict):
    phoneme: str
    avg_attention: float
    appearance_count: int
    intervals: List[int]


class BranchTrends(TypedDict):
    visual_avg: float
    geometry_avg: float
    identity_avg: float
    most_dominant: Literal["visual", "geometry", "identity"]


class MARSummary(TypedDict):
    intervals_with_anomalies: int
    total_anomalous_frames: int
    avg_deviation_percent: float


class AggregatedInsights(TypedDict):
    top_suspicious_phonemes: List[TopSuspiciousPhoneme]
    branch_trends: BranchTrends
    mar_summary: MARSummary


class MMSBAModelInfo(TypedDict):
    name: str
    checkpoint: str
    training_accuracy: float
    stage: str


class PIAModelInfo(TypedDict):
    name: str
    checkpoint: str
    training_accuracy: float
    stage: str
    supported_phonemes: int
    xai_methods: List[str]


class PipelineInfo(TypedDict):
    version: str
    threshold: float
    resampling_strategy: str


class ModelInfo(TypedDict):
    mmms_ba: MMSBAModelInfo
    pia: PIAModelInfo
    pipeline: PipelineInfo


class Outputs(TypedDict):
    stage1_timeline: str
    stage2_intervals: List[str]
    combined_json: str
    combined_summary: str


class HybridDeepfakeXAIResult(TypedDict):
    metadata: Metadata
    video_info: VideoInfo
    detection: Detection
    summary: Summary
    stage1_timeline: Stage1Timeline
    stage2_interval_analysis: List[Stage2IntervalAnalysis]
    aggregated_insights: AggregatedInsights
    model_info: ModelInfo
    outputs: Outputs


# ===================================
# Validation Functions
# ===================================

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Validate metadata section"""
    errors = []

    required_fields = ["video_id", "request_id", "processed_at", "processing_time_ms", "pipeline_version"]
    for field in required_fields:
        if field not in metadata:
            errors.append(f"metadata.{field} is missing")

    # Validate ISO 8601 timestamp
    if "processed_at" in metadata:
        try:
            datetime.fromisoformat(metadata["processed_at"].replace('Z', '+00:00'))
        except ValueError:
            errors.append(f"metadata.processed_at must be ISO 8601 format, got: {metadata['processed_at']}")

    # Validate processing time
    if "processing_time_ms" in metadata and not isinstance(metadata["processing_time_ms"], (int, float)):
        errors.append(f"metadata.processing_time_ms must be number, got: {type(metadata['processing_time_ms'])}")

    return errors


def validate_video_info(video_info: Dict[str, Any]) -> List[str]:
    """Validate video_info section"""
    errors = []

    required_fields = ["duration_sec", "total_frames", "fps", "resolution", "original_path"]
    for field in required_fields:
        if field not in video_info:
            errors.append(f"video_info.{field} is missing")

    # Validate numeric fields
    if "duration_sec" in video_info and not isinstance(video_info["duration_sec"], (int, float)):
        errors.append(f"video_info.duration_sec must be number")

    if "total_frames" in video_info and not isinstance(video_info["total_frames"], int):
        errors.append(f"video_info.total_frames must be integer")

    if "fps" in video_info and not isinstance(video_info["fps"], (int, float)):
        errors.append(f"video_info.fps must be number")

    # Validate resolution format (e.g., "1080x1920")
    if "resolution" in video_info:
        res = video_info["resolution"]
        if not isinstance(res, str) or 'x' not in res:
            errors.append(f"video_info.resolution must be format 'WxH', got: {res}")

    return errors


def validate_detection(detection: Dict[str, Any]) -> List[str]:
    """Validate detection section"""
    errors = []

    # Validate verdict
    if "verdict" not in detection:
        errors.append("detection.verdict is missing")
    elif detection["verdict"] not in ["real", "fake"]:
        errors.append(f"detection.verdict must be 'real' or 'fake', got: {detection['verdict']}")

    # Validate confidence (0~1)
    if "confidence" in detection:
        conf = detection["confidence"]
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            errors.append(f"detection.confidence must be 0~1, got: {conf}")

    # Validate probabilities
    if "probabilities" in detection:
        probs = detection["probabilities"]
        if "real" not in probs or "fake" not in probs:
            errors.append("detection.probabilities must have 'real' and 'fake'")
        else:
            if not (0 <= probs["real"] <= 1):
                errors.append(f"detection.probabilities.real must be 0~1, got: {probs['real']}")
            if not (0 <= probs["fake"] <= 1):
                errors.append(f"detection.probabilities.fake must be 0~1, got: {probs['fake']}")
            # Should sum to ~1.0
            prob_sum = probs["real"] + probs["fake"]
            if not (0.99 <= prob_sum <= 1.01):
                errors.append(f"detection.probabilities should sum to 1.0, got: {prob_sum}")

    # Validate suspicious_frame_ratio (0~100)
    if "suspicious_frame_ratio" in detection:
        ratio = detection["suspicious_frame_ratio"]
        if not isinstance(ratio, (int, float)) or not (0 <= ratio <= 100):
            errors.append(f"detection.suspicious_frame_ratio must be 0~100, got: {ratio}")

    return errors


def validate_summary(summary: Dict[str, Any]) -> List[str]:
    """Validate summary section"""
    errors = []

    required_fields = ["title", "risk_level", "primary_reason", "suspicious_interval_count",
                      "top_suspicious_phonemes", "detailed_explanation"]
    for field in required_fields:
        if field not in summary:
            errors.append(f"summary.{field} is missing")

    # Validate risk_level
    if "risk_level" in summary:
        if summary["risk_level"] not in ["low", "medium", "high", "critical"]:
            errors.append(f"summary.risk_level must be low/medium/high/critical, got: {summary['risk_level']}")

    # Validate top_suspicious_phonemes is array
    if "top_suspicious_phonemes" in summary:
        if not isinstance(summary["top_suspicious_phonemes"], list):
            errors.append(f"summary.top_suspicious_phonemes must be array")

    return errors


def validate_stage1_timeline(stage1: Dict[str, Any]) -> List[str]:
    """Validate stage1_timeline section"""
    errors = []

    # Validate frame_probabilities array
    if "frame_probabilities" not in stage1:
        errors.append("stage1_timeline.frame_probabilities is missing")
    else:
        frame_probs = stage1["frame_probabilities"]
        if not isinstance(frame_probs, list):
            errors.append("stage1_timeline.frame_probabilities must be array")
        else:
            # Validate first few entries
            for i, fp in enumerate(frame_probs[:5]):
                required = ["frame_index", "timestamp_sec", "fake_probability", "is_suspicious"]
                for field in required:
                    if field not in fp:
                        errors.append(f"stage1_timeline.frame_probabilities[{i}].{field} is missing")

                if "fake_probability" in fp:
                    prob = fp["fake_probability"]
                    if not (0 <= prob <= 1):
                        errors.append(f"stage1_timeline.frame_probabilities[{i}].fake_probability must be 0~1, got: {prob}")

    # Validate suspicious_intervals array
    if "suspicious_intervals" not in stage1:
        errors.append("stage1_timeline.suspicious_intervals is missing")
    else:
        intervals = stage1["suspicious_intervals"]
        if not isinstance(intervals, list):
            errors.append("stage1_timeline.suspicious_intervals must be array")
        else:
            for i, interval in enumerate(intervals):
                required = ["interval_id", "start_frame", "end_frame", "start_time_sec",
                           "end_time_sec", "duration_sec", "frame_count", "mean_fake_prob",
                           "max_fake_prob", "severity"]
                for field in required:
                    if field not in interval:
                        errors.append(f"stage1_timeline.suspicious_intervals[{i}].{field} is missing")

                # Validate severity
                if "severity" in interval:
                    if interval["severity"] not in ["low", "medium", "high", "critical"]:
                        errors.append(f"stage1_timeline.suspicious_intervals[{i}].severity invalid: {interval['severity']}")

    # Validate statistics
    if "statistics" not in stage1:
        errors.append("stage1_timeline.statistics is missing")
    else:
        stats = stage1["statistics"]
        required = ["mean_fake_prob", "std_fake_prob", "max_fake_prob", "min_fake_prob", "threshold_used"]
        for field in required:
            if field not in stats:
                errors.append(f"stage1_timeline.statistics.{field} is missing")

    return errors


def validate_stage2_interval_analysis(stage2_array: List[Dict[str, Any]]) -> List[str]:
    """Validate stage2_interval_analysis section"""
    errors = []

    if not isinstance(stage2_array, list):
        errors.append("stage2_interval_analysis must be array")
        return errors

    for i, interval in enumerate(stage2_array):
        # Validate prediction
        if "prediction" not in interval:
            errors.append(f"stage2_interval_analysis[{i}].prediction is missing")
        else:
            pred = interval["prediction"]
            if "verdict" in pred and pred["verdict"] not in ["real", "fake"]:
                errors.append(f"stage2_interval_analysis[{i}].prediction.verdict invalid: {pred['verdict']}")

        # Validate branch_contributions
        if "branch_contributions" not in interval:
            errors.append(f"stage2_interval_analysis[{i}].branch_contributions is missing")
        else:
            branches = interval["branch_contributions"]
            for branch in ["visual", "geometry", "identity"]:
                if branch not in branches:
                    errors.append(f"stage2_interval_analysis[{i}].branch_contributions.{branch} is missing")

            if "top_branch" in branches:
                if branches["top_branch"] not in ["visual", "geometry", "identity"]:
                    errors.append(f"stage2_interval_analysis[{i}].branch_contributions.top_branch invalid")

        # Validate phoneme_analysis
        if "phoneme_analysis" not in interval:
            errors.append(f"stage2_interval_analysis[{i}].phoneme_analysis is missing")
        else:
            phoneme = interval["phoneme_analysis"]
            if "phoneme_scores" not in phoneme:
                errors.append(f"stage2_interval_analysis[{i}].phoneme_analysis.phoneme_scores is missing")
            elif not isinstance(phoneme["phoneme_scores"], list):
                errors.append(f"stage2_interval_analysis[{i}].phoneme_analysis.phoneme_scores must be array")

        # Validate temporal_analysis
        if "temporal_analysis" not in interval:
            errors.append(f"stage2_interval_analysis[{i}].temporal_analysis is missing")
        else:
            temporal = interval["temporal_analysis"]
            if "heatmap_data" not in temporal:
                errors.append(f"stage2_interval_analysis[{i}].temporal_analysis.heatmap_data is missing")
            elif not isinstance(temporal["heatmap_data"], list):
                errors.append(f"stage2_interval_analysis[{i}].temporal_analysis.heatmap_data must be array")
            else:
                # Validate 14×5 matrix shape
                heatmap = temporal["heatmap_data"]
                if len(heatmap) != 14:
                    errors.append(f"stage2_interval_analysis[{i}].temporal_analysis.heatmap_data must have 14 rows, got: {len(heatmap)}")
                elif len(heatmap[0]) != 5:
                    errors.append(f"stage2_interval_analysis[{i}].temporal_analysis.heatmap_data rows must have 5 cols, got: {len(heatmap[0])}")

        # Validate geometry_analysis
        if "geometry_analysis" not in interval:
            errors.append(f"stage2_interval_analysis[{i}].geometry_analysis is missing")

        # Validate korean_explanation
        if "korean_explanation" not in interval:
            errors.append(f"stage2_interval_analysis[{i}].korean_explanation is missing")
        else:
            korean = interval["korean_explanation"]
            required = ["summary", "key_findings", "detailed_analysis"]
            for field in required:
                if field not in korean:
                    errors.append(f"stage2_interval_analysis[{i}].korean_explanation.{field} is missing")

    return errors


def validate_aggregated_insights(aggregated: Dict[str, Any]) -> List[str]:
    """Validate aggregated_insights section"""
    errors = []

    required_fields = ["top_suspicious_phonemes", "branch_trends", "mar_summary"]
    for field in required_fields:
        if field not in aggregated:
            errors.append(f"aggregated_insights.{field} is missing")

    # Validate branch_trends.most_dominant
    if "branch_trends" in aggregated:
        trends = aggregated["branch_trends"]
        if "most_dominant" in trends:
            if trends["most_dominant"] not in ["visual", "geometry", "identity"]:
                errors.append(f"aggregated_insights.branch_trends.most_dominant invalid: {trends['most_dominant']}")

    return errors


def validate_model_info(model_info: Dict[str, Any]) -> List[str]:
    """Validate model_info section"""
    errors = []

    if "mmms_ba" not in model_info:
        errors.append("model_info.mmms_ba is missing")

    if "pia" not in model_info:
        errors.append("model_info.pia is missing")
    else:
        pia = model_info["pia"]
        if "xai_methods" in pia:
            if not isinstance(pia["xai_methods"], list):
                errors.append("model_info.pia.xai_methods must be array")

    if "pipeline" not in model_info:
        errors.append("model_info.pipeline is missing")

    return errors


def validate_outputs(outputs: Dict[str, Any]) -> List[str]:
    """Validate outputs section"""
    errors = []

    required_fields = ["stage1_timeline", "stage2_intervals", "combined_json", "combined_summary"]
    for field in required_fields:
        if field not in outputs:
            errors.append(f"outputs.{field} is missing")

    # Validate stage2_intervals is array
    if "stage2_intervals" in outputs:
        if not isinstance(outputs["stage2_intervals"], list):
            errors.append("outputs.stage2_intervals must be array")

    return errors


def validate_hybrid_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate complete HybridDeepfakeXAIResult

    Args:
        result: Dictionary containing the pipeline output

    Returns:
        (is_valid, errors): Tuple of boolean and list of error messages
    """
    errors = []

    # Check top-level sections
    required_sections = [
        "metadata", "video_info", "detection", "summary",
        "stage1_timeline", "stage2_interval_analysis",
        "aggregated_insights", "model_info", "outputs"
    ]

    for section in required_sections:
        if section not in result:
            errors.append(f"Top-level section '{section}' is missing")

    # Validate each section
    if "metadata" in result:
        errors.extend(validate_metadata(result["metadata"]))

    if "video_info" in result:
        errors.extend(validate_video_info(result["video_info"]))

    if "detection" in result:
        errors.extend(validate_detection(result["detection"]))

    if "summary" in result:
        errors.extend(validate_summary(result["summary"]))

    if "stage1_timeline" in result:
        errors.extend(validate_stage1_timeline(result["stage1_timeline"]))

    if "stage2_interval_analysis" in result:
        errors.extend(validate_stage2_interval_analysis(result["stage2_interval_analysis"]))

    if "aggregated_insights" in result:
        errors.extend(validate_aggregated_insights(result["aggregated_insights"]))

    if "model_info" in result:
        errors.extend(validate_model_info(result["model_info"]))

    if "outputs" in result:
        errors.extend(validate_outputs(result["outputs"]))

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_json_file(json_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a JSON file containing HybridDeepfakeXAIResult

    Args:
        json_path: Path to JSON file

    Returns:
        (is_valid, errors): Tuple of boolean and list of error messages
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        return validate_hybrid_result(result)

    except FileNotFoundError:
        return False, [f"File not found: {json_path}"]
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {str(e)}"]
    except Exception as e:
        return False, [f"Unexpected error: {str(e)}"]


# ===================================
# CLI Interface
# ===================================

def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate Hybrid MMMS-BA + PIA XAI output JSON against TypeScript interface"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to JSON file to validate"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all validation errors (default: show first 20)"
    )

    args = parser.parse_args()

    # Validate file
    is_valid, errors = validate_json_file(args.json_file)

    # Print results
    print(f"\n{'='*70}")
    print(f"Validation Result: {args.json_file}")
    print(f"{'='*70}\n")

    if is_valid:
        print("✅ VALID - Output matches hybrid_xai_interface.ts\n")
        return 0
    else:
        print(f"❌ INVALID - Found {len(errors)} error(s):\n")

        # Limit errors shown unless verbose
        max_errors = None if args.verbose else 20
        for i, error in enumerate(errors[:max_errors], 1):
            print(f"  {i}. {error}")

        if not args.verbose and len(errors) > 20:
            print(f"\n  ... and {len(errors) - 20} more errors (use --verbose to see all)\n")

        return 1


if __name__ == "__main__":
    sys.exit(main())