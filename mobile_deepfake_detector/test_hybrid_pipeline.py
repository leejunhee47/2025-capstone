"""
Hybrid MMMS-BA + PIA XAI Pipeline E2E Tests

Comprehensive test suite for the 2-stage hybrid deepfake detection pipeline.

Test Coverage:
- Stage1Scanner (MMMS-BA temporal scanning)
- Stage2Analyzer (PIA XAI analysis)
- HybridXAIPipeline (full orchestration)
- Output format validation
- Performance benchmarks

Created: 2025-11-17
"""

import pytest
import json
import time
from pathlib import Path
from typing import Dict, Any

# Import pipeline components (using refactored modules)
from src.xai.hybrid_pipeline import HybridXAIPipeline
from src.xai.stage1_scanner import Stage1Scanner
from src.xai.stage2_analyzer import Stage2Analyzer

# Import validator
from validate_hybrid_output import validate_hybrid_result


# ===================================
# Test Configuration
# ===================================

# Test video paths (adjust to your dataset)
TEST_VIDEO_REAL = Path("E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터/train_탐지방해/원본영상/02_원본영상/02_원본영상/13968_019.mp4")
TEST_VIDEO_FAKE = Path("E:/capstone/real_deepfake_dataset/youtube_3.mp4")  # youtube_3.mp4 (fake video)

# Model checkpoints (adjust to your trained models)
MMMS_BA_CHECKPOINT = Path("models/checkpoints/mmms-ba_fulldata_best.pth")
MMMS_BA_CONFIG = Path("configs/train_teacher_korean.yaml")

PIA_CHECKPOINT = Path("F:/preprocessed_data_pia_optimized/checkpoints/best.pth")
PIA_CONFIG = Path("configs/train_pia.yaml")

# Output directory for test results
TEST_OUTPUT_DIR = Path("E:/capstone/mobile_deepfake_detector/outputs/xai/hybrid_tests")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================================
# Fixtures
# ===================================

@pytest.fixture(scope="module")
def stage1_scanner():
    """Initialize Stage1Scanner once for all tests"""
    if not MMMS_BA_CHECKPOINT.exists():
        pytest.skip(f"MMMS-BA checkpoint not found: {MMMS_BA_CHECKPOINT}")

    scanner = Stage1Scanner(
        model_path=str(MMMS_BA_CHECKPOINT),
        config_path=str(MMMS_BA_CONFIG),
        device="cuda"
    )
    return scanner


@pytest.fixture(scope="module")
def stage2_analyzer():
    """Initialize Stage2Analyzer once for all tests"""
    if not PIA_CHECKPOINT.exists():
        pytest.skip(f"PIA checkpoint not found: {PIA_CHECKPOINT}")

    analyzer = Stage2Analyzer(
        pia_model_path=str(PIA_CHECKPOINT),
        pia_config_path=str(PIA_CONFIG),
        device="cuda"
    )
    return analyzer


@pytest.fixture(scope="module")
def hybrid_pipeline():
    """Initialize HybridXAIPipeline once for all tests"""
    if not MMMS_BA_CHECKPOINT.exists():
        pytest.skip(f"MMMS-BA checkpoint not found: {MMMS_BA_CHECKPOINT}")
    if not PIA_CHECKPOINT.exists():
        pytest.skip(f"PIA checkpoint not found: {PIA_CHECKPOINT}")

    pipeline = HybridXAIPipeline(
        mmms_model_path=str(MMMS_BA_CHECKPOINT),
        pia_model_path=str(PIA_CHECKPOINT),
        mmms_config_path=str(MMMS_BA_CONFIG),
        pia_config_path=str(PIA_CONFIG),
        device="cuda"
    )
    return pipeline


@pytest.fixture(scope="module")
def stage1_result_with_intervals(stage1_scanner):
    """
    Stage1 결과와 첫 번째 suspicious interval을 반환하는 fixture.
    Stage2 테스트들에서 재사용하여 중복 코드 제거 및 실행 시간 단축.
    """
    if not TEST_VIDEO_FAKE.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")
    
    stage1_result = stage1_scanner.scan_video(
        video_path=str(TEST_VIDEO_FAKE),
        threshold=0.6
    )
    
    if len(stage1_result["suspicious_intervals"]) == 0:
        pytest.skip("No suspicious intervals detected")
    
    extracted_features = stage1_result.get('extracted_features', {})
    if not extracted_features:
        pytest.skip("extracted_features not found in stage1_result")
    
    return {
        'stage1_result': stage1_result,
        'interval': stage1_result["suspicious_intervals"][0],
        'extracted_features': extracted_features
    }


# ===================================
# Stage 1 Tests
# ===================================

class TestStage1Scanner:
    """Test Stage1Scanner (MMMS-BA temporal scanning)"""

    def test_scan_real_video(self, stage1_scanner):
        """Test Stage1 on REAL video"""
        if not TEST_VIDEO_REAL.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_REAL}")

        result = stage1_scanner.scan_video(
            video_path=str(TEST_VIDEO_REAL),
            threshold=0.6,
            output_dir=str(TEST_OUTPUT_DIR / "stage1_real")
        )

        # Basic structure validation
        assert "frame_probabilities" in result
        assert "suspicious_intervals" in result
        assert "statistics" in result

        # Validate frame probabilities
        frame_probs = result["frame_probabilities"]
        assert len(frame_probs) > 0
        assert all("frame_index" in fp for fp in frame_probs)
        assert all("fake_probability" in fp for fp in frame_probs)
        assert all(0 <= fp["fake_probability"] <= 1 for fp in frame_probs)

        # For REAL videos, expect low mean fake probability
        mean_fake_prob = result["statistics"]["mean_fake_prob"]
        print(f"\n[REAL Video] Mean Fake Probability: {mean_fake_prob:.3f}")

        # Save result for inspection (exclude numpy arrays for JSON serialization)
        output_path = TEST_OUTPUT_DIR / "stage1_real_result.json"
        result_for_json = {k: v for k, v in result.items() if k != 'extracted_features'}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_for_json, f, indent=2, ensure_ascii=False)
        print(f"Saved Stage1 REAL result to: {output_path}")

    def test_scan_fake_video(self, stage1_scanner):
        """Test Stage1 on FAKE video"""
        if not TEST_VIDEO_FAKE.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")

        result = stage1_scanner.scan_video(
            video_path=str(TEST_VIDEO_FAKE),
            threshold=0.6,
            output_dir=str(TEST_OUTPUT_DIR / "stage1_fake")
        )

        # Basic structure validation
        assert "frame_probabilities" in result
        assert "suspicious_intervals" in result
        assert "statistics" in result

        # For FAKE videos, expect at least some suspicious intervals
        suspicious_intervals = result["suspicious_intervals"]
        mean_fake_prob = result["statistics"]["mean_fake_prob"]
        print(f"\n[FAKE Video] Mean Fake Probability: {mean_fake_prob:.3f}")
        print(f"[FAKE Video] Suspicious Intervals Detected: {len(suspicious_intervals)}")

        # Save result for inspection (exclude numpy arrays for JSON serialization)
        output_path = TEST_OUTPUT_DIR / "stage1_fake_result.json"
        result_for_json = {k: v for k, v in result.items() if k != 'extracted_features'}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_for_json, f, indent=2, ensure_ascii=False)
        print(f"Saved Stage1 FAKE result to: {output_path}")

    def test_suspicious_interval_format(self, stage1_scanner):
        """Test that suspicious intervals have correct format"""
        if not TEST_VIDEO_FAKE.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")

        result = stage1_scanner.scan_video(
            video_path=str(TEST_VIDEO_FAKE),
            threshold=0.6
        )

        suspicious_intervals = result["suspicious_intervals"]
        if len(suspicious_intervals) > 0:
            interval = suspicious_intervals[0]

            # Required fields
            required_fields = [
                "interval_id", "start_frame", "end_frame",
                "start_time_sec", "end_time_sec", "duration_sec",
                "frame_count", "mean_fake_prob", "max_fake_prob", "severity"
            ]
            for field in required_fields:
                assert field in interval, f"Missing field: {field}"

            # Validate severity
            assert interval["severity"] in ["low", "medium", "high", "critical"]

            # Validate frame counts
            assert interval["end_frame"] > interval["start_frame"]
            assert interval["frame_count"] == interval["end_frame"] - interval["start_frame"] + 1

    def test_visualization_creation(self, stage1_scanner):
        """Test that Stage1 creates timeline visualization"""
        if not TEST_VIDEO_FAKE.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")

        output_dir = TEST_OUTPUT_DIR / "stage1_viz_test"
        result = stage1_scanner.scan_video(
            video_path=str(TEST_VIDEO_FAKE),
            threshold=0.6,
            output_dir=str(output_dir)
        )

        # Check visualization field
        assert "visualization" in result

        # Check if plot was created
        if "timeline_plot_url" in result["visualization"]:
            plot_path = Path(result["visualization"]["timeline_plot_url"])
            assert plot_path.exists(), f"Timeline plot not created: {plot_path}"
            print(f"\n[OK] Stage1 visualization created: {plot_path}")


# ===================================
# Stage 2 Tests
# ===================================

class TestStage2Analyzer:
    """Test Stage2Analyzer (PIA XAI analysis)"""

    def test_analyze_interval_structure(self, stage2_analyzer, stage1_result_with_intervals):
        """Test Stage2 output structure on a suspicious interval"""
        interval = stage1_result_with_intervals['interval']
        extracted_features = stage1_result_with_intervals['extracted_features']
        output_dir = TEST_OUTPUT_DIR / "stage2_test"

        result = stage2_analyzer.analyze_interval(
            interval=interval,
            video_path=str(TEST_VIDEO_FAKE),
            extracted_features=extracted_features,
            output_dir=str(output_dir)
        )

        # Validate top-level structure
        required_sections = [
            "interval_id", "time_range", "prediction",
            "branch_contributions", "phoneme_analysis",
            "temporal_analysis", "geometry_analysis",
            "korean_explanation", "visualization"
        ]
        for section in required_sections:
            assert section in result, f"Missing section: {section}"

        # Validate prediction
        pred = result["prediction"]
        assert pred["verdict"] in ["real", "fake"]
        assert 0 <= pred["confidence"] <= 1

        # Validate branch contributions
        branches = result["branch_contributions"]
        assert "visual" in branches
        assert "geometry" in branches
        assert "identity" in branches
        assert branches["top_branch"] in ["visual", "geometry", "identity"]

        # Validate temporal analysis heatmap (14×5)
        heatmap = result["temporal_analysis"]["heatmap_data"]
        assert len(heatmap) == 14, f"Heatmap should have 14 rows, got {len(heatmap)}"
        assert len(heatmap[0]) == 5, f"Heatmap rows should have 5 cols, got {len(heatmap[0])}"

        # Save result
        output_path = TEST_OUTPUT_DIR / "stage2_interval_result.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Stage2 result saved to: {output_path}")

    def test_phoneme_analysis(self, stage2_analyzer, stage1_result_with_intervals):
        """Test phoneme analysis output"""
        interval = stage1_result_with_intervals['interval']
        extracted_features = stage1_result_with_intervals['extracted_features']
        
        result = stage2_analyzer.analyze_interval(
            interval=interval,
            video_path=str(TEST_VIDEO_FAKE),
            extracted_features=extracted_features
        )

        # Validate phoneme analysis
        phoneme_analysis = result["phoneme_analysis"]
        assert "phoneme_scores" in phoneme_analysis
        assert "top_phoneme" in phoneme_analysis
        assert "total_phonemes" in phoneme_analysis

        # Validate phoneme scores structure
        phoneme_scores = phoneme_analysis["phoneme_scores"]
        assert len(phoneme_scores) > 0

        for score in phoneme_scores:
            assert "phoneme_ipa" in score
            assert "phoneme_korean" in score
            assert "attention_weight" in score
            assert "rank" in score
            assert 0 <= score["attention_weight"] <= 1

        print(f"\n[Phoneme Analysis] Top Phoneme: {phoneme_analysis['top_phoneme']}")
        print(f"[Phoneme Analysis] Total Phonemes: {phoneme_analysis['total_phonemes']}")

    def test_korean_explanation(self, stage2_analyzer, stage1_result_with_intervals):
        """Test Korean explanation generation"""
        interval = stage1_result_with_intervals['interval']
        extracted_features = stage1_result_with_intervals['extracted_features']
        
        result = stage2_analyzer.analyze_interval(
            interval=interval,
            video_path=str(TEST_VIDEO_FAKE),
            extracted_features=extracted_features
        )

        # Validate Korean explanation
        korean = result["korean_explanation"]
        assert "summary" in korean
        assert "key_findings" in korean
        assert "detailed_analysis" in korean

        assert isinstance(korean["summary"], str)
        assert isinstance(korean["key_findings"], list)
        assert isinstance(korean["detailed_analysis"], str)

        print(f"\n[Korean Explanation] Summary: {korean['summary']}")
        print(f"[Korean Explanation] Key Findings: {korean['key_findings']}")


# ===================================
# Full Pipeline Tests (Agent 3)
# ===================================

class TestHybridXAIPipeline:
    """Test HybridXAIPipeline (full E2E orchestration)"""

    def test_full_pipeline_real_video(self, hybrid_pipeline):
        """Test full pipeline on REAL video"""
        if not TEST_VIDEO_REAL.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_REAL}")

        result = hybrid_pipeline.process_video(
            video_path=str(TEST_VIDEO_REAL),
            video_id="test_real",
            threshold=0.6,
            save_visualizations=True
        )

        # Validate structure
        assert "metadata" in result
        assert "detection" in result
        assert "summary" in result
        assert result["detection"]["verdict"] in ["real", "fake"]

        print(f"\n[REAL Video Pipeline] Verdict: {result['detection']['verdict']}")
        print(f"[REAL Video Pipeline] Confidence: {result['detection']['confidence']:.1%}")

    def test_full_pipeline_fake_video(self, hybrid_pipeline):
        """Test full pipeline on FAKE video"""
        if not TEST_VIDEO_FAKE.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")

        result = hybrid_pipeline.process_video(
            video_path=str(TEST_VIDEO_FAKE),
            video_id="test_fake",
            threshold=0.6,
            save_visualizations=True
        )

        # Validate structure
        assert "metadata" in result
        assert "detection" in result
        assert "summary" in result
        assert result["detection"]["verdict"] in ["real", "fake"]

        print(f"\n[FAKE Video Pipeline] Verdict: {result['detection']['verdict']}")
        print(f"[FAKE Video Pipeline] Confidence: {result['detection']['confidence']:.1%}")
        print(f"[FAKE Video Pipeline] Risk Level: {result['summary']['risk_level']}")

    def test_output_format_validation(self, hybrid_pipeline):
        """Test that full pipeline output matches TypeScript interface"""
        if not TEST_VIDEO_FAKE.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")

        result = hybrid_pipeline.process_video(
            video_path=str(TEST_VIDEO_FAKE),
            video_id="test_validation",
            threshold=0.6
        )

        # Validate against TypeScript interface
        is_valid, errors = validate_hybrid_result(result)

        if not is_valid:
            print(f"\n[ERROR] Validation errors:")
            for error in errors[:10]:
                print(f"  - {error}")

        assert is_valid, f"Output validation failed: {errors}"
        print(f"\n[OK] Full pipeline output is valid")

    def test_aggregated_insights(self, hybrid_pipeline):
        """Test aggregated insights across all intervals"""
        if not TEST_VIDEO_FAKE.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")

        result = hybrid_pipeline.process_video(
            video_path=str(TEST_VIDEO_FAKE),
            video_id="test_aggregated",
            threshold=0.6
        )

        # Validate aggregated insights
        assert "aggregated_insights" in result
        insights = result["aggregated_insights"]

        assert "top_suspicious_phonemes" in insights
        assert "branch_trends" in insights
        assert "mar_summary" in insights

        print(f"\n[Aggregated Insights] Top phonemes: {[p['phoneme'] for p in insights['top_suspicious_phonemes'][:3]]}")
        print(f"[Aggregated Insights] Dominant branch: {insights['branch_trends']['most_dominant']}")


# ===================================
# Performance Tests
# ===================================

class TestPerformance:
    """Performance benchmarks"""

    def test_stage1_processing_time(self, stage1_scanner):
        """Benchmark Stage1 processing time"""
        if not TEST_VIDEO_FAKE.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")

        start_time = time.time()
        result = stage1_scanner.scan_video(
            video_path=str(TEST_VIDEO_FAKE),
            threshold=0.6
        )
        elapsed_time = time.time() - start_time

        print(f"\n[TIME] Stage1 Processing Time: {elapsed_time:.2f} seconds")

        # Assert reasonable processing time (adjust threshold as needed)
        # For 30-second video, should complete in < 120 seconds (includes preprocessing)
        assert elapsed_time < 120, f"Stage1 too slow: {elapsed_time:.2f}s"

    def test_stage2_processing_time(self, stage2_analyzer, stage1_result_with_intervals):
        """Benchmark Stage2 processing time per interval"""
        interval = stage1_result_with_intervals['interval']
        extracted_features = stage1_result_with_intervals['extracted_features']

        start_time = time.time()
        result = stage2_analyzer.analyze_interval(
            interval=interval,
            video_path=str(TEST_VIDEO_FAKE),
            extracted_features=extracted_features
        )
        elapsed_time = time.time() - start_time

        print(f"\n[TIME] Stage2 Processing Time (per interval): {elapsed_time:.2f} seconds")

        # Assert reasonable processing time
        # For one interval, should complete in < 30 seconds
        assert elapsed_time < 30, f"Stage2 too slow: {elapsed_time:.2f}s"

    @pytest.mark.skip(reason="HybridXAIPipeline not implemented yet")
    def test_full_pipeline_processing_time(self, hybrid_pipeline):
        """Benchmark full pipeline E2E processing time"""
        # Will be implemented after Agent 3 completes HybridXAIPipeline
        pass


# ===================================
# Validation Tests
# ===================================

class TestOutputValidation:
    """Test output format compliance with TypeScript interface"""

    @pytest.mark.skip(reason="HybridXAIPipeline not implemented yet")
    def test_validate_full_output_format(self, hybrid_pipeline):
        """Test that full pipeline output passes validate_hybrid_output.py"""
        # Will be implemented after Agent 3 completes HybridXAIPipeline
        #
        # result = hybrid_pipeline.process_video(...)
        # is_valid, errors = validate_hybrid_result(result)
        # assert is_valid, f"Output validation failed: {errors}"
        pass

    def test_stage1_output_partial_validation(self, stage1_scanner):
        """Test that Stage1 output sections are valid"""
        if not TEST_VIDEO_FAKE.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")

        result = stage1_scanner.scan_video(
            video_path=str(TEST_VIDEO_FAKE),
            threshold=0.6
        )

        # Manually validate stage1_timeline structure
        from validate_hybrid_output import validate_stage1_timeline

        errors = validate_stage1_timeline(result)

        if len(errors) > 0:
            print(f"\n[ERROR] Stage1 validation errors:")
            for error in errors:
                print(f"  - {error}")

        assert len(errors) == 0, f"Stage1 output invalid: {errors}"
        print(f"\n[OK] Stage1 output format is valid")

    def test_stage2_output_partial_validation(self, stage2_analyzer, stage1_result_with_intervals):
        """Test that Stage2 output sections are valid"""
        interval = stage1_result_with_intervals['interval']
        extracted_features = stage1_result_with_intervals['extracted_features']
        
        result = stage2_analyzer.analyze_interval(
            interval=interval,
            video_path=str(TEST_VIDEO_FAKE),
            extracted_features=extracted_features
        )

        # Manually validate stage2_interval_analysis structure
        from validate_hybrid_output import validate_stage2_interval_analysis

        errors = validate_stage2_interval_analysis([result])

        if len(errors) > 0:
            print(f"\n[ERROR] Stage2 validation errors:")
            for error in errors:
                print(f"  - {error}")

        assert len(errors) == 0, f"Stage2 output invalid: {errors}"
        print(f"\n[OK] Stage2 output format is valid")


# ===================================
# Regression Tests
# ===================================

class TestRegression:
    """Regression tests to ensure pipeline stability"""

    def test_stage1_consistent_results(self, stage1_scanner):
        """Test that Stage1 produces consistent results on same video"""
        if not TEST_VIDEO_FAKE.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_FAKE}")

        # Run twice
        result1 = stage1_scanner.scan_video(
            video_path=str(TEST_VIDEO_FAKE),
            threshold=0.6
        )
        result2 = stage1_scanner.scan_video(
            video_path=str(TEST_VIDEO_FAKE),
            threshold=0.6
        )

        # Compare statistics
        stats1 = result1["statistics"]
        stats2 = result2["statistics"]

        assert abs(stats1["mean_fake_prob"] - stats2["mean_fake_prob"]) < 0.01, \
            "Stage1 results inconsistent between runs"

        print(f"\n[OK] Stage1 produces consistent results")

    def test_stage1_no_crashes_on_edge_cases(self, stage1_scanner):
        """Test that Stage1 handles edge cases gracefully"""
        # Test with high threshold (should find few/no intervals)
        if not TEST_VIDEO_REAL.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO_REAL}")

        result = stage1_scanner.scan_video(
            video_path=str(TEST_VIDEO_REAL),
            threshold=0.95  # Very high threshold
        )

        # Should complete without errors
        assert "frame_probabilities" in result
        assert "suspicious_intervals" in result
        print(f"\n[OK] Stage1 handles high threshold edge case")


# ===================================
# Main Entry Point
# ===================================

if __name__ == "__main__":
    """Run tests from command line"""
    pytest.main([
        __file__,
        "-v",                    # Verbose output
        "-s",                    # Show print statements
        "--tb=short",            # Shorter traceback format
        "--maxfail=3",           # Stop after 3 failures
        "-k", "not skip",        # Skip tests marked with @pytest.mark.skip
    ])
