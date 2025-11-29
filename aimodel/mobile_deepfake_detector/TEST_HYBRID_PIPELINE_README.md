# Hybrid XAI Pipeline E2E Test Guide

## Quick Start

### Prerequisites

1. **Python 환경**: `pia_test` conda 가상환경 활성화
   ```bash
   conda activate pia_test
   ```

2. **CUDA**: GPU 사용 (CUDA 12.4 권장)
   ```bash
   python scripts/check_cuda.py
   ```

3. **Model Checkpoints** (필수):
   - `models/checkpoints/mmms-ba_fulldata_best.pth` (MMMS-BA 모델)
   - `models/checkpoints/pia-best.pth` (PIA 모델)

4. **Test Videos** (자동 생성 가능):
   - `cache/0e105f8ec5146f9737d0_132.mp4` (Real 영상)
   - `cache/0e105f8ec5146f9737d0_026f9b9514f28f37a3fd_2_0035.mp4` (Fake 영상)

---

## Running Tests

### 1. 전체 파이프라인 테스트 (Real/Fake 영상)

```bash
cd mobile_deepfake_detector

# E2E 테스트만 실행
pytest test_hybrid_pipeline.py::TestHybridXAIPipeline::test_full_pipeline_real_video -v -s
pytest test_hybrid_pipeline.py::TestHybridXAIPipeline::test_full_pipeline_fake_video -v -s

# 둘 다 실행
pytest test_hybrid_pipeline.py::TestHybridXAIPipeline -v -s
```

### 2. 개별 스테이지 테스트

```bash
# Stage1 (MMMS-BA) 테스트
pytest test_hybrid_pipeline.py::TestStage1Scanner -v -s

# Stage2 (PIA) 테스트
pytest test_hybrid_pipeline.py::TestStage2Analyzer -v -s

# Feature Extractor 테스트
pytest test_hybrid_pipeline.py::TestUnifiedFeatureExtractor -v -s
```

### 3. 전체 테스트 스위트

```bash
# 모든 테스트 실행 (약 5-10분 소요)
pytest test_hybrid_pipeline.py -v -s --tb=short

# 첫 3개 실패 시 중단
pytest test_hybrid_pipeline.py -v -s --maxfail=3
```

---

## Test Classes Overview

| Class | Description | Tests |
|-------|-------------|-------|
| `TestUnifiedFeatureExtractor` | 입력 데이터 검증 | 5 tests |
| `TestStage1Scanner` | MMMS-BA 모델 테스트 | 3 tests |
| `TestStage2Analyzer` | PIA 모델 테스트 | 4 tests |
| `TestHybridXAIPipeline` | **E2E 통합 테스트** | 2 tests |
| `TestPerformance` | 성능 벤치마크 | 3 tests |
| `TestRegression` | 안정성 테스트 | 3 tests |
| `TestOutputValidation` | 출력 스키마 검증 | 3 tests |

---

## Feature Caching (빠른 테스트)

테스트는 `cache/unified_features/` 폴더의 NPZ 캐시를 우선 사용합니다.

### 캐시 재생성
```bash
python tests/fixtures/generate_test_cache.py
```

### 캐시 구조
```
cache/
├── unified_features/
│   ├── 0e105f8ec5146f9737d0_132_dc362f32.npz         # Real video features
│   ├── 0e105f8ec5146f9737d0_026f9b9514f28f37a3fd_..._6ff42c2f.npz  # Fake video features
│   └── cache_index.json                              # Cache metadata
├── 0e105f8ec5146f9737d0_132.mp4                      # Real test video
└── 0e105f8ec5146f9737d0_026f9b9514f28f37a3fd_2_0035.mp4  # Fake test video
```

### 캐시 효과
- **With cache**: ~30초 (features 로드만)
- **Without cache**: ~3-5분 (WhisperX, MediaPipe, ArcFace 추출)

---

## Output Files

테스트 실행 후 생성되는 파일들:

```
outputs/xai/hybrid_tests/
├── pipeline_real_test/
│   ├── result.json           # 분석 결과 JSON
│   ├── stage1_timeline.png   # MMMS-BA 타임라인
│   └── pia_xai_*.png         # PIA 시각화
└── pipeline_fake_test/
    ├── result.json
    ├── stage1_timeline.png
    └── pia_xai_*.png
```

---

## Troubleshooting

### 1. "Checkpoint not found" 에러
```
pytest.skip: MMMS-BA checkpoint not found
```
**해결**: `models/checkpoints/` 폴더에 체크포인트 파일 확인

### 2. "Video not found" 에러
```
pytest.skip: Real video not found
```
**해결**: `cache/` 폴더에 테스트 영상 복사

### 3. CUDA Out of Memory
```bash
# GPU 메모리 확인
nvidia-smi

# 배치 크기 줄이기 또는 CPU 모드 (느림)
# test_hybrid_pipeline.py의 device="cuda" → "cpu" 변경
```

### 4. WhisperX 관련 에러
```bash
# WhisperX 모델 다운로드 확인
python -c "import whisperx; print(whisperx.__version__)"
```

---

## CI/CD Integration

```yaml
# GitHub Actions example
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run E2E Tests
        run: |
          conda activate pia_test
          pytest test_hybrid_pipeline.py::TestHybridXAIPipeline -v --tb=short
```

---

## Module Structure

```
test_hybrid_pipeline.py
├── Imports
│   ├── src.xai.hybrid_pipeline.HybridXAIPipeline
│   ├── src.xai.stage1_scanner.Stage1Scanner
│   ├── src.xai.stage2_analyzer.Stage2Analyzer
│   ├── src.xai.unified_feature_extractor.UnifiedFeatureExtractor
│   ├── src.xai.result_aggregator.ResultAggregator
│   └── src.utils.feature_cache.FeatureCache
│
├── Configuration
│   ├── TEST_VIDEO_REAL, TEST_VIDEO_FAKE  (테스트 영상 경로)
│   ├── MMMS_BA_CHECKPOINT, PIA_CHECKPOINT (모델 경로)
│   └── TEST_OUTPUT_DIR (출력 폴더)
│
├── Fixtures (pytest fixtures)
│   ├── feature_extractor      (UnifiedFeatureExtractor 인스턴스)
│   ├── real_video_path        (Real 영상 경로)
│   ├── fake_video_path        (Fake 영상 경로)
│   ├── extracted_features_*   (캐시된 features)
│   ├── stage1_scanner         (Stage1Scanner 인스턴스)
│   ├── stage2_analyzer        (Stage2Analyzer 인스턴스)
│   └── hybrid_pipeline        (HybridXAIPipeline 인스턴스)
│
└── Test Classes (7개)
    ├── TestUnifiedFeatureExtractor (입력 검증)
    ├── TestStage1Scanner (MMMS-BA)
    ├── TestStage2Analyzer (PIA)
    ├── TestHybridXAIPipeline (E2E) ★
    ├── TestPerformance (벤치마크)
    ├── TestRegression (안정성)
    └── TestOutputValidation (스키마)
```

---

