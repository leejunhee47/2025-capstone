# Hybrid MMMS-BA + PIA XAI Pipeline

하나의 비디오에 대해 **MMMS-BA 기반 Stage1 스캐닝**과 **PIA XAI 기반 Stage2 분석**을 순차적으로 실행해 심층 위변조 탐지를 수행하는 파이프라인입니다.

---

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

> `requirements.txt`는 CUDA 12.4 + Torch 2.5 조합을 기준으로 하며, CPU만 사용할 경우 `--device cpu` 옵션을 사용하세요.

---

## E2E 테스트 실행

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

## 파이프라인 흐름

1. **MMMS-BA용 전처리**
   - `unified_feature_extractor.py`에서 프레임, 오디오(MFCC), 입영역 데이터를 추출하고 30fps→10fps로 다운샘플합니다.
2. **Stage1 (MMMS-BA Temporal Scan)**
   - `stage1_scanner.py`가 체크포인트를 로드해 프레임별 Fake 확률을 산출합니다.
3. **의심 구간 추출/전처리**
   - `hybrid_utils.group_consecutive_frames()`가 의심 프레임을 묶고, Stage2에 필요한 75프레임(2.5초) 단위로 재가공합니다.
4. **Stage2 (PIA XAI Analysis)**
   - `stage2_analyzer.py`가 PIA 모델과 `HybridPhonemeAligner`, `EnhancedMARExtractor`, `ArcFaceExtractor`를 호출해 음소/입술/얼굴 특징을 추출한 뒤, 분기별 중요도를 계산합니다.
5. **시각화 및 결과 집계**
   - `result_aggregator.py`가 Stage2 결과를 통합해 전체 Verdict, 한국어 요약, 위험도, 시각화 경로를 포함한 최종 JSON을 제공합니다.

---

## 실행 방법

```bash
python -m src.xai.hybrid_pipeline \
  --video "real_deepfake_dataset/.../your_video.mp4" \
  --mmms-model "models/checkpoints/mmms-ba_fulldata_best.pth" \
  --pia-model "models/checkpoints/pia-best.pth" \
  --output-dir "outputs/xai/hybrid/demo_run" \
  --threshold 0.6 \
  --device cuda
```

| 옵션 | 설명 |
|------|------|
| `--video` | 분석할 원본/변조 영상 경로 |
| `--mmms-model` | MMMS-BA 체크포인트 경로 |
| `--pia-model` | PIA 체크포인트 경로 |
| `--output-dir` | JSON/PNG 결과가 저장될 경로 |
| `--threshold` | Stage1 Fake 확률 임계값 |
| `--device` | `cuda` 또는 `cpu` |

---

## 출력물 구조

```
outputs/xai/hybrid/<run_id>/
├── result.json                         # HybridDeepfakeXAIResult 전체 구조
├── <video_id>_stage1_timeline.png      # Stage1 확률/히트맵 시각화
└── <video_id>_interval_<n>_xai.png     # Stage2 Interval별 XAI 시각화
```

`result.json`은 `validate_hybrid_output.py`로 타입 검증이 가능하며, TypeScript 인터페이스 `plans/hybrid_xai_interface.ts`와 1:1 매칭됩니다.

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
nvidia-smi  # GPU 메모리 확인
# test_hybrid_pipeline.py의 device="cuda" → "cpu" 변경
```

### 4. WhisperX 관련 에러
```bash
python -c "import whisperx; print(whisperx.__version__)"
```

---

## 다운로드 자료

- 예시 fake 영상: https://drive.google.com/file/d/1NtsfoI-gJ1YHdgAylpW8CC6D6ddq-qd7/view?usp=sharing
- 예시 mmms-ba 모델 파일: https://drive.google.com/file/d/13I52pthEDoTij4pKKK-cY_1GKhle9zfm/view?usp=sharing
- 예시 pia 모델 파일: https://drive.google.com/file/d/1KeBVQ9MRj4IPU9Qc8njuz6QaUrdoCMQP/view?usp=sharing

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

필요한 이슈나 제안이 있으면 `AI` 브랜치를 기반으로 PR을 올려주세요!
