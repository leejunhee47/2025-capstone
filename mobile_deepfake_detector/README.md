# Hybrid MMMS-BA + PIA XAI Pipeline

하나의 비디오에 대해 **MMMS-BA 기반 Stage1 스캐닝**과 **PIA XAI 기반 Stage2 분석**을 순차적으로 실행해 심층 위변조 탐지를 수행하는 파이프라인입니다. 이 레포는 `src/xai` 모듈로 스테이지별 로직을 분리했으며, 테스트 스크립트와 결과 검증 도구까지 포함합니다.

---

## 1. 필수 요구 사항

```bash
cd mobile_deepfake_detector
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` 는 `C:\ce\pia_test` 환경을 기준으로 `pip freeze` 한 버전입니다. CUDA 12.4 + Torch 2.5 조합을 기준으로 하며, CPU만 사용할 경우 `--device cpu` 옵션을 사용하세요.

---

## 2. 파이프라인 흐름

1. **MMMS-BA용 전처리**  
   - `feature_extractor.py` 에서 프레임, 오디오(MFCC), 입영역 데이터를 추출하고 30fps→10fps로 다운샘플합니다.
2. **Stage1 (MMMS-BA Temporal Scan)**  
   - `stage1_scanner.py` 가 체크포인트(`mmms_model_path`)를 로드해 프레임별 Fake 확률을 산출합니다.
3. **의심 구간 추출/전처리**  
   - `interval_detector.py` 와 `hybrid_utils.group_consecutive_frames()` 가 의심 프레임을 묶고, Stage2에 필요한 75프레임(2.5초) 단위로 재가공합니다.
4. **Stage2 (PIA XAI Analysis)**  
   - `stage2_analyzer.py` 가 PIA 모델(`pia_model_path`)과 `HybridPhonemeAligner`, `EnhancedMARExtractor`, `ArcFaceExtractor`를 호출해 음소/입술/얼굴 특징을 추출한 뒤, 분기별 중요도를 계산합니다.
5. **시각화 및 결과 집계**  
   - `result_aggregator.py` 가 Stage2 결과를 통합해 전체 Verdict, 한국어 요약, 위험도, 시각화 경로를 포함한 최종 JSON을 제공합니다.  
   - Stage1 타임라인, Stage2 Interval XAI 시각화 PNG가 `outputs/xai/hybrid/<run_id>/` 아래 생성됩니다.

---

## 3. 실행 방법

1. **프로젝트 루트 이동 및 환경 활성화**
   ```bash
   cd mobile_deepfake_detector
   .venv\Scripts\activate
   ```
2. **하이브리드 파이프라인 실행**
   ```bash
   python -m src.xai.hybrid_pipeline ^
     --video "real_deepfake_dataset/.../your_video.mp4" ^
     --mmms-model "mobile_deepfake_detector/models/mmms-ba_best.pth" ^
     --pia-model "mobile_deepfake_detector/outputs/pia/checkpoints/best.pth" ^
     --output-dir "mobile_deepfake_detector/outputs/xai/hybrid/demo_run" ^
     --threshold 0.6 ^
     --device cuda
   ```
   - `--video` : 분석할 원본/변조 영상 경로  
   - `--mmms-model`, `--pia-model` : 학습된 체크포인트 경로  
   - `--output-dir` : JSON/PNG 결과가 저장될 경로 (없으면 자동 생성)  
   - `--threshold` : Stage1 Fake 확률 임계값  
   - `--device` : `cuda` 또는 `cpu`

---

## 4. 출력물 구조

```
outputs/xai/hybrid/<run_id>/
├── result.json                         # HybridDeepfakeXAIResult 전체 구조
├── <video_id>_stage1_timeline.png      # Stage1 확률/히트맵 시각화
└── <video_id>_interval_<n>_xai.png     # Stage2 Interval별 XAI 시각화
```

`result.json` 은 `validate_hybrid_output.py` 로 타입 검증이 가능하며, TypeScript 인터페이스 `plans/hybrid_xai_interface.ts` 와 1:1 매칭됩니다.

---

## 5. 테스트와 검증

| 명령 | 설명 |
|------|------|
| `pytest test_hybrid_pipeline.py -v` | REAL/FAKE 영상으로 Stage1~Stage2 전체 동작 및 시각화 검증 |
| `pytest test_stage1_scanner.py` | Stage1 모듈 단위 테스트 |
| `pytest test_stage2_analyzer.py` | Stage2 모듈 단위 테스트 |
| `python test_pipeline_init.py` | 파이프라인 초기화 확인 (체크포인트 경로/모듈 import) |
| `python validate_hybrid_output.py outputs/xai/hybrid/.../result.json` | 결과 JSON 구조 검증 |

---

## 6. 참고 모듈

- `src/xai/stage1_scanner.py`, `src/xai/stage2_analyzer.py` : 핵심 Stage 모듈  
- `src/xai/feature_extractor.py`, `src/xai/interval_detector.py` : Stage1/Stage2 연결부  
- `src/xai/result_aggregator.py`, `src/xai/pia_visualizer.py` : 결과 요약 및 시각화  
- `test_hybrid_pipeline.py`, `tests/test_stage1_scanner.py`, `tests/test_stage2_analyzer.py` : 단위 및 통합 테스트

---

필요한 이슈나 제안이 있으면 `AI` 브랜치를 기반으로 PR을 올려주세요. 즐거운 디버깅 되세요!

