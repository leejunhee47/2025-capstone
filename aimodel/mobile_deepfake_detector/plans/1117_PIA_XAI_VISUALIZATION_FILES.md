# PIA XAI Visualization Files Tree

PIA (Phoneme-Identity-Appearance) 모델의 전체 파이프라인 - 전처리부터 학습, XAI 시각화까지

**생성일**: 2025-11-17
**참조 보고서**:
- `diary/2025-11-14_MAR_수정_연구보고서.md`
- `diary/2025-11-15_PIA_XAI_결과_비교분석.md`
- `diary/2025-11-15_XAI_구현_및_모델_분석.md`

---

## 📁 Complete PIA Pipeline Tree

```
E:\capstone\
├── 📂 전처리 데이터 (Preprocessed Data)
│   ├── preprocessed_data_phoneme/          # ✅ PIA 전용 전처리 데이터
│   │   ├── train/                          # 학습 데이터 (NPZ 파일들)
│   │   │   ├── 00000.npz                   # frames, audio, lip, arcface, geometry(MAR), phoneme_labels, timestamps
│   │   │   ├── 00001.npz
│   │   │   └── ...
│   │   ├── val/                            # 검증 데이터
│   │   ├── test/                           # 테스트 데이터
│   │   ├── train_index.json               # 학습 데이터 인덱스
│   │   ├── val_index.json                 # 검증 데이터 인덱스
│   │   └── test_index.json                # 테스트 데이터 인덱스
│   │
│   └── preprocessed_data_real/             # ⚠️ 구버전 (음소 라벨 없음)
│       └── train/test/val/
│
├── 📂 전처리 스크립트 (Preprocessing Scripts)
│   ├── preprocess_parallel.py              # ✅ 핵심 전처리 스크립트
│   │   └── 기능:
│   │       ├── HybridPhonemeAligner로 음소 추출
│   │       ├── EnhancedMARExtractor로 MAR 추출
│   │       ├── ArcFaceExtractor로 얼굴 임베딩 추출
│   │       ├── 멀티프로세싱 병렬 처리
│   │       └── 출력: preprocessed_data_phoneme/
│   │
│   ├── test_phoneme_preprocessing.py
│   ├── test_single_video_phoneme.py
│   └── verify_phoneme_accuracy.py
│
├── 📂 모바일 딥페이크 탐지기 (Mobile Deepfake Detector)
│   └── mobile_deepfake_detector/
│       │
│       ├── 📂 configs/                     # 설정 파일
│       │   ├── train_pia.yaml              # ✅ PIA 학습 설정
│       │   │   └── 내용: num_phonemes=14, frames_per_phoneme=5, arcface_dim=512, geo_dim=1
│       │   ├── train_teacher_korean.yaml   # MMMS-BA 학습 설정
│       │   └── phoneme_vocab.json          # 14개 핵심 한국어 음소 vocabulary
│       │
│       ├── 📂 src/                         # 소스 코드
│       │   │
│       │   ├── 📂 data/                    # 데이터 로딩
│       │   │   ├── dataset.py              # MMMS-BA Dataset
│       │   │   ├── phoneme_dataset.py      # ✅ PIA Dataset (KoreanPhonemeDataset)
│       │   │   │   └── __getitem__():
│       │   │   │       ├── NPZ 파일 로드
│       │   │   │       ├── 음소 → 14×5 그리드 매칭
│       │   │   │       ├── MAR, ArcFace, Frames 추출
│       │   │   │       └── 출력: {geometry, images, arcface, mask, phonemes, label}
│       │   │   ├── preprocessing.py        # ShortsPreprocessor (비디오→NPZ)
│       │   │   └── korean_phoneme_vocab.py # 음소 vocabulary 관리
│       │   │
│       │   ├── 📂 models/                  # 모델 아키텍처
│       │   │   ├── teacher.py              # MMMS-BA 모델 (Tri-modal)
│       │   │   └── pia_model.py            # ✅ PIA 모델 (Tri-branch)
│       │   │       └── class PIAModel:
│       │   │           ├── GeometryBranch (MAR → GRU)
│       │   │           ├── ImageBranch (Frames → ResNet → GRU)
│       │   │           ├── ArcBranch (ArcFace → GRU)
│       │   │           ├── CrossAttention (3-branch fusion)
│       │   │           └── forward() → (logits, branch_outputs)
│       │   │
│       │   ├── 📂 utils/                   # 유틸리티
│       │   │   │
│       │   │   ├── 🎤 음소 추출 (Phoneme Extraction)
│       │   │   ├── hybrid_phoneme_aligner_v2.py  # ✅ 실제 사용 (WhisperX + Wav2Vec2)
│       │   │   │   └── class HybridPhonemeAligner:
│       │   │   │       ├── align_video(video_path) → {phonemes, intervals, transcription}
│       │   │   │       ├── _align_segment_pia_style()  # 3단계: 균등→WhisperX→자모 분배
│       │   │   │       └── _extract_and_distribute_chars()  # 음절→자모 분해
│       │   │   │
│       │   │   ├── wav2vec2_korean_phoneme_aligner.py  # ⚠️ 사용 안 함 (구버전)
│       │   │   ├── hybrid_phoneme_aligner.py           # ⚠️ 사용 안 함 (v1)
│       │   │   ├── hybrid_phoneme_aligner_v3_failed.py # ⚠️ 실패 버전
│       │   │   ├── pia_main_phoneme_aligner.py
│       │   │   ├── phoneme_classifier.py
│       │   │   ├── phoneme_filter.py
│       │   │   ├── phoneme_mar_matcher.py
│       │   │   └── korean_phoneme_config.py      # KEEP_PHONEMES_KOREAN (14개)
│       │   │   │
│       │   │   ├── 👄 기하학 특징 추출 (Geometry Feature)
│       │   │   ├── enhanced_mar_extractor.py     # ✅ MAR 추출 (v3.2)
│       │   │   │   └── class EnhancedMARExtractor:
│       │   │   │       ├── extract_from_video() → {mar_vertical, mar_horizontal, ...}
│       │   │   │       └── _calculate_multi_features_relative()  # Face-height 정규화
│       │   │   │
│       │   │   ├── 😀 얼굴 임베딩 추출 (Identity Feature)
│       │   │   ├── arcface_extractor.py          # ✅ ArcFace 추출 (buffalo_l)
│       │   │   │   └── class ArcFaceExtractor:
│       │   │   │       └── extract_from_video() → (T, 512)
│       │   │   │
│       │   │   ├── 🔧 기타 유틸리티
│       │   │   ├── config.py                     # YAML 설정 로더
│       │   │   ├── logger.py
│       │   │   ├── metrics.py
│       │   │   └── mmms_ba_adapter.py            # MMMS-BA 어댑터
│       │   │
│       │   └── 📂 xai/                     # ✅ XAI 모듈
│       │       ├── pia_explainer.py              # ✅ PIA XAI 분석 엔진 (640 lines)
│       │       │   └── class PIAExplainer:
│       │       │       ├── explain(geoms, imgs, arcs, mask, phonemes, timestamps)
│       │       │       │   └── 출력: {
│       │       │       │       'prediction': 'FAKE'/'REAL',
│       │       │       │       'confidence': 1.00,
│       │       │       │       'branch_contributions': {'geometry': 15%, 'image': 78%, 'arcface': 7%},
│       │       │       │       'top_branch': 'image',
│       │       │       │       'phoneme_attention': (14, 5) attention weights,
│       │       │       │       'temporal_analysis': {...},
│       │       │       │       'geometry_analysis': {...},
│       │       │       │       'korean_summary': "..."
│       │       │       │   }
│       │       │       ├── _compute_branch_contributions()
│       │       │       ├── _compute_phoneme_attention()
│       │       │       ├── _analyze_temporal_patterns()
│       │       │       ├── _analyze_geometry_anomalies()
│       │       │       └── _generate_korean_explanation()
│       │       │
│       │       ├── pia_visualizer.py             # ✅ PIA XAI 시각화 (664 lines)
│       │       │   └── class PIAVisualizer:
│       │       │       ├── visualize_full_analysis(xai_result, output_path, video_id)
│       │       │       │   └── 출력: 4-subplot 그래프 PNG
│       │       │       │       ├── Branch Contribution (Bar chart)
│       │       │       │       ├── Phoneme Attention Heatmap (14×5)
│       │       │       │       ├── Temporal Analysis (Line plot)
│       │       │       │       └── Korean Explanations (Text box)
│       │       │       ├── _create_branch_contribution_plot()
│       │       │       ├── _create_phoneme_attention_heatmap()
│       │       │       ├── _create_temporal_analysis_plot()
│       │       │       ├── _create_geometry_analysis_plot()
│       │       │       └── _add_korean_explanations()
│       │       │
│       │       └── hybrid_mmms_pia_explainer.py  # 🚧 하이브리드 파이프라인 (작업 중)
│       │
│       ├── 📂 scripts/                    # 실행 스크립트
│       │   ├── train_pia.py                      # ✅ PIA 학습 스크립트
│       │   │   └── 기능:
│       │   │       ├── KoreanPhonemeDataset 로드
│       │   │       ├── PIAModel 학습 (CrossEntropyLoss)
│       │   │       ├── Early stopping (patience=10)
│       │   │       └── 출력: outputs/pia_*/checkpoints/best.pth
│       │   │
│       │   ├── evaluate_pia.py                   # PIA 평가 스크립트
│       │   ├── test_pia_from_urls.py             # URL 테스트
│       │   ├── train.py                          # MMMS-BA 학습
│       │   └── evaluate.py                       # MMMS-BA 평가
│       │
│       ├── 📂 tests/                      # 테스트 코드
│       │   ├── test_phoneme_dataset.py           # Dataset 테스트
│       │   ├── test_pia_alignment.py             # 음소 정렬 테스트
│       │   ├── test_korean_phoneme_extraction.py
│       │   └── analyze_mismatch_phonemes.py
│       │
│       ├── 📂 outputs/                    # 출력 결과
│       │   ├── pia_aug50/                        # ✅ PIA 학습 결과 (Real 증강)
│       │   │   ├── checkpoints/
│       │   │   │   ├── best.pth                  # ✅ 최고 성능 체크포인트 (epoch 26)
│       │   │   │   └── last.pth
│       │   │   └── logs/
│       │   │       └── train_*.log
│       │   │
│       │   ├── pia_baseline/                     # PIA 베이스라인 (증강 없음)
│       │   │
│       │   ├── korean/                           # MMMS-BA 학습 결과
│       │   │   └── evaluation/
│       │   │       ├── xai_analysis_00000.png    # XAI 분석 결과
│       │   │       ├── xai_analysis_00001.png
│       │   │       └── test_results.json
│       │   │
│       │   └── xai_comparisons/                  # ✅ XAI 비교 분석
│       │       ├── fake_sample_xai.png           # Fake 샘플 XAI
│       │       └── real_sample_xai.png           # Real 샘플 XAI
│       │
│       ├── 📂 분석 스크립트 (Analysis Scripts)
│       │   ├── test_pia_xai.py                   # ✅ PIA XAI 테스트 (FAKE)
│       │   ├── test_pia_xai_real.py              # ✅ PIA XAI 테스트 (REAL)
│       │   ├── test_pia_dataset.py
│       │   ├── test_pia_main_aligner.py
│       │   ├── analyze_phoneme_alignment.py
│       │   ├── analyze_phoneme_discriminability_v3.py
│       │   ├── analyze_phoneme_mar_overlap.py
│       │   ├── test_phoneme_mar_matching.py
│       │   ├── test_key_phonemes.py
│       │   └── debug_phoneme_frame_matching.py
│       │
│       └── 📄 문서 (Documentation)
│           ├── README.md
│           ├── PIA_UNSUITABILITY_ANALYSIS_1023.md
│           ├── KOREAN_PHONEME_EXTRACTION_1023.md
│           └── PHONEME_MATCHING_IMPLEMENTATION.md
│
└── 📂 연구 일지 (Research Diaries)
    └── diary/
        ├── 2025-11-14_MAR_수정_연구보고서.md       # ✅ MAR 알고리즘 개선 (v3.1→v3.2)
        ├── 2025-11-15_PIA_XAI_결과_비교분석.md     # ✅ Real vs Fake XAI 비교
        └── 2025-11-15_XAI_구현_및_모델_분석.md     # ✅ XAI 구현 상세
```

---

## 🔍 핵심 파일 상세 설명

### 1. 전처리 파이프라인

#### `preprocess_parallel.py` (585 lines)
```python
# 비디오 → NPZ 변환 (PIA 전용)
def process_single_video(video_path):
    # Step 1: 음소 추출
    aligner = HybridPhonemeAligner(whisper_model="base", device="cuda")
    alignment = aligner.align_video(video_path)  # → phonemes, intervals

    # Step 2: MAR 추출
    mar_extractor = EnhancedMARExtractor()
    geometry = mar_extractor.extract_from_video(video_path)  # → (T, 1)

    # Step 3: ArcFace 추출
    arcface_extractor = ArcFaceExtractor(device="cuda", model_name="buffalo_l")
    arcface = arcface_extractor.extract_from_video(video_path)  # → (T, 512)

    # Step 4: 프레임/오디오/립 추출 (ShortsPreprocessor)
    preprocessor = ShortsPreprocessor(config)
    result = preprocessor.process_video(video_path)

    # Step 5: NPZ 저장
    np.savez_compressed(
        output_path,
        frames=result['frames'],        # (50, 224, 224, 3)
        audio=result['audio'],          # (T_audio, 40)
        lip=result['lip'],              # (50, 96, 96, 3)
        arcface=arcface,                # (T, 512) ✅ REAL
        geometry=geometry,              # (T, 1) ✅ REAL MAR
        phoneme_labels=phoneme_labels,  # (T,) ✅ REAL
        timestamps=timestamps,          # (T,) ✅ REAL
        label=1 if label == 'fake' else 0
    )
```

**출력 위치**: `preprocessed_data_phoneme/train/00000.npz`

---

### 2. 데이터 로딩

#### `src/data/phoneme_dataset.py` - `KoreanPhonemeDataset`
```python
def __getitem__(self, idx):
    # NPZ 로드
    data = np.load(npz_path)

    # 음소 라벨 추출
    phoneme_labels = data['phoneme_labels']  # (T,) - 프레임별 음소
    timestamps = data['timestamps']          # (T,) - 타임스탬프

    # 14×5 그리드 생성
    phoneme_indices, phoneme_labels_14 = sample_phonemes_from_timestamps(
        phoneme_labels, timestamps, num_phonemes=14, frames_per_phoneme=5
    )  # → (14, 5) 인덱스

    # 특징 추출
    geometry = data['geometry'][phoneme_indices]  # (14, 5, 1)
    images = data['frames'][phoneme_indices]      # (14, 5, 224, 224, 3)
    arcface = data['arcface'][phoneme_indices]    # (14, 5, 512)

    return {
        'geometry': geometry,
        'images': images,
        'arcface': arcface,
        'mask': mask,             # (14, 5) - 유효한 프레임
        'phonemes': phoneme_labels_14,  # List[str] - 14개 음소
        'label': label
    }
```

---

### 3. 모델 아키텍처

#### `src/models/pia_model.py` - `PIAModel`
```python
class PIAModel(nn.Module):
    def __init__(self, num_phonemes=14, frames_per_phoneme=5, num_classes=2):
        # Branch 1: Geometry (MAR)
        self.geometry_branch = GeometryBranch(geo_dim=1, hidden_dim=128)

        # Branch 2: Image (ResNet + GRU)
        self.image_branch = ImageBranch(resnet_model='resnet18', hidden_dim=256)

        # Branch 3: ArcFace (Identity)
        self.arc_branch = ArcBranch(arcface_dim=512, hidden_dim=128)

        # Fusion: Cross-Attention
        self.cross_attention = CrossAttention(hidden_dim=512)

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, geoms, imgs, arcs, mask):
        # (B, P, F, ...) → Branch outputs
        geo_out = self.geometry_branch(geoms, mask)     # (B, P, 128)
        img_out = self.image_branch(imgs, mask)         # (B, P, 256)
        arc_out = self.arc_branch(arcs, mask)           # (B, P, 128)

        # Fusion
        fused = self.cross_attention(geo_out, img_out, arc_out)  # (B, 512)

        # Classification
        logits = self.classifier(fused)  # (B, 2)

        return logits, {
            'geometry': geo_out,
            'image': img_out,
            'arcface': arc_out
        }
```

---

### 4. 학습 스크립트

#### `scripts/train_pia.py`
```bash
# 사용 예시
python scripts/train_pia.py \
    --config configs/train_pia.yaml \
    --data-dir ../preprocessed_data_phoneme/ \
    --epochs 30 \
    --batch-size 8 \
    --augment-real \
    --augment-ratio 1.0

# 출력
outputs/pia_aug50/
├── checkpoints/
│   ├── best.pth      # 최고 성능 (epoch 26, val_acc=...)
│   └── last.pth
└── logs/
    └── train_20251115_*.log
```

---

### 5. XAI 분석

#### `src/xai/pia_explainer.py` - `PIAExplainer`
```python
def explain(self, geoms, imgs, arcs, mask, phonemes, timestamps):
    # 1. Forward pass
    logits, branch_outputs = self.model(geoms, imgs, arcs, mask)
    prediction = 'FAKE' if logits[0, 1] > logits[0, 0] else 'REAL'
    confidence = torch.softmax(logits, dim=1)[0, 1].item()

    # 2. Branch Contribution Analysis
    branch_contributions = self._compute_branch_contributions(branch_outputs)
    # → {'geometry': 15.41%, 'image': 78.33%, 'arcface': 6.26%}

    # 3. Phoneme Attention Analysis
    phoneme_attention = self._compute_phoneme_attention(branch_outputs, mask)
    # → (14, 5) attention weights

    # 4. Temporal Pattern Analysis
    temporal_analysis = self._analyze_temporal_patterns(
        branch_outputs, timestamps
    )

    # 5. Geometry Anomaly Analysis
    geometry_analysis = self._analyze_geometry_anomalies(
        geoms, phonemes, timestamps
    )

    # 6. Korean Explanation Generation
    korean_summary = self._generate_korean_explanation(
        prediction, confidence, branch_contributions,
        phoneme_attention, geometry_analysis
    )

    return {
        'prediction': prediction,
        'confidence': confidence,
        'branch_contributions': branch_contributions,
        'top_branch': max(branch_contributions, key=branch_contributions.get),
        'phoneme_attention': phoneme_attention,
        'temporal_analysis': temporal_analysis,
        'geometry_analysis': geometry_analysis,
        'korean_summary': korean_summary
    }
```

---

#### `src/xai/pia_visualizer.py` - `PIAVisualizer`
```python
def visualize_full_analysis(self, xai_result, output_path, video_id):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Branch Contribution
    self._create_branch_contribution_plot(axes[0, 0], xai_result)

    # Subplot 2: Phoneme Attention Heatmap
    self._create_phoneme_attention_heatmap(axes[0, 1], xai_result)

    # Subplot 3: Temporal Analysis
    self._create_temporal_analysis_plot(axes[1, 0], xai_result)

    # Subplot 4: Korean Explanations
    self._add_korean_explanations(axes[1, 1], xai_result)

    plt.suptitle(f"PIA XAI Analysis - {video_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

**출력 예시**: `outputs/xai_comparisons/fake_sample_xai.png`

---

### 6. XAI 테스트 스크립트

#### `test_pia_xai.py` (FAKE 샘플)
```bash
python test_pia_xai.py

# 출력
[PIA XAI Analysis - FAKE Sample]
Prediction: FAKE (100% confidence)
Top Branch: image (78.33%)
Top Phoneme: ㅊ (51.86%)

Saved: outputs/xai_comparisons/fake_sample_xai.png
```

#### `test_pia_xai_real.py` (REAL 샘플)
```bash
python test_pia_xai_real.py

# 출력
[PIA XAI Analysis - REAL Sample]
Prediction: REAL (100% confidence)
Top Branch: image (84.55%)
Top Phoneme: ㅏ (98.74%)

Saved: outputs/xai_comparisons/real_sample_xai.png
```

---

## 📊 PIA XAI 분석 결과 요약 (2025-11-15 연구)

### Real vs Fake 비교

| 특징 | FAKE 영상 | REAL 영상 |
|------|-----------|-----------|
| **Prediction** | FAKE (100%) | REAL (100%) |
| **Top Phoneme** | ㅊ (51.86%) | ㅏ (98.74%) |
| **Attention 분포** | 다중 분산 | 단일 집중 |
| **Visual 기여도** | 78.33% | 84.55% |
| **Geometry 기여도** | 15.41% | 13.16% |
| **ArcFace 기여도** | 6.26% | 2.29% |
| **MAR 평균** | 0.059 | 0.017 |
| **MAR 최대** | 0.322 | 0.599 |

### 핵심 발견

1. ✅ **Visual Branch 지배성**: 78-85% 기여도 (입 모양이 핵심)
2. ✅ **Real vs Fake 음소 패턴**: ㅏ (자연) vs ㅊ (이상)
3. ✅ **Attention 분포 차이**: Real은 집중, Fake는 분산
4. ⚠️ **MAR 낮음**: 한국어 특성 반영 필요

---

## 🚀 사용 예시 (Full Pipeline)

### 1단계: 전처리
```bash
cd E:\capstone

# 학습 데이터 전처리
python preprocess_parallel.py --split train --workers 6
# 출력: preprocessed_data_phoneme/train/00000.npz, ...
```

### 2단계: 학습
```bash
cd mobile_deepfake_detector

# PIA 모델 학습
python scripts/train_pia.py \
    --config configs/train_pia.yaml \
    --data-dir ../preprocessed_data_phoneme/ \
    --epochs 30 \
    --augment-real \
    --augment-ratio 1.0

# 출력: outputs/pia_aug50/checkpoints/best.pth
```

### 3단계: XAI 테스트
```bash
# Fake 샘플 XAI 분석
python test_pia_xai.py
# 출력: outputs/xai_comparisons/fake_sample_xai.png

# Real 샘플 XAI 분석
python test_pia_xai_real.py
# 출력: outputs/xai_comparisons/real_sample_xai.png
```

---

## ⚙️ 핵심 설정

### `configs/train_pia.yaml`
```yaml
data:
  num_phonemes: 14              # 한국어 핵심 음소 개수
  frames_per_phoneme: 5         # 음소당 프레임 개수
  data_dir: ../preprocessed_data_phoneme/

model:
  arcface_dim: 512              # ArcFace 임베딩 차원
  geo_dim: 1                    # MAR (Mouth Aspect Ratio)
  embed_dim: 512                # Fusion 임베딩 차원
  num_heads: 8                  # Cross-Attention 헤드 수
  num_classes: 2                # Real/Fake
  use_temporal_loss: false

training:
  batch_size: 8
  learning_rate: 0.0001
  num_epochs: 30
  early_stopping_patience: 10

output:
  checkpoint_dir: outputs/pia_aug50/checkpoints/
  log_dir: outputs/pia_aug50/logs/
```

---

## 📚 관련 문서

1. **MAR 알고리즘 개선**: `diary/2025-11-14_MAR_수정_연구보고서.md`
   - Inner lip → Outer lip 변경
   - Mouth-box → Face-height 정규화
   - MAR 값 17배 증가 (0.03 → 0.51)

2. **PIA XAI 비교 분석**: `diary/2025-11-15_PIA_XAI_결과_비교분석.md`
   - Real vs Fake attention 패턴
   - Visual Branch 지배성 (78-85%)
   - 음소별 attention 분포

3. **XAI 구현 상세**: `diary/2025-11-15_XAI_구현_및_모델_분석.md`
   - Branch contribution 계산
   - Phoneme attention 분석
   - Korean explanation 생성

---

## 🔧 음소 추출기 변천사

| 버전 | 파일명 | 상태 | 방식 |
|------|--------|------|------|
| v1 | `hybrid_phoneme_aligner.py` | ⚠️ 구버전 | WhisperX + Wav2Vec2 |
| v2 | `hybrid_phoneme_aligner_v2.py` | ✅ **현재 사용** | WhisperX + Wav2Vec2 + 자모 분배 |
| v3 | `hybrid_phoneme_aligner_v3_failed.py` | ❌ 실패 | 실험 버전 |
| - | `wav2vec2_korean_phoneme_aligner.py` | ⚠️ 사용 안 함 | Wav2Vec2 단독 |
| - | `pia_main_phoneme_aligner.py` | ⚠️ 보조 | PIA-main 원본 스타일 |

**결론**: `hybrid_phoneme_aligner_v2.py`가 실제 전처리에 사용됨!

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-11-17
**검증 완료**: PIA 전처리 → 학습 → XAI 전체 파이프라인