# Audio and Video Extraction Module

오디오 및 프레임 추출 모듈 - 딥페이크 탐지를 위한 비디오 전처리 파이프라인

## 📋 개요

이 모듈은 비디오 파일에서 오디오와 프레임을 추출하고 전처리하는 기능을 제공합니다. FFmpeg와 OpenCV를 사용하여 고품질 추출을 수행하며, 멀티프로세싱을 통한 배치 처리를 지원합니다.

## 🎯 주요 기능

- ✅ **FFmpeg 기반 오디오 추출**: 16kHz, mono WAV 형식
- ✅ **OpenCV 기반 프레임 추출**: 30fps, 224x224 해상도
- ✅ **자동 전처리**: 리사이징, 정규화, RGB 변환
- ✅ **임시 파일 자동 관리**: Context manager 기반 안전한 파일 처리
- ✅ **배치 처리**: 멀티프로세싱 지원
- ✅ **진행 상태 추적**: tqdm 기반 진행 표시
- ✅ **에러 핸들링**: 재시도 로직 및 상세한 에러 로깅

## 📦 설치

### 필수 요구사항

```bash
# Python 3.7+
# FFmpeg (시스템에 설치 필요)

# Windows
# https://ffmpeg.org/download.html 에서 다운로드 및 PATH 설정

# Ubuntu/Debian
sudo apt-get install ffmpeg

# MacOS
brew install ffmpeg
```

### Python 패키지

```bash
pip install numpy opencv-python librosa tqdm psutil
```

## 🚀 빠른 시작

### 1. 단일 비디오 처리

```python
from av_module import VideoProcessor

# VideoProcessor 생성
processor = VideoProcessor()

# 비디오 처리
result = processor.process_video(
    video_path='path/to/video.mp4',
    extract_audio=True,
    extract_frames=True,
    max_frames=100
)

# 결과 확인
print(f"Audio shape: {result['audio'].shape}")      # (T,)
print(f"Frames shape: {result['frames'].shape}")    # (N, 224, 224, 3)
```

### 2. 배치 처리 (데이터셋)

```python
from av_module import PreprocessingPipeline, get_dataset_videos

# 데이터셋에서 비디오 파일 찾기
video_paths = get_dataset_videos(
    dataset_root='dataset_sample/원천데이터/train_변조',
    pattern='**/*.mp4'
)

# 파이프라인 생성
pipeline = PreprocessingPipeline(
    output_dir='preprocessed_data',
    num_workers=4
)

# 전체 데이터셋 처리
results = pipeline.preprocess_dataset(
    video_paths=video_paths,
    extract_audio=True,
    extract_frames=True,
    use_multiprocessing=True,
    save_results=True
)
```

### 3. 개별 추출기 사용

```python
from av_module import AudioExtractor, FrameExtractor, TempFileManager

# 오디오만 추출
audio_extractor = AudioExtractor()
with TempFileManager() as temp_mgr:
    audio = audio_extractor.extract_audio('video.mp4', temp_mgr)
    print(f"Audio: {audio.shape}")

# 프레임만 추출
frame_extractor = FrameExtractor()
frames = frame_extractor.extract_frames('video.mp4', max_frames=50)
print(f"Frames: {frames.shape}")
```

## 📁 모듈 구조

```
97_av_module/
├── __init__.py                    # 패키지 초기화
├── config.py                      # 설정 관리
├── temp_file_manager.py           # 임시 파일 관리
├── audio_extractor.py             # 오디오 추출
├── frame_extractor.py             # 프레임 추출
├── video_processor.py             # 비디오 처리 조율자
├── preprocessing_pipeline.py      # 배치 처리 파이프라인
├── utils.py                       # 유틸리티 함수
├── test_module.py                 # 테스트 스크립트
└── README.md                      # 이 파일
```

## 🔧 설정 (Config)

```python
from av_module import Config

# 기본 설정 사용
config = Config()

# 설정 확인
print(f"Audio sample rate: {config.AUDIO_SAMPLE_RATE}")  # 16000
print(f"Frame size: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")  # 224x224
print(f"Target FPS: {config.TARGET_FPS}")  # 30

# 설정 커스터마이징
config.MAX_FRAMES_PER_VIDEO = 100
config.NUM_WORKERS = 8
```

### 주요 설정 옵션

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `AUDIO_SAMPLE_RATE` | 16000 | 오디오 샘플링 레이트 (Hz) |
| `AUDIO_CHANNELS` | 1 | 오디오 채널 수 (1=mono) |
| `TARGET_FPS` | 30 | 프레임 추출 FPS |
| `FRAME_WIDTH` | 224 | 프레임 너비 |
| `FRAME_HEIGHT` | 224 | 프레임 높이 |
| `NORMALIZE_FRAMES` | True | 프레임 정규화 (0-1) |
| `NUM_WORKERS` | 4 | 멀티프로세싱 워커 수 |
| `MAX_FRAMES_PER_VIDEO` | None | 최대 프레임 수 (None=전체) |

## 📊 출력 형식

### 처리 결과 구조

```python
{
    'audio': np.ndarray,           # (T,) - 오디오 샘플
    'frames': np.ndarray,          # (N, 224, 224, 3) - 프레임
    'metadata': {
        'video_path': str,
        'video_name': str,
        'success': bool,
        'audio_shape': tuple,
        'audio_duration': float,
        'frames_shape': tuple,
        'num_frames': int,
        'video_info': {...},
        'processing_time': float
    }
}
```

### 저장된 파일 구조

```
preprocessed_data/
├── audio/
│   ├── video1.npy
│   ├── video2.npy
│   └── ...
├── frames/
│   ├── video1.npy
│   ├── video2.npy
│   └── ...
├── metadata/
│   ├── video1.json
│   ├── video2.json
│   └── ...
└── dataset_index.json
```

## 🧪 테스트

```bash
# 모듈 테스트 실행
cd 97_av_module
python test_module.py
```

테스트 항목:
1. 단일 비디오 처리
2. 비디오 검증
3. 오디오 추출
4. 프레임 추출
5. 배치 처리

## 📖 사용 예제

### 예제 1: 오디오 특징 추출

```python
from av_module import AudioExtractor, TempFileManager

extractor = AudioExtractor()

with TempFileManager() as temp_mgr:
    # Raw 오디오
    audio = extractor.extract_audio_features(
        'video.mp4',
        temp_mgr,
        feature_type='raw'
    )

    # MFCC 특징
    mfcc = extractor.extract_audio_features(
        'video.mp4',
        temp_mgr,
        feature_type='mfcc'
    )
```

### 예제 2: 특정 FPS로 프레임 추출

```python
from av_module import FrameExtractor

extractor = FrameExtractor()

# 30 FPS로 프레임 추출
frames = extractor.extract_frames_at_fps(
    'video.mp4',
    target_fps=30,
    preprocess=True
)
```

### 예제 3: 비디오 검증

```python
from av_module import VideoProcessor

processor = VideoProcessor()

# 비디오 파일 검증
validation = processor.validate_video('video.mp4')

if validation['can_open'] and validation['has_video']:
    print("Valid video file!")
    print(f"Duration: {validation['video_info']['duration']:.2f}s")
else:
    print("Invalid video file!")
```

### 예제 4: 재개 가능한 전처리

```python
from av_module import PreprocessingPipeline

pipeline = PreprocessingPipeline(output_dir='preprocessed_data')

# 이미 처리된 비디오는 건너뛰고 재개
results = pipeline.resume_preprocessing(
    video_paths=all_videos,
    extract_audio=True,
    extract_frames=True
)
```

## 🔍 워크플로우

```
Video File (.mp4)
    ↓
[TempFileManager] - 임시 디렉토리 생성
    ↓
[VideoProcessor] - 조율자
    ├→ [AudioExtractor] → Audio Array (T,)
    └→ [FrameExtractor] → Frames Array (N,224,224,3)
    ↓
결과 통합
    ↓
[PreprocessingPipeline] - 배치 처리
    ↓
저장: audio/*.npy, frames/*.npy, metadata/*.json
```

## ⚙️ 고급 기능

### 멀티프로세싱 제어

```python
pipeline = PreprocessingPipeline(
    output_dir='output',
    num_workers=8  # 워커 프로세스 수
)

# 멀티프로세싱 비활성화 (순차 처리)
results = pipeline.preprocess_dataset(
    videos,
    use_multiprocessing=False
)
```

### 로깅 설정

```python
from av_module import setup_logging
import logging

# 로깅 활성화
setup_logging(
    log_file='preprocessing.log',
    level=logging.INFO
)
```

### 진행 상태 저장

```python
# 데이터셋 인덱스와 함께 처리
pipeline = PreprocessingPipeline(output_dir='output')
results = pipeline.preprocess_dataset(videos, save_results=True)

# dataset_index.json 파일이 자동 생성됨
```

## 🐛 문제 해결

### FFmpeg 관련 오류

```
RuntimeError: FFmpeg not found
```

**해결**: FFmpeg를 설치하고 PATH에 추가하세요.

### 메모리 부족

**해결**: `num_workers`를 줄이거나 `max_frames`를 제한하세요.

```python
config.MAX_FRAMES_PER_VIDEO = 50
pipeline = PreprocessingPipeline(output_dir='output', num_workers=2)
```

### OpenCV 오류

```
cv2.error: Could not open video
```

**해결**: 비디오 파일 경로와 코덱을 확인하세요.

## 📝 라이센스

이 모듈은 Audio-Visual Deepfake Detection 프로젝트의 일부입니다.

## 👥 기여자

Audio-Visual Deepfake Detection Team

## 📧 연락처

문제나 질문이 있으시면 이슈를 등록해주세요.
