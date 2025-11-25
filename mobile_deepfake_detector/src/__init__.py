"""
Mobile Deepfake Detector
========================

모바일 쇼츠 영상용 경량 멀티모달 딥페이크 탐지 시스템

주요 모듈:
    - models: Teacher/Student 모델 아키텍처
    - data: 데이터 로딩 및 전처리
    - xai: 설명 가능 AI (Grad-CAM, SHAP, LIME)
    - utils: 유틸리티 함수
    - mobile: 모바일 배포 도구
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.models import MMMSBA
from src.data import DeepfakeDataset
from src.utils import load_config, setup_logger

__all__ = [
    "MMMSBA",
    "DeepfakeDataset",
    "load_config",
    "setup_logger",
]
