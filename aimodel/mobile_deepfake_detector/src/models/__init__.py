"""
Model architectures for deepfake detection
"""

from src.models.teacher import MMMSBA
from src.models.attention import BiModalAttention

__all__ = [
    "MMMSBA",
    "BiModalAttention",
]
