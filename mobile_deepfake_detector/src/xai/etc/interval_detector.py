"""
Interval Detector Module

Groups suspicious frames into intervals for Stage 2 analysis.
Handles consecutive frame grouping and interval merging.

Author: Claude
Date: 2025-11-17
"""

import numpy as np
from typing import List, Dict
import logging

from .hybrid_utils import group_consecutive_frames

# Setup logger
logger = logging.getLogger(__name__)


class IntervalDetector:
    """
    Interval Detector for suspicious frame grouping.

    Groups consecutive suspicious frames into intervals suitable for
    Stage 2 PIA XAI analysis. Handles minimum frame filtering and
    gap merging.
    """

    def __init__(self):
        """Initialize IntervalDetector."""
        logger.info("Initializing IntervalDetector...")

    def detect_intervals(
        self,
        suspicious_indices: np.ndarray,
        fps: float,
        min_interval_frames: int = 14,
        merge_gap_sec: float = 1.0
    ) -> List[Dict]:
        """
        Detect suspicious intervals from suspicious frame indices.

        Args:
            suspicious_indices: Array of suspicious frame indices
            fps: Video FPS
            min_interval_frames: Minimum frames per interval (PIA needs 14+)
            merge_gap_sec: Gap in seconds to merge nearby intervals

        Returns:
            intervals: List of interval dictionaries with keys:
                - 'interval_id': int
                - 'start_frame': int
                - 'end_frame': int
                - 'start_time': float (seconds)
                - 'end_time': float (seconds)
                - 'duration': float (seconds)
                - 'frame_count': int
                - 'frame_indices': np.ndarray
        """
        logger.info(f"Detecting intervals from {len(suspicious_indices)} suspicious frames...")

        # Use existing group_consecutive_frames function
        intervals = group_consecutive_frames(
            suspicious_indices=suspicious_indices,
            fps=fps,
            min_interval_frames=min_interval_frames,
            merge_gap_sec=merge_gap_sec
        )

        logger.info(f"  Found {len(intervals)} suspicious intervals")
        for i, interval in enumerate(intervals):
            logger.info(f"    Interval {i}: {interval['start_time']:.1f}s - {interval['end_time']:.1f}s "
                       f"({interval['frame_count']} frames)")

        return intervals

    def _merge_nearby_intervals(
        self,
        intervals: List[Dict],
        gap_sec: float,
        fps: float
    ) -> List[Dict]:
        """
        Merge intervals that are close together.

        This is already handled by group_consecutive_frames, but kept
        for potential future customization.

        Args:
            intervals: List of interval dictionaries
            gap_sec: Maximum gap in seconds to merge
            fps: Video FPS

        Returns:
            merged_intervals: List of merged interval dictionaries
        """
        # This functionality is already in group_consecutive_frames
        # Keep this method for potential future extensions
        return intervals

