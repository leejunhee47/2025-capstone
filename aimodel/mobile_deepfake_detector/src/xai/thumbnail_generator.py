"""
Thumbnail generator for mobile deepfake detection app.
Generates detection card thumbnails with verdict overlays.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ThumbnailGenerator:
    """Generate thumbnails for detection cards."""

    def __init__(self):
        self.card_size = (300, 200)  # Width x Height

    def generate_detection_card(
        self,
        video_path: str,
        detection: Dict,
        output_path: str,
        frame_index: Optional[int] = None
    ) -> str:
        """
        Generate detection card thumbnail with verdict overlay.

        Args:
            video_path: Path to video file or npz file
            detection: Detection result dict with keys:
                - verdict: "real" or "fake"
                - confidence: float (0-1)
            output_path: Where to save thumbnail PNG
            frame_index: Specific frame to use (default: middle frame)

        Returns:
            Path to saved thumbnail PNG
        """
        logger.info(f"Generating detection card thumbnail for {video_path}")

        # Extract representative frame
        if video_path.endswith('.npz'):
            frame = self._extract_frame_from_npz(video_path, frame_index)
        else:
            frame = self._extract_frame_from_video(video_path, frame_index)

        # Resize to card size
        thumbnail = cv2.resize(frame, self.card_size)

        # Add verdict badge overlay
        thumbnail = self._add_verdict_badge(thumbnail, detection)

        # Save thumbnail
        cv2.imwrite(output_path, thumbnail)
        logger.info(f"  Saved thumbnail: {output_path}")

        return output_path

    def _extract_frame_from_npz(
        self,
        npz_path: str,
        frame_index: Optional[int] = None
    ) -> np.ndarray:
        """Extract frame from preprocessed npz file."""
        data = np.load(npz_path, allow_pickle=True)
        frames = data['frames']  # (T, 224, 224, 3) uint8 [0, 255]

        # Default to middle frame
        if frame_index is None:
            frame_index = frames.shape[0] // 2

        frame = frames[frame_index]  # (224, 224, 3)

        # Convert to BGR for OpenCV
        if frame.dtype == np.float32:
            frame = (frame * 255).astype(np.uint8)

        return frame

    def _extract_frame_from_video(
        self,
        video_path: str,
        frame_index: Optional[int] = None
    ) -> np.ndarray:
        """Extract frame from video file."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Default to middle frame
        if frame_index is None:
            frame_index = total_frames // 2

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Cannot read frame {frame_index} from {video_path}")

        return frame

    def _add_verdict_badge(
        self,
        thumbnail: np.ndarray,
        detection: Dict
    ) -> np.ndarray:
        """
        Add verdict badge overlay to thumbnail.

        Badge format: "FAKE 70.3%" or "REAL 95.2%"
        Color: Red for FAKE, Green for REAL
        Position: Top-right corner
        """
        verdict = detection['verdict'].upper()
        confidence = detection['confidence']

        # Badge text
        text = f"{verdict} {confidence*100:.1f}%"

        # Badge colors
        if verdict == "FAKE":
            bg_color = (0, 0, 200)  # Red (BGR)
            text_color = (255, 255, 255)  # White
        else:
            bg_color = (0, 180, 0)  # Green (BGR)
            text_color = (255, 255, 255)  # White

        # Calculate text size
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Badge position (top-right corner with padding)
        padding = 10
        badge_width = text_width + 2 * padding
        badge_height = text_height + 2 * padding

        x = thumbnail.shape[1] - badge_width - 10
        y = 10

        # Draw badge background (semi-transparent)
        overlay = thumbnail.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + badge_width, y + badge_height),
            bg_color,
            -1  # Filled
        )

        # Blend overlay (70% opacity)
        alpha = 0.7
        thumbnail = cv2.addWeighted(overlay, alpha, thumbnail, 1 - alpha, 0)

        # Draw text
        text_x = x + padding
        text_y = y + padding + text_height
        cv2.putText(
            thumbnail,
            text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

        return thumbnail


# Convenience function
def generate_detection_card(
    video_path: str,
    detection: Dict,
    output_path: str,
    frame_index: Optional[int] = None
) -> str:
    """
    Generate detection card thumbnail.

    Convenience wrapper around ThumbnailGenerator.generate_detection_card().
    """
    generator = ThumbnailGenerator()
    return generator.generate_detection_card(
        video_path, detection, output_path, frame_index
    )
