"""
Enhanced MAR (Mouth Aspect Ratio) Extraction with Mouth Cropping and Multi-Features

This module extracts multiple mouth features from video frames using MediaPipe FaceMesh
with mouth region cropping for improved phoneme discriminability.

Features:
1. MAR_vertical: Vertical mouth opening (lip_height / lip_width)
2. MAR_horizontal: Horizontal mouth extension (inner_width / outer_width)
3. Aspect_ratio: Mouth bounding box aspect ratio (width / height)
4. Lip_roundness: Lip contour roundness (4πA / P²)

Process:
1. Detect face with MediaPipe FaceMesh (full frame)
2. Extract mouth ROI with 20% padding
3. Re-run MediaPipe on cropped mouth for precise landmarks
4. Calculate 4 features from enhanced landmarks

Author: Deepfake Detection Team
Created: 2025-11-02
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging


class EnhancedMARExtractor:
    """
    Extract multiple mouth features using mouth cropping for enhanced precision

    MediaPipe FaceMesh Lip Landmarks (468-point model):
    - Outer lip contour: 40 points
    - Inner lip contour: 40 points
    - Key points used:
        Upper outer: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        Lower outer: [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        Upper inner: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        Lower inner: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    """

    # MediaPipe landmark indices
    UPPER_OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_OUTER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    UPPER_INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    LOWER_INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

    # Full lip contour (for area/perimeter calculation)
    FULL_LIP_CONTOUR = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                         375, 321, 405, 314, 17, 84, 181, 91, 146]

    LEFT_CORNER = 61
    RIGHT_CORNER = 291

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize Enhanced MAR extractor (v3.1: Coordinate Transformation)

        Uses coordinate transformation instead of mouth cropping:
        - MediaPipe runs once on full frame
        - Mouth landmarks normalized to mouth bounding box
        - 3-5x sensitivity increase without crop failures

        Args:
            static_image_mode: Treat each frame independently
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.logger = logging.getLogger(__name__)

        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.logger.info("Enhanced MAR extractor initialized (v3.1: coordinate transformation)")

    def extract_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        save_landmarks: bool = False
    ) -> Dict:
        """
        Extract multi-feature MAR timeline from video

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all frames)
            save_landmarks: If True, save frame and landmark data for visualization

        Returns:
            dict: {
                'mar_vertical': [0.25, 0.28, ...],
                'mar_horizontal': [0.75, 0.73, ...],
                'aspect_ratio': [1.85, 1.92, ...],
                'lip_roundness': [0.45, 0.42, ...],
                'timestamps': [0.0, 0.033, ...],
                'fps': 30.0,
                'total_frames': 600,
                'detected_frames': 580,
                'landmark_frames': []  # Only if save_landmarks=True
            }
        """
        self.logger.info(f"Extracting enhanced MAR from: {Path(video_path).name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        self.logger.info(f"Video: {total_frames} frames @ {fps:.2f} fps")

        # Initialize result arrays
        mar_vertical = []
        mar_horizontal = []
        aspect_ratio = []
        lip_roundness = []
        timestamps = []
        landmark_frames = [] if save_landmarks else None  # NEW: Store frames for visualization

        detected_count = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            timestamp = frame_idx / fps

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Step 1: Detect face in full frame
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # v3.1: Calculate multi-features using coordinate transformation
                # No cropping needed - MediaPipe runs once, coordinates transformed to mouth box
                features = self._calculate_multi_features_relative(face_landmarks)

                mar_vertical.append(features['mar_vertical'])
                mar_horizontal.append(features['mar_horizontal'])
                aspect_ratio.append(features['aspect_ratio'])
                lip_roundness.append(features['lip_roundness'])
                timestamps.append(timestamp)
                detected_count += 1

                # NEW: Save landmark data for visualization (if enabled)
                if save_landmarks:
                    landmark_frames.append({
                        'frame': frame.copy(),  # BGR frame
                        'landmarks': face_landmarks,
                        'timestamp': timestamp,
                        'frame_idx': frame_idx,
                        'features': features.copy()
                    })
            else:
                # No face detected
                self._append_nan(mar_vertical, mar_horizontal, aspect_ratio, lip_roundness, timestamps, timestamp)

            frame_idx += 1

            if frame_idx % 100 == 0:
                self.logger.info(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()

        self.logger.info(
            f"Enhanced MAR extraction complete: {detected_count}/{total_frames} frames detected "
            f"(v3.1: coordinate transformation)"
        )

        result = {
            'mar_vertical': mar_vertical,
            'mar_horizontal': mar_horizontal,
            'aspect_ratio': aspect_ratio,
            'lip_roundness': lip_roundness,
            'timestamps': timestamps,
            'fps': fps,
            'total_frames': total_frames,
            'detected_frames': detected_count
        }

        # Add landmark frames if visualization enabled
        if save_landmarks:
            result['landmark_frames'] = landmark_frames
            self.logger.info(f"Saved {len(landmark_frames)} landmark frames for visualization")

        return result

    def _append_nan(self, mar_v, mar_h, ar, lr, ts, timestamp):
        """Helper to append NaN values"""
        mar_v.append(np.nan)
        mar_h.append(np.nan)
        ar.append(np.nan)
        lr.append(np.nan)
        ts.append(timestamp)

    def _calculate_multi_features_relative(self, face_landmarks) -> Dict[str, float]:
        """
        Calculate 4 mouth features using face-height normalization (v3.2 - FIXED)

        Option 2: Normalize by face height instead of mouth box
        - Prevents ratio distortion from mouth box normalization
        - Expected MAR range: 0.1-0.9 (vs 0.03-0.1 in v3.1)

        Args:
            face_landmarks: MediaPipe face landmarks (468 points)

        Returns:
            dict: {
                'mar_vertical': float,      # Vertical opening (face-height normalized)
                'mar_horizontal': float,    # Horizontal extension
                'aspect_ratio': float,      # Width/Height ratio
                'lip_roundness': float      # Circularity (0-1)
            }
        """
        lm = face_landmarks.landmark

        # Step 1: Calculate face height from all 468 landmarks
        all_y_coords = [lm[i].y for i in range(len(lm))]
        face_height = max(all_y_coords) - min(all_y_coords)

        # Prevent division by zero
        if face_height < 1e-6:
            return {
                'mar_vertical': 0.0,
                'mar_horizontal': 0.0,
                'aspect_ratio': 0.0,
                'lip_roundness': 0.0
            }

        # Step 2: Calculate mouth bounding box (for aspect_ratio and roundness)
        mouth_x_coords = [lm[i].x for i in self.FULL_LIP_CONTOUR]
        mouth_y_coords = [lm[i].y for i in self.FULL_LIP_CONTOUR]

        x_min = min(mouth_x_coords)
        x_max = max(mouth_x_coords)
        y_min = min(mouth_y_coords)
        y_max = max(mouth_y_coords)

        mouth_width = x_max - x_min
        mouth_height = y_max - y_min

        # Define relative coordinate transformation functions for other features
        def to_relative_x(x):
            """Transform x to mouth box relative coordinate (0-1)"""
            return (x - x_min) / (mouth_width + 1e-6)

        def to_relative_y(y):
            """Transform y to mouth box relative coordinate (0-1)"""
            return (y - y_min) / (mouth_height + 1e-6)

        # Step 3: Feature 1 - MAR_vertical (FIXED: face-height normalized + outer lip)
        # Calculate absolute lip height and width using OUTER lip for true opening
        upper_ys = [lm[i].y for i in self.UPPER_OUTER_LIP]
        lower_ys = [lm[i].y for i in self.LOWER_OUTER_LIP]
        heights = [abs(u - l) for u, l in zip(upper_ys, lower_ys)]
        avg_height = np.mean(heights)

        left_x = lm[self.LEFT_CORNER].x
        right_x = lm[self.RIGHT_CORNER].x
        width = abs(right_x - left_x)

        # Normalize by face height (not mouth box!)
        height_norm = avg_height / face_height
        width_norm = width / face_height

        # MAR = normalized_height / normalized_width
        mar_vertical = height_norm / (width_norm + 1e-6)

        # Calculate width_rel for mar_horizontal (using mouth box relative coords)
        left_x_rel = to_relative_x(left_x)
        right_x_rel = to_relative_x(right_x)
        width_rel = abs(right_x_rel - left_x_rel)

        # Step 4: Feature 2 - MAR_horizontal (inner width / outer width)
        inner_left_rel = to_relative_x(lm[78].x)  # Inner left corner
        inner_right_rel = to_relative_x(lm[308].x)  # Inner right corner
        inner_width_rel = abs(inner_right_rel - inner_left_rel)

        outer_width_rel = width_rel  # Already calculated above

        mar_horizontal = inner_width_rel / (outer_width_rel + 1e-6)

        # Step 5: Feature 3 - Aspect_ratio (mouth bbox width/height)
        # This is already in mouth box coordinates (0-1), so bbox is [0,0] to [1,1]
        # Width and height are both 1.0 in relative coordinates
        # So we use the original mouth box dimensions
        aspect_ratio = mouth_width / (mouth_height + 1e-6)

        # Step 6: Feature 4 - Lip_roundness (circularity = 4πA / P²)
        # Transform all lip contour points to relative coordinates
        relative_points = [
            (to_relative_x(lm[i].x), to_relative_y(lm[i].y))
            for i in self.FULL_LIP_CONTOUR
        ]

        area = self._calculate_polygon_area(relative_points)
        perimeter = self._calculate_polygon_perimeter(relative_points)

        # Circularity: 1 = perfect circle, 0 = line
        lip_roundness = (4 * np.pi * area) / (perimeter**2 + 1e-6)

        # Clamp to [0, 1] (can exceed 1 due to noise)
        lip_roundness = min(1.0, max(0.0, lip_roundness))

        return {
            'mar_vertical': float(mar_vertical),
            'mar_horizontal': float(mar_horizontal),
            'aspect_ratio': float(aspect_ratio),
            'lip_roundness': float(lip_roundness)
        }

    def _calculate_polygon_area(self, points: List[Tuple[float, float]]) -> float:
        """Calculate area of polygon using Shoelace formula"""
        n = len(points)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return abs(area) / 2.0

    def _calculate_polygon_perimeter(self, points: List[Tuple[float, float]]) -> float:
        """Calculate perimeter of polygon"""
        n = len(points)
        if n < 2:
            return 0.0

        perimeter = 0.0
        for i in range(n):
            j = (i + 1) % n
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            perimeter += np.sqrt(dx**2 + dy**2)

        return perimeter

    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

    def draw_phoneme_landmarks(
        self,
        frame: np.ndarray,
        face_landmarks,
        phoneme: str,
        start_time: float,
        end_time: float,
        features: Dict[str, float],
        frame_position: str  # 'start', 'middle', 'end'
    ) -> np.ndarray:
        """
        Draw mouth landmarks on frame with phoneme information

        Args:
            frame: BGR frame (H, W, 3)
            face_landmarks: MediaPipe face landmarks
            phoneme: Phoneme Jamo character
            start_time: Phoneme start time
            end_time: Phoneme end time
            features: Dict with MAR feature values
            frame_position: Position within phoneme ('start', 'middle', 'end')

        Returns:
            Annotated frame with landmarks and info
        """
        vis_frame = frame.copy()
        h, w = frame.shape[:2]

        # Convert landmarks to pixel coordinates
        landmarks = face_landmarks.landmark

        # 1. Draw lip contour (20 points) - Green line
        lip_points = []
        for idx in self.FULL_LIP_CONTOUR:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            lip_points.append((x, y))

        # Draw lip contour as connected lines
        lip_points_np = np.array(lip_points, np.int32)
        cv2.polylines(vis_frame, [lip_points_np], True, (0, 255, 0), 2)

        # 2. Draw corner points - Red circles
        left_corner = landmarks[self.LEFT_CORNER]
        right_corner = landmarks[self.RIGHT_CORNER]

        left_x, left_y = int(left_corner.x * w), int(left_corner.y * h)
        right_x, right_y = int(right_corner.x * w), int(right_corner.y * h)

        cv2.circle(vis_frame, (left_x, left_y), 5, (0, 0, 255), -1)
        cv2.circle(vis_frame, (right_x, right_y), 5, (0, 0, 255), -1)

        # 3. Draw phoneme info at top
        phoneme_text = f"Phoneme: {phoneme} [{start_time:.2f}s - {end_time:.2f}s] ({frame_position})"
        cv2.putText(vis_frame, phoneme_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 4. Draw MAR feature values
        y_offset = 60
        feature_abbr = {
            'mar_vertical': 'V',
            'mar_horizontal': 'H',
            'aspect_ratio': 'A',
            'lip_roundness': 'R'
        }

        for name, value in features.items():
            abbr = feature_abbr.get(name, name[:1])
            text = f"{abbr}={value:.3f}"
            cv2.putText(vis_frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        return vis_frame


def main():
    """Test function"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_mar_extractor.py <video_path>")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Extract enhanced MAR (v3.1: coordinate transformation)
    extractor = EnhancedMARExtractor()
    result = extractor.extract_from_video(sys.argv[1])

    # Print results
    print("\n=== Enhanced MAR Extraction Results (v3.1) ===")
    print(f"Total frames: {result['total_frames']}")
    print(f"Detected frames: {result['detected_frames']}")
    print(f"FPS: {result['fps']:.2f}")
    print(f"Duration: {result['timestamps'][-1]:.2f}s")

    print(f"\n=== Multi-Feature Statistics ===")
    for feature_name in ['mar_vertical', 'mar_horizontal', 'aspect_ratio', 'lip_roundness']:
        values = np.array([v for v in result[feature_name] if not np.isnan(v)])
        if len(values) > 0:
            print(f"\n{feature_name}:")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  Std: {np.std(values):.3f}")
            print(f"  Min: {np.min(values):.3f}")
            print(f"  Max: {np.max(values):.3f}")
            print(f"  Range: [{np.percentile(values, 10):.3f}, {np.percentile(values, 90):.3f}]")


if __name__ == "__main__":
    main()
