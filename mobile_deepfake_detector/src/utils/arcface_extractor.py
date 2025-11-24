"""
ArcFace Per-Frame Feature Extraction for Korean Deepfake Detection

This module extracts 512-dimensional ArcFace embeddings from video frames using
InsightFace's buffalo_l model, following the PIA paper's approach.

ArcFace embeddings capture facial identity information and are used to detect
temporal inconsistencies in deepfake videos (identity drift across frames).

Reference:
    PIA: "Phoneme-Temporal and Identity-Dynamic Analysis" (Section 4.3)
    InsightFace: https://github.com/deepinsight/insightface
"""

import os
import cv2
import time
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None  # Type hint placeholder


class ArcFaceExtractor:
    """
    Extract ArcFace embeddings from video frames using InsightFace.

    This extractor uses the buffalo_l model from InsightFace, which provides
    high-quality 512-dimensional face embeddings with ArcFace loss training.

    Attributes:
        device (str): Device for computation ('cuda' or 'cpu')
        model_name (str): InsightFace model name (default: 'buffalo_l')
        face_app (FaceAnalysis): InsightFace face analysis application
        logger (logging.Logger): Logger instance
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ArcFace extractor.

        Args:
            device: Device for computation ('cuda' or 'cpu')
            model_name: InsightFace model name (buffalo_l, buffalo_s, etc.)
            det_size: Detection input size (width, height)
            logger: Optional logger instance

        Raises:
            ImportError: If insightface is not installed
            RuntimeError: If model cannot be loaded
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "insightface is not installed. Install with: "
                "pip install insightface onnxruntime-gpu"
            )

        self.device = device
        self.model_name = model_name
        self.det_size = det_size
        self.logger = logger or logging.getLogger(__name__)

        # Initialize InsightFace
        self.logger.info(f"Initializing ArcFace extractor (model={model_name}, device={device})...")
        self.face_app = self._load_face_app()
        self.logger.info("ArcFace extractor initialized successfully")

    def _load_face_app(self) -> FaceAnalysis:
        """
        Load InsightFace FaceAnalysis application.

        Returns:
            Initialized FaceAnalysis instance

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Set execution providers (GPU or CPU)
            if self.device == "cuda" and torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ctx_id = 0  # GPU device ID
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1  # CPU

            # Initialize FaceAnalysis
            face_app = FaceAnalysis(
                name=self.model_name,
                providers=providers
            )

            # Prepare model
            face_app.prepare(
                ctx_id=ctx_id,
                det_size=self.det_size
            )

            return face_app

        except Exception as e:
            self.logger.error(f"Failed to load InsightFace model: {e}")
            raise RuntimeError(f"InsightFace model loading failed: {e}")

    def extract_from_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        frame_indices: Optional[List[int]] = None,
        skip_frames: int = 0,
        show_progress: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract ArcFace embeddings from video frames with batch processing.

        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (None = all, legacy parameter)
            frame_indices: Specific frame indices to extract (overrides max_frames)
                - If provided, extracts embeddings only for these frames
                - Example: [0, 12, 24, 36, ...] for uniform sampling
                - Ensures synchronization with VideoPreprocessor
            skip_frames: Number of frames to skip between extractions (0 = no skip)
            show_progress: Whether to log progress updates
            batch_size: Number of frames to load into memory at once (default: 32)
                - Larger batch = better GPU utilization but more memory
                - Recommended: 16-64 depending on VRAM availability

        Returns:
            Array of shape (num_frames, 512) containing ArcFace embeddings.
            Frames without detected faces are filled with zeros.

        Example:
            >>> extractor = ArcFaceExtractor(device="cuda")
            >>> embeddings = extractor.extract_from_video("test.mp4", frame_indices=[0, 10, 20], batch_size=32)
            >>> print(embeddings.shape)  # (3, 512) for 3 frames
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.logger.info(f"Extracting ArcFace embeddings from: {video_path}")
        t0 = time.time()

        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Determine which frames to extract
        if frame_indices is not None:
            # PIA mode: Extract specific frames only (synchronized with VideoPreprocessor)
            frames_to_extract = sorted(frame_indices)
            total_frames = len(frames_to_extract)
            self.logger.info(f"Frame-specific extraction: {total_frames} frames from {frames_to_extract[0]} to {frames_to_extract[-1]}")
        else:
            # Legacy mode: All frames or limited by max_frames
            if max_frames:
                frames_to_extract = list(range(min(max_frames, total_frames_in_video)))
                total_frames = len(frames_to_extract)
            else:
                frames_to_extract = list(range(total_frames_in_video))
                total_frames = total_frames_in_video

        self.logger.info(f"Video: {total_frames} frames @ {fps:.2f} FPS (batch_size={batch_size})")

        # ===== Batch Processing =====
        # Process frames in batches to optimize GPU utilization
        embeddings = []
        num_batches = (total_frames + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_frames)
            batch_frame_indices = frames_to_extract[batch_start:batch_end]

            # Step 1: Load all frames in batch into memory (minimize I/O latency)
            batch_frames = []
            for frame_idx in batch_frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    batch_frames.append(frame)
                else:
                    batch_frames.append(None)  # Failed frame

            # Step 2: Process loaded frames consecutively (maximize GPU throughput)
            for frame in batch_frames:
                if frame is not None:
                    embedding = self._extract_from_frame(frame)
                else:
                    embedding = np.zeros(512, dtype=np.float32)
                embeddings.append(embedding)

            # Progress logging per batch
            if show_progress:
                processed = batch_end
                elapsed = time.time() - t0
                fps_proc = processed / elapsed if elapsed > 0 else 0
                self.logger.info(
                    f"[Batch {batch_idx+1}/{num_batches}] Processed {processed}/{total_frames} frames "
                    f"({fps_proc:.1f} FPS)"
                )

        cap.release()

        # Convert to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)

        elapsed = time.time() - t0
        self.logger.info(
            f"Extracted {len(embeddings)} embeddings in {elapsed:.2f}s "
            f"({len(embeddings)/elapsed:.1f} FPS)"
        )

        return embeddings

    def _extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract ArcFace embedding from a single frame.

        Args:
            frame: BGR image from cv2.VideoCapture (H, W, 3)

        Returns:
            512-dimensional ArcFace embedding. Returns zeros if no face detected.
        """
        try:
            # Detect faces and extract features
            faces = self.face_app.get(frame)

            # Get embedding from first detected face
            if len(faces) > 0 and hasattr(faces[0], "embedding"):
                embedding = faces[0].embedding  # (512,) numpy array
                return embedding
            else:
                # No face detected - return zero vector
                return np.zeros(512, dtype=np.float32)

        except Exception as e:
            # Error during extraction - return zero vector
            self.logger.warning(f"Face extraction failed: {e}")
            return np.zeros(512, dtype=np.float32)

    def extract_from_frames(
        self,
        frames: List[np.ndarray],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Extract ArcFace embeddings from a list of frames.

        Args:
            frames: List of BGR images (each H x W x 3)
            show_progress: Whether to log progress

        Returns:
            Array of shape (num_frames, 512) containing ArcFace embeddings

        Example:
            >>> frames = [cv2.imread(f"frame_{i:04d}.jpg") for i in range(50)]
            >>> extractor = ArcFaceExtractor(device="cuda")
            >>> embeddings = extractor.extract_from_frames(frames)
            >>> print(embeddings.shape)  # (50, 512)
        """
        self.logger.info(f"Extracting ArcFace from {len(frames)} frames...")
        t0 = time.time()

        embeddings = []
        for i, frame in enumerate(frames):
            embedding = self._extract_from_frame(frame)
            embeddings.append(embedding)

            if show_progress and (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i+1}/{len(frames)} frames")

        embeddings = np.array(embeddings, dtype=np.float32)

        elapsed = time.time() - t0
        self.logger.info(
            f"Extracted {len(embeddings)} embeddings in {elapsed:.2f}s"
        )

        return embeddings

    def get_embedding_stats(self, embeddings: np.ndarray) -> dict:
        """
        Get statistics about extracted embeddings.

        Args:
            embeddings: Array of shape (num_frames, 512)

        Returns:
            Dictionary with embedding statistics
        """
        num_frames = len(embeddings)
        zero_frames = np.sum(np.all(embeddings == 0, axis=1))
        valid_frames = num_frames - zero_frames

        # Compute embedding norms (magnitude)
        norms = np.linalg.norm(embeddings, axis=1)
        avg_norm = np.mean(norms[norms > 0]) if valid_frames > 0 else 0

        # Compute temporal consistency (cosine similarity between consecutive frames)
        if valid_frames > 1:
            similarities = []
            for i in range(len(embeddings) - 1):
                emb1, emb2 = embeddings[i], embeddings[i + 1]
                # Skip if either embedding is zero
                if np.any(emb1) and np.any(emb2):
                    cos_sim = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2)
                    )
                    similarities.append(cos_sim)

            avg_similarity = np.mean(similarities) if similarities else 0
            std_similarity = np.std(similarities) if similarities else 0
        else:
            avg_similarity = 0
            std_similarity = 0

        return {
            "total_frames": num_frames,
            "valid_frames": valid_frames,
            "zero_frames": zero_frames,
            "detection_rate": valid_frames / num_frames if num_frames > 0 else 0,
            "avg_embedding_norm": float(avg_norm),
            "avg_temporal_similarity": float(avg_similarity),
            "std_temporal_similarity": float(std_similarity)
        }


# ============================================================================
# Utility Functions
# ============================================================================

def extract_arcface_features(
    video_path: str,
    device: str = "cuda",
    logger: Optional[logging.Logger] = None
) -> List[List[float]]:
    """
    Extract ArcFace features from video (PIA-compatible interface).

    This function matches PIA's original API for easy integration.

    Args:
        video_path: Path to input video
        device: Device for computation ('cuda' or 'cpu')
        logger: Optional logger instance

    Returns:
        List of 512-dimensional embeddings (one per frame)

    Example:
        >>> features = extract_arcface_features("test.mp4", device="cuda")
        >>> print(len(features), len(features[0]))  # num_frames, 512
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("[STEP] ArcFace embeddings...")
    t0 = time.time()

    # Initialize extractor
    extractor = ArcFaceExtractor(device=device, logger=logger)

    # Extract embeddings
    embeddings = extractor.extract_from_video(video_path, show_progress=False)

    # Convert to list of lists (PIA format)
    features = embeddings.tolist()

    elapsed = time.time() - t0
    logger.info(
        f"[OK] ArcFace in {elapsed:.1f}s (frames={len(features)})"
    )

    return features


def compute_temporal_consistency_loss(
    arcface_embeddings: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute ArcFace temporal consistency loss (PIA Section 4.3).

    L_arcface = Σ(1 - cos(a_t, a_{t+1})) · m_t · m_{t+1} / Σ(m_t · m_{t+1}) + ε

    where:
        a_t = ArcFace embedding at frame t (512-dim)
        m_t = binary mask (1 if valid frame, 0 if silence/no face)
        cos(a_t, a_{t+1}) = cosine similarity between consecutive embeddings

    Args:
        arcface_embeddings: Array of shape (T, 512) where T = num_frames
        mask: Binary mask of shape (T,) indicating valid frames (default: all 1s)

    Returns:
        Temporal consistency loss value (lower = more consistent identity)

    Example:
        >>> embeddings = np.random.randn(100, 512)  # 100 frames
        >>> loss = compute_temporal_consistency_loss(embeddings)
        >>> print(loss)  # ~1.0 for random embeddings
    """
    T = len(arcface_embeddings)
    if T < 2:
        return 0.0

    # Default mask: all frames valid
    if mask is None:
        mask = np.ones(T, dtype=np.float32)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(arcface_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized_embs = arcface_embeddings / norms

    # Compute cosine similarities between consecutive frames
    cos_similarities = np.sum(
        normalized_embs[:-1] * normalized_embs[1:],
        axis=1
    )  # (T-1,)

    # Apply mask: only consider consecutive valid frames
    mask_pairs = mask[:-1] * mask[1:]  # (T-1,)

    # Compute loss: (1 - cosine_similarity) weighted by mask
    losses = (1.0 - cos_similarities) * mask_pairs

    # Average over valid pairs (add epsilon to avoid division by zero)
    epsilon = 1e-8
    total_loss = np.sum(losses) / (np.sum(mask_pairs) + epsilon)

    return float(total_loss)
