"""
Utility functions for audio and video processing
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def save_array_to_npy(array: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save numpy array to .npy file

    Args:
        array: Numpy array to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    logger.debug(f"Saved array to {output_path}: shape={array.shape}")


def load_array_from_npy(input_path: Union[str, Path]) -> np.ndarray:
    """
    Load numpy array from .npy file

    Args:
        input_path: Input file path

    Returns:
        Loaded numpy array
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    array = np.load(input_path)
    logger.debug(f"Loaded array from {input_path}: shape={array.shape}")
    return array


def save_metadata_to_json(metadata: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save metadata to JSON file

    Args:
        metadata: Metadata dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    metadata = convert_numpy_types(metadata)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.debug(f"Saved metadata to {output_path}")


def load_metadata_from_json(input_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metadata from JSON file

    Args:
        input_path: Input file path

    Returns:
        Metadata dictionary
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    logger.debug(f"Loaded metadata from {input_path}")
    return metadata


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization

    Args:
        obj: Object to convert

    Returns:
        Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return 0.0
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_video_id_from_path(video_path: Union[str, Path]) -> str:
    """
    Extract video ID from file path

    Args:
        video_path: Path to video file

    Returns:
        Video ID (filename without extension)
    """
    video_path = Path(video_path)
    return video_path.stem


def create_output_structure(
    output_dir: Union[str, Path],
    subdirs: List[str] = None
) -> Dict[str, Path]:
    """
    Create output directory structure

    Args:
        output_dir: Base output directory
        subdirs: List of subdirectories to create

    Returns:
        Dict mapping subdirectory names to paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subdirs = subdirs or ['audio', 'frames', 'metadata']
    paths = {'root': output_dir}

    for subdir in subdirs:
        subdir_path = output_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        paths[subdir] = subdir_path

    logger.info(f"Created output structure at {output_dir}")
    return paths


def get_dataset_videos(
    dataset_root: Union[str, Path],
    pattern: str = '**/*.mp4'
) -> List[Path]:
    """
    Get list of video files from dataset directory

    Args:
        dataset_root: Root directory of dataset
        pattern: Glob pattern for video files (default: '**/*.mp4' for recursive search)

    Returns:
        List of video file paths
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # Use rglob for recursive search if pattern starts with **
    if pattern.startswith('**'):
        # Extract extension from pattern (e.g., '**/*.mp4' -> '*.mp4')
        file_pattern = pattern.split('/')[-1] if '/' in pattern else pattern.replace('**', '*')
        video_files = sorted(dataset_root.rglob(file_pattern))
    else:
        video_files = sorted(dataset_root.glob(pattern))

    logger.info(f"Found {len(video_files)} videos in {dataset_root}")
    return video_files


def split_list(items: list, n_splits: int) -> List[list]:
    """
    Split list into n roughly equal parts

    Args:
        items: List to split
        n_splits: Number of splits

    Returns:
        List of sublists
    """
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")

    if n_splits >= len(items):
        return [[item] for item in items]

    chunk_size = len(items) // n_splits
    remainder = len(items) % n_splits

    splits = []
    start = 0

    for i in range(n_splits):
        # Add one extra item to first 'remainder' chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        splits.append(items[start:end])
        start = end

    return splits


def validate_array_shape(
    array: np.ndarray,
    expected_ndim: int,
    name: str = "array"
) -> None:
    """
    Validate numpy array dimensions

    Args:
        array: Array to validate
        expected_ndim: Expected number of dimensions
        name: Name for error messages

    Raises:
        ValueError if validation fails
    """
    if array.ndim != expected_ndim:
        raise ValueError(
            f"{name} expected {expected_ndim}D array, got {array.ndim}D: shape={array.shape}"
        )


def print_processing_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print summary of processing results

    Args:
        results: List of processing results
    """
    total = len(results)
    successful = sum(1 for r in results if r['metadata']['success'])
    with_audio = sum(1 for r in results if r['audio'] is not None)
    with_frames = sum(1 for r in results if r['frames'] is not None)

    total_time = sum(r['metadata'].get('processing_time', 0) for r in results)
    avg_time = total_time / total if total > 0 else 0

    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total videos:      {total}")
    print(f"Successful:        {successful} ({successful/total*100:.1f}%)")
    print(f"With audio:        {with_audio}")
    print(f"With frames:       {with_frames}")
    print(f"Total time:        {format_time(total_time)}")
    print(f"Average time:      {format_time(avg_time)}")
    print("="*60 + "\n")


def get_system_info() -> Dict[str, Any]:
    """
    Get system information

    Returns:
        Dict with system info
    """
    import platform
    import psutil

    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3)
    }

    return info


def setup_logging(
    log_file: Union[str, Path] = None,
    level: int = logging.INFO
) -> None:
    """
    Setup logging configuration

    Args:
        log_file: Path to log file (None = console only)
        level: Logging level
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger.info("Logging configured")
