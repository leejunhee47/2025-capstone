"""
Preprocessing pipeline for batch video processing
Handles large-scale dataset preprocessing with multiprocessing
"""

import os
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
from tqdm import tqdm

from config import Config
from video_processor import VideoProcessor
from utils import (
    save_array_to_npy,
    save_metadata_to_json,
    create_output_structure,
    get_video_id_from_path,
    split_list,
    print_processing_summary
)

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Pipeline for preprocessing video datasets
    Supports batch processing with multiprocessing
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Optional[Config] = None,
        num_workers: Optional[int] = None
    ):
        """
        Initialize PreprocessingPipeline

        Args:
            output_dir: Output directory for preprocessed data
            config: Configuration object
            num_workers: Number of worker processes (None = use config)
        """
        self.config = config or Config()
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers or self.config.NUM_WORKERS

        # Create output structure
        self.output_paths = create_output_structure(
            self.output_dir,
            subdirs=['audio', 'frames', 'metadata']
        )

        logger.info(f"Initialized pipeline with output dir: {self.output_dir}")
        logger.info(f"Number of workers: {self.num_workers}")

    def preprocess_dataset(
        self,
        video_paths: List[Union[str, Path]],
        extract_audio: bool = True,
        extract_frames: bool = True,
        audio_feature_type: str = 'raw',
        max_frames: Optional[int] = None,
        use_multiprocessing: bool = True,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Preprocess entire dataset

        Args:
            video_paths: List of video file paths
            extract_audio: Extract audio
            extract_frames: Extract frames
            audio_feature_type: Type of audio features
            max_frames: Maximum frames per video
            use_multiprocessing: Use multiprocessing
            save_results: Save extracted features to disk

        Returns:
            List of processing results
        """
        logger.info(f"Starting preprocessing of {len(video_paths)} videos")
        start_time = time.time()

        if use_multiprocessing and self.num_workers > 1:
            results = self._process_multiprocessing(
                video_paths,
                extract_audio,
                extract_frames,
                audio_feature_type,
                max_frames
            )
        else:
            results = self._process_sequential(
                video_paths,
                extract_audio,
                extract_frames,
                audio_feature_type,
                max_frames
            )

        # Save results to disk
        if save_results:
            logger.info("Saving preprocessed data to disk")
            self._save_results(results)

        # Save dataset index
        dataset_index = self._create_dataset_index(results)
        save_metadata_to_json(
            dataset_index,
            self.output_paths['root'] / 'dataset_index.json'
        )

        total_time = time.time() - start_time
        logger.info(f"Preprocessing completed in {total_time:.2f}s")

        # Print summary
        print_processing_summary(results)

        return results

    def _process_sequential(
        self,
        video_paths: List[Union[str, Path]],
        extract_audio: bool,
        extract_frames: bool,
        audio_feature_type: str,
        max_frames: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Process videos sequentially"""
        processor = VideoProcessor(self.config)
        results = []

        for video_path in tqdm(video_paths, desc="Processing videos"):
            try:
                result = processor.process_video(
                    video_path,
                    extract_audio=extract_audio,
                    extract_frames=extract_frames,
                    audio_feature_type=audio_feature_type,
                    max_frames=max_frames
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results.append({
                    'audio': None,
                    'frames': None,
                    'metadata': {
                        'video_path': str(video_path),
                        'success': False,
                        'error': str(e)
                    }
                })

        return results

    def _process_multiprocessing(
        self,
        video_paths: List[Union[str, Path]],
        extract_audio: bool,
        extract_frames: bool,
        audio_feature_type: str,
        max_frames: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Process videos using multiprocessing"""
        logger.info(f"Using multiprocessing with {self.num_workers} workers")

        # Create worker function
        worker_fn = partial(
            _process_single_video,
            extract_audio=extract_audio,
            extract_frames=extract_frames,
            audio_feature_type=audio_feature_type,
            max_frames=max_frames,
            config=self.config
        )

        # Process in parallel
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_fn, video_paths),
                total=len(video_paths),
                desc="Processing videos"
            ))

        return results

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save processing results to disk"""
        saved_count = 0

        for result in tqdm(results, desc="Saving results"):
            if not result['metadata']['success']:
                continue

            video_id = get_video_id_from_path(result['metadata']['video_path'])

            # Save audio
            if result['audio'] is not None:
                audio_path = self.output_paths['audio'] / f"{video_id}.npy"
                save_array_to_npy(result['audio'], audio_path)

            # Save frames
            if result['frames'] is not None:
                frames_path = self.output_paths['frames'] / f"{video_id}.npy"
                save_array_to_npy(result['frames'], frames_path)

            # Save metadata
            metadata_path = self.output_paths['metadata'] / f"{video_id}.json"
            save_metadata_to_json(result['metadata'], metadata_path)

            saved_count += 1

        logger.info(f"Saved {saved_count} processed videos")

    def _create_dataset_index(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create dataset index from results"""
        index = {
            'total_videos': len(results),
            'successful': sum(1 for r in results if r['metadata']['success']),
            'with_audio': sum(1 for r in results if r['audio'] is not None),
            'with_frames': sum(1 for r in results if r['frames'] is not None),
            'videos': []
        }

        for result in results:
            video_entry = {
                'video_path': result['metadata']['video_path'],
                'video_id': get_video_id_from_path(result['metadata']['video_path']),
                'success': result['metadata']['success']
            }

            if result['audio'] is not None:
                video_entry['audio_file'] = f"audio/{video_entry['video_id']}.npy"
                video_entry['audio_shape'] = result['metadata'].get('audio_shape')

            if result['frames'] is not None:
                video_entry['frames_file'] = f"frames/{video_entry['video_id']}.npy"
                video_entry['frames_shape'] = result['metadata'].get('frames_shape')

            index['videos'].append(video_entry)

        return index

    def preprocess_from_index(
        self,
        index_file: Union[str, Path],
        video_root: Union[str, Path],
        extract_audio: bool = True,
        extract_frames: bool = True,
        audio_feature_type: str = 'raw',
        max_frames: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Preprocess videos listed in an index file

        Args:
            index_file: Path to dataset index JSON file
            video_root: Root directory containing videos
            extract_audio: Extract audio
            extract_frames: Extract frames
            audio_feature_type: Type of audio features
            max_frames: Maximum frames per video

        Returns:
            List of processing results
        """
        import json

        index_file = Path(index_file)
        video_root = Path(video_root)

        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        # Build video paths
        video_paths = []
        for entry in index:
            video_path = video_root / entry.get('video_path', '')
            if video_path.exists():
                video_paths.append(video_path)
            else:
                logger.warning(f"Video not found: {video_path}")

        logger.info(f"Found {len(video_paths)}/{len(index)} videos from index")

        return self.preprocess_dataset(
            video_paths,
            extract_audio=extract_audio,
            extract_frames=extract_frames,
            audio_feature_type=audio_feature_type,
            max_frames=max_frames
        )

    def resume_preprocessing(
        self,
        video_paths: List[Union[str, Path]],
        extract_audio: bool = True,
        extract_frames: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Resume preprocessing, skipping already processed videos

        Args:
            video_paths: List of video paths
            extract_audio: Extract audio
            extract_frames: Extract frames

        Returns:
            List of processing results
        """
        # Find videos that haven't been processed yet
        remaining_videos = []

        for video_path in video_paths:
            video_id = get_video_id_from_path(video_path)

            audio_exists = (self.output_paths['audio'] / f"{video_id}.npy").exists()
            frames_exists = (self.output_paths['frames'] / f"{video_id}.npy").exists()

            needs_processing = False
            if extract_audio and not audio_exists:
                needs_processing = True
            if extract_frames and not frames_exists:
                needs_processing = True

            if needs_processing:
                remaining_videos.append(video_path)

        logger.info(
            f"Resume preprocessing: {len(remaining_videos)}/{len(video_paths)} "
            f"videos remaining"
        )

        if len(remaining_videos) == 0:
            logger.info("All videos already processed")
            return []

        return self.preprocess_dataset(
            remaining_videos,
            extract_audio=extract_audio,
            extract_frames=extract_frames
        )


def _process_single_video(
    video_path: Union[str, Path],
    extract_audio: bool,
    extract_frames: bool,
    audio_feature_type: str,
    max_frames: Optional[int],
    config: Config
) -> Dict[str, Any]:
    """
    Worker function for multiprocessing

    Args:
        video_path: Path to video
        extract_audio: Extract audio
        extract_frames: Extract frames
        audio_feature_type: Type of audio features
        max_frames: Maximum frames
        config: Configuration

    Returns:
        Processing result
    """
    try:
        processor = VideoProcessor(config)
        result = processor.process_video(
            video_path,
            extract_audio=extract_audio,
            extract_frames=extract_frames,
            audio_feature_type=audio_feature_type,
            max_frames=max_frames
        )
        return result
    except Exception as e:
        logger.error(f"Worker failed to process {video_path}: {e}")
        return {
            'audio': None,
            'frames': None,
            'metadata': {
                'video_path': str(video_path),
                'success': False,
                'error': str(e)
            }
        }
