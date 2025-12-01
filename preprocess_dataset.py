"""
Dataset preprocessing script for Korean deepfake dataset
Converts dataset_sample to format compatible with audio-visual-deepfake models
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import librosa
import pickle


def extract_audio_from_video(video_path, sr=16000):
    """Extract audio from video file"""
    import subprocess
    import tempfile

    # Create temporary audio file
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()

    try:
        # Extract audio using ffmpeg
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sr), '-ac', '1',
            temp_audio_path, '-y'
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Load audio
        audio, _ = librosa.load(temp_audio_path, sr=sr)
        return audio
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return None
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


def extract_frames_from_video(video_path, max_frames=None):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames


def get_video_info(video_path):
    """Get video metadata"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height
    }


def create_dataset_index(dataset_root):
    """
    Create dataset index from Korean deepfake dataset

    Structure:
    - dataset_sample/
      - 원천데이터/
        - train_변조/ (fake videos)
        - train_원본/ (original videos)
      - 라벨링데이터/
        - train_meta_data/
          - 변조영상_training_메타데이터.csv
          - 원본영상_training_메타데이터.csv
    """
    dataset_root = Path(dataset_root)

    # Read metadata
    fake_meta_path = dataset_root / "라벨링데이터" / "train_meta_data" / "변조영상_training_메타데이터.csv"
    orig_meta_path = dataset_root / "라벨링데이터" / "train_meta_data" / "원본영상_training_메타데이터.csv"

    fake_meta = pd.read_csv(fake_meta_path, encoding='utf-8-sig')
    orig_meta = pd.read_csv(orig_meta_path, encoding='utf-8-sig')

    # Find all video files
    fake_video_root = dataset_root / "원천데이터" / "train_변조"
    orig_video_root = dataset_root / "원천데이터" / "train_원본"

    dataset_index = []

    # Process fake videos
    print("Processing fake videos...")
    for idx, row in tqdm(fake_meta.iterrows(), total=len(fake_meta)):
        video_id = row['영상ID']

        # Find video file
        video_files = list(fake_video_root.rglob(f"{video_id}"))

        if video_files:
            video_path = video_files[0]
            dataset_index.append({
                'video_path': str(video_path),
                'video_id': video_id,
                'label': 'fake',
                'manipulation_type': row.get('변조모델', 'unknown'),
                'gender': row.get('인물성별', 'unknown'),
                'target_video': row.get('타겟영상', ''),
                'source_uuid': row.get('소스UUID', ''),
                'target_uuid': row.get('타겟UUID', '')
            })

    # Process original videos
    print("Processing original videos...")
    for idx, row in tqdm(orig_meta.iterrows(), total=len(orig_meta)):
        video_id = row['영상ID']

        # Find video file
        video_files = list(orig_video_root.rglob(f"{video_id}"))

        if video_files:
            video_path = video_files[0]
            dataset_index.append({
                'video_path': str(video_path),
                'video_id': video_id,
                'label': 'real',
                'manipulation_type': 'none',
                'gender': row.get('인물성별', 'unknown'),
                'uuid': row.get('UUID', ''),
                'script_file': row.get('스크립트파일', '')
            })

    return dataset_index


def save_dataset_index(dataset_index, output_path):
    """Save dataset index to JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_index, f, indent=2, ensure_ascii=False)
    print(f"Dataset index saved to {output_path}")


def create_train_test_split(dataset_index, test_ratio=0.2, random_seed=42):
    """Split dataset into train and test sets"""
    np.random.seed(random_seed)

    # Separate by label
    fake_samples = [s for s in dataset_index if s['label'] == 'fake']
    real_samples = [s for s in dataset_index if s['label'] == 'real']

    # Shuffle
    np.random.shuffle(fake_samples)
    np.random.shuffle(real_samples)

    # Split
    fake_split_idx = int(len(fake_samples) * (1 - test_ratio))
    real_split_idx = int(len(real_samples) * (1 - test_ratio))

    train_set = fake_samples[:fake_split_idx] + real_samples[:real_split_idx]
    test_set = fake_samples[fake_split_idx:] + real_samples[real_split_idx:]

    # Shuffle again
    np.random.shuffle(train_set)
    np.random.shuffle(test_set)

    return train_set, test_set


def main():
    """Main preprocessing function"""

    # Configuration
    dataset_root = "dataset_sample"
    output_dir = "preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset index
    print("Creating dataset index...")
    dataset_index = create_dataset_index(dataset_root)

    print(f"\nDataset statistics:")
    print(f"Total videos: {len(dataset_index)}")
    print(f"Fake videos: {sum(1 for s in dataset_index if s['label'] == 'fake')}")
    print(f"Real videos: {sum(1 for s in dataset_index if s['label'] == 'real')}")

    # Save full index
    save_dataset_index(dataset_index, os.path.join(output_dir, "dataset_index.json"))

    # Create train/test split
    print("\nCreating train/test split...")
    train_set, test_set = create_train_test_split(dataset_index)

    print(f"\nSplit statistics:")
    print(f"Train set: {len(train_set)} videos")
    print(f"  - Fake: {sum(1 for s in train_set if s['label'] == 'fake')}")
    print(f"  - Real: {sum(1 for s in train_set if s['label'] == 'real')}")
    print(f"Test set: {len(test_set)} videos")
    print(f"  - Fake: {sum(1 for s in test_set if s['label'] == 'fake')}")
    print(f"  - Real: {sum(1 for s in test_set if s['label'] == 'real')}")

    # Save splits
    save_dataset_index(train_set, os.path.join(output_dir, "train_index.json"))
    save_dataset_index(test_set, os.path.join(output_dir, "test_index.json"))

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
