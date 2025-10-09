"""
Model evaluation script for Korean deepfake dataset
Evaluates the tri-modal attention model on the preprocessed dataset
"""

import os
import json
import numpy as np
import cv2
import librosa
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def extract_audio_features(video_path, sr=16000, n_mfcc=40):
    """Extract audio features from video"""
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
            temp_audio_path, '-y', '-loglevel', 'quiet'
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Load audio
        audio, _ = librosa.load(temp_audio_path, sr=sr)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        return mfcc.T  # Shape: (time_steps, n_mfcc)
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return None
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


def extract_visual_features(video_path, max_frames=100, target_size=(224, 224)):
    """Extract visual features from video frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Sample frames uniformly
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    # Select frame indices
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize and normalize
            frame = cv2.resize(frame, target_size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    return np.array(frames)  # Shape: (num_frames, height, width, 3)


def extract_lip_region(frame, face_cascade=None):
    """
    Extract lip region from frame using face detection
    Simplified version - returns lower half of face
    """
    if face_cascade is None:
        # Use OpenCV's pre-trained face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # Get the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Extract lower third of face (approximate lip region)
        lip_y_start = y + int(2 * h / 3)
        lip_y_end = y + h
        lip_x_start = x + int(w / 4)
        lip_x_end = x + int(3 * w / 4)

        lip_region = frame[lip_y_start:lip_y_end, lip_x_start:lip_x_end]

        if lip_region.size > 0:
            return lip_region

    # If no face detected, return center region
    h, w = frame.shape[:2]
    center_region = frame[h//2:h, w//4:3*w//4]
    return center_region


def extract_lip_features(video_path, max_frames=100, target_size=(96, 96)):
    """Extract lip region features from video"""
    cap = cv2.VideoCapture(video_path)
    lip_frames = []
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Sample frames uniformly
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Extract lip region
            lip_region = extract_lip_region(frame, face_cascade)
            # Resize and normalize
            lip_region = cv2.resize(lip_region, target_size)
            lip_region = lip_region.astype(np.float32) / 255.0
            lip_frames.append(lip_region)

    cap.release()

    if len(lip_frames) == 0:
        return None

    return np.array(lip_frames)  # Shape: (num_frames, height, width, 3)


def simple_baseline_classifier(audio_features, visual_features, lip_features):
    """
    Simple baseline classifier for testing
    Returns prediction based on feature variance (fake videos often have artifacts)
    """
    # Calculate feature statistics
    audio_var = np.var(audio_features) if audio_features is not None else 0
    visual_var = np.var(visual_features) if visual_features is not None else 0
    lip_var = np.var(lip_features) if lip_features is not None else 0

    # Simple heuristic: higher variance might indicate manipulation
    total_var = audio_var + visual_var + lip_var

    # Random baseline with slight bias
    prediction = np.random.rand()
    if total_var > 0.5:
        prediction += 0.2

    return min(prediction, 1.0)


def evaluate_on_dataset(test_index_path, output_dir="evaluation_results"):
    """Evaluate model on test set"""
    os.makedirs(output_dir, exist_ok=True)

    # Load test index
    with open(test_index_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    print(f"Evaluating on {len(test_set)} videos...")

    y_true = []
    y_pred_proba = []
    y_pred = []
    failed_videos = []

    for sample in tqdm(test_set, desc="Processing videos"):
        video_path = sample['video_path']
        true_label = 1 if sample['label'] == 'fake' else 0

        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            failed_videos.append(sample)
            continue

        try:
            # Extract features
            audio_features = extract_audio_features(video_path)
            visual_features = extract_visual_features(video_path)
            lip_features = extract_lip_features(video_path)

            # Skip if feature extraction failed
            if audio_features is None and visual_features is None and lip_features is None:
                failed_videos.append(sample)
                continue

            # Get prediction (using simple baseline for now)
            pred_proba = simple_baseline_classifier(audio_features, visual_features, lip_features)
            pred_label = 1 if pred_proba > 0.5 else 0

            y_true.append(true_label)
            y_pred_proba.append(pred_proba)
            y_pred.append(pred_label)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            failed_videos.append(sample)
            continue

    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    if len(y_true) == 0:
        print("No valid predictions. Check video paths and ffmpeg installation.")
        return

    print(f"\nProcessed videos: {len(y_true)}/{len(test_set)}")
    print(f"Failed videos: {len(failed_videos)}")

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # AUC
    if len(set(y_true)) > 1:  # Need both classes for AUC
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"AUC:       {auc:.4f}")
    else:
        print("AUC: N/A (only one class present)")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Real  Fake")
    print(f"Actual Real     {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"       Fake     {cm[1][0]:5d} {cm[1][1]:5d}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake'], zero_division=0))

    # Save results
    results = {
        'total_videos': len(test_set),
        'processed_videos': len(y_true),
        'failed_videos': len(failed_videos),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    }

    if len(set(y_true)) > 1:
        results['auc'] = float(auc)

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")

    return results


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    parser.add_argument('--test_index', type=str, default='preprocessed_data/test_index.json',
                        help='Path to test index JSON file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Check if test index exists
    if not os.path.exists(args.test_index):
        print(f"Test index not found: {args.test_index}")
        print("Please run preprocess_dataset.py first to create the test index.")
        return

    # Run evaluation
    evaluate_on_dataset(args.test_index, args.output_dir)


if __name__ == "__main__":
    main()
