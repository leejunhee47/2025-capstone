"""
E2E Test Script V4 - Fresh Cache Generation with New Key Format

기존 캐시를 무시하고 50개 영상에 대해 새 형식(video_id만 사용)으로
전처리 및 캐시를 생성합니다.

Usage:
    python test_e2e_v4.py
    python test_e2e_v4.py --skip-existing  # 이미 처리된 영상 스킵
"""

# IMPORTANT: Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.xai.hybrid_pipeline import HybridXAIPipeline
from src.utils.feature_cache import FeatureCache

# sklearn metrics
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Metrics will be limited.")


# Default paths
DEFAULT_INPUT = Path(__file__).parent / "test_e2e_v2" / "test_result.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "test_e2e_v4"
MMMS_MODEL = "models/checkpoints/mmms-ba_fulldata_best.pth"
PIA_MODEL = "models/checkpoints/pia-best.pth"


def load_previous_results(input_path: str) -> Dict:
    """Load previous test results (test_e2e_v2)"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clear_cache_for_video(video_id: str):
    """Clear existing cache for video to force re-extraction"""
    cache = FeatureCache()

    # New format (video_id only)
    new_cache_file = cache.cache_dir / f"{video_id}.npz"
    if new_cache_file.exists():
        new_cache_file.unlink()
        print(f"    Cleared cache: {new_cache_file.name}")

    # Update index
    if video_id in cache.index:
        del cache.index[video_id]
        cache._save_index()


def run_single_video(
    pipeline: HybridXAIPipeline,
    entry: Dict,
    output_base_dir: Path,
    force_extract: bool = True
) -> Dict:
    """Run pipeline on a single video with fresh extraction"""
    video_id = entry['video_id']
    video_path = entry['video_path']
    ground_truth = entry['ground_truth']

    # Output directory: real/video_id or fake/video_id
    video_output_dir = output_base_dir / ground_truth / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing cache if force_extract
    if force_extract:
        clear_cache_for_video(video_id)

    start_time = time.time()

    try:
        # Process video (will extract fresh and save to new cache format)
        result = pipeline.process_video(
            video_path=video_path,
            video_id=video_id,
            output_dir=str(video_output_dir)
        )

        processing_time = time.time() - start_time

        predicted = result['detection']['verdict']
        confidence = result['detection']['confidence']
        fake_prob = result['detection']['probabilities']['fake']
        correct = (predicted == ground_truth)

        return {
            'video_id': video_id,
            'video_path': video_path,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'confidence': confidence,
            'fake_prob': fake_prob,
            'correct': correct,
            'processing_time': processing_time,
            'output_dir': str(video_output_dir),
            'error': None
        }

    except Exception as e:
        processing_time = time.time() - start_time
        return {
            'video_id': video_id,
            'video_path': video_path,
            'ground_truth': ground_truth,
            'predicted': 'error',
            'confidence': 0.0,
            'fake_prob': 0.5,
            'correct': False,
            'processing_time': processing_time,
            'output_dir': str(video_output_dir),
            'error': str(e)
        }


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
    valid_results = [r for r in results if r['predicted'] != 'error']

    if not valid_results:
        return {"error": "No valid results"}

    y_true = [1 if r['ground_truth'] == 'fake' else 0 for r in valid_results]
    y_pred = [1 if r['predicted'] == 'fake' else 0 for r in valid_results]
    y_prob = [r['fake_prob'] for r in valid_results]

    metrics = {
        'total_samples': len(results),
        'valid_samples': len(valid_results),
        'error_samples': len(results) - len(valid_results),
        'accuracy': sum(1 for r in valid_results if r['correct']) / len(valid_results)
    }

    if SKLEARN_AVAILABLE:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        if len(set(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['auc_roc'] = None

        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        report = classification_report(y_true, y_pred, target_names=['real', 'fake'], output_dict=True)
        metrics['per_class'] = {
            'real': {
                'precision': report['real']['precision'],
                'recall': report['real']['recall'],
                'f1': report['real']['f1-score'],
                'support': report['real']['support']
            },
            'fake': {
                'precision': report['fake']['precision'],
                'recall': report['fake']['recall'],
                'f1': report['fake']['f1-score'],
                'support': report['fake']['support']
            }
        }

    return metrics


def print_progress(idx: int, total: int, result: Dict, start_time: float):
    """Print progress information"""
    elapsed = time.time() - start_time
    avg_time = elapsed / (idx + 1)
    remaining = avg_time * (total - idx - 1)

    status = "✓" if result['correct'] else "✗"
    if result['error']:
        status = "!"

    print(f"[{idx+1:3d}/{total}] {status} {result['video_id'][:30]:30} | "
          f"GT: {result['ground_truth']:4} | Pred: {result['predicted']:4} | "
          f"Conf: {result['confidence']:.1%} | {result['processing_time']:.1f}s | "
          f"ETA: {remaining/60:.1f}m")


def print_metrics(metrics: Dict):
    """Print performance metrics in a clean format"""
    print("\n" + "="*50)
    print("📊 성능 지표 (Performance Metrics)")
    print("="*50)
    print(f"  Total Samples:  {metrics.get('total_samples', 0)}")
    print(f"  Valid Samples:  {metrics.get('valid_samples', 0)}")
    print(f"  Error Samples:  {metrics.get('error_samples', 0)}")
    print("-"*50)
    print(f"  Accuracy:   {metrics.get('accuracy', 0)*100:6.2f}%")
    print(f"  Precision:  {metrics.get('precision', 0)*100:6.2f}%")
    print(f"  Recall:     {metrics.get('recall', 0)*100:6.2f}%")
    print(f"  F1 Score:   {metrics.get('f1_score', 0)*100:6.2f}%")
    if metrics.get('auc_roc') is not None:
        print(f"  AUC-ROC:    {metrics.get('auc_roc', 0):6.4f}")
    print("-"*50)

    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print("\n📋 Confusion Matrix:")
        print(f"              Pred Real  Pred Fake")
        print(f"  True Real:    {cm[0][0]:4d}       {cm[0][1]:4d}")
        print(f"  True Fake:    {cm[1][0]:4d}       {cm[1][1]:4d}")

    if 'per_class' in metrics:
        print("\n📈 Per-Class Metrics:")
        for cls, cls_metrics in metrics['per_class'].items():
            print(f"  {cls.upper():5}: P={cls_metrics['precision']:.1%}, "
                  f"R={cls_metrics['recall']:.1%}, "
                  f"F1={cls_metrics.get('f1', 0):.1%}")

    print("="*50)


def save_results(output_dir: Path, results: List[Dict], metrics: Dict, args):
    """Save final results to JSON"""
    result_path = output_dir / "test_result.json"

    final_output = {
        "metadata": {
            "test_date": datetime.now().isoformat(),
            "source_file": str(args.input),
            "mmms_model": MMMS_MODEL,
            "pia_model": PIA_MODEL,
            "device": args.device,
            "cache_format": "video_id_only (no path hash)",
            "force_extract": not args.skip_existing
        },
        "summary": metrics,
        "details": results
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {result_path}")


def main():
    parser = argparse.ArgumentParser(
        description="E2E Test V4 - Fresh Cache Generation with New Key Format"
    )
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT),
                        help='Path to source test_result.json (default: test_e2e_v2/test_result.json)')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Output directory (default: test_e2e_v4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip videos that already have output (use existing cache)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load previous results
    print(f"📂 Loading source: {args.input}")
    prev_data = load_previous_results(args.input)
    all_entries = prev_data.get('details', [])

    # Filter to valid entries only (skip previous errors)
    entries = [e for e in all_entries if e.get('error') is None]
    print(f"📊 Processing {len(entries)} valid videos (skipping {len(all_entries) - len(entries)} errors)")

    if args.skip_existing:
        existing_count = 0
        filtered_entries = []
        for e in entries:
            output_path = output_dir / e['ground_truth'] / e['video_id'] / f"{e['video_id']}_pia_xai.png"
            if output_path.exists():
                existing_count += 1
            else:
                filtered_entries.append(e)
        entries = filtered_entries
        if existing_count > 0:
            print(f"⏭️  Skipping {existing_count} already processed videos")

    total = len(entries)
    if total == 0:
        print("No videos to process!")
        return

    # Initialize pipeline
    print(f"\n🚀 Initializing Hybrid XAI Pipeline (device={args.device})...")
    pipeline = HybridXAIPipeline(
        mmms_model_path=MMMS_MODEL,
        pia_model_path=PIA_MODEL,
        device=args.device
    )

    # Process videos
    force_extract = not args.skip_existing
    print(f"\n🎬 Processing {total} videos (force_extract={force_extract})...")
    print("="*70)

    results = []
    start_time = time.time()

    for idx, entry in enumerate(entries):
        result = run_single_video(pipeline, entry, output_dir, force_extract=force_extract)
        results.append(result)
        print_progress(idx, total, result, start_time)

    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(results)

    # Print metrics
    print_metrics(metrics)

    # Save results
    save_results(output_dir, results, metrics, args)

    # Summary
    total_time = time.time() - start_time
    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📈 Average: {total_time/len(results):.1f}s per video")

    # Cache verification
    print("\n📦 Cache verification:")
    cache = FeatureCache()
    new_format_count = sum(1 for f in cache.cache_dir.iterdir()
                          if f.suffix == '.npz' and '_' not in f.stem[-8:])
    print(f"  New format cache files: {new_format_count}")


if __name__ == "__main__":
    main()
