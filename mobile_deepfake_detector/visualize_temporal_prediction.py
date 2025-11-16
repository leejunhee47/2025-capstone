"""
Temporal Visualization - Frame-level Deepfake Probability

Phase 3 of Temporal Visualization Implementation
- 프레임별 fake 확률을 시간축으로 시각화
- 3-row 레이아웃: 대표 프레임, 확률 그래프, 히트맵
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cv2
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mobile_deepfake_detector.src.models.teacher import MMMSBA
from mobile_deepfake_detector.src.utils.mmms_ba_adapter import MMSBAdapter


def load_config(config_path: str = "configs/train_teacher_korean.yaml"):
    """Load training configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def visualize_temporal_prediction(
    npz_path: str,
    original_video_root: str,
    model_path: str = None,
    config_path: str = "configs/train_teacher_korean.yaml",
    output_path: str = "temporal_visualization.png",
    num_sample_frames: int = 6
):
    """
    Temporal visualization 생성

    Args:
        npz_path: NPZ 파일 경로
        original_video_root: 원본 영상 루트
        model_path: 학습된 모델 경로 (None이면 random prediction)
        config_path: Training config 경로
        output_path: 출력 이미지 경로
        num_sample_frames: 표시할 샘플 프레임 개수
    """
    print("="*80)
    print("TEMPORAL DEEPFAKE VISUALIZATION")
    print("="*80)

    # Device (use CPU for large videos to avoid OOM)
    device = torch.device('cpu')  # Force CPU to handle large videos

    # Load config
    config = load_config(config_path)

    # 1. Load data
    print("\n[1/4] Loading data...")
    adapter = MMSBAdapter(original_video_root=original_video_root)
    data = adapter.load_npz_with_full_frames(npz_path=npz_path, extract_lip=True)

    frames_np = data['frames']  # (T, H, W, 3)
    audio_np = data['audio']    # (T', 40)
    lip_np = data['lip']        # (T, lip_H, lip_W, 3)
    timestamps = data['timestamps']  # (T,)
    total_frames = data['total_frames']
    video_fps = data['video_fps']
    video_id = data['video_id']
    label = data['label']

    print(f"  Video: {video_id}")
    print(f"  Frames: {total_frames} @ {video_fps:.2f} FPS")
    print(f"  Duration: {timestamps[-1]:.2f}s")
    print(f"  Label: {'Fake' if label == 1 else 'Real'}")

    # 2. Initialize model matching train.py (hardcoded feature extractor dims)
    print("\n[2/4] Initializing model...")

    model = MMMSBA(
        audio_dim=config['dataset']['audio']['n_mfcc'],  # 40
        visual_dim=256,  # From feature extractor (hardcoded in train.py)
        lip_dim=128,     # From feature extractor (hardcoded in train.py)
        gru_hidden_dim=config['model']['gru']['hidden_size'],  # 300
        gru_num_layers=config['model']['gru']['num_layers'],  # 1
        gru_dropout=config['model']['gru']['dropout'],  # 0.5
        dense_hidden_dim=config['model']['dense']['hidden_size'],  # 100
        dense_dropout=config['model']['dense']['dropout'],  # 0.7
        attention_type=config['model']['attention']['type'],  # "bi_modal"
        num_classes=config['model']['num_classes']  # 2
    )

    if model_path and Path(model_path).exists():
        print(f"  Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  [OK] Model loaded (epoch {checkpoint.get('epoch', '?')})")
    else:
        print(f"  [WARNING] No model weights - using random prediction")

    model = model.to(device)
    model.eval()

    # 3. Get frame-level predictions
    print("\n[3/4] Running frame-level prediction...")

    # Prepare tensors
    frames_torch = torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0).to(device).float()
    audio_torch = torch.from_numpy(audio_np).unsqueeze(0).to(device).float()
    lip_torch = torch.from_numpy(lip_np).permute(0, 3, 1, 2).unsqueeze(0).to(device).float()

    with torch.no_grad():
        frame_logits = model(
            audio=audio_torch,
            frames=frames_torch,
            lip=lip_torch,
            frame_level=True  # Frame-level prediction
        )

    frame_probs = torch.softmax(frame_logits, dim=-1)  # (1, T, 2)
    fake_probs = frame_probs[0, :, 1].cpu().numpy()  # (T,)

    print(f"  Frame predictions: {frame_logits.shape}")
    print(f"  Mean fake prob: {fake_probs.mean():.3f}")
    print(f"  Max fake prob: {fake_probs.max():.3f}")
    print(f"  Min fake prob: {fake_probs.min():.3f}")

    # 4. Create visualization
    print("\n[4/4] Creating visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1.5, 0.5], hspace=0.3)

    # === Row 1: Sample Frames ===
    ax_frames = fig.add_subplot(gs[0])
    ax_frames.set_title(
        f"Video: {Path(video_id).stem} | Label: {'FAKE' if label == 1 else 'REAL'} | "
        f"Duration: {timestamps[-1]:.2f}s ({total_frames} frames)",
        fontsize=14,
        fontweight='bold'
    )

    # Select evenly spaced frames
    frame_indices = np.linspace(0, total_frames-1, num_sample_frames, dtype=int)

    # Create montage
    montage_frames = []
    for idx in frame_indices:
        frame = frames_np[idx]  # (224, 224, 3), [0, 1]
        frame_uint8 = (frame * 255).astype(np.uint8)
        # Add frame number text
        cv2.putText(
            frame_uint8,
            f"#{idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame_uint8,
            f"{timestamps[idx]:.1f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        # Add fake probability
        prob = fake_probs[idx]
        color = (255, 0, 0) if prob > 0.5 else (0, 255, 0)
        cv2.putText(
            frame_uint8,
            f"P(Fake)={prob:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        montage_frames.append(frame_uint8)

    # Concatenate horizontally
    montage = np.concatenate(montage_frames, axis=1)
    ax_frames.imshow(montage)
    ax_frames.axis('off')

    # === Row 2: Temporal Probability Graph ===
    ax_graph = fig.add_subplot(gs[1])
    ax_graph.set_title("Frame-Level Fake Probability Over Time", fontsize=12, fontweight='bold')

    # Plot probability line
    ax_graph.plot(timestamps, fake_probs, linewidth=2, color='darkblue', alpha=0.8)
    ax_graph.fill_between(timestamps, fake_probs, alpha=0.3, color='lightblue')

    # Threshold line
    ax_graph.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (0.5)')

    # Highlight high-risk regions
    high_risk_mask = fake_probs > 0.5
    if high_risk_mask.any():
        ax_graph.fill_between(
            timestamps,
            0,
            1,
            where=high_risk_mask,
            alpha=0.2,
            color='red',
            label='High Risk (>0.5)'
        )

    # Styling
    ax_graph.set_xlabel("Time (seconds)", fontsize=11)
    ax_graph.set_ylabel("P(Fake)", fontsize=11)
    ax_graph.set_ylim([0, 1])
    ax_graph.set_xlim([timestamps[0], timestamps[-1]])
    ax_graph.grid(True, alpha=0.3, linestyle=':')
    ax_graph.legend(loc='upper right', fontsize=10)

    # Add frame markers
    for idx in frame_indices:
        ax_graph.axvline(x=timestamps[idx], color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    # === Row 3: Temporal Heatmap ===
    ax_heatmap = fig.add_subplot(gs[2])
    ax_heatmap.set_title("Temporal Heatmap", fontsize=11)

    # Create heatmap (1D)
    heatmap_data = fake_probs.reshape(1, -1)  # (1, T)

    im = ax_heatmap.imshow(
        heatmap_data,
        cmap='RdYlGn_r',  # Red (high fake prob) to Green (low fake prob)
        aspect='auto',
        vmin=0,
        vmax=1,
        extent=[timestamps[0], timestamps[-1], 0, 1]
    )

    ax_heatmap.set_xlabel("Time (seconds)", fontsize=11)
    ax_heatmap.set_yticks([])
    ax_heatmap.set_xlim([timestamps[0], timestamps[-1]])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, orientation='horizontal', pad=0.15, fraction=0.05)
    cbar.set_label('P(Fake)', fontsize=10)

    # === Overall Statistics ===
    high_risk_count = (fake_probs > 0.5).sum()
    high_risk_ratio = high_risk_count / len(fake_probs) * 100
    mean_fake_prob = fake_probs.mean()

    stats_text = (
        f"Statistics:\n"
        f"High-risk frames: {high_risk_count}/{len(fake_probs)} ({high_risk_ratio:.1f}%)\n"
        f"Mean P(Fake): {mean_fake_prob:.3f}\n"
        f"Overall: {'FAKE' if high_risk_ratio > 50 else 'REAL'} (Majority voting)"
    )

    fig.text(
        0.98, 0.02,
        stats_text,
        fontsize=10,
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to: {output_path}")
    plt.close()

    print(f"\n{'='*80}")
    print(f"Temporal Visualization Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Configuration
    npz_path = "E:/capstone/preprocessed_data_phoneme/test/00000.npz"
    original_video_root = "E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터"
    output_path = "E:/capstone/mobile_deepfake_detector/temporal_visualization_00000.png"

    # Use trained model (97.74% Val Acc at Epoch 1)
    model_path = "E:/capstone/mobile_deepfake_detector/models/checkpoints/mmms-ba_best.pth"
    config_path = "E:/capstone/mobile_deepfake_detector/configs/train_teacher_korean.yaml"

    visualize_temporal_prediction(
        npz_path=npz_path,
        original_video_root=original_video_root,
        model_path=model_path if Path(model_path).exists() else None,
        config_path=config_path,
        output_path=output_path,
        num_sample_frames=6
    )
