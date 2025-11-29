"""
Training script for PIA (Phoneme-Temporal and Identity-Dynamic Analysis) Model

Phase 1: Baseline training with dummy ArcFace embeddings
- Uses Val split (133 samples) for initial validation
- Simple CrossEntropyLoss (no temporal loss yet)
- Verifies entire pipeline works correctly

Usage:
    python scripts/train_pia.py --config configs/train_pia.yaml
    python scripts/train_pia.py --data-dir preprocessed_data_phoneme/ --epochs 30
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.phoneme_dataset import KoreanPhonemeDataset
from src.models.pia_model import create_pia_model


def setup_logging(log_dir: Path):
    """Setup logging to file and console"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML"""
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        geoms = batch['geometry'].to(device)    # (B, P, F, 4)
        imgs = batch['images'].to(device)       # (B, P, F, 3, H, W)
        arcs = batch['arcface'].to(device)      # (B, P, F, 512)
        mask = batch['mask'].to(device)         # (B, P, F)
        labels = batch['label'].to(device)      # (B,)

        # Safety: Replace NaN/Inf with 0 (belt-and-suspenders approach)
        geoms = torch.nan_to_num(geoms, nan=0.0, posinf=0.0, neginf=0.0)
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)
        arcs = torch.nan_to_num(arcs, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(geoms, imgs, arcs, mask)

        # Loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent explosion/NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Log progress (adaptive: ~5 times per epoch)
        log_interval = max(1, len(dataloader) // 5)
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(dataloader):
            logger.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger
):
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            geoms = batch['geometry'].to(device)
            imgs = batch['images'].to(device)
            arcs = batch['arcface'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)

            # Safety: Replace NaN/Inf with 0 (consistent with training)
            geoms = torch.nan_to_num(geoms, nan=0.0, posinf=0.0, neginf=0.0)
            imgs = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)
            arcs = torch.nan_to_num(arcs, nan=0.0, posinf=0.0, neginf=0.0)

            # Forward pass
            logits, _ = model(geoms, imgs, arcs, mask)

            # Loss
            loss = criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    logger.info(f"Validation - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train PIA Model')
    parser.add_argument('--config', type=str, default='configs/train_pia.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='../preprocessed_data_phoneme/',
                        help='Path to preprocessed data directory')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='F:/preprocessed_data_pia_optimized/',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--augment-real', action='store_true',
                        help='Apply train-time augmentation to Real samples')
    parser.add_argument('--augment-ratio', type=float, default=1.0,
                        help='Augmentation ratio for Real samples (default: 1.0 = same as original)')

    args = parser.parse_args()

    # Setup output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("="*60)
    logger.info("PIA Model Training - Phase 1 Baseline")
    logger.info("="*60)

    # Load config (if exists)
    config = load_config(args.config)

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Dataset setup
    logger.info(f"Loading datasets from {args.data_dir}")
    logger.info("Phase 2 Mode: Loading train and val splits")

    # Load train dataset (with augmentation)
    try:
        train_dataset = KoreanPhonemeDataset(
            args.data_dir,
            split='train',
            augment_real=args.augment_real,
            augment_ratio=args.augment_ratio
        )
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        if args.augment_real:
            logger.info(f"Real augmentation enabled (ratio={args.augment_ratio})")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Train dataset not found: {e}")
        logger.error("Cannot proceed without train data")
        return

    # Load validation dataset
    try:
        val_dataset = KoreanPhonemeDataset(args.data_dir, split='val')
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Validation dataset not found: {e}")
        logger.error("Cannot proceed without validation data")
        return

    # DataLoader setup
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Model setup
    logger.info("Creating PIA model")
    model = create_pia_model(
        num_phonemes=config['data']['num_phonemes'],
        frames_per_phoneme=config['data']['frames_per_phoneme'],
        arcface_dim=config['model']['arcface_dim'],
        geo_dim=config['model']['geo_dim'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_classes=config['model']['num_classes'],
        use_temporal_loss=config['model']['use_temporal_loss']
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    logger.info("="*60)
    logger.info("Starting training")
    logger.info("="*60)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 10)
    logger.info(f"Early stopping patience: {early_stopping_patience} epochs")
    logger.info("Best model selection: Based on lowest validation loss")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, logger)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter
            best_path = checkpoint_dir / 'best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, best_path)
            logger.info(f"Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%) to {best_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{early_stopping_patience} epochs")

            # Early stopping check
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                logger.info(f"Best validation loss: {best_val_loss:.4f}")
                break

        # Save last model
        last_path = checkpoint_dir / 'last.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss
        }, last_path)

    logger.info("="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
