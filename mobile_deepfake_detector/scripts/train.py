"""
Training script for Teacher model (MMMS-BA)
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.teacher import MMMSBA
from src.data.dataset import DeepfakeDataset, PreprocessedDeepfakeDataset, create_dataloader
from src.data.augmentation import MultiModalAugmentation
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metrics


class Trainer:
    """
    Trainer for MMMS-BA model
    """

    def __init__(self, config: dict):
        """
        Initialize trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(
            name="trainer",
            log_file=f"{config['paths']['log_dir']}/train.log"
        )

        # Set device and CUDA optimizations
        if torch.cuda.is_available():
            self.device = torch.device(config['experiment']['device'])

            # CUDA optimization settings
            torch.backends.cudnn.benchmark = True  # Auto-tune algorithms
            torch.backends.cudnn.deterministic = False  # Faster but not deterministic

            # Print CUDA info
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU count: {torch.cuda.device_count()}")
            self.logger.info(f"Current GPU memory: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        else:
            self.device = torch.device("cpu")
            self.logger.warning("CUDA not available, using CPU")

        self.logger.info(f"Using device: {self.device}")

        # Set seed
        torch.manual_seed(config['experiment']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['experiment']['seed'])

        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', False) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision training (AMP) enabled")

        # Build model
        self.model = self._build_model()
        self.logger.info(f"Model built: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")

        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Loss function with optional class weights
        if config['training'].get('use_weighted_loss', False):
            class_weights = torch.tensor(
                config['training'].get('class_weights', [1.0, 1.0]),
                dtype=torch.float32
            ).to(self.device)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config['training']['loss'].get('label_smoothing', 0.0)
            )
            self.logger.info(f"Using weighted loss with class weights: {class_weights.cpu().tolist()}")
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=config['training']['loss'].get('label_smoothing', 0.0)
            )

        # Metrics
        self.best_val_acc = 0.0
        self.current_epoch = 0

        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=f"{config['paths']['log_dir']}/tensorboard"
        )

    def _build_model(self) -> nn.Module:
        """Build model"""
        model_config = self.config['model']

        model = MMMSBA(
            audio_dim=self.config['dataset']['audio']['n_mfcc'],
            visual_dim=256,  # From feature extractor
            lip_dim=128,     # From feature extractor
            gru_hidden_dim=model_config['gru']['hidden_size'],
            gru_num_layers=model_config['gru']['num_layers'],
            gru_dropout=model_config['gru']['dropout'],
            dense_hidden_dim=model_config['dense']['hidden_size'],
            dense_dropout=model_config['dense']['dropout'],
            attention_type=model_config['attention']['type'],
            num_classes=model_config['num_classes']
        )

        model = model.to(self.device)

        # Multi-GPU if available
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        return model

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        opt_config = self.config['training']['optimizer']

        if opt_config['type'].lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config.get('betas', [0.9, 0.999]),
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        elif opt_config['type'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")

        return optimizer

    def _build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler"""
        sched_config = self.config['training']['scheduler']

        if sched_config['type'].lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', 50),
                eta_min=sched_config.get('eta_min', 1e-5)
            )
        elif sched_config['type'].lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 10),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_config['type']}")

        return scheduler

    def train_epoch(self, train_loader) -> dict:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader

        Returns:
            metrics: Dictionary of metrics
        """
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device (non-blocking for faster transfer)
            audio = batch['audio'].to(self.device, non_blocking=True)
            frames = batch['frames'].to(self.device, non_blocking=True)
            lip = batch['lip'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            audio_mask = batch['audio_mask'].to(self.device, non_blocking=True)
            frames_mask = batch['frames_mask'].to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()

            # Mixed precision training
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        audio=audio,
                        frames=frames,
                        lip=lip,
                        audio_mask=audio_mask,
                        frames_mask=frames_mask
                    )
                    loss = self.criterion(logits, labels)

                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['training']['grad_clip']['enabled']:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']['max_norm']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(
                    audio=audio,
                    frames=frames,
                    lip=lip,
                    audio_mask=audio_mask,
                    frames_mask=frames_mask
                )
                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config['training']['grad_clip']['enabled']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']['max_norm']
                    )

                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Logging
            if (batch_idx + 1) % self.config['training']['logging']['interval'] == 0:
                step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)

        # Compute epoch metrics
        metrics = compute_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(train_loader)

        return metrics

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        """
        Validate model

        Args:
            val_loader: Validation data loader

        Returns:
            metrics: Dictionary of metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(val_loader, desc="Validation"):
            # Move to device (non-blocking)
            audio = batch['audio'].to(self.device, non_blocking=True)
            frames = batch['frames'].to(self.device, non_blocking=True)
            lip = batch['lip'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            audio_mask = batch['audio_mask'].to(self.device, non_blocking=True)
            frames_mask = batch['frames_mask'].to(self.device, non_blocking=True)

            # Forward pass (with AMP if enabled)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        audio=audio,
                        frames=frames,
                        lip=lip,
                        audio_mask=audio_mask,
                        frames_mask=frames_mask
                    )
                    # Loss 계산도 autocast 안에서 수행 (타입 일치를 위해)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(
                    audio=audio,
                    frames=frames,
                    lip=lip,
                    audio_mask=audio_mask,
                    frames_mask=frames_mask
                )
                loss = self.criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(val_loader)

        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """
        Save checkpoint

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(self.config['training']['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }

        # Save last checkpoint
        if self.config['training']['checkpoint']['save_last']:
            torch.save(checkpoint, checkpoint_dir / 'last.pth')

        # Save best checkpoint
        if is_best and self.config['training']['checkpoint']['save_best']:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            self.logger.info(f"Saved best checkpoint (val_acc: {self.best_val_acc:.4f})")

    def train(self, train_loader, val_loader):
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("TRAINING START")
        self.logger.info("="*70)

        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping'].get('patience', 10)
        patience_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            self.logger.info(f"\nEpoch {epoch}/{epochs}")
            self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            self.logger.info(f"  Train Acc: {train_metrics['accuracy']:.4f}")

            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_acc', train_metrics['accuracy'], epoch)

            # Validate
            if (epoch + 1) % self.config['validation']['interval'] == 0:
                val_metrics = self.validate(val_loader)

                self.logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
                self.logger.info(f"  Val Acc: {val_metrics['accuracy']:.4f}")
                self.logger.info(f"  Val Precision: {val_metrics['precision']:.4f}")
                self.logger.info(f"  Val Recall: {val_metrics['recall']:.4f}")
                self.logger.info(f"  Val F1: {val_metrics['f1_score']:.4f}")

                # Log to tensorboard
                self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/val_acc', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('epoch/val_f1', val_metrics['f1_score'], epoch)

                # Check if best
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Save checkpoint
                self.save_checkpoint(is_best=is_best)

                # Early stopping
                if self.config['training']['early_stopping']['enabled']:
                    if patience_counter >= patience:
                        self.logger.info(f"\nEarly stopping at epoch {epoch}")
                        break

            # Step scheduler
            self.scheduler.step()

        self.logger.info("\n" + "="*70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        self.logger.info("="*70 + "\n")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train MMMS-BA model with preprocessed data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_teacher_korean.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of data loading workers (overrides config)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Override num_workers if specified
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers

    # Create datasets (전처리된 npz 파일 사용!)
    logger = logging.getLogger("main")
    logger.info("Loading preprocessed datasets (npz files)...")

    # Augmentation (train only)
    train_augmentation = None
    if config['dataset']['augmentation'].get('enabled', False):
        train_augmentation = MultiModalAugmentation(
            config=config['dataset']['augmentation'],
            train=True
        )
        logger.info("[OK] Data augmentation enabled for training")

    train_dataset = PreprocessedDeepfakeDataset(
        data_root=config['dataset']['root_dir'],
        split='train',
        config=config['dataset'],
        augmentation=train_augmentation
    )

    val_dataset = PreprocessedDeepfakeDataset(
        data_root=config['dataset']['root_dir'],
        split='val',
        config=config['dataset'],
        augmentation=None  # No augmentation for validation
    )

    logger.info(f"[OK] Loaded {len(train_dataset)} train samples")
    logger.info(f"[OK] Loaded {len(val_dataset)} val samples")

    # Weighted sampler for class imbalance (optional)
    train_sampler = None
    if config['training'].get('use_weighted_sampler', False):
        # Calculate class distribution
        # label은 문자열 'real' 또는 'fake'이므로 숫자로 변환
        labels = [sample['label'] for sample in train_dataset.samples]
        label_to_int = {'real': 0, 'fake': 1}
        labels_int = [label_to_int[label] for label in labels]
        
        class_counts = [labels_int.count(0), labels_int.count(1)]  # [Real, Fake]
        logger.info(f"Class distribution: Real={class_counts[0]}, Fake={class_counts[1]}")

        # 0으로 나누기 방지: 클래스 개수가 0인 경우 체크
        if min(class_counts) == 0:
            logger.warning(f"⚠️ 경고: 클래스 불균형이 심각합니다 (Real={class_counts[0]}, Fake={class_counts[1]})")
            logger.warning(f"⚠️ Weighted sampler를 비활성화합니다.")
        else:
            # Calculate sample weights
            class_weights = [1.0 / count for count in class_counts]
            sample_weights = [class_weights[label] for label in labels_int]

            # Create sampler
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            logger.info(f"[OK] Weighted sampler enabled with class weights: {class_weights}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),  # Disable shuffle if using sampler
        num_workers=config['training']['num_workers'],
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None
    )

    # Create trainer
    trainer = Trainer(config)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
