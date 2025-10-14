"""
모델 평가 스크립트 (Test 세트)
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.teacher import MMMSBA
from src.data.dataset import PreprocessedDeepfakeDataset, create_dataloader
from src.utils.config import load_config
from src.utils.metrics import compute_metrics


def evaluate_model(model, test_loader, device, use_amp=False):
    """
    모델 평가
    
    Args:
        model: 평가할 모델
        test_loader: Test 데이터 로더
        device: 디바이스
        use_amp: Mixed precision 사용 여부
        
    Returns:
        results: 평가 결과 딕셔너리
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n" + "="*70)
    print("MODEL EVALUATION - TEST SET")
    print("="*70 + "\n")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 빈 배치 스킵
            if batch is None:
                continue
            
            # GPU로 전송
            audio = batch['audio'].to(device, non_blocking=True)
            frames = batch['frames'].to(device, non_blocking=True)
            lip = batch['lip'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            audio_mask = batch['audio_mask'].to(device, non_blocking=True)
            frames_mask = batch['frames_mask'].to(device, non_blocking=True)
            
            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(
                        audio=audio,
                        frames=frames,
                        lip=lip,
                        audio_mask=audio_mask,
                        frames_mask=frames_mask
                    )
            else:
                logits = model(
                    audio=audio,
                    frames=frames,
                    lip=lip,
                    audio_mask=audio_mask,
                    frames_mask=frames_mask
                )
            
            # 예측 및 확률
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # 수집
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Fake 클래스 확률
    
    # NumPy 배열로 변환
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics 계산
    metrics = compute_metrics(all_labels, all_preds)
    
    # AUC 계산
    try:
        auc = roc_auc_score(all_labels, all_probs)
        metrics['auc'] = auc
    except:
        metrics['auc'] = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification Report
    class_names = ['Real', 'Fake']
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4
    )
    
    # 결과 출력
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)
    print(f"\n             Predicted")
    print(f"              Real  Fake")
    print(f"Actual Real   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Fake   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    print("\n" + "-"*70)
    print("CLASSIFICATION REPORT")
    print("-"*70)
    print(report)
    print("="*70 + "\n")
    
    # 결과 딕셔너리
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
        'classification_report': report
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMMS-BA model on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_teacher_korean.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=['train', 'val', 'test'],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Build model
    print("Building model...")
    model = MMMSBA(
        audio_dim=config['dataset']['audio']['n_mfcc'],
        visual_dim=256,
        lip_dim=128,
        gru_hidden_dim=config['model']['gru']['hidden_size'],
        gru_num_layers=config['model']['gru']['num_layers'],
        gru_dropout=config['model']['gru']['dropout'],
        dense_hidden_dim=config['model']['dense']['hidden_size'],
        dense_dropout=config['model']['dense']['dropout'],
        attention_type=config['model']['attention']['type'],
        num_classes=config['model']['num_classes']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Best val accuracy: {checkpoint.get('best_val_acc', 'Unknown'):.4f}\n")
    
    # Load test dataset
    print(f"Loading {args.split} dataset...")
    test_dataset = PreprocessedDeepfakeDataset(
        data_root=config['dataset']['root_dir'],
        split=args.split,
        config=config['dataset']
    )
    
    print(f"Loaded {len(test_dataset)} samples\n")
    
    # Create dataloader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 평가는 단순하게
        pin_memory=True
    )
    
    # Evaluate
    use_amp = config['training'].get('mixed_precision', False) and torch.cuda.is_available()
    results = evaluate_model(model, test_loader, device, use_amp=use_amp)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
    else:
        # 기본 저장 위치
        output_dir = Path("outputs/korean/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.split}_results.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()




