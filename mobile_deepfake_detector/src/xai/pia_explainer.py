"""
PIA Model Explainer for XAI (Explainable AI)

Provides interpretability for Korean short-form deepfake detection using:
- Phoneme-level attention analysis
- Branch contribution analysis (Visual, Geometry, Identity)
- Temporal heatmap generation
- Geometry (MAR) analysis
- Korean language explanations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import json


class PIAExplainer:
    """
    Explainable AI (XAI) for PIA deepfake detection model.

    Captures intermediate activations and attention weights to provide
    human-interpretable explanations for model predictions.

    Args:
        model: PIAModel or PIAModelWithTemporalLoss instance
        phoneme_vocab: List of 14 Korean phonemes (MFA format)
        device: torch device ('cuda' or 'cpu')
    """

    def __init__(
        self,
        model: nn.Module,
        phoneme_vocab: List[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.phoneme_vocab = phoneme_vocab
        self.device = device

        # Hook handles for cleanup
        self.hook_handles = []

        # Captured activations (filled during forward pass)
        self.activations = {
            'visual': None,      # (B, P, E) - Visual branch features
            'geometry': None,    # (B, P, E) - Geometry branch features
            'identity': None,    # (B, P, E) - Identity (ArcFace) branch features
            'fused': None,       # (B, P, 3*E) - Fused features
            'attention_weights': None,  # (B, H, 1, P) - Multi-head attention weights
        }

        # Input data (saved during explain() for analysis)
        self.input_data = {
            'geoms': None,   # (B, P, F, 1) - MAR features
            'imgs': None,    # (B, P, F, 3, H, W) - Lip crops
            'arcs': None,    # (B, P, F, 512) - ArcFace embeddings
        }

        # Analysis results (filled during explain())
        self.results = {}

    def register_hooks(self):
        """
        Register forward hooks to capture intermediate activations.

        Hooks are attached to:
        1. Visual branch: after EfficientNet projector
        2. Geometry branch: after geo_enc
        3. Identity branch: after arc_enc
        4. Attention pooling: attention weights
        """
        # Clear previous hooks
        self.remove_hooks()

        # Access base model (handle PIAModelWithTemporalLoss wrapper)
        base_model = self.model.model if hasattr(self.model, 'model') else self.model

        # ===== 1. Visual Branch Hook =====
        def visual_hook(module, input, output):
            """
            Captures visual features after projector.
            Output: (B*P, E) → reshape to (B, P, E)
            """
            B = 1  # XAI always uses batch size 1
            E = output.size(1)
            P = output.size(0) // B  # Infer P from output shape (e.g., 14 for actual phonemes)
            self.activations['visual'] = output.detach().view(B, P, E)

        h1 = base_model.projector.register_forward_hook(visual_hook)
        self.hook_handles.append(h1)

        # ===== 2. Geometry Branch Hook =====
        def geometry_hook(module, input, output):
            """
            Captures geometry features after geo_enc.
            Output: (B, P, E) - already correct shape
            """
            self.activations['geometry'] = output.detach()

        h2 = base_model.geo_enc.register_forward_hook(geometry_hook)
        self.hook_handles.append(h2)

        # ===== 3. Identity Branch Hook =====
        def identity_hook(module, input, output):
            """
            Captures identity features after arc_enc.
            Output: (B*P, F, E) → mean over F → reshape to (B, P, E)
            """
            B = 1
            # Average across F frames
            features = output.mean(dim=1)  # (B*P, E)
            E = features.size(1)
            P = features.size(0) // B  # Infer P from output shape (e.g., 14 for actual phonemes)
            self.activations['identity'] = features.detach().view(B, P, E)

        h3 = base_model.arc_enc.register_forward_hook(identity_hook)
        self.hook_handles.append(h3)

        # ===== 4. Attention Pooling Hook =====
        def attention_hook(module, input, output):
            """
            Captures attention weights from MultiHeadAttentionPooling.
            Weights are stored in module.attention_weights: (B, H, 1, P)
            """
            if hasattr(module, 'attention_weights') and module.attention_weights is not None:
                self.activations['attention_weights'] = module.attention_weights.detach()

        h4 = base_model.attn_pool.register_forward_hook(attention_hook)
        self.hook_handles.append(h4)

        # ===== 5. Fused Features Hook (for accurate branch contribution) =====
        def fused_hook(module, input, output):
            """
            Captures fused features BEFORE attention pooling.
            Input[0]: (B, P, 3*E) - Concatenated [visual, geometry, identity]
            This is what the attention pooling actually sees.
            """
            # input is a tuple, input[0] is the actual fused tensor
            if len(input) > 0:
                fused = input[0]  # (B, P, 3*E)
                self.activations['fused'] = fused.detach()

        h5 = base_model.attn_pool.register_forward_hook(fused_hook)
        self.hook_handles.append(h5)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def explain(
        self,
        geoms: torch.Tensor,
        imgs: torch.Tensor,
        arcs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        video_id: str = "unknown",
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate comprehensive XAI explanation for a single video.

        Args:
            geoms: (1, P, F, 1) - Geometry features (batch size 1)
            imgs: (1, P, F, 3, H, W) - Lip crop images
            arcs: (1, P, F, 512) - ArcFace embeddings
            mask: (1, P, F) - Valid data mask (optional)
            video_id: Video identifier
            confidence_threshold: Threshold for fake/real classification

        Returns:
            DeepfakeXAIResult dictionary with 8 sections:
            - metadata
            - video_info
            - detection
            - summary (Korean)
            - phoneme_analysis
            - temporal_analysis
            - geometry_analysis
            - model_info
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Register hooks to capture activations
        self.register_hooks()

        # Move inputs to device
        geoms = geoms.to(self.device)
        imgs = imgs.to(self.device)
        arcs = arcs.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Save input data for analysis
        self.input_data['geoms'] = geoms.detach()
        self.input_data['imgs'] = imgs.detach()
        self.input_data['arcs'] = arcs.detach()

        # Forward pass (hooks will capture activations)
        with torch.no_grad():
            logits, _ = self.model(geoms, imgs, arcs, mask)
            probs = torch.softmax(logits, dim=-1)

        # DEBUG: Print prediction details
        print(f"\n[PIA DEBUG] video_id={video_id}")
        print(f"  Logits: {logits}")
        print(f"  Probs (Real, Fake): {probs}")
        print(f"  Mask shape: {mask.shape if mask is not None else None}, Valid frames: {mask.sum().item() if mask is not None else 'N/A'}")

        # Extract prediction
        pred_class = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, pred_class].item()
        is_fake = pred_class == 1

        # DEBUG: Log prediction details
        print(f"\n[PREDICTION DEBUG]")
        print(f"  Logits: {logits[0].detach().cpu().numpy()}")
        print(f"  Probs: {probs[0].detach().cpu().numpy()}")
        print(f"  Predicted class: {pred_class} ({'FAKE' if is_fake else 'REAL'})")
        print(f"  Confidence: {confidence:.4f}")
        if confidence > 0.95:
            print(f"  [WARNING] Overconfident prediction (>95%) - possible overfitting or easy sample")

        # TODO: Implement 5 analysis functions
        # 1. Branch contribution analysis
        branch_contributions = self._analyze_branch_contributions()

        # 2. Phoneme attention analysis
        phoneme_scores = self._analyze_phoneme_attention()

        # 3. Temporal heatmap analysis
        temporal_heatmap = self._analyze_temporal_patterns()

        # 4. Geometry (MAR) analysis
        geometry_analysis = self._analyze_geometry()

        # 5. Korean language summary
        korean_summary = self._generate_korean_summary(
            is_fake, confidence, branch_contributions, phoneme_scores
        )

        # 6. Temporal probabilities (frame-level and 1-second intervals)
        temporal_probs = self._analyze_temporal_probabilities(probs, fps=30.0)

        # Build final result
        result = self._build_result(
            video_id=video_id,
            is_fake=is_fake,
            confidence=confidence,
            probs=probs,
            branch_contributions=branch_contributions,
            phoneme_scores=phoneme_scores,
            temporal_heatmap=temporal_heatmap,
            geometry_analysis=geometry_analysis,
            korean_summary=korean_summary,
            temporal_probs=temporal_probs
        )

        # Cleanup hooks
        self.remove_hooks()

        return result

    def _analyze_branch_contributions(self) -> Dict[str, float]:
        """
        Analyze contribution of each branch from FUSED features.

        NOTE: We now use the concatenated fused features (B, P, 3*E)
        to measure each branch's contribution, as this reflects what
        the attention pooling actually sees.

        Returns:
            Dictionary with branch names and contribution scores (0-1)
        """
        # Extract fused features: (B, P, 3*E)
        fused = self.activations.get('fused')  # (1, P, 3*E)

        if fused is None:
            # Fallback: use individual activations (old method)
            print("\n[BRANCH CONTRIBUTION WARNING] Fused features not captured, using fallback method")
            visual = self.activations.get('visual')
            geometry = self.activations.get('geometry')
            identity = self.activations.get('identity')

            if visual is None or geometry is None or identity is None:
                return {'visual': 0.33, 'geometry': 0.33, 'identity': 0.34}

            visual_norm = torch.norm(visual, p=2).item()
            geometry_norm = torch.norm(geometry, p=2).item()
            identity_norm = torch.norm(identity, p=2).item()
        else:
            # NEW: Extract each branch from fused tensor
            E = fused.size(-1) // 3  # Embedding dim per branch

            # Split fused features: (B, P, 3*E) → 3 × (B, P, E)
            visual_part = fused[:, :, 0:E]       # First E dims
            geometry_part = fused[:, :, E:2*E]   # Middle E dims
            identity_part = fused[:, :, 2*E:3*E] # Last E dims

            # Compute L2 norm for each branch
            visual_norm = torch.norm(visual_part, p=2).item()
            geometry_norm = torch.norm(geometry_part, p=2).item()
            identity_norm = torch.norm(identity_part, p=2).item()

        # Normalize to 0-1
        total = visual_norm + geometry_norm + identity_norm
        if total == 0:
            return {'visual': 0.33, 'geometry': 0.33, 'identity': 0.34}

        contributions = {
            'visual': visual_norm / total,
            'geometry': geometry_norm / total,
            'identity': identity_norm / total
        }

        # DEBUG: Log contributions
        print(f"\n[BRANCH CONTRIBUTION DEBUG]")
        print(f"  Visual norm: {visual_norm:.4f} → {contributions['visual']:.1%}")
        print(f"  Geometry norm: {geometry_norm:.4f} → {contributions['geometry']:.1%}")
        print(f"  Identity norm: {identity_norm:.4f} → {contributions['identity']:.1%}")

        return contributions

    def _analyze_phoneme_attention(self) -> List[Dict[str, Any]]:
        """
        Analyze phoneme-level attention scores.

        Returns:
            List of 14 phoneme dictionaries with:
            - phoneme: Korean character
            - score: attention weight (0-1)
            - importance_level: "high" | "medium" | "low"
        """
        # MFA → 한글 매핑
        phoneme_to_korean = {
            'A': 'ㅏ', 'B': 'ㅂ', 'BB': 'ㅃ', 'CHh': 'ㅊ',
            'E': 'ㅔ', 'EU': 'ㅡ', 'I': 'ㅣ', 'M': 'ㅁ',
            'O': 'ㅗ', 'Ph': 'ㅍ', 'U': 'ㅜ', 'iA': 'ㅑ',
            'iO': 'ㅛ', 'iU': 'ㅠ'
        }

        # Extract attention weights: (B, H, 1, P) → (P,)
        attn_weights = self.activations.get('attention_weights', None)  # (1, H, 1, P)

        # DEBUG: Print attention details
        print(f"\n[PHONEME ATTENTION DEBUG]")
        if 'attention_weights' not in self.activations:
            print(f"  [ERROR] 'attention_weights' not found in activations!")
            print(f"  Available keys: {list(self.activations.keys())}")
        if attn_weights is None:
            print(f"  Attention weights: None (using uniform fallback)")
            # Fallback: uniform attention
            scores = [1.0 / len(self.phoneme_vocab)] * len(self.phoneme_vocab)
        else:
            print(f"  Attention weights shape: {attn_weights.shape}")
            print(f"  Attention weights (raw):\n{attn_weights}")
            # Average across heads: (1, H, 1, P) → (P,)
            scores = attn_weights.mean(dim=1).squeeze().cpu().numpy()  # (P,)
            print(f"  Attention scores (averaged): {scores}")
            print(f"  Sum of scores: {scores.sum():.4f} (should be ~1.0)")

        # Build phoneme analysis
        phoneme_analysis = []
        for phoneme, score in zip(self.phoneme_vocab, scores):
            score_float = float(score)

            # Classify importance level
            if score_float > 0.1:
                importance = "high"
            elif score_float > 0.05:
                importance = "medium"
            else:
                importance = "low"

            phoneme_analysis.append({
                'phoneme': phoneme_to_korean.get(phoneme, phoneme),
                'phoneme_mfa': phoneme,
                'score': score_float,
                'importance_level': importance
            })

        return phoneme_analysis

    def _analyze_temporal_patterns(self) -> List[List[float]]:
        """
        Generate temporal heatmap (P x F).

        Returns:
            2D array of attention scores across phonemes and frames
        """
        # Extract attention weights (B, H, 1, P)
        attn_weights = self.activations['attention_weights']

        if attn_weights is None:
            # Fallback: uniform heatmap
            P = len(self.phoneme_vocab)
            F = 5  # Default frames per phoneme
            return [[1.0 / P] * F for _ in range(P)]

        # Average across heads: (1, H, 1, P) → (P,)
        scores = attn_weights.mean(dim=1).squeeze().cpu().numpy()  # (P,)

        # Replicate across F frames (temporal dimension)
        F = 5  # Frames per phoneme (from PIA config)
        heatmap = []
        for score in scores:
            # Each phoneme's attention score replicated across F frames
            heatmap.append([float(score)] * F)

        return heatmap  # (P, F)

    def _analyze_temporal_probabilities(
        self,
        probs: torch.Tensor,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Calculate frame-level fake probability contributions.

        Args:
            probs: (1, 2) tensor with [P(real), P(fake)]
            fps: Frames per second (default: 30)

        Returns:
            Dictionary with:
            - frame_probabilities: List[Dict] with frame-level probs
            - interval_probabilities: List[Dict] with 1-second intervals
        """
        fake_prob = probs[0, 1].item()

        # Get attention weights: (1, H, 1, P) → (P,)
        attn_weights = self.activations['attention_weights']
        if attn_weights is None:
            # Uniform fallback
            scores = np.ones(14) / 14
        else:
            scores = attn_weights.mean(dim=1).squeeze().cpu().numpy()

        # Expand to frame-level (14 phonemes × 5 frames = 70 frames)
        frame_probs = []
        frame_idx = 0

        for p_idx, phoneme_score in enumerate(scores):
            phoneme = self.phoneme_vocab[p_idx]

            for f_idx in range(5):  # 5 frames per phoneme
                timestamp = frame_idx / fps

                # Frame's fake contribution = phoneme_attention × overall_fake_prob
                frame_fake_prob = phoneme_score * fake_prob

                frame_probs.append({
                    'frame_index': frame_idx,
                    'timestamp_sec': timestamp,
                    'phoneme': phoneme,
                    'fake_probability': float(frame_fake_prob),
                    'real_probability': 1.0 - float(frame_fake_prob)
                })

                frame_idx += 1

        # Group by 1-second intervals
        max_time = frame_probs[-1]['timestamp_sec']
        num_intervals = int(np.ceil(max_time))

        interval_probs = []
        for interval_idx in range(num_intervals):
            start_sec = interval_idx
            end_sec = interval_idx + 1

            # Collect frames in this interval
            interval_frames = [
                f for f in frame_probs
                if start_sec <= f['timestamp_sec'] < end_sec
            ]

            if interval_frames:
                avg_fake_prob = np.mean([f['fake_probability'] for f in interval_frames])
                avg_real_prob = np.mean([f['real_probability'] for f in interval_frames])

                interval_probs.append({
                    'interval': f'[{start_sec:.2f} - {end_sec:.2f})',
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'fake_probability': float(avg_fake_prob),
                    'real_probability': float(avg_real_prob),
                    'frame_count': len(interval_frames)
                })

        return {
            'frame_probabilities': frame_probs,
            'interval_probabilities': interval_probs
        }

    def _analyze_geometry(self) -> Dict[str, Any]:
        """
        Analyze geometry (MAR) features using statistical baseline.

        Returns:
            Dictionary with:
            - mean_mar: float
            - std_mar: float
            - abnormal_phonemes: List of phonemes with unusual MAR
            - baseline_info: Baseline collection metadata
        """
        # Load statistical baseline from Real videos (525 videos, 156k frames)
        baseline_path = Path(__file__).parent.parent.parent / 'mar_baseline_pia_real_fixed.json'

        try:
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline = json.load(f)
            phoneme_stats = baseline['phoneme_stats']
            baseline_info = baseline['collection_info']
            print(f"\n[MAR BASELINE] Loaded from {baseline_info['num_videos']} Real videos")
        except Exception as e:
            print(f"\n[MAR BASELINE ERROR] Failed to load baseline: {e}")
            print(f"  Using fallback: uniform range [0.15, 0.45]")
            phoneme_stats = None
            baseline_info = None

        # MFA → 한글 매핑
        phoneme_to_korean = {
            'A': 'ㅏ', 'B': 'ㅂ', 'BB': 'ㅃ', 'CHh': 'ㅊ',
            'E': 'ㅔ', 'EU': 'ㅡ', 'I': 'ㅣ', 'M': 'ㅁ',
            'O': 'ㅗ', 'Ph': 'ㅍ', 'U': 'ㅜ', 'iA': 'ㅑ',
            'iO': 'ㅛ', 'iU': 'ㅠ'
        }

        # Extract MAR values: (1, P, F, 1) → (P, F)
        geoms = self.input_data['geoms']  # (1, P, F, 1)
        mar_values = geoms[0, :, :, 0].cpu().numpy()  # (P, F)

        # Compute statistics (exclude padding - 0 values)
        mar_nonzero = mar_values[mar_values != 0]
        if len(mar_nonzero) > 0:
            mean_mar = float(mar_nonzero.mean())
            std_mar = float(mar_nonzero.std())
        else:
            mean_mar = 0.0
            std_mar = 0.0

        # Find abnormal phonemes using statistical baseline
        abnormal_phonemes = []
        P_actual = mar_values.shape[0]

        for pi, phoneme in enumerate(self.phoneme_vocab):
            if pi >= P_actual:
                break

            # Average MAR across F frames for this phoneme (exclude padding - 0 values)
            phoneme_frames = mar_values[pi]
            phoneme_frames_nonzero = phoneme_frames[phoneme_frames != 0]

            # Skip if all frames are padding
            if len(phoneme_frames_nonzero) == 0:
                continue

            phoneme_mar = float(phoneme_frames_nonzero.mean())

            if phoneme_stats and phoneme in phoneme_stats:
                stats = phoneme_stats[phoneme]

                # Use P10-P90 as expected range (80% of Real video distribution)
                expected_min = stats['p10']
                expected_max = stats['p90']
                expected_mean = stats['mean']
                expected_std = stats['std']

                # Calculate Z-score (how many std devs from mean)
                z_score = (phoneme_mar - expected_mean) / expected_std if expected_std > 0 else 0

                # Check if outside P10-P90 range (abnormal)
                if not (expected_min <= phoneme_mar <= expected_max):
                    # Determine severity
                    if abs(z_score) > 3:
                        severity = "high"  # >3 std devs
                    elif abs(z_score) > 2:
                        severity = "medium"  # >2 std devs
                    else:
                        severity = "low"  # within 2 std devs

                    deviation = float(phoneme_mar - expected_mean)
                    deviation_percent = abs(deviation / expected_mean * 100) if expected_mean > 0 else 0
                    
                    abnormal_phonemes.append({
                        'phoneme': phoneme_to_korean.get(phoneme, phoneme),
                        'phoneme_mfa': phoneme,
                        'measured_mar': float(phoneme_mar),
                        'expected_mean': expected_mean,
                        'expected_range': [expected_min, expected_max],
                        'deviation': deviation,
                        'deviation_percent': deviation_percent,
                        'z_score': float(z_score),
                        'severity': severity,
                        'explanation': self._generate_mar_explanation(
                            phoneme, phoneme_mar, expected_mean, expected_min, expected_max, z_score
                        )
                    })
            else:
                # Fallback: use simple range check if baseline not available
                expected_min, expected_max = 0.15, 0.45
                expected_mean_fallback = 0.30
                if not (expected_min <= phoneme_mar <= expected_max):
                    deviation = float(phoneme_mar - expected_mean_fallback)
                    deviation_percent = abs(deviation / expected_mean_fallback * 100) if expected_mean_fallback > 0 else 0
                    
                    abnormal_phonemes.append({
                        'phoneme': phoneme_to_korean.get(phoneme, phoneme),
                        'phoneme_mfa': phoneme,
                        'measured_mar': float(phoneme_mar),
                        'expected_mean': expected_mean_fallback,
                        'expected_range': [expected_min, expected_max],
                        'deviation': deviation,
                        'deviation_percent': deviation_percent,
                        'z_score': 0.0,
                        'severity': 'unknown',
                        'explanation': f'Baseline 없음 (측정값: {phoneme_mar:.3f})'
                    })

        return {
            'mean_mar': mean_mar,
            'std_mar': std_mar,
            'abnormal_phonemes': abnormal_phonemes,
            'num_abnormal': len(abnormal_phonemes),
            'baseline_info': {
                'num_videos': baseline_info['num_videos'] if baseline_info else 0,
                'total_frames': baseline_info['total_frames'] if baseline_info else 0,
                'mar_version': baseline_info['mar_version'] if baseline_info else 'unknown'
            } if baseline_info else None
        }

    def _generate_mar_explanation(
        self,
        phoneme: str,
        measured: float,
        expected_mean: float,
        expected_min: float,
        expected_max: float,
        z_score: float
    ) -> str:
        """
        Generate Korean explanation for MAR anomaly.

        Args:
            phoneme: Phoneme MFA code
            measured: Measured MAR value
            expected_mean: Expected mean from baseline
            expected_min: P10 from baseline
            expected_max: P90 from baseline
            z_score: Standard deviations from mean

        Returns:
            Korean explanation string
        """
        phoneme_to_korean = {
            'A': 'ㅏ', 'B': 'ㅂ', 'BB': 'ㅃ', 'CHh': 'ㅊ',
            'E': 'ㅔ', 'EU': 'ㅡ', 'I': 'ㅣ', 'M': 'ㅁ',
            'O': 'ㅗ', 'Ph': 'ㅍ', 'U': 'ㅜ', 'iA': 'ㅑ',
            'iO': 'ㅛ', 'iU': 'ㅠ'
        }

        phoneme_kr = phoneme_to_korean.get(phoneme, phoneme)

        if measured > expected_max:
            direction = "크게"
            comparison = f"정상 범위({expected_min:.3f}~{expected_max:.3f})보다 {measured - expected_max:.3f} 더 큼"
        else:
            direction = "작게"
            comparison = f"정상 범위({expected_min:.3f}~{expected_max:.3f})보다 {expected_min - measured:.3f} 더 작음"

        # Z-score interpretation
        if abs(z_score) > 3:
            severity_desc = "매우 비정상적"
        elif abs(z_score) > 2:
            severity_desc = "비정상적"
        else:
            severity_desc = "약간 비정상적"

        return f"'{phoneme_kr}' 음소의 입 벌림이 {direction} 측정됨 ({measured:.3f}). " \
               f"{comparison}. " \
               f"Z-score: {z_score:.2f} ({severity_desc})"

    def _generate_korean_summary(
        self,
        is_fake: bool,
        confidence: float,
        branch_contributions: Dict[str, float],
        phoneme_scores: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate Korean language explanation.

        Returns:
            Dictionary with:
            - overall: Overall detection summary
            - reasoning: Detailed reasoning
            - key_findings: Key findings (bullet points)
        """
        # Overall summary
        label = "딥페이크" if is_fake else "진짜 영상"
        conf_pct = confidence * 100

        overall = f"이 영상은 {label}으로 판정되었습니다 (신뢰도: {conf_pct:.1f}%)."

        # Reasoning based on branch contributions
        top_branch = max(branch_contributions, key=branch_contributions.get)
        top_contrib = branch_contributions[top_branch] * 100

        branch_names_ko = {
            'visual': '시각 정보 (입 모양)',
            'geometry': '기하학적 특징 (입술 벌림 정도)',
            'identity': '신원 정보 (얼굴 특징)'
        }

        reasoning = f"판정 근거: {branch_names_ko[top_branch]}가 가장 큰 영향을 주었습니다 ({top_contrib:.1f}% 기여도). "

        # Add top suspicious phonemes
        top_phonemes = sorted(phoneme_scores, key=lambda x: x['score'], reverse=True)[:3]
        phoneme_names = ', '.join([p['phoneme'] for p in top_phonemes])
        reasoning += f"특히 '{phoneme_names}' 음소에서 의심스러운 패턴이 감지되었습니다."

        # Key findings
        findings = []

        # Finding 1: Branch analysis
        visual_contrib = branch_contributions['visual'] * 100
        geometry_contrib = branch_contributions['geometry'] * 100
        identity_contrib = branch_contributions['identity'] * 100

        findings.append(
            f"- 시각 정보 {visual_contrib:.1f}%, 기하학 {geometry_contrib:.1f}%, "
            f"신원 {identity_contrib:.1f}% 비율로 판정에 기여"
        )

        # Finding 2: Phoneme attention
        high_attention_phonemes = [p for p in phoneme_scores if p['importance_level'] == 'high']
        if high_attention_phonemes:
            phonemes_str = ', '.join([p['phoneme'] for p in high_attention_phonemes])
            findings.append(f"- 주목도가 높은 음소: {phonemes_str}")

        # Finding 3: Confidence assessment
        if confidence > 0.8:
            findings.append(f"- 신뢰도가 매우 높음 (80% 이상)")
        elif confidence > 0.6:
            findings.append(f"- 신뢰도가 양호함 (60-80%)")
        else:
            findings.append(f"- 신뢰도가 낮음 (60% 미만) - 추가 검증 권장")

        key_findings = '\n'.join(findings)

        return {
            'overall': overall,
            'reasoning': reasoning,
            'key_findings': key_findings
        }

    def _build_result(
        self,
        video_id: str,
        is_fake: bool,
        confidence: float,
        probs: torch.Tensor,
        branch_contributions: Dict[str, float],
        phoneme_scores: List[Dict[str, Any]],
        temporal_heatmap: List[List[float]],
        geometry_analysis: Dict[str, Any],
        korean_summary: Dict[str, str],
        temporal_probs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build final DeepfakeXAIResult matching TypeScript interface.

        Returns:
            Complete XAI result dictionary
        """
        from datetime import datetime

        # [DEPRECATED] Temporal Analysis 제거 - Temporal Heatmap은 단순히 attention score를 5프레임에 복제한 것으로
        # 실제 시간 변화를 보여주지 않아 유용성이 낮음
        # Geometry Analysis는 실제 MAR deviation과 Z-score 기반 이상 탐지를 제공

        # Geometry Analysis에 시각화 정보 추가
        geometry_analysis_with_viz = geometry_analysis.copy()
        if 'abnormal_phonemes' in geometry_analysis and len(geometry_analysis['abnormal_phonemes']) > 0:
            visualization_summary = []
            for phoneme_info in geometry_analysis['abnormal_phonemes']:
                phoneme = phoneme_info.get('phoneme', '')
                deviation = phoneme_info.get('deviation', 0)
                deviation_percent = phoneme_info.get('deviation_percent', abs(deviation / phoneme_info.get('expected_mean', 1) * 100) if phoneme_info.get('expected_mean', 0) > 0 else 0)
                
                if deviation > 0:
                    viz_desc = f"'{phoneme}' 발음 시 입을 {deviation_percent:.1f}% 더 크게 벌렸습니다"
                else:
                    viz_desc = f"'{phoneme}' 발음 시 입을 {deviation_percent:.1f}% 더 작게 벌렸습니다"
                
                visualization_summary.append({
                    'phoneme': phoneme,
                    'description': viz_desc,
                    'severity': phoneme_info.get('severity', 'unknown'),
                    'deviation_percent': deviation_percent,
                    'measured_mar': phoneme_info.get('measured_mar', 0),
                    'expected_mar': phoneme_info.get('expected_mean', 0)
                })
            
            geometry_analysis_with_viz['visualization'] = {
                'summary': visualization_summary,
                'plot_type': 'mar_deviation_bar_chart',
                'description': 'MAR Deviation 분석: 각 음소별 입 벌림 정도를 정상 범위와 비교하여 시각화'
            }

        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': '1.0.0',
                'xai_version': '1.0.0'
            },
            'video_info': {
                'video_id': video_id,
                'duration_seconds': 0,  # TODO: Add from input
                'num_phonemes': len(self.phoneme_vocab),
                'num_frames': 0  # TODO: Add from input
            },
            'detection': {
                'is_fake': is_fake,
                'confidence': confidence,
                'prediction_label': 'FAKE' if is_fake else 'REAL',
                'probabilities': {
                    'real': probs[0, 0].item(),
                    'fake': probs[0, 1].item()
                }
            },
            'summary': korean_summary,
            'phoneme_analysis': {
                'phoneme_scores': phoneme_scores,
                'top_suspicious_phonemes': self._get_top_phonemes(phoneme_scores, 3)
            },
            # [DEPRECATED] temporal_analysis 제거
            # 'temporal_analysis': {
            #     'heatmap': temporal_heatmap,
            #     'peak_frames': [],
            #     'frame_probabilities': temporal_probs['frame_probabilities'],
            #     'interval_probabilities': temporal_probs['interval_probabilities']
            # },
            'geometry_analysis': geometry_analysis_with_viz,
            'model_info': {
                'architecture': 'PIA',
                'branches': ['visual', 'geometry', 'identity'],
                'branch_contributions': branch_contributions
            }
        }

    def _get_top_phonemes(
        self,
        phoneme_scores: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Get top-k phonemes by attention score."""
        sorted_phonemes = sorted(
            phoneme_scores,
            key=lambda x: x.get('score', 0.0),
            reverse=True
        )
        return sorted_phonemes[:top_k]
