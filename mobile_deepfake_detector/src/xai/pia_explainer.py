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
            P = len(self.phoneme_vocab)
            E = output.size(1)
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
            P = len(self.phoneme_vocab)
            # Average across F frames
            features = output.mean(dim=1)  # (B*P, E)
            E = features.size(1)
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

        # Extract prediction
        pred_class = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, pred_class].item()
        is_fake = pred_class == 1

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
        Analyze contribution of each branch (Visual, Geometry, Identity).

        Returns:
            Dictionary with branch names and contribution scores (0-1)
        """
        # Extract activations (B, P, E)
        visual = self.activations['visual']  # (1, P, E)
        geometry = self.activations['geometry']  # (1, P, E)
        identity = self.activations['identity']  # (1, P, E)

        # Compute L2 norm as contribution metric
        visual_norm = torch.norm(visual, p=2).item()
        geometry_norm = torch.norm(geometry, p=2).item()
        identity_norm = torch.norm(identity, p=2).item()

        # Normalize to 0-1 (relative contributions)
        total = visual_norm + geometry_norm + identity_norm
        if total == 0:
            return {'visual': 0.33, 'geometry': 0.33, 'identity': 0.34}

        return {
            'visual': visual_norm / total,
            'geometry': geometry_norm / total,
            'identity': identity_norm / total
        }

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
        attn_weights = self.activations['attention_weights']  # (1, H, 1, P)

        if attn_weights is None:
            # Fallback: uniform attention
            scores = [1.0 / len(self.phoneme_vocab)] * len(self.phoneme_vocab)
        else:
            # Average across heads: (1, H, 1, P) → (P,)
            scores = attn_weights.mean(dim=1).squeeze().cpu().numpy()  # (P,)

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
        Analyze geometry (MAR) features.

        Returns:
            Dictionary with:
            - mean_mar: float
            - std_mar: float
            - abnormal_phonemes: List of phonemes with unusual MAR
        """
        # Expected MAR ranges (from korean_phoneme_config.py)
        expected_mar_ranges = {
            'M': (0.10, 0.30), 'B': (0.10, 0.30), 'BB': (0.10, 0.30), 'Ph': (0.10, 0.30),
            'A': (0.60, 0.90), 'E': (0.50, 0.80), 'iA': (0.55, 0.85),
            'O': (0.20, 0.50), 'U': (0.20, 0.50), 'iO': (0.25, 0.55), 'iU': (0.25, 0.55),
            'I': (0.30, 0.60), 'EU': (0.30, 0.60),
            'CHh': (0.25, 0.55)
        }

        # MFA → 한글 매핑
        phoneme_to_korean = {
            'A': 'ㅏ', 'B': 'ㅂ', 'BB': 'ㅃ', 'CHh': 'ㅊ',
            'E': 'ㅔ', 'EU': 'ㅡ', 'I': 'ㅣ', 'M': 'ㅁ',
            'O': 'ㅗ', 'Ph': 'ㅍ', 'U': 'ㅜ', 'iA': 'ㅑ',
            'iO': 'ㅛ', 'iU': 'ㅠ'
        }

        # Extract MAR values: (1, P, F, 1) → (P, F)
        geoms = self.input_data['geoms']  # (1, P, F, 1)
        mar_values = geoms.squeeze().cpu().numpy()  # (P, F)

        # Compute statistics (exclude padding - 0 values)
        mar_nonzero = mar_values[mar_values != 0]
        if len(mar_nonzero) > 0:
            mean_mar = float(mar_nonzero.mean())
            std_mar = float(mar_nonzero.std())
        else:
            mean_mar = 0.0
            std_mar = 0.0

        # Find abnormal phonemes (MAR outside expected range)
        abnormal_phonemes = []
        for pi, phoneme in enumerate(self.phoneme_vocab):
            # Average MAR across F frames for this phoneme
            phoneme_mar = mar_values[pi].mean()

            # Check if within expected range (±20% tolerance)
            if phoneme in expected_mar_ranges:
                min_mar, max_mar = expected_mar_ranges[phoneme]
                tolerance = 0.2

                if not ((min_mar - tolerance) <= phoneme_mar <= (max_mar + tolerance)):
                    abnormal_phonemes.append({
                        'phoneme': phoneme_to_korean.get(phoneme, phoneme),
                        'phoneme_mfa': phoneme,
                        'measured_mar': float(phoneme_mar),
                        'expected_range': [min_mar, max_mar],
                        'deviation': float(phoneme_mar - ((min_mar + max_mar) / 2))
                    })

        return {
            'mean_mar': mean_mar,
            'std_mar': std_mar,
            'abnormal_phonemes': abnormal_phonemes,
            'num_abnormal': len(abnormal_phonemes)
        }

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
            f"• 시각 정보 {visual_contrib:.1f}%, 기하학 {geometry_contrib:.1f}%, "
            f"신원 {identity_contrib:.1f}% 비율로 판정에 기여"
        )

        # Finding 2: Phoneme attention
        high_attention_phonemes = [p for p in phoneme_scores if p['importance_level'] == 'high']
        if high_attention_phonemes:
            phonemes_str = ', '.join([p['phoneme'] for p in high_attention_phonemes])
            findings.append(f"• 주목도가 높은 음소: {phonemes_str}")

        # Finding 3: Confidence assessment
        if confidence > 0.8:
            findings.append(f"• 신뢰도가 매우 높음 (80% 이상)")
        elif confidence > 0.6:
            findings.append(f"• 신뢰도가 양호함 (60-80%)")
        else:
            findings.append(f"• 신뢰도가 낮음 (60% 미만) - 추가 검증 권장")

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
            'temporal_analysis': {
                'heatmap': temporal_heatmap,
                'peak_frames': [],  # TODO: Implement
                'frame_probabilities': temporal_probs['frame_probabilities'],
                'interval_probabilities': temporal_probs['interval_probabilities']
            },
            'geometry_analysis': geometry_analysis,
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
