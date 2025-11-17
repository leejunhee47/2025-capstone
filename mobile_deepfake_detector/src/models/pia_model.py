"""
PIA (Phoneme-Temporal and Identity-Dynamic Analysis) Model

Korean Deepfake Detection using phoneme-based multimodal analysis.

Architecture:
- Visual Branch: 3D CNN + EfficientNet-B0
- Geometry Branch: MLP encoder
- Identity Branch: ArcFace MLP encoder
- Fusion: Multi-head Attention Pooling
- Classifier: MLP (2 classes: Real/Fake)

Reference: PIA-main/src/model.py
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling for phoneme-level features.

    Args:
        dim: Input feature dimension
        num_heads: Number of attention heads
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Parameter(torch.randn(num_heads, 1, dim))
        self.dropout = nn.Dropout(0.1)
        self.attention_weights = None  # For XAI: (B, H, 1, P)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, P, D) - Batch, Phonemes, Dimension
            mask: (B, P) - Valid phoneme mask (1=valid, 0=padding), optional

        Returns:
            (B, D) - Pooled features
        """
        B, P, D = x.size()

        # Expand for multi-head attention
        x = x.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # (B, H, P, D)
        q = self.q.unsqueeze(0).repeat(B, 1, 1, 1)          # (B, H, 1, D)

        # Attention scores
        attn = (q @ x.transpose(2, 3)) / (D ** 0.5)         # (B, H, 1, P)

        # Apply mask: set padding positions to -inf before softmax
        if mask is not None:
            # Expand mask: (B, P) → (B, H, 1, P)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, P)
            mask_expanded = mask_expanded.repeat(1, self.num_heads, 1, 1)  # (B, H, 1, P)

            # Set padding to -inf (becomes 0 after softmax)
            attn = attn.masked_fill(mask_expanded == 0, float('-inf'))

        w = attn.softmax(dim=-1)

        # Save attention weights for XAI (before dropout)
        self.attention_weights = w.detach()  # (B, H, 1, P)

        w = self.dropout(w)

        # Weighted aggregation
        out = w @ x                                         # (B, H, 1, D)
        return out.mean(1).squeeze(1)                       # (B, D)


class PIAModel(nn.Module):
    """
    PIA (Phoneme-Temporal and Identity-Dynamic Analysis) Model

    Tri-modal deepfake detection using:
    - Visual (lip movements)
    - Geometry (lip shape metrics)
    - Identity (ArcFace embeddings)

    Args:
        num_phonemes: Number of phoneme classes (default: 14 for Korean)
        frames_per_phoneme: Frames per phoneme (default: 5)
        arcface_dim: ArcFace embedding dimension (default: 512)
        geo_dim: Geometry feature dimension (default: 1, MAR only)
        embed_dim: Hidden embedding dimension (default: 128)
        num_heads: Multi-head attention heads (default: 4)
        num_classes: Output classes (default: 2 for binary)
    """

    def __init__(
        self,
        num_phonemes: int = 14,
        frames_per_phoneme: int = 5,
        arcface_dim: int = 512,
        geo_dim: int = 1,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 2
    ):
        super().__init__()

        self.P = num_phonemes
        self.F = frames_per_phoneme
        self.E = embed_dim

        # ===== Visual Branch =====
        # 3D CNN for temporal modeling across F frames
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 3, kernel_size=1),
            nn.BatchNorm3d(3),
            nn.ReLU()
        )

        # EfficientNet-B0 backbone (pretrained)
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone = eff.features

        # Fine-tune only last blocks (6, 7)
        for n, p in self.backbone.named_parameters():
            p.requires_grad = ("6" in n or "7" in n)

        # Feature projector (1280 → embed_dim)
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, embed_dim),
            nn.ReLU()
        )

        # ===== Geometry Branch =====
        self.geo_enc = nn.Sequential(
            nn.Linear(geo_dim, embed_dim),
            nn.ReLU()
        )

        # ===== Identity Branch (ArcFace) =====
        self.arc_enc = nn.Sequential(
            nn.Linear(arcface_dim, embed_dim),
            nn.ReLU()
        )

        # ===== Fusion & Classification =====
        self.dropout = nn.Dropout(0.2)
        self.attn_pool = MultiHeadAttentionPooling(embed_dim * 3, num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        geoms: torch.Tensor,
        imgs: torch.Tensor,
        arcs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass

        Args:
            geoms: (B, P, F, 1) - Geometry features (vertical MAR only)
            imgs: (B, P, F, 3, H, W) - Lip crop images
            arcs: (B, P, F, 512) - ArcFace embeddings
            mask: (B, P, F) - Valid data mask (optional)

        Returns:
            (logits, arcface_features):
                - logits: (B, num_classes)
                - arcface_features: (B, P, E) for temporal consistency loss
        """
        B, P, F, C, H, W = imgs.shape

        # ===== Visual Features =====
        # Process P phonemes independently
        x = imgs.view(B * P, C, F, H, W)   # (B*P, 3, F, H, W)

        # 3D CNN temporal modeling
        x = self.cnn3d(x)                  # (B*P, 3, F, H, W)
        x = x.mean(dim=2)                  # (B*P, 3, H, W) - average across time

        # EfficientNet feature extraction
        v = self.backbone(x)               # (B*P, 1280, h, w)
        v = self.projector(v)              # (B*P, E)
        v = v.view(B, P, self.E)           # (B, P, E)

        # ===== Geometry Features =====
        # Average geometry across F frames per phoneme
        g = geoms.mean(dim=2)              # (B, P, 1)
        g = self.geo_enc(g)                # (B, P, E)

        # ===== Identity Features (ArcFace) =====
        # Process and average across F frames
        a = arcs.view(B * P, F, -1)        # (B*P, F, 512)
        a = self.arc_enc(a)                # (B*P, F, E)
        a = a.mean(dim=1)                  # (B*P, E)
        a = a.view(B, P, self.E)           # (B, P, E)

        # ===== Multimodal Fusion =====
        fused = torch.cat([v, g, a], dim=-1)  # (B, P, 3*E)
        fused = self.dropout(fused)

        # Convert frame-level mask to phoneme-level mask
        # (B, P, F) → (B, P): phoneme is valid if any frame is valid
        mask_phoneme = None
        if mask is not None:
            mask_phoneme = (mask.sum(dim=-1) > 0).float()  # (B, P)

        # Multi-head attention pooling across phonemes (with mask)
        pooled = self.attn_pool(fused, mask_phoneme)     # (B, 3*E)

        # Classification
        logits = self.classifier(pooled)   # (B, num_classes)

        return logits, a  # Return ArcFace features for temporal loss


class PIAModelWithTemporalLoss(nn.Module):
    """
    PIA Model with Temporal Consistency Loss

    Adds ArcFace temporal consistency loss to detect identity drift.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = PIAModel(**kwargs)
        self.temporal_weight = 0.1  # λ in paper

    def forward(
        self,
        geoms: torch.Tensor,
        imgs: torch.Tensor,
        arcs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Forward with temporal consistency loss

        Returns:
            (logits, temporal_loss)
        """
        logits, arcface_features = self.model(geoms, imgs, arcs, mask)

        # Temporal consistency loss (optional, requires real ArcFace)
        temporal_loss = self.compute_temporal_consistency_loss(
            arcface_features, mask
        )

        return logits, temporal_loss

    def compute_temporal_consistency_loss(
        self,
        arcface_features: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss for ArcFace embeddings.

        L_temporal = Σ (1 - cos_similarity(a_t, a_{t+1})) * mask

        Args:
            arcface_features: (B, P, E)
            mask: (B, P) - Valid phoneme mask

        Returns:
            Scalar loss
        """
        if arcface_features.abs().sum() == 0:
            # Skip if dummy ArcFace (all zeros)
            return torch.tensor(0.0, device=arcface_features.device)

        B, P, E = arcface_features.shape

        if P < 2:
            return torch.tensor(0.0, device=arcface_features.device)

        # Normalize embeddings
        normalized = torch.nn.functional.normalize(arcface_features, p=2, dim=-1)

        # Cosine similarity between consecutive phonemes
        cos_sim = (normalized[:, :-1] * normalized[:, 1:]).sum(dim=-1)  # (B, P-1)

        # Loss = 1 - similarity (penalize drift)
        losses = 1.0 - cos_sim

        # Apply mask if provided
        if mask is not None:
            mask_pairs = mask[:, :-1] * mask[:, 1:]  # Both phonemes must be valid
            losses = losses * mask_pairs
            return losses.sum() / (mask_pairs.sum() + 1e-8)
        else:
            return losses.mean()


def create_pia_model(
    num_phonemes: int = 14,
    frames_per_phoneme: int = 5,
    use_temporal_loss: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create PIA model

    Args:
        num_phonemes: Number of phoneme classes
        frames_per_phoneme: Frames per phoneme
        use_temporal_loss: Whether to use temporal consistency loss
        **kwargs: Additional model arguments

    Returns:
        PIA model instance
    """
    if use_temporal_loss:
        return PIAModelWithTemporalLoss(
            num_phonemes=num_phonemes,
            frames_per_phoneme=frames_per_phoneme,
            **kwargs
        )
    else:
        return PIAModel(
            num_phonemes=num_phonemes,
            frames_per_phoneme=frames_per_phoneme,
            **kwargs
        )


if __name__ == "__main__":
    # Test model
    print("Testing PIA Model...")

    model = create_pia_model(
        num_phonemes=14,
        frames_per_phoneme=5,
        use_temporal_loss=False
    )

    # Dummy input
    B = 2  # Batch size
    P = 14  # Phonemes
    F = 5  # Frames per phoneme
    H, W = 112, 112  # Image size

    geoms = torch.randn(B, P, F, 1)  # Vertical MAR only
    imgs = torch.randn(B, P, F, 3, H, W)
    arcs = torch.randn(B, P, F, 512)

    logits, _ = model(geoms, imgs, arcs)

    print(f"Input shapes:")
    print(f"  geoms: {geoms.shape}")
    print(f"  imgs: {imgs.shape}")
    print(f"  arcs: {arcs.shape}")
    print(f"Output:")
    print(f"  logits: {logits.shape}")
    print(f"\n✅ PIA Model test passed!")
