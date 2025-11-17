"""
Teacher Model: MMMS-BA (Multi-Modal Multi-Sequence Bi-Modal Attention)

Based on:
- Contextual Cross-Modal Attention for Audio-Visual Deepfake Detection
- audio-visual-deepfake-main repository
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from src.models.attention import BiModalAttention


class ModalityEncoder(nn.Module):
    """
    Encoder for single modality using Bi-GRU
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 300,
        num_layers: int = 1,
        dropout: float = 0.5,
        recurrent_dropout: float = 0.5
    ):
        """
        Initialize modality encoder

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (per direction)
            num_layers: Number of GRU layers
            dropout: Dropout rate
            recurrent_dropout: Recurrent dropout (not directly supported in PyTorch)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bi-directional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # Output dimension: hidden_dim * 2 (bidirectional)
        self.output_dim = hidden_dim * 2

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (B, T, input_dim)
            mask: Mask (B, T)

        Returns:
            output: GRU output (B, T, hidden_dim*2)
        """
        # Pack padded sequence if mask provided
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
            output_packed, _ = self.gru(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed,
                batch_first=True
            )
        else:
            output, _ = self.gru(x)

        output = self.dropout(output)

        return output


class MMMSBA(nn.Module):
    """
    Multi-Modal Multi-Sequence Bi-Modal Attention (MMMS-BA) Model

    Architecture:
    1. Three Bi-GRU encoders (Audio, Visual, Lip)
    2. Dense layers for each modality
    3. Bi-modal attention between modality pairs
    4. Classification head
    """

    def __init__(
        self,
        # Input dimensions
        audio_dim: int = 40,  # MFCC features
        visual_dim: int = 2048,  # Visual features (e.g., ResNet)
        lip_dim: int = 512,  # Lip features

        # GRU configuration
        gru_hidden_dim: int = 300,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.5,

        # Dense configuration
        dense_hidden_dim: int = 100,
        dense_dropout: float = 0.7,

        # Attention
        attention_type: str = "bi_modal",  # bi_modal | self | none

        # Output
        num_classes: int = 2
    ):
        """
        Initialize MMMS-BA model

        Args:
            audio_dim: Audio feature dimension
            visual_dim: Visual feature dimension
            lip_dim: Lip feature dimension
            gru_hidden_dim: GRU hidden dimension
            gru_num_layers: Number of GRU layers
            gru_dropout: GRU dropout
            dense_hidden_dim: Dense layer hidden dimension
            dense_dropout: Dense dropout
            attention_type: Type of attention mechanism
            num_classes: Number of output classes
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.lip_dim = lip_dim
        self.attention_type = attention_type

        # === Feature Extractors (for raw inputs) ===
        # If input is already features, these will be identity
        self.visual_feature_extractor = self._build_visual_extractor()
        self.lip_feature_extractor = self._build_lip_extractor()

        # === GRU Encoders ===
        self.audio_encoder = ModalityEncoder(
            input_dim=audio_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout
        )

        self.visual_encoder = ModalityEncoder(
            input_dim=visual_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout
        )

        self.lip_encoder = ModalityEncoder(
            input_dim=lip_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout
        )

        gru_output_dim = gru_hidden_dim * 2  # Bidirectional

        # === Dense Layers ===
        self.audio_dense = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(gru_output_dim, dense_hidden_dim),
            nn.Tanh()
        )

        self.visual_dense = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(gru_output_dim, dense_hidden_dim),
            nn.Tanh()
        )

        self.lip_dense = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(gru_output_dim, dense_hidden_dim),
            nn.Tanh()
        )

        # === Attention Layers ===
        if attention_type == "bi_modal":
            self.attention_vl = BiModalAttention(dense_hidden_dim)  # Visual-Lip
            self.attention_av = BiModalAttention(dense_hidden_dim)  # Audio-Visual
            self.attention_la = BiModalAttention(dense_hidden_dim)  # Lip-Audio

            # After bi-modal attention, features are concatenated
            # Each attention outputs 2*dense_hidden_dim
            # Total: 3 * 2 * dense_hidden_dim + 3 * dense_hidden_dim
            final_dim = 9 * dense_hidden_dim

        elif attention_type == "self":
            from src.models.attention import SelfAttention
            self.attention_audio = SelfAttention(dense_hidden_dim)
            self.attention_visual = SelfAttention(dense_hidden_dim)
            self.attention_lip = SelfAttention(dense_hidden_dim)

            # After self-attention: 3 * dense_hidden_dim + 3 * dense_hidden_dim
            final_dim = 6 * dense_hidden_dim

        else:  # none
            # Just concatenate
            final_dim = 3 * dense_hidden_dim

        # === Classification Head ===
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, num_classes)
        )

    def _build_visual_extractor(self) -> nn.Module:
        """
        Build visual feature extractor

        For now, use a simple CNN
        """
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(start_dim=1)  # (B*T, 256)
        )

    def _build_lip_extractor(self) -> nn.Module:
        """
        Build lip feature extractor

        Smaller CNN for lip region
        """
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(start_dim=1)  # (B*T, 128)
        )

    def extract_visual_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from frames

        Args:
            frames: (B, T, 3, H, W)

        Returns:
            features: (B, T, visual_dim)
        """
        B, T = frames.size(0), frames.size(1)

        # Reshape: (B, T, 3, H, W) -> (B*T, 3, H, W)
        frames_flat = frames.view(B * T, *frames.shape[2:])

        # Extract features
        features_flat = self.visual_feature_extractor(frames_flat)  # (B*T, visual_dim)

        # Reshape back: (B*T, visual_dim) -> (B, T, visual_dim)
        features = features_flat.view(B, T, -1)

        return features

    def extract_lip_features(self, lip_frames: torch.Tensor) -> torch.Tensor:
        """
        Extract lip features from lip frames

        Args:
            lip_frames: (B, T, 3, lip_H, lip_W)

        Returns:
            features: (B, T, lip_dim)
        """
        B, T = lip_frames.size(0), lip_frames.size(1)

        # Reshape: (B, T, 3, H, W) -> (B*T, 3, H, W)
        lip_flat = lip_frames.view(B * T, *lip_frames.shape[2:])

        # Extract features
        features_flat = self.lip_feature_extractor(lip_flat)  # (B*T, lip_dim)

        # Reshape back
        features = features_flat.view(B, T, -1)

        return features

    def forward(
        self,
        audio: torch.Tensor,
        frames: torch.Tensor,
        lip: torch.Tensor,
        audio_mask: torch.Tensor = None,
        frames_mask: torch.Tensor = None,
        frame_level: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            audio: Audio features (B, T_audio, audio_dim)
            frames: Visual frames (B, T_frames, 3, H, W)
            lip: Lip frames (B, T_lip, 3, lip_H, lip_W)
            audio_mask: Audio mask (B, T_audio)
            frames_mask: Frames mask (B, T_frames)
            frame_level: If True, return frame-level predictions (B, T, num_classes)
                        If False, return video-level prediction (B, num_classes)

        Returns:
            logits: Classification logits
                - (B, num_classes) if frame_level=False
                - (B, T_frames, num_classes) if frame_level=True
        """
        # === Extract Features ===
        visual_features = self.extract_visual_features(frames)  # (B, T, visual_dim)
        lip_features = self.extract_lip_features(lip)  # (B, T, lip_dim)

        # === GRU Encoding ===
        audio_encoded = self.audio_encoder(audio, audio_mask)  # (B, T_audio, gru_dim*2)
        visual_encoded = self.visual_encoder(visual_features, frames_mask)  # (B, T_frames, gru_dim*2)
        lip_encoded = self.lip_encoder(lip_features, frames_mask)  # (B, T_lip, gru_dim*2)

        # === Dense Projection ===
        audio_dense = self.audio_dense(audio_encoded)  # (B, T_audio, dense_dim)
        visual_dense = self.visual_dense(visual_encoded)  # (B, T_frames, dense_dim)
        lip_dense = self.lip_dense(lip_encoded)  # (B, T_lip, dense_dim)

        # === Attention ===
        if self.attention_type == "bi_modal":
            # Visual-Lip attention
            vl_att = self.attention_vl(visual_dense, lip_dense, frames_mask, frames_mask)

            # Audio-Visual attention
            av_att = self.attention_av(audio_dense, visual_dense, audio_mask, frames_mask)

            # Lip-Audio attention
            la_att = self.attention_la(lip_dense, audio_dense, frames_mask, audio_mask)

            if frame_level:
                # Frame-level prediction: Keep temporal dimension
                # Interpolate audio features to match frame length
                T_frames = visual_dense.size(1)
                T_audio = audio_dense.size(1)

                if T_audio != T_frames:
                    # Interpolate audio and audio-related features to T_frames
                    audio_dense_interp = F.interpolate(
                        audio_dense.transpose(1, 2),  # (B, D, T_audio)
                        size=T_frames,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # (B, T_frames, D)

                    av_att_interp = F.interpolate(
                        av_att.transpose(1, 2),  # (B, D, T_audio)
                        size=T_frames,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # (B, T_frames, D)

                    la_att_interp = la_att  # Already at T_frames
                else:
                    audio_dense_interp = audio_dense
                    av_att_interp = av_att
                    la_att_interp = la_att

                # Concatenate along feature dimension (keep time dimension)
                merged = torch.cat([
                    vl_att, av_att_interp, la_att_interp,
                    audio_dense_interp, visual_dense, lip_dense
                ], dim=-1)  # (B, T_frames, 9*dense_dim)

            else:
                # Video-level prediction: Pool over time
                vl_pooled = self._masked_pool(vl_att, frames_mask)
                av_pooled = self._masked_pool(av_att, audio_mask)
                la_pooled = self._masked_pool(la_att, frames_mask)

                audio_pooled = self._masked_pool(audio_dense, audio_mask)
                visual_pooled = self._masked_pool(visual_dense, frames_mask)
                lip_pooled = self._masked_pool(lip_dense, frames_mask)

                merged = torch.cat([
                    vl_pooled, av_pooled, la_pooled,
                    audio_pooled, visual_pooled, lip_pooled
                ], dim=-1)  # (B, 9*dense_dim)

        elif self.attention_type == "self":
            audio_att = self.attention_audio(audio_dense, audio_mask)
            visual_att = self.attention_visual(visual_dense, frames_mask)
            lip_att = self.attention_lip(lip_dense, frames_mask)

            if frame_level:
                # Frame-level: Keep temporal dimension
                T_frames = visual_dense.size(1)
                T_audio = audio_dense.size(1)

                if T_audio != T_frames:
                    audio_att_interp = F.interpolate(
                        audio_att.transpose(1, 2), size=T_frames, mode='linear', align_corners=False
                    ).transpose(1, 2)
                    audio_dense_interp = F.interpolate(
                        audio_dense.transpose(1, 2), size=T_frames, mode='linear', align_corners=False
                    ).transpose(1, 2)
                else:
                    audio_att_interp = audio_att
                    audio_dense_interp = audio_dense

                merged = torch.cat([
                    audio_att_interp, visual_att, lip_att,
                    audio_dense_interp, visual_dense, lip_dense
                ], dim=-1)  # (B, T_frames, 6*dense_dim)

            else:
                # Video-level: Pool
                audio_att_pooled = self._masked_pool(audio_att, audio_mask)
                visual_att_pooled = self._masked_pool(visual_att, frames_mask)
                lip_att_pooled = self._masked_pool(lip_att, frames_mask)

                audio_pooled = self._masked_pool(audio_dense, audio_mask)
                visual_pooled = self._masked_pool(visual_dense, frames_mask)
                lip_pooled = self._masked_pool(lip_dense, frames_mask)

                merged = torch.cat([
                    audio_att_pooled, visual_att_pooled, lip_att_pooled,
                    audio_pooled, visual_pooled, lip_pooled
                ], dim=-1)  # (B, 6*dense_dim)

        else:  # none
            if frame_level:
                # Frame-level: Keep temporal dimension
                T_frames = visual_dense.size(1)
                T_audio = audio_dense.size(1)

                if T_audio != T_frames:
                    audio_dense_interp = F.interpolate(
                        audio_dense.transpose(1, 2), size=T_frames, mode='linear', align_corners=False
                    ).transpose(1, 2)
                else:
                    audio_dense_interp = audio_dense

                merged = torch.cat([audio_dense_interp, visual_dense, lip_dense], dim=-1)  # (B, T_frames, 3*dense_dim)
            else:
                # Video-level: Pool
                audio_pooled = self._masked_pool(audio_dense, audio_mask)
                visual_pooled = self._masked_pool(visual_dense, frames_mask)
                lip_pooled = self._masked_pool(lip_dense, frames_mask)

                merged = torch.cat([audio_pooled, visual_pooled, lip_pooled], dim=-1)  # (B, 3*dense_dim)

        # === Classification ===
        if frame_level:
            # Apply classifier to each frame (TimeDistributed)
            B, T, D = merged.shape
            merged_flat = merged.view(B * T, D)  # (B*T, D)
            logits_flat = self.classifier(merged_flat)  # (B*T, num_classes)
            logits = logits_flat.view(B, T, -1)  # (B, T, num_classes)
        else:
            # Single prediction for video
            logits = self.classifier(merged)  # (B, num_classes)

        return logits

    def _masked_pool(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Masked mean pooling over time dimension

        Args:
            x: Input (B, T, D)
            mask: Mask (B, T)

        Returns:
            pooled: (B, D)
        """
        if mask is None:
            return x.mean(dim=1)

        # Expand mask: (B, T) -> (B, T, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # Masked sum
        masked_sum = (x * mask_expanded).sum(dim=1)

        # Masked count
        masked_count = mask_expanded.sum(dim=1).clamp(min=1)

        # Masked mean
        pooled = masked_sum / masked_count

        return pooled
