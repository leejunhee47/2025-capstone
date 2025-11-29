"""
Attention mechanisms for multimodal fusion
"""

from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiModalAttention(nn.Module):
    """
    Bi-modal attention mechanism for two modalities

    Given modalities X and Y:
    - Compute attention scores: X @ Y^T
    - Apply softmax
    - Compute weighted representations
    - Element-wise multiplication with original
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize bi-modal attention

        Args:
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor = None,
        y_mask: torch.Tensor = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass

        Args:
            x: First modality (B, T_x, D)
            y: Second modality (B, T_y, D)
            x_mask: Mask for x (B, T_x)
            y_mask: Mask for y (B, T_y)
            return_attention: If True, also return attention weights (default: False)

        Returns:
            If return_attention=False:
                attended: Concatenated attended features (B, T_x, 2*D)
            If return_attention=True:
                attended: Concatenated attended features (B, T_x, 2*D)
                (attn_xy, attn_yx): Tuple of attention weight tensors
                    attn_xy: (B, T_x, T_y) - X -> Y attention
                    attn_yx: (B, T_y, T_x) - Y -> X attention
        """
        # X -> Y attention
        # scores: (B, T_x, T_y)
        scores_xy = torch.bmm(x, y.transpose(1, 2))

        if y_mask is not None:
            # Expand mask: (B, T_y) -> (B, 1, T_y)
            # Convert to boolean type for proper masking
            y_mask_expanded = y_mask.unsqueeze(1).bool()
            # Use -1e4 instead of -1e9 for float16 compatibility
            scores_xy = scores_xy.masked_fill(~y_mask_expanded, -1e4)

        attn_xy = F.softmax(scores_xy, dim=-1)  # (B, T_x, T_y)
        attended_xy = torch.bmm(attn_xy, y)  # (B, T_x, D)
        output_x = attended_xy * x  # Element-wise multiplication

        # Y -> X attention
        # scores: (B, T_y, T_x)
        scores_yx = torch.bmm(y, x.transpose(1, 2))

        if x_mask is not None:
            # Convert to boolean type for proper masking
            x_mask_expanded = x_mask.unsqueeze(1).bool()
            # Use -1e4 instead of -1e9 for float16 compatibility
            scores_yx = scores_yx.masked_fill(~x_mask_expanded, -1e4)

        attn_yx = F.softmax(scores_yx, dim=-1)  # (B, T_y, T_x)
        attended_yx = torch.bmm(attn_yx, x)  # (B, T_y, D)
        output_y = attended_yx * y

        # For X -> Y attention, we need to bring output_y to T_x length
        # Use the attention weights from X->Y to aggregate Y's output
        # attn_xy: (B, T_x, T_y)
        # output_y: (B, T_y, D)
        # We want: (B, T_x, D)
        output_y_to_x = torch.bmm(attn_xy, output_y)  # (B, T_x, D)

        # Concatenate at X's length
        # attended: (B, T_x, 2*D)
        attended = torch.cat([output_x, output_y_to_x], dim=-1)

        if return_attention:
            return attended, (attn_xy, attn_yx)
        else:
            return attended


class SelfAttention(nn.Module):
    """
    Self-attention mechanism
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize self-attention

        Args:
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input (B, T, D)
            mask: Mask (B, T)

        Returns:
            attended: Attended features (B, T, D)
        """
        # Compute self-attention scores
        scores = torch.bmm(x, x.transpose(1, 2))  # (B, T, T)

        if mask is not None:
            # Convert to boolean type for proper masking
            mask_expanded = mask.unsqueeze(1).bool()  # (B, 1, T)
            # Use -1e4 instead of -1e9 for float16 compatibility
            scores = scores.masked_fill(~mask_expanded, -1e4)

        attn = F.softmax(scores, dim=-1)  # (B, T, T)
        attended = torch.bmm(attn, x)  # (B, T, D)

        # Element-wise multiplication
        output = attended * x

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention

        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            query: Query tensor (B, T_q, D)
            key: Key tensor (B, T_k, D)
            value: Value tensor (B, T_v, D)
            mask: Mask (B, T_k)

        Returns:
            output: Attended features (B, T_q, D)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)  # (B, T_q, D)
        K = self.k_linear(key)    # (B, T_k, D)
        V = self.v_linear(value)  # (B, T_v, D)

        # Reshape for multi-head attention
        # (B, T, D) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # scores: (B, num_heads, T_q, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # Expand mask: (B, T_k) -> (B, 1, 1, T_k)
            # Convert to boolean type for proper masking
            mask = mask.unsqueeze(1).unsqueeze(2).bool()
            # Use -1e4 instead of -1e9 for float16 compatibility
            scores = scores.masked_fill(~mask, -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        # output: (B, num_heads, T_q, head_dim)
        output = torch.matmul(attn, V)

        # Concatenate heads
        # (B, num_heads, T_q, head_dim) -> (B, T_q, num_heads, head_dim) -> (B, T_q, D)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # Final linear projection
        output = self.out_linear(output)

        return output
