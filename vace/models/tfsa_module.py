# -*- coding: utf-8 -*-
# Training-Free Self-Attention (TFSA) Module
# Adapted from RepLDM: Training-Free Self-Attention for High-Resolution Image Generation
# Integrated into VACE for video generation enhancement

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TrainingFreeSelfAttention(nn.Module):
    """
    Training-Free Self-Attention (TFSA) as described in RepLDM.

    This parameter-free self-attention mechanism clusters semantically related
    tokens in latent representations, enabling enhanced structural consistency
    without any learnable parameters.

    Formula:
        TFSA(z) = f^(-1)(Softmax(f(z)f(z)^T/λ))f(z)

    Where:
        f: reshape operation that transforms latent to (hw) × c format
        f^(-1): inverse reshape operation
        λ = √c: scaling factor (c is the channel dimension)

    Args:
        embed_dim: Channel dimension of the latent representation
        spatial_only: If True, apply TFSA only to spatial dimensions (ignore temporal)
        temporal_only: If True, apply TFSA only to temporal dimension
    """

    def __init__(
        self,
        embed_dim: int,
        spatial_only: bool = False,
        temporal_only: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial_only = spatial_only
        self.temporal_only = temporal_only
        self.lambda_sqrt = float(embed_dim ** 0.5)

        assert not (spatial_only and temporal_only), \
            "Cannot set both spatial_only and temporal_only to True"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply TFSA to hidden states.

        Args:
            hidden_states: Input tensor of shape [B, C, T, H, W] for 3D video data
                            or [B, C, H, W] for 2D image data

        Returns:
            TFSA-enhanced hidden states with same shape as input
        """
        if hidden_states.dim() == 4:  # [B, C, H, W]
            return self._tfsa_2d(hidden_states)
        elif hidden_states.dim() == 5:  # [B, C, T, H, W]
            return self._tfsa_3d(hidden_states)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {hidden_states.dim()}D")

    def _tfsa_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply TFSA to 2D input [B, C, H, W]"""
        B, C, H, W = x.shape

        # f: reshape to (HW) × C
        f = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]

        # Compute attention without projection matrices
        # attn = softmax(f @ f^T / λ)
        attn = torch.softmax(f @ f.T / self.lambda_sqrt, dim=-1)  # [B*H*W, B*H*W]

        # Apply attention: attn @ f
        output = attn @ f  # [B*H*W, C]

        # f^(-1): reshape back to [B, C, H, W]
        output = output.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return output

    def _tfsa_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TFSA to 3D video input [B, C, T, H, W]

        For video data, we can apply TFSA in different ways:
        - Spatio-temporal: Treat all dimensions as one (default)
        - Spatial-only: Apply TFSA per frame (T independent)
        - Temporal-only: Apply TFSA across time (H*W independent)
        """
        B, C, T, H, W = x.shape

        if self.spatial_only:
            # Apply TFSA per frame (spatial consistency within each frame)
            outputs = []
            for t in range(T):
                frame = x[:, :, t, :, :]  # [B, C, H, W]
                output = self._tfsa_2d(frame)
                outputs.append(output.unsqueeze(2))
            return torch.cat(outputs, dim=2)

        elif self.temporal_only:
            # Apply TFSA across temporal dimension (temporal consistency)
            x = x.permute(0, 3, 4, 1, 2)  # [B, H, W, C, T]
            B, H, W, C, T = x.shape
            f = x.reshape(B * H * W, C * T)  # [B*H*W, C*T]

            attn = torch.softmax(f @ f.T / self.lambda_sqrt, dim=-1)
            output = attn @ f
            output = output.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)
            return output

        else:
            # Apply TFSA spatio-temporally (default)
            # Reshape to merge spatial and temporal: [B, T*H*W, C]
            f = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)  # [B, T*H*W, C]

            attn = torch.softmax(f @ f.T / self.lambda_sqrt, dim=-1)
            output = attn @ f

            output = output.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
            return output


class AdaptiveTFSA(nn.Module):
    """
    Adaptive TFSA with guidance scale modulation.

    This allows dynamic control over how much TFSA influence to apply,
    similar to the adaptive guidance in RepLDM.

    Args:
        embed_dim: Channel dimension
        initial_guidance: Initial guidance scale γ (default: 1.0)
        enabled: If False, TFSA is disabled (passes through)
    """

    def __init__(
        self,
        embed_dim: int,
        initial_guidance: float = 1.0,
        enabled: bool = True
    ):
        super().__init__()
        self.tfsa = TrainingFreeSelfAttention(embed_dim)
        self.guidance_scale = initial_guidance
        self.enabled = enabled

    def set_guidance(self, scale: float):
        """Update guidance scale at runtime."""
        self.guidance_scale = scale

    def enable(self):
        """Enable TFSA."""
        self.enabled = True

    def disable(self):
        """Disable TFSA (passes through original input)."""
        self.enabled = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply TFSA with adaptive guidance.

        Formula: z̃ = γ·TFSA(z) + (1-γ)z

        Where:
            γ: guidance scale
            z: original latent
            TFSA(z): TFSA-enhanced latent
        """
        if not self.enabled:
            return hidden_states

        tfsa_output = self.tfsa(hidden_states)

        # Linear combination with guidance
        gamma = self.guidance_scale
        output = gamma * tfsa_output + (1 - gamma) * hidden_states

        return output


class TFSALayerWrapper(nn.Module):
    """
    Wrapper to inject TFSA into existing attention blocks.

    This wrapper can replace or augment standard attention in DiT models.

    Args:
        original_attention: The original attention module to wrap
        embed_dim: Embedding dimension
        guidance_scale: Initial guidance scale
        mode: 'replace' to replace original attention,
               'augment' to add TFSA as additional parallel branch
    """

    def __init__(
        self,
        original_attention: nn.Module,
        embed_dim: int,
        guidance_scale: float = 1.0,
        mode: str = 'augment'
    ):
        super().__init__()
        self.original = original_attention
        self.tfsa = AdaptiveTFSA(embed_dim, guidance_scale)
        self.mode = mode
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with TFSA augmentation.

        During training, use original attention only.
        During inference, apply TFSA based on mode.
        """
        if self.training:
            return self.original(x, **kwargs)

        if self.mode == 'replace':
            # Replace with TFSA entirely
            return self.tfsa(x)

        elif self.mode == 'augment':
            # Run both and combine results
            original_output = self.original(x, **kwargs)
            tfsa_output = self.tfsa(x)

            # Average the two outputs
            return (original_output + tfsa_output) / 2.0

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def set_guidance(self, scale: float):
        """Update TFSA guidance scale."""
        self.tfsa.set_guidance(scale)

    def enable_tfsa(self):
        """Enable TFSA."""
        self.tfsa.enable()

    def disable_tfsa(self):
        """Disable TFSA."""
        self.tfsa.disable()


def add_tfsa_to_attention_module(
    attention_module: nn.Module,
    guidance_scale: float = 1.0,
    mode: str = 'augment',
    enabled: bool = True
) -> TFSALayerWrapper:
    """
    Utility function to wrap an existing attention module with TFSA.

    Args:
        attention_module: Original attention module (e.g., BasicTransformerBlock)
        guidance_scale: TFSA guidance scale (default: 1.0)
        mode: 'replace' or 'augment' (default: 'augment')
        enabled: Whether TFSA is enabled (default: True)

    Returns:
        Wrapped attention module with TFSA

    Example:
        >>> from ltx_video.models.transformers.attention import BasicTransformerBlock
        >>> original_attn = BasicTransformerBlock(...)
        >>> tfsa_attn = add_tfsa_to_attention_module(original_attn, guidance_scale=0.5)
        >>> # During inference:
        >>> output = tfsa_attn(hidden_states)
    """
    embed_dim = getattr(attention_module, 'embed_dim', None)

    if embed_dim is None:
        # Try to infer from other attributes
        if hasattr(attention_module, 'attention_head_dim'):
            num_heads = getattr(attention_module, 'num_attention_heads', 1)
            head_dim = attention_module.attention_head_dim
            embed_dim = num_heads * head_dim
        else:
            raise ValueError("Cannot infer embed_dim from attention module")

    wrapper = TFSALayerWrapper(
        original_attention=attention_module,
        embed_dim=embed_dim,
        guidance_scale=guidance_scale,
        mode=mode
    )

    if not enabled:
        wrapper.disable_tfsa()

    return wrapper
