# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Modified to add TFSA (Training-Free Self-Attention) support

import torch
from torch import nn

from diffusers.utils.torch_utils import maybe_allow_in_graph

from ltx_video.models.transformers.attention import BasicTransformerBlock

# Import TFSA module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tfsa_module import add_tfsa_to_attention_module, TFSALayerWrapper


@maybe_allow_in_graph
class BasicTransformerMainBlock(BasicTransformerBlock):
    def __init__(self, *args, **kwargs):
        self.block_id = kwargs.pop('block_id')
        # TFSA configuration
        self.use_tfsa = kwargs.pop('use_tfsa', False)
        self.tfsa_guidance = kwargs.pop('tfsa_guidance', 1.0)
        self.tfsa_mode = kwargs.pop('tfsa_mode', 'augment')  # 'replace' or 'augment'

        super().__init__(*args, **kwargs)

        # Wrap with TFSA if enabled
        if self.use_tfsa:
            self._original_forward = super().forward
            # Note: We'll apply TFSA in forward() instead of wrapping here
            # to preserve the original module structure

    def forward(self, *args, **kwargs) -> torch.FloatTensor:
        context_hints = kwargs.pop('context_hints')
        context_scale = kwargs.pop('context_scale')

        # Apply original attention
        hidden_states = super().forward(*args, **kwargs)

        # Apply TFSA during inference if enabled
        if self.use_tfsa and not self.training:
            # Import TFSA here to avoid circular import
            from tfsa_module import AdaptiveTFSA
            tfsa = AdaptiveTFSA(
                embed_dim=self.attention_head_dim * self.num_attention_heads,
                enabled=True
            )
            tfsa.set_guidance(self.tfsa_guidance)

            # Apply TFSA and blend
            tfsa_output = tfsa(hidden_states)
            if self.tfsa_mode == 'replace':
                hidden_states = tfsa_output
            else:  # augment
                # Blend original and TFSA outputs
                gamma = self.tfsa_guidance
                hidden_states = gamma * tfsa_output + (1 - gamma) * hidden_states

        # Apply context hints (VACE original functionality)
        if self.block_id < len(context_hints) and context_hints[self.block_id] is not None:
            hidden_states = hidden_states + context_hints[self.block_id] * context_scale

        return hidden_states

    def set_tfsa_guidance(self, scale: float):
        """Update TFSA guidance scale at runtime."""
        self.tfsa_guidance = scale

    def enable_tfsa(self):
        """Enable TFSA for this block."""
        self.use_tfsa = True

    def disable_tfsa(self):
        """Disable TFSA for this block."""
        self.use_tfsa = False


@maybe_allow_in_graph
class BasicTransformerBypassBlock(BasicTransformerBlock):
    def __init__(self, *args, **kwargs):
        self.dim = args[0]
        self.block_id = kwargs.pop('block_id')
        # TFSA configuration
        self.use_tfsa = kwargs.pop('use_tfsa', False)
        self.tfsa_guidance = kwargs.pop('tfsa_guidance', 1.0)
        self.tfsa_mode = kwargs.pop('tfsa_mode', 'augment')

        super().__init__(*args, **kwargs)
        if self.block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, *args, **kwargs):
        hidden_states = kwargs.pop('hidden_states')
        context_hidden_states = kwargs.pop('context_hidden_states')

        if self.block_id == 0:
            context_hidden_states = self.before_proj(context_hidden_states) + hidden_states

        kwargs['hidden_states'] = context_hidden_states
        bypass_context_hidden_states = super().forward(*args, **kwargs)

        # Apply TFSA during inference if enabled
        if self.use_tfsa and not self.training:
            from tfsa_module import AdaptiveTFSA
            tfsa = AdaptiveTFSA(
                embed_dim=self.dim,
                enabled=True
            )
            tfsa.set_guidance(self.tfsa_guidance)

            tfsa_output = tfsa(bypass_context_hidden_states)
            if self.tfsa_mode == 'replace':
                bypass_context_hidden_states = tfsa_output
            else:  # augment
                gamma = self.tfsa_guidance
                bypass_context_hidden_states = gamma * tfsa_output + (1 - gamma) * bypass_context_hidden_states

        main_context_hidden_states = self.after_proj(bypass_context_hidden_states)
        return (main_context_hidden_states, bypass_context_hidden_states)

    def set_tfsa_guidance(self, scale: float):
        """Update TFSA guidance scale at runtime."""
        self.tfsa_guidance = scale

    def enable_tfsa(self):
        """Enable TFSA for this block."""
        self.use_tfsa = True

    def disable_tfsa(self):
        """Disable TFSA for this block."""
        self.use_tfsa = False
