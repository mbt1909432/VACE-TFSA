# -*- coding: utf-8 -*-
"""
VACE with TFSA (Training-Free Self-Attention) Inference Script

This script demonstrates how to use VACE with the TFSA enhancement from RepLDM.
TFSA improves structural consistency in generated videos without requiring additional training.

Usage:
    python run_vace_with_tfsa.py \
        --prompt "A cat playing with a ball" \
        --tfsa_enabled \
        --tfsa_guidance 0.5 \
        --output output_video.mp4
"""

import argparse
import torch
from pathlib import Path

# Import VACE
from vace.models.ltx.ltx_vace import LTXVace


class VACEWithTFSA(LTXVace):
    """
    Extended VACE class with TFSA (Training-Free Self-Attention) support.

    This class adds TFSA capabilities to the original VACE implementation,
    allowing for improved structural consistency in generated videos.
    """

    def __init__(
        self,
        ckpt_path,
        text_encoder_path,
        precision='bfloat16',
        stg_skip_layers="19",
        stg_mode="stg_a",
        offload_to_cpu=False,
        # TFSA specific parameters
        tfsa_enabled=False,
        tfsa_guidance=1.0,
        tfsa_mode='augment',  # 'replace' or 'augment'
        tfsa_spatial_only=False,
        tfsa_temporal_only=False
    ):
        # Initialize parent VACE class
        super().__init__(
            ckpt_path=ckpt_path,
            text_encoder_path=text_encoder_path,
            precision=precision,
            stg_skip_layers=stg_skip_layers,
            stg_mode=stg_mode,
            offload_to_cpu=offload_to_cpu
        )

        # TFSA configuration
        self.tfsa_enabled = tfsa_enabled
        self.tfsa_guidance = tfsa_guidance
        self.tfsa_mode = tfsa_mode
        self.tfsa_spatial_only = tfsa_spatial_only
        self.tfsa_temporal_only = tfsa_temporal_only

        # If TFSA is enabled, modify the pipeline
        if self.tfsa_enabled:
            self._patch_pipeline_with_tfsa()

    def _patch_pipeline_with_tfsa(self):
        """
        Patch the VACE pipeline to use TFSA-enhanced attention blocks.

        This modifies the attention blocks in the transformer to include TFSA.
        """
        # Import TFSA-enabled attention blocks
        from vace.models.ltx.models.transformers.attention_tfsa import (
            BasicTransformerMainBlock,
            BasicTransformerBypassBlock
        )

        # Replace attention blocks in the transformer
        transformer = self.pipeline.transformer
        if hasattr(transformer, 'transformers'):
            for i, block in enumerate(transformer.transformers):
                # Create new block with TFSA enabled
                block_config = {
                    'block_id': i,
                    'use_tfsa': True,
                    'tfsa_guidance': self.tfsa_guidance,
                    'tfsa_mode': self.tfsa_mode
                }

                # Re-create block with TFSA support
                # Note: This requires the block's original configuration
                # For now, we'll store the TFSA config to be applied during forward pass
                block.use_tfsa = True
                block.tfsa_guidance = self.tfsa_guidance
                block.tfsa_mode = self.tfsa_mode

    def set_tfsa_guidance(self, scale: float):
        """
        Update TFSA guidance scale at runtime.

        Args:
            scale: Guidance scale γ (default 1.0)
                   - Higher values: More TFSA influence
                   - Lower values: Less TFSA influence
                   - 0.0: Disable TFSA (equivalent to original VACE)
        """
        self.tfsa_guidance = scale

        # Update all transformer blocks
        transformer = self.pipeline.transformer
        if hasattr(transformer, 'transformers'):
            for block in transformer.transformers:
                if hasattr(block, 'tfsa_guidance'):
                    block.tfsa_guidance = scale

    def enable_tfsa(self):
        """Enable TFSA enhancement."""
        self.tfsa_enabled = True

        transformer = self.pipeline.transformer
        if hasattr(transformer, 'transformers'):
            for block in transformer.transformers:
                if hasattr(block, 'use_tfsa'):
                    block.use_tfsa = True

    def disable_tfsa(self):
        """Disable TFSA enhancement (revert to original VACE)."""
        self.tfsa_enabled = False

        transformer = self.pipeline.transformer
        if hasattr(transformer, 'transformers'):
            for block in transformer.transformers:
                if hasattr(block, 'use_tfsa'):
                    block.use_tfsa = False

    def generate(
        self,
        src_video=None,
        src_mask=None,
        src_ref_images=[],
        prompt="",
        negative_prompt="",
        seed=42,
        num_inference_steps=40,
        num_images_per_prompt=1,
        context_scale=1.0,
        guidance_scale=3,
        stg_scale=1,
        stg_rescale=0.7,
        frame_rate=25,
        image_cond_noise_scale=0.15,
        decode_timestep=0.05,
        decode_noise_scale=0.025,
        output_height=512,
        output_width=768,
        num_frames=97
    ):
        """
        Generate video with optional TFSA enhancement.

        Args:
            (Same as VACE original parameters)

        Returns:
            dict: Generated video and metadata
        """
        # Call parent generate method
        result = super().generate(
            src_video=src_video,
            src_mask=src_mask,
            src_ref_images=src_ref_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            context_scale=context_scale,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            stg_rescale=stg_rescale,
            frame_rate=frame_rate,
            image_cond_noise_scale=image_cond_noise_scale,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_height=output_height,
            output_width=output_width,
            num_frames=num_frames
        )

        # Add TFSA info to result
        result['tfsa_enabled'] = self.tfsa_enabled
        result['tfsa_guidance'] = self.tfsa_guidance
        result['tfsa_mode'] = self.tfsa_mode

        return result


def main():
    parser = argparse.ArgumentParser(
        description='VACE with TFSA (Training-Free Self-Attention)'
    )

    # VACE parameters
    parser.add_argument('--ckpt_path', type=str,
                        default='models/ltx-vace',
                        help='Path to VACE checkpoint')
    parser.add_argument('--text_encoder_path', type=str,
                        default='models/LTX-Video-2B',
                        help='Path to text encoder')
    parser.add_argument('--precision', type=str, default='bfloat16',
                        choices=['bfloat16', 'float32', 'mixed_precision'],
                        help='Model precision')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for video generation')
    parser.add_argument('--negative_prompt', type=str, default='',
                        help='Negative text prompt')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Output video path')
    parser.add_argument('--num_inference_steps', type=int, default=40,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--output_height', type=int, default=512,
                        help='Output video height')
    parser.add_argument('--output_width', type=int, default=768,
                        help='Output video width')
    parser.add_argument('--num_frames', type=int, default=97,
                        help='Number of frames to generate')

    # TFSA specific parameters
    parser.add_argument('--tfsa_enabled', action='store_true',
                        help='Enable TFSA (Training-Free Self-Attention)')
    parser.add_argument('--tfsa_guidance', type=float, default=0.5,
                        help='TFSA guidance scale (default: 0.5)')
    parser.add_argument('--tfsa_mode', type=str, default='augment',
                        choices=['augment', 'replace'],
                        help='TFSA mode: augment (blend with original) or replace (use TFSA only)')
    parser.add_argument('--tfsa_spatial_only', action='store_true',
                        help='Apply TFSA only to spatial dimensions')
    parser.add_argument('--tfsa_temporal_only', action='store_true',
                        help='Apply TFSA only to temporal dimension')

    args = parser.parse_args()

    print("=" * 60)
    print("VACE with TFSA (Training-Free Self-Attention)")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"TFSA Enabled: {args.tfsa_enabled}")
    print(f"TFSA Guidance: {args.tfsa_guidance}")
    print(f"TFSA Mode: {args.tfsa_mode}")
    print("=" * 60)

    # Initialize VACE with TFSA
    vace_tfsa = VACEWithTFSA(
        ckpt_path=args.ckpt_path,
        text_encoder_path=args.text_encoder_path,
        precision=args.precision,
        tfsa_enabled=args.tfsa_enabled,
        tfsa_guidance=args.tfsa_guidance,
        tfsa_mode=args.tfsa_mode,
        tfsa_spatial_only=args.tfsa_spatial_only,
        tfsa_temporal_only=args.tfsa_temporal_only
    )

    # Generate video
    print("Generating video...")
    result = vace_tfsa.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_height=args.output_height,
        output_width=args.output_width,
        num_frames=args.num_frames
    )

    # Save output
    print(f"Saving video to {args.output}...")
    import torchvision.io as io
    io.write_video(args.output, result['out_video'], fps=25)

    print("Done!")
    print(f"Output saved to: {args.output}")
    print(f"TFSA was used: {result['tfsa_enabled']}")
    print(f"TFSA guidance: {result['tfsa_guidance']}")


if __name__ == "__main__":
    main()
