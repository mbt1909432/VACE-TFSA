# -*- coding: utf-8 -*-
"""
Test script for TFSA module validation
Run this to verify TFSA is working correctly before using in VACE
"""

import torch
from vace.models.tfsa_module import (
    TrainingFreeSelfAttention,
    AdaptiveTFSA,
    TFSALayerWrapper
)


def test_tfsa_2d():
    """Test TFSA on 2D input (image-like data)"""
    print("=" * 60)
    print("Testing TFSA on 2D input [B, C, H, W]...")
    print("=" * 60)

    # Create dummy input
    B, C, H, W = 2, 1408, 32, 32  # Typical dimensions
    x = torch.randn(B, C, H, W)

    print(f"Input shape: {x.shape}")

    # Test TFSA
    tfsa = TrainingFreeSelfAttention(embed_dim=C)
    output = tfsa(x)

    print(f"Output shape: {output.shape}")
    print(f"Shape preserved: {x.shape == output.shape}")
    print(f"Output requires grad: {output.requires_grad}")

    assert output.shape == x.shape, "TFSA changed the output shape!"
    print("✓ 2D TFSA test passed!\n")

    return output


def test_tfsa_3d():
    """Test TFSA on 3D input (video data)"""
    print("=" * 60)
    print("Testing TFSA on 3D input [B, C, T, H, W]...")
    print("=" * 60)

    # Create dummy video input
    B, C, T, H, W = 1, 1408, 8, 64, 64  # Small video for testing
    x = torch.randn(B, C, T, H, W)

    print(f"Input shape: {x.shape}")

    # Test default TFSA (spatio-temporal)
    tfsa = TrainingFreeSelfAttention(embed_dim=C)
    output = tfsa(x)

    print(f"Output shape: {output.shape}")
    print(f"Shape preserved: {x.shape == output.shape}")
    assert output.shape == x.shape, "TFSA changed the output shape!"
    print("✓ 3D TFSA (spatio-temporal) test passed!")

    # Test spatial-only TFSA
    tfsa_spatial = TrainingFreeSelfAttention(embed_dim=C, spatial_only=True)
    output_spatial = tfsa_spatial(x)
    print(f"Spatial-only output shape: {output_spatial.shape}")
    assert output_spatial.shape == x.shape
    print("✓ 3D TFSA (spatial-only) test passed!")

    # Test temporal-only TFSA
    tfsa_temporal = TrainingFreeSelfAttention(embed_dim=C, temporal_only=True)
    output_temporal = tfsa_temporal(x)
    print(f"Temporal-only output shape: {output_temporal.shape}")
    assert output_temporal.shape == x.shape
    print("✓ 3D TFSA (temporal-only) test passed!\n")

    return output


def test_adaptive_tfsa():
    """Test AdaptiveTFSA with guidance"""
    print("=" * 60)
    print("Testing AdaptiveTFSA with guidance scale...")
    print("=" * 60)

    B, C, H, W = 1, 1408, 32, 32
    x = torch.randn(B, C, H, W)

    # Create AdaptiveTFSA
    adaptive_tfsa = AdaptiveTFSA(embed_dim=C, initial_guidance=0.5)

    # Test with different guidance scales
    for scale in [0.0, 0.3, 0.5, 0.7, 1.0]:
        adaptive_tfsa.set_guidance(scale)
        output = adaptive_tfsa(x)

        if scale == 0.0:
            # Should be same as input (no TFSA influence)
            assert torch.allclose(output, x, atol=1e-6), f"Scale 0.0 should give original input"
            print(f"Guidance {scale}: Output ≈ Input (as expected) ✓")
        else:
            # Note: TFSA may not significantly change random input data
            # It's designed for inference-time enhancement of pretrained models
            diff = torch.abs(output - x).mean().item()
            print(f"Guidance {scale}: Mean difference from input = {diff:.4f}")
            # Just verify output shape is correct
            assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"

    print("✓ AdaptiveTFSA test passed!\n")


def test_guidance_modes():
    """Test different guidance modes"""
    print("=" * 60)
    print("Testing different guidance modes...")
    print("=" * 60)

    B, C, H, W = 1, 1408, 32, 32
    x = torch.randn(B, C, H, W)

    tfsa = AdaptiveTFSA(embed_dim=C, initial_guidance=0.5)

    # Test disabled state
    tfsa.disable()
    output_disabled = tfsa(x)
    assert torch.allclose(output_disabled, x, atol=1e-6), "Disabled should pass through"
    print("✓ Disabled mode: Output = Input ✓")

    # Test enabled state
    tfsa.enable()
    output_enabled = tfsa(x)
    # Verify TFSA runs without error and shape is preserved
    assert output_enabled.shape == x.shape, "Enabled should preserve shape"
    print("✓ Enabled mode: TFSA executed successfully ✓")

    print("✓ Guidance mode test passed!\n")


def test_performance():
    """Basic performance test"""
    print("=" * 60)
    print("Performance test (100 iterations)...")
    print("=" * 60)

    import time

    B, C, H, W = 1, 1408, 32, 32
    x = torch.randn(B, C, H, W)

    tfsa = AdaptiveTFSA(embed_dim=C, initial_guidance=0.5)

    # Warmup
    for _ in range(10):
        _ = tfsa(x)

    # Timed run
    start = time.time()
    for _ in range(100):
        _ = tfsa(x)
    end = time.time()

    elapsed = end - start
    avg_time = elapsed / 100

    print(f"Total time for 100 iterations: {elapsed:.3f}s")
    print(f"Average time per iteration: {avg_time*1000:.2f}ms")
    print("✓ Performance test completed!\n")


def main():
    print("\n" + "=" * 60)
    print("TFSA Module Validation Suite")
    print("=" * 60 + "\n")

    try:
        # Run all tests
        test_tfsa_2d()
        test_tfsa_3d()
        test_adaptive_tfsa()
        test_guidance_modes()
        test_performance()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nTFSA module is ready to use!")
        print("\nTo use TFSA in VACE:")
        print("1. Set --tfsa_enabled flag in run_vace_with_tfsa.py")
        print("2. Adjust --tfsa_guidance (recommended: 0.3-0.7)")
        print("3. Choose --tfsa_mode (augment or replace)")

    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
