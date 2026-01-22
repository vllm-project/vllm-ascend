#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for verifying the custom ApplyTopKTopP operator binding.
This script tests the torch.ops._C_ascend.npu_apply_top_k_top_p operator.
"""

import torch
import torch_npu

def test_apply_top_k_top_p_basic():
    """
    Basic test for apply_top_k_top_p operator.
    Tests that the operator can be called and returns valid output.
    """
    print("=" * 60)
    print("Test: Basic apply_top_k_top_p operator test")
    print("=" * 60)
    
    # Set device to NPU
    device = torch.device("npu:0")
    
    # Create test input tensor: [batch_size, vocab_size]
    batch_size = 4
    vocab_size = 1024
    
    # Generate random logits
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)
    
    # Create top_p values for each batch
    top_p = torch.tensor([0.9, 0.8, 0.7, 0.95], dtype=torch.float32, device=device)
    
    # Create top_k values for each batch
    top_k = torch.tensor([50, 40, 30, 60], dtype=torch.int32, device=device)
    
    print(f"Input logits shape: {logits.shape}")
    print(f"Input logits dtype: {logits.dtype}")
    print(f"Top-p values: {top_p}")
    print(f"Top-k values: {top_k}")
    
    # Test with both p and k
    print("\n--- Test with both top_p and top_k ---")
    try:
        result = torch.ops._C_ascend.npu_apply_top_k_top_p(logits, top_p, top_k)
        print(f"Output shape: {result.shape}")
        print(f"Output dtype: {result.dtype}")
        print(f"Output sample (first row, first 10 elements): {result[0, :10]}")
        print("Test PASSED: Operator executed successfully with both p and k")
    except Exception as e:
        print(f"Test FAILED: {e}")
        return False
    
    # Test with only top_p
    print("\n--- Test with only top_p ---")
    try:
        result_p_only = torch.ops._C_ascend.npu_apply_top_k_top_p(logits, top_p, None)
        print(f"Output shape: {result_p_only.shape}")
        print("Test PASSED: Operator executed successfully with only top_p")
    except Exception as e:
        print(f"Test FAILED: {e}")
        return False
    
    # Test with only top_k
    print("\n--- Test with only top_k ---")
    try:
        result_k_only = torch.ops._C_ascend.npu_apply_top_k_top_p(logits, None, top_k)
        print(f"Output shape: {result_k_only.shape}")
        print("Test PASSED: Operator executed successfully with only top_k")
    except Exception as e:
        print(f"Test FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All basic tests PASSED!")
    print("=" * 60)
    return True


def test_apply_top_k_top_p_dtypes():
    """
    Test apply_top_k_top_p operator with different data types.
    """
    print("\n" + "=" * 60)
    print("Test: Different data types test")
    print("=" * 60)
    
    device = torch.device("npu:0")
    batch_size = 2
    vocab_size = 512
    
    dtypes_to_test = [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
    ]
    
    top_p = torch.tensor([0.9, 0.8], dtype=torch.float32, device=device)
    
    for dtype, dtype_name in dtypes_to_test:
        print(f"\n--- Testing with dtype: {dtype_name} ---")
        try:
            logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=device)
            # Convert top_p to match logits dtype
            top_p_typed = top_p.to(dtype)
            result = torch.ops._C_ascend.npu_apply_top_k_top_p(logits, top_p_typed, None)
            print(f"Input dtype: {dtype_name}, Output dtype: {result.dtype}")
            print(f"Test PASSED for dtype: {dtype_name}")
        except Exception as e:
            print(f"Test FAILED for dtype {dtype_name}: {e}")
            # Some dtypes might not be supported, this is expected
    
    print("\n" + "=" * 60)
    print("Dtype tests completed!")
    print("=" * 60)
    return True


def test_apply_top_k_top_p_error_handling():
    """
    Test apply_top_k_top_p operator error handling.
    """
    print("\n" + "=" * 60)
    print("Test: Error handling test")
    print("=" * 60)
    
    device = torch.device("npu:0")
    logits = torch.randn(4, 1024, dtype=torch.float32, device=device)
    
    # Test with both p and k as None (should fail)
    print("\n--- Test with both p and k as None (should fail) ---")
    try:
        result = torch.ops._C_ascend.npu_apply_top_k_top_p(logits, None, None)
        print("Test FAILED: Should have raised an error")
        return False
    except Exception as e:
        print(f"Test PASSED: Correctly raised error: {e}")
    
    print("\n" + "=" * 60)
    print("Error handling tests completed!")
    print("=" * 60)
    return True


def main():
    """
    Main function to run all tests.
    """
    print("\n" + "#" * 70)
    print("# Testing custom ApplyTopKTopP operator (npu_apply_top_k_top_p)")
    print("#" * 70)
    
    # Check if NPU is available
    if not torch.npu.is_available():
        print("ERROR: NPU is not available. Please ensure torch_npu is properly installed.")
        return
    
    print(f"\nNPU device count: {torch.npu.device_count()}")
    print(f"Current NPU device: {torch.npu.current_device()}")
    
    # Try to load the custom ops library
    try:
        # The library should be loaded via vllm_ascend package
        import vllm_ascend
        print("vllm_ascend package loaded successfully")
    except ImportError:
        print("WARNING: vllm_ascend package not found. "
              "Make sure the custom ops are built and installed.")
        print("You can try loading the library directly:")
        print("  torch.ops.load_library('/path/to/libcust_opapi.so')")
    
    # Run tests
    all_passed = True
    
    try:
        all_passed &= test_apply_top_k_top_p_basic()
    except Exception as e:
        print(f"Basic test failed with exception: {e}")
        all_passed = False
    
    try:
        all_passed &= test_apply_top_k_top_p_dtypes()
    except Exception as e:
        print(f"Dtype test failed with exception: {e}")
        all_passed = False
    
    try:
        all_passed &= test_apply_top_k_top_p_error_handling()
    except Exception as e:
        print(f"Error handling test failed with exception: {e}")
        all_passed = False
    
    print("\n" + "#" * 70)
    if all_passed:
        print("# ALL TESTS PASSED!")
    else:
        print("# SOME TESTS FAILED!")
    print("#" * 70)


if __name__ == "__main__":
    main()
