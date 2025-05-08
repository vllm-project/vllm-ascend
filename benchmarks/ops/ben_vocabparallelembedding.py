import numpy as np
import torch
import torch_npu
from typing import Optional, Tuple
import pytest
import vllm
import vllm_ascend.platform
def benchmark_npu(fn, num_iterations=100, num_warmup_iterations=50):
    """
    Benchmark function for NPU operations
    
    Args:
        fn: Function to benchmark
        num_iterations: Number of timing iterations
        num_warmup_iterations: Number of warmup iterations
    
    Returns:
        float: Minimum elapsed time in seconds
    """
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(num_iterations + num_warmup_iterations)
    
    # Run iterations
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            fn()  # Execute the function
            end.record()
        torch.npu.synchronize()
        times[i] = start.elapsed_time(end)
    
    # Remove warmup iterations and convert to seconds
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000
    return elapsed_time

'''
def get_masked_input_and_mask_ref(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for verification"""
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
    added_offset = added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    masked_input = vocab_mask * (input_ - valid_offset)
    return masked_input, ~vocab_mask
'''

def get_masked_input_and_mask_ref(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for verification"""
    input_ = input_.to(torch.int64)

    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    org_vocab_mask = org_vocab_mask.to(torch.int64)

    added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
    added_vocab_mask = added_vocab_mask.to(torch.int64)

    added_offset = torch.tensor(added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding,
                              dtype=torch.int64,
                              device=input_.device)

    valid_offset = torch.where(
        org_vocab_mask == 1,
        torch.tensor(org_vocab_start_index, dtype=torch.int64, device=input_.device),
        torch.tensor(0, dtype=torch.int64, device=input_.device)
    )
    valid_offset += torch.where(
        added_vocab_mask == 1,
        added_offset,
        torch.tensor(0, dtype=torch.int64, device=input_.device)
    )

    vocab_mask = (org_vocab_mask | added_vocab_mask).to(torch.int64)

    masked_input = torch.where(
        vocab_mask == 1,
        input_ - valid_offset,
        torch.tensor(0, dtype=torch.int64, device=input_.device)
    )

    masked_input = masked_input.to(input_.dtype)
    return masked_input, ~vocab_mask.to(torch.bool)

DTYPES = [torch.int32]
SHAPES = [(3, 4, 5)]
DEVICES = [f"npu:{0}"]
SEEDS = [0]

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_get_masked_input_and_mask(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> None:
    # Set random seed and device
    torch.manual_seed(seed)
    torch.set_default_device(device)
    
    # Generate random input tensor
    input_tensor = torch.randint(0, 1000, shape, dtype=dtype)
    
    # Test parameters
    test_case = {
        "org_start": 100,
        "org_end": 200,
        "padding": 0,
        "added_start": 300,
        "added_end": 400,
    }
    
    # Define reference function
    def ref_fn():
        return get_masked_input_and_mask_ref(
            input_tensor,
            test_case["org_start"],
            test_case["org_end"],
            test_case["padding"],
            test_case["added_start"],
            test_case["added_end"]
        )
    
    # Define custom function
    def custom_fn():
        return torch.ops._C.get_masked_input_and_mask(
            input_tensor,
            test_case["org_start"],
            test_case["org_end"],
            test_case["padding"],
            test_case["added_start"],
            test_case["added_end"]
        )
    
    # Get results for correctness testing
    ref_masked_input, ref_mask = ref_fn()
    custom_masked_input, custom_mask = custom_fn()
    
    # Benchmark both implementations
    ref_time = benchmark_npu(ref_fn)
    custom_time = benchmark_npu(custom_fn)
    
    # Print performance results
    print(f"\nPerformance Results:")
    print(f"Reference implementation: {ref_time*1000:.3f} ms")
    print(f"Custom implementation: {custom_time*1000:.3f} ms")
    print(f"Speedup: {ref_time/custom_time:.2f}x")
    
    # Compare results for correctness
    ref_masked_input = ref_masked_input.to(dtype)
    print("\nResults comparison:")
    print("custom_masked_input:", custom_masked_input)
    print("ref_masked_input:", ref_masked_input)
    print("custom_mask:", custom_mask)
    print("ref_mask:", ref_mask)
