import gc
from typing import Tuple

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.platform  # noqa: F401
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# Test parameters
DTYPES = [torch.int32]
#SHAPES = [(100,), (5, 20), (3, 4, 5)]  # Various tensor shapes
#SHAPES = [(3, 4, 8), (3, 4, 5)]  # Various tensor shapes
SHAPES = [(3, 4, 3)]
DEVICES = [f"npu:{0}"]
SEEDS = [0]


def get_masked_input_and_mask_ref(
        input_: torch.Tensor, org_vocab_start_index: int,
        org_vocab_end_index: int, num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for verification"""
    org_vocab_mask = (input_ >= org_vocab_start_index) & (
        input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index)
    added_offset = added_vocab_start_index - (
        org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
    valid_offset = (org_vocab_start_index *
                    org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    masked_input = vocab_mask * (input_ - valid_offset)
    return masked_input, ~vocab_mask


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
    # Set random seed
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

    # Get reference result
    ref_masked_input, ref_mask = get_masked_input_and_mask_ref(
        input_tensor, test_case["org_start"], test_case["org_end"],
        test_case["padding"], test_case["added_start"], test_case["added_end"])

    # Get custom op result
    print("input_tensor:", input_tensor)
    custom_masked_input, custom_mask = torch.ops._C_ascend.get_masked_input_and_mask(
        input_tensor, test_case["org_start"], test_case["org_end"],
        test_case["padding"], test_case["added_start"], test_case["added_end"])

    ref_masked_input = ref_masked_input.to(dtype)
    print("custom_masked_input:", custom_masked_input)
    print("ref_masked_input:", ref_masked_input)
    print("custom_mask:", custom_mask)
    print("ref_mask:", ref_mask)
    # Compare results
    torch.testing.assert_close(
        custom_masked_input,
        ref_masked_input,
        rtol=1e-5,
        atol=1e-5,
        msg=f"Masked input mismatch for case: {test_case}")
    torch.testing.assert_close(custom_mask,
                               ref_mask,
                               rtol=1e-5,
                               atol=1e-5,
                               msg=f"Mask mismatch for case: {test_case}")
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
