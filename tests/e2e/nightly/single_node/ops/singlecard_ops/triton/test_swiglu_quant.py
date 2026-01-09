import gc

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.activation import swiglu_quant
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


DTYPES = [torch.bfloat16, torch.float16]
NUM_TOKENS = [1, 4, 8, 16, 1024]
HIDDEN_SIZES = [12800, 36864]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3



def _swiglu_quant_pytorch_native(
        x) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch-native implementation equivalent to forward()."""
    x = torch_npu.npu_swiglu(x)
    x, swiglu_out_scale = torch_npu.npu_dynamic_quant(
        x)
    return x, swiglu_out_scale


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_swiglu_quant_triton_kernel(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    x_trt.copy_(x)
    x_trt, scale_trt = swiglu_quant(x_trt, need_quant=True)
    x_gold, scale_gold = _swiglu_quant_pytorch_native(x_gold)
    # Compare the results.
    torch.testing.assert_close(x_trt.view(x_gold.size()),
                               x_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    torch.testing.assert_close(scale_trt.view(scale_gold.size()),
                               scale_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
