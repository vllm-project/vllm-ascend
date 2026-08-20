import gc

import numpy as np
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def generate_non_decreasing_sequence(length, upper_limit):
    random_increments = torch.randint(0, 128, (length,))
    sequence = torch.cumsum(random_increments, dim=0)
    if sequence[-1] >= upper_limit:
        scale_factor = upper_limit / sequence[-1]
        sequence = (sequence * scale_factor).to(torch.int64)
    return sequence


@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_single_tensor():
    """W4A8 uses two INT4 values packed into INT8, NZ as INT8, then viewed as INT32."""
    E = 16
    M = 512
    K = 7168
    packed_n = 2048
    logical_n = packed_n * 2
    torch.npu.config.allow_internal_format = True
    x = torch.randint(-5, 5, (M, K), dtype=torch.int8).npu()
    weight_ori = torch.randint(-8, 8, (E, K, packed_n), dtype=torch.int8)
    pack_weight = torch_npu.npu_format_cast(weight_ori.npu().contiguous(), 29).view(torch.int32).contiguous()

    weight_scale = torch.randn(E, logical_n)
    scale_np = weight_scale.cpu().numpy()
    scale_np.dtype = np.uint32
    scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
    pertoken_scale = torch.randn(M).to(torch.float32).npu()
    group_list = generate_non_decreasing_sequence(E, M).npu()
    bias = torch.zeros((E, logical_n), dtype=torch.float32, device="npu").uniform_(-5, 5)

    output, output_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
        x=x,
        weight=[pack_weight],
        weight_assist_matrix=[bias],
        group_list=group_list,
        weight_scale=[scale_uint64_tensor],
        x_scale=pertoken_scale,
        dequant_mode=0,
        group_list_type=0,
    )
    assert tuple(output.shape) == (M, logical_n // 2)
    assert tuple(output_scale.shape) == (M,)
    assert output.dtype == torch.int8
    assert torch.isfinite(output_scale).all()

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
