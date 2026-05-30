import gc
import math

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.utils import enable_custom_op

# enable internal format
torch_npu.npu.config.allow_internal_format = True
# enable vllm-ascend custom ops
enable_custom_op()


def _has_effective_swiglu_limit(swiglu_limit: int | float) -> bool:
    limit = float(swiglu_limit)
    return math.isfinite(limit) and 0.0 < limit < 1_000_000.0


def _shared_dequant_swiglu_quant(
    hidden_states: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    swiglu_limit: int | float,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _has_effective_swiglu_limit(swiglu_limit):
        return torch.ops._C_ascend.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=None,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=1,
            clamp_limit=swiglu_limit,
        )

    if hidden_states.shape[0] == 0:
        output_shape = hidden_states.shape[:-1] + (hidden_states.shape[-1] // 2,)
        return (
            hidden_states.new_empty(output_shape, dtype=torch.int8),
            torch.empty(hidden_states.shape[:-1], dtype=torch.float32, device=hidden_states.device),
        )

    weight_scale = weight_scale.to(torch.float32).reshape((1,) * (hidden_states.dim() - 1) + (-1,))
    activation_scale = activation_scale.to(torch.float32).reshape(hidden_states.shape[:-1] + (1,))
    gate_up = hidden_states.to(torch.float32) * weight_scale * activation_scale

    half = gate_up.shape[-1] // 2
    limit = float(swiglu_limit)
    gate = torch.clamp(gate_up[..., :half], max=limit)
    up = torch.clamp(gate_up[..., half:], min=-limit, max=limit)
    swiglu = F.silu(gate) * up
    if swiglu.dtype not in (torch.float16, torch.bfloat16):
        swiglu = swiglu.to(output_dtype if output_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16)
    return torch_npu.npu_dynamic_quant(swiglu)


@torch.inference_mode()
def test_npu_dequant_swiglu_quant_with_limit():
    swiglu_mode = 1
    x_shape = [4608, 2048]
    x = torch.randint(-10, 10, x_shape, dtype=torch.int32)
    weight_scale = torch.randn(x_shape[1], dtype=torch.float32)
    activate_scale = torch.randn((x_shape[0], 1), dtype=torch.float32)
    clamp_limit = 2.0
    quant_mode = 1

    output_golden, output_scale_golden = _shared_dequant_swiglu_quant(
        x.npu(),
        weight_scale.npu(),
        activate_scale.npu(),
        clamp_limit,
        torch.bfloat16,
    )

    x = x.npu()
    weight_scale = weight_scale.npu()
    activate_scale = activate_scale.npu()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        output, output_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
            x=x,
            weight_scale=weight_scale,
            activation_scale=activate_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=None,
            activate_left=True,
            quant_mode=quant_mode,
            swiglu_mode=swiglu_mode,
            clamp_limit=clamp_limit,
            glu_alpha=1.0,
            glu_bias=0.0,
        )
    graph.replay()

    torch.testing.assert_close(output.cpu(), output_golden.cpu(), atol=1, rtol=0.1)
    torch.testing.assert_close(output_scale.cpu(), output_scale_golden.cpu(), atol=1e-4, rtol=5e-3)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
