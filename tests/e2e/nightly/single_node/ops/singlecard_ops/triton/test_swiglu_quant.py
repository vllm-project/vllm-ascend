import contextlib
import gc
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import build_mlp_compute_input
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

DEVICE = "npu"
INT8_MAX = 127
TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}
SCALE_TOLERANCES = {
    torch.float16: (5e-3, 1e-5),
    torch.bfloat16: (2e-2, 1e-4),
}


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    init_device_properties_triton()


def _swiglu_reference(x: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.cpu().float()
    gate, up = x_fp32.chunk(2, dim=-1)
    return F.silu(gate) * up


def _dynamic_quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax(dim=-1) / INT8_MAX
    quantized = torch.round(x / scale.unsqueeze(-1)).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return quantized, scale


def _assert_quantized_result(x, actual, actual_scale):
    swiglu_ref = _swiglu_reference(x)
    expected, expected_scale = _dynamic_quant_reference(swiglu_ref)
    scale_rtol, scale_atol = SCALE_TOLERANCES[x.dtype]

    assert actual.shape == swiglu_ref.shape
    assert actual.dtype == torch.int8
    assert actual_scale.shape == (x.shape[0],)
    assert actual_scale.dtype == torch.float32
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=scale_rtol, atol=scale_atol)
    # The kernel casts through the input dtype before converting to int8.
    # Permit one quantization bin of error relative to the FP32 reference.
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=0, atol=1)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize(
    ("shape", "dtype", "group_counts", "group_list_dtype"),
    [
        pytest.param((1, 4096), torch.float16, [1], torch.int32, id="single-token-fp16"),
        pytest.param(
            (65, 14336),
            torch.bfloat16,
            [0, 17, 1, 0, 31, 16],
            torch.int64,
            id="multi-core-bf16-empty-experts",
        ),
    ],
)
@pytest.mark.parametrize("group_list_type", [0, 1], ids=["cumsum", "count"])
@torch.inference_mode()
def test_swiglu_quant_correctness(shape, dtype, group_counts, group_list_dtype, group_list_type):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    group_list = torch.tensor(group_counts, dtype=group_list_dtype, device=DEVICE)
    if group_list_type == 0:
        group_list = group_list.cumsum(dim=0)

    actual, actual_scale = swiglu_quant(x, group_list, group_list_type=group_list_type)
    _assert_quantized_result(x, actual, actual_scale)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize(
    ("shape", "group_counts"),
    [
        pytest.param((2, 256), [1, 1] + [0] * 6, id="moe-small"),
        pytest.param((384, 4096), [6] * 64, id="moe-large"),
        pytest.param((198, 4096), [25] * 6 + [24] * 2, id="moe-irregular-tokens"),
        pytest.param((2, 2048), [1, 1] + [0] * 62, id="moe-empty-experts"),
    ],
)
@torch.inference_mode()
def test_swiglu_quant_moe_routed_shapes(shape, group_counts):
    # Extracted from the W8A8 AllGather cases: GMM1 produces
    # [tokens * top_k, 2 * intermediate_size], with INT64 expert counts.
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    group_list = torch.tensor(group_counts, dtype=torch.int64, device=DEVICE)

    actual, actual_scale = swiglu_quant(x, group_list, group_list_type=1)
    _assert_quantized_result(x, actual, actual_scale)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_swiglu_quant_without_quantization(dtype):
    torch.manual_seed(42)
    x = torch.randn((9, 4096), dtype=dtype, device=DEVICE)
    group_list = torch.tensor([2, 0, 7], dtype=torch.int64, device=DEVICE)

    actual, _ = swiglu_quant(x, group_list, group_list_type=1, need_quant=False)
    expected = _swiglu_reference(x)
    rtol, atol = TOLERANCES[dtype]

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
def test_swiglu_quant_rejects_invalid_group_list_type():
    x = torch.empty((1, 4096), dtype=torch.float16, device=DEVICE)
    group_list = torch.tensor([1], dtype=torch.int32, device=DEVICE)

    with pytest.raises(ValueError, match="group_list_type must be 0 or 1"):
        swiglu_quant(x, group_list, group_list_type=2)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
def test_swiglu_quant_rejects_invalid_group_list_dtype():
    x = torch.empty((1, 4096), dtype=torch.float16, device=DEVICE)
    group_list = torch.tensor([1], dtype=torch.float32, device=DEVICE)

    with pytest.raises(ValueError, match="group_list dtype must be torch.int32 or torch.int64"):
        swiglu_quant(x, group_list, group_list_type=1)


# Caller-contract regression for the W8A8 AllGather activation path.
@contextlib.contextmanager
def moe_comm_type(comm_type: MoECommType):
    """Stub the MoE comm type that `apply_moe_mlp` reads off the forward context.

    The quant MLP path branches on ``_EXTRA_CTX.moe_comm_type``, which resolves
    through vLLM's `get_forward_context()`. That context only exists inside a
    model forward, so an op-level test has to supply the one attribute the
    kernel path actually consumes.
    """
    with patch(
        "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic._EXTRA_CTX",
        SimpleNamespace(moe_comm_type=comm_type),
    ):
        yield


@pytest.mark.parametrize(
    "m, n, k, e, topk",
    [
        pytest.param(1, 128, 128, 8, 2, id="typical-small"),
        pytest.param(64, 2048, 1024, 64, 6, id="typical-large"),
        pytest.param(33, 2048, 511, 8, 6, id="regression-irregular-tokens-nonaligned-K-wide-activation"),
        pytest.param(1, 1024, 511, 64, 2, id="regression-mid-N-nonaligned-K-empty-experts"),
    ],
)
@pytest.mark.parametrize("ep_size", [1])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="BF16 W8A8 AllGather fallback UT is limited to A2/A3/A5",
)
def test_token_dispatcher_with_all_gather_quant(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    device: str,
):
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1 = torch.randint(-8, 8, (e, k, 2 * n), device=device, dtype=torch.int8)
    w1_scale = torch.full((e, 2 * n), 1 / 32, device=device, dtype=dtype)
    w2 = torch.randint(-8, 8, (e, n, k), device=device, dtype=torch.int8)
    w2_scale = torch.full((e, k), 1 / 32, device=device, dtype=dtype)

    score = torch.randn((m, e), device=device, dtype=dtype)
    expert_map = None
    local_e = e

    score = torch.softmax(score, dim=-1, dtype=dtype)
    topk_weights, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    dispatcher_kwargs = {
        "num_experts": e,
        "top_k": topk,
        "num_local_experts": local_e,
    }
    dispatcher = TokenDispatcherWithAllGather(**dispatcher_kwargs)

    apply_router_weight_on_input = False
    token_dispatch_output = dispatcher.token_dispatch(
        token_dispatch_input=MoETokenDispatchInput(
            hidden_states=a,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            routing=MoeRouterInput(
                expert_map=expert_map,
                global_redundant_expert_num=0,
                mc2_mask=None,
                apply_router_weight_on_input=apply_router_weight_on_input,
            ),
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
        )
    )

    combine_metadata = token_dispatch_output.combine_metadata

    layer = SimpleNamespace(
        w13_weight=w1,
        w13_weight_scale_fp32=w1_scale,
        w2_weight=w2,
        w2_weight_scale=w2_scale,
        activation="silu",
    )
    mlp_compute_input = build_mlp_compute_input(
        fused_experts_input=build_fused_experts_input(
            hidden_states=a,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            layer=layer,
            quant_type=QuantType.W8A8,
            dynamic_eplb=False,
            expert_map=expert_map,
        ),
        token_dispatch_output=token_dispatch_output,
        use_fusion_ops=False,
    )
    quant_method = AscendW8A8DynamicFusedMoEMethod.__new__(AscendW8A8DynamicFusedMoEMethod)
    quant_method.use_expert_weight_list = False
    assert mlp_compute_input.fusion is False

    def check_swiglu_input(hidden_states, group_list, group_list_type, need_quant=True):
        assert hidden_states.shape == (m * topk, 2 * n)
        assert hidden_states.dtype == dtype
        assert hidden_states.is_contiguous()
        assert group_list.shape == (e,)
        assert group_list.dtype == torch.int64
        assert group_list_type == 1
        assert need_quant is True
        assert (group_list >= 0).all().item()
        assert group_list.sum().item() == hidden_states.shape[0]
        result = swiglu_quant(hidden_states, group_list, group_list_type, need_quant=need_quant)
        _assert_quantized_result(hidden_states, *result)
        return result

    # Exercise the real local kernel and verify the MLP hands it valid data.
    with patch(
        "vllm_ascend.ops.triton.activation.swiglu_quant.swiglu_quant",
        side_effect=check_swiglu_input,
    ) as activation:
        with moe_comm_type(MoECommType.ALLGATHER):
            expert_output, _ = apply_moe_mlp(mlp_compute_input, quant_method)
        activation.assert_called_once()
    combined_output = dispatcher.token_combine(
        hidden_states=expert_output, combine_metadata=combine_metadata, bias=None
    )
    assert combined_output.shape == (m, k)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
