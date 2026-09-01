from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ops.fused_moe.mega_moe import MegaMoEBackend, _view_mxfp_scales_as_e8m0
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithMegaMoE
from vllm_ascend.quantization.quant_type import QuantType


class _TestMoEActivation(Enum):
    SILU = 1
    SWIGLUOAI = 2
    SWIGLUSTEP = 3


def _make_moe_config():
    moe_config = MagicMock(spec=FusedMoEConfig)
    moe_config.num_experts = 8
    moe_config.experts_per_token = 2
    return moe_config


def _make_ascend_config(mega_moe_max_tokens=16, max_num_batched_tokens=8192):
    return SimpleNamespace(
        mega_moe_max_tokens=mega_moe_max_tokens,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        ),
    )


def _make_mega_moe_group(device_group=None, world_size=4):
    return SimpleNamespace(
        device_group=object() if device_group is None else device_group,
        world_size=world_size,
    )


def _make_fused_input(
    *,
    swiglu_limit=0.0,
    log2phy=None,
    scale_dtype=torch.uint8,
    activation="silu",
    dynamic_eplb=False,
    global_redundant_expert_num=0,
    w1=None,
):
    return build_fused_experts_input(
        hidden_states=torch.randn(4, 8),
        topk_weights=torch.randn(4, 2),
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]]),
        w1=torch.randn(2, 16, 8) if w1 is None else w1,
        w2=torch.randn(2, 8, 8),
        quant_type=QuantType.MXFP8,
        dynamic_eplb=dynamic_eplb,
        log2phy=log2phy,
        global_redundant_expert_num=global_redundant_expert_num,
        activation=activation,
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=torch.float8_e4m3fn,
        mxfp_scale_dtype=torch.uint8,
        mxfp_per_token_scale_dtype=torch.uint8,
        w1_scale=torch.ones(2, 16, 1, 2, dtype=scale_dtype),
        w2_scale=torch.ones(2, 8, 1, 2, dtype=scale_dtype),
        swiglu_limit=swiglu_limit,
    )


def test_mega_moe_backend_builds_buffer_and_operator_args():
    fused_input = _make_fused_input(log2phy=torch.tensor([3, 2, 1, 0]))
    mega_moe_device_group = object()
    sym_buffer = SimpleNamespace(num_max_tokens_per_rank=None)
    get_symm_buffer = MagicMock(return_value=sym_buffer)
    mega_moe = MagicMock(return_value=(torch.randn(4, 8), torch.tensor([1, 3], dtype=torch.int32)))

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops", return_value=(get_symm_buffer, mega_moe)),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config",
            return_value=_make_ascend_config(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group",
            return_value=_make_mega_moe_group(mega_moe_device_group),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ) as get_capacity,
        patch("vllm_ascend.ops.fused_moe.mega_moe.logger.debug") as mock_debug,
    ):
        backend = MegaMoEBackend(_make_moe_config())
        backend.fused_experts(fused_input)

    get_symm_buffer.assert_called_once()
    assert get_symm_buffer.call_args.args[0] is mega_moe_device_group
    assert get_symm_buffer.call_args.kwargs["intermediate_hidden"] == 16
    assert get_symm_buffer.call_args.kwargs["num_max_tokens_per_rank"] == 8
    assert get_symm_buffer.call_args.kwargs["dispatch_quant_mode"] == 4
    assert sym_buffer.num_max_tokens_per_rank is None
    get_capacity.assert_called_once()

    mega_moe.assert_called_once()
    mega_moe_kwargs = mega_moe.call_args.kwargs
    assert torch.equal(mega_moe_kwargs["topk_ids"], fused_input.routing.log2phy[fused_input.topk_ids].to(torch.int32))
    assert mega_moe_kwargs["activation"] == "swiglu"
    assert mega_moe_kwargs["activation_clamp"] is None
    assert "activation_params" not in mega_moe_kwargs
    assert mega_moe_kwargs["weight1_type"] == torch.float8_e4m3fn
    assert mega_moe_kwargs["weight2_type"] == torch.float8_e4m3fn
    assert mega_moe_kwargs["l1_weights_sf"][0].dtype == torch.float8_e8m0fnu
    assert mega_moe_kwargs["l2_weights_sf"][0].dtype == torch.float8_e8m0fnu
    messages = [call.args[0] for call in mock_debug.call_args_list]
    assert any(message.startswith("A5 MegaMoE creates the process-wide symmetric buffer") for message in messages)
    assert any(message.startswith("A5 MegaMoE call") for message in messages)
    assert any(message.startswith("A5 MegaMoE output") for message in messages)


def test_mega_moe_prepare_pads_to_common_token_count_without_filling_buffer():
    prepare_finalize = object.__new__(PrepareAndFinalizeWithMegaMoE)
    hidden_states = torch.randn(4, 8)
    router_logits = torch.randn(4, 2)
    input_ids = torch.tensor([1, 2, 3, 4])

    # Gate multistream overlap can select experts before prepare().
    torch.testing.assert_close(prepare_finalize.pad_and_split_input_ids(input_ids), input_ids)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX",
            SimpleNamespace(max_tokens_across_dp=6),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_ascend_config",
            return_value=_make_ascend_config(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ) as get_capacity,
    ):
        prepare_output = prepare_finalize.prepare(hidden_states, router_logits)

    get_capacity.assert_called_once()
    assert prepare_output.hidden_states.shape == (6, 8)
    assert prepare_output.router_logits.shape == (6, 2)
    torch.testing.assert_close(prepare_output.hidden_states[:4], hidden_states)
    torch.testing.assert_close(prepare_output.router_logits[:4], router_logits)
    assert torch.count_nonzero(prepare_output.hidden_states[4:]) == 0
    assert torch.count_nonzero(prepare_output.router_logits[4:]) == 0

    torch.testing.assert_close(
        prepare_finalize.pad_and_split_input_ids(input_ids),
        torch.tensor([1, 2, 3, 4, 0, 0]),
    )

    padded_output = torch.randn(6, 8)
    torch.testing.assert_close(
        prepare_finalize.finalize(padded_output, reduce_results=False),
        padded_output[:4],
    )
    torch.testing.assert_close(prepare_finalize.pad_and_split_input_ids(input_ids), input_ids)


def test_mega_moe_prepare_rejects_target_smaller_than_local_tokens():
    prepare_finalize = object.__new__(PrepareAndFinalizeWithMegaMoE)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX",
            SimpleNamespace(max_tokens_across_dp=3),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_ascend_config",
            return_value=_make_ascend_config(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        pytest.raises(ValueError, match="cannot be smaller than the local token count"),
    ):
        prepare_finalize.prepare(torch.randn(4, 8), torch.randn(4, 2))


def test_mega_moe_prepare_rejects_common_token_count_over_buffer_capacity():
    prepare_finalize = object.__new__(PrepareAndFinalizeWithMegaMoE)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX",
            SimpleNamespace(max_tokens_across_dp=9),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_ascend_config",
            return_value=_make_ascend_config(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        pytest.raises(ValueError, match="exceeds the symmetric buffer token capacity"),
    ):
        prepare_finalize.prepare(torch.randn(4, 8), torch.randn(4, 2))


def test_main_and_draft_mega_moe_backends_share_single_buffer():
    fused_input = _make_fused_input()
    sym_buffer = SimpleNamespace(num_max_tokens_per_rank=None)
    get_symm_buffer = MagicMock(return_value=sym_buffer)
    mega_moe = MagicMock(return_value=(torch.randn(4, 8), torch.tensor([1, 3], dtype=torch.int32)))

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops", return_value=(get_symm_buffer, mega_moe)),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config",
            return_value=_make_ascend_config(mega_moe_max_tokens=16),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group",
            return_value=_make_mega_moe_group(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=4,
        ),
        patch("vllm_ascend.ops.fused_moe.mega_moe.logger.debug") as mock_debug,
    ):
        main_backend = MegaMoEBackend(_make_moe_config())
        draft_backend = MegaMoEBackend(_make_moe_config())
        main_backend.fused_experts(fused_input)
        draft_backend.fused_experts(fused_input)

    get_symm_buffer.assert_called_once()
    assert get_symm_buffer.call_args.kwargs["num_max_tokens_per_rank"] == 4
    assert sym_buffer.num_max_tokens_per_rank is None
    assert mega_moe.call_count == 2
    assert all(call.kwargs["sym_buffer"] is sym_buffer for call in mega_moe.call_args_list)
    assert any(
        call.args[0].startswith("A5 MegaMoE reuses the process-wide symmetric buffer")
        for call in mock_debug.call_args_list
    )


def test_mega_moe_backend_rejects_buffer_reinitialization():
    fused_input = _make_fused_input()
    get_symm_buffer = MagicMock(return_value=SimpleNamespace(num_max_tokens_per_rank=None))

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops", return_value=(get_symm_buffer, MagicMock())),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config",
            side_effect=(
                _make_ascend_config(mega_moe_max_tokens=16),
                _make_ascend_config(mega_moe_max_tokens=16),
            ),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group",
            return_value=_make_mega_moe_group(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=4,
        ),
    ):
        backend = MegaMoEBackend(_make_moe_config())
        backend._get_sym_buffer(fused_input, projected_hidden=16)
        with pytest.raises(RuntimeError, match="shared by the main and draft models"):
            backend._get_sym_buffer(fused_input, projected_hidden=8)

    get_symm_buffer.assert_called_once()


def test_view_mxfp_scales_as_e8m0_reinterprets_uint8_storage():
    scale = torch.arange(64, dtype=torch.uint8).reshape(2, 16, 1, 2)

    normalized_scale = _view_mxfp_scales_as_e8m0([scale], "w1_scale")[0]

    assert normalized_scale.dtype == torch.float8_e8m0fnu
    assert normalized_scale.shape == scale.shape
    assert normalized_scale.stride() == scale.stride()
    assert normalized_scale.numel() == scale.numel()
    assert normalized_scale.data_ptr() == scale.data_ptr()


def test_view_mxfp_scales_as_e8m0_requires_native_torch_dtype():
    scale = torch.ones(2, 16, 1, 2, dtype=torch.uint8)

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._TORCH_FLOAT8_E8M0FNU_DTYPE", None),
        pytest.raises(RuntimeError, match=r"torch\.float8_e8m0fnu"),
    ):
        _view_mxfp_scales_as_e8m0([scale], "w1_scale")


def test_mega_moe_backend_passes_positive_swiglu_limit_as_clamp():
    fused_input = _make_fused_input(swiglu_limit=7.0)
    get_symm_buffer = MagicMock(return_value=SimpleNamespace(num_max_tokens_per_rank=None))
    mega_moe = MagicMock(return_value=(torch.randn(4, 8), torch.tensor([1, 3], dtype=torch.int32)))

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops", return_value=(get_symm_buffer, mega_moe)),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config",
            return_value=_make_ascend_config(mega_moe_max_tokens=16),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group",
            return_value=_make_mega_moe_group(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=4,
        ),
    ):
        backend = MegaMoEBackend(_make_moe_config())
        backend.fused_experts(fused_input)

    assert mega_moe.call_args.kwargs["activation_clamp"] == 7.0


def test_mega_moe_backend_accepts_enum_activation():
    assert MegaMoEBackend._normalize_activation(_TestMoEActivation.SILU) == "swiglu"


@pytest.mark.parametrize("activation", [_TestMoEActivation.SWIGLUOAI, _TestMoEActivation.SWIGLUSTEP, "situglu"])
def test_mega_moe_backend_rejects_activation_with_different_semantics(activation):
    with pytest.raises(ValueError, match="without changing its semantics"):
        MegaMoEBackend._normalize_activation(activation)


@pytest.mark.parametrize(
    ("input_kwargs", "message"),
    [
        ({"dynamic_eplb": True}, "dynamic EPLB"),
        ({"global_redundant_expert_num": 1}, "redundant physical experts"),
    ],
)
def test_mega_moe_backend_rejects_unsupported_eplb_layouts(input_kwargs, message):
    backend = MegaMoEBackend(_make_moe_config())
    with pytest.raises(RuntimeError, match=message):
        backend.fused_experts(_make_fused_input(**input_kwargs))


def test_mega_moe_backend_rejects_transposed_grouped_matmul_layout():
    w1 = torch.randn(2, 8, 16).transpose(1, 2)
    fused_input = _make_fused_input(w1=w1)

    backend = MegaMoEBackend(_make_moe_config())
    with pytest.raises(ValueError, match="requires contiguous w1"):
        backend.fused_experts(fused_input)


def test_mega_moe_backend_validates_fp4_packed_dimensions():
    fused_input = build_fused_experts_input(
        hidden_states=torch.randn(4, 128),
        topk_weights=torch.randn(4, 2),
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]]),
        w1=torch.randint(0, 255, (2, 128, 64), dtype=torch.uint8),
        w2=torch.randint(0, 255, (2, 128, 32), dtype=torch.uint8),
        quant_type=QuantType.W4A8MXFP,
        dynamic_eplb=False,
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=torch.uint8,
        mxfp_scale_dtype=torch.uint8,
        mxfp_per_token_scale_dtype=torch.uint8,
        w1_scale=torch.ones(2, 128, 2, 2, dtype=torch.uint8),
        w2_scale=torch.ones(2, 128, 1, 2, dtype=torch.uint8),
    )

    key = MegaMoEBackend(_make_moe_config())._make_buffer_key(fused_input, buffer_tokens_per_rank=4)

    assert key.hidden_size == 128
    assert key.intermediate_hidden == 128


def test_mega_moe_backend_accepts_input_smaller_than_configured_capacity():
    fused_input = _make_fused_input()
    backend = MegaMoEBackend(_make_moe_config())

    key = backend._make_buffer_key(fused_input, buffer_tokens_per_rank=8)

    assert key.buffer_tokens_per_rank == 8


def test_mega_moe_backend_rejects_input_over_configured_capacity():
    fused_input = _make_fused_input()
    backend = MegaMoEBackend(_make_moe_config())
    with pytest.raises(ValueError, match="exceeds the symmetric buffer token capacity"):
        backend._make_buffer_key(fused_input, buffer_tokens_per_rank=3)


def test_mega_moe_backend_rejects_packed_uint64_scales():
    fused_input = _make_fused_input(scale_dtype=torch.uint64)

    backend = MegaMoEBackend(_make_moe_config())
    try:
        backend.fused_experts(fused_input)
    except RuntimeError as exc:
        assert "Do not pass packed INT/Fused-MC2 UINT64 scales into MegaMoE" in str(exc)
    else:
        raise AssertionError("Expected uint64 scales to be rejected.")
