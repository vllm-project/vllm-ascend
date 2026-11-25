# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ops import mmrs
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

ATTENTION_LAYER_NAME = "model.layers.0.self_attn.attn"


def make_layer(tp_size=2):
    return SimpleNamespace(
        custom_op=None,
        reduce_results=False,
        input_is_parallel=True,
        disable_tp=False,
        bias=None,
        tp_size=tp_size,
        tp_rank=0,
        quant_method=UnquantizedLinearMethod(),
        weight=torch.ones(32, 256, dtype=torch.bfloat16),
        skip_bias_add=False,
        return_bias=True,
        input_size_per_partition=256,
        output_size=32,
        prefix="model.layers.0.self_attn.o_proj",
    )


def make_config(**overrides):
    config = SimpleNamespace(
        lora_config=None,
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_op(layer, min_tokens=1, max_tokens=65535):
    return mmrs.SequenceParallelMMReduceScatterOp(
        layer, ATTENTION_LAYER_NAME, min_tokens=min_tokens, max_tokens=max_tokens
    )


@pytest.fixture
def harness():
    layer = make_layer()
    group = Mock(world_size=2, rank_in_group=0)
    group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "original-tp"
    metadata = SimpleNamespace(num_prefills=1, num_decodes=0)
    context = SimpleNamespace(attn_metadata={ATTENTION_LAYER_NAME: metadata})
    profile = Mock()
    profile.supports.return_value = True
    with (
        patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=group),
        patch.object(mmrs, "is_forward_context_available", return_value=True),
        patch.object(mmrs, "get_forward_context", return_value=context),
        patch.object(mmrs, "get_current_vllm_config", return_value=make_config()),
        patch.object(mmrs, "get_current_hardware_profile", return_value=profile),
        patch.object(
            mmrs.torch_npu, "get_npu_format", return_value=mmrs.ACL_FORMAT_FRACTAL_ND, create=True
        ) as weight_format,
        # The CPU torch_npu wheel has no fused MM+ReduceScatter entry point;
        # provide one so the static gate can pass on any platform.
        patch.object(mmrs.torch_npu, "npu_mm_reduce_scatter_base", create=True),
        patch.object(mmrs.DeviceOperator, "npu_mm_reduce_scatter_base") as fused,
        patch.object(mmrs, "sp_reduce_scatter") as reduce_scatter,
    ):
        op = make_op(layer)

        def projection(owner, x, bias):
            return x.float().matmul(owner.weight.t().float()).to(x.dtype)

        def scatter(x):
            padded = torch.nn.functional.pad(x, (0, 0, 0, (-x.shape[0]) % group.world_size))
            return (padded * group.world_size).chunk(group.world_size, dim=0)[group.rank_in_group]

        def fused_projection(x1, x2, hcom, world_size, **kwargs):
            assert x1.shape[0] % world_size == 0
            return scatter(x1.float().matmul(x2.float()).to(x1.dtype))

        layer.quant_method.apply = Mock(side_effect=projection)
        fused.side_effect = fused_projection
        reduce_scatter.side_effect = scatter
        yield SimpleNamespace(
            layer=layer,
            op=op,
            group=group,
            metadata=metadata,
            context=context,
            profile=profile,
            fused=fused,
            reduce_scatter=reduce_scatter,
            projection=projection,
            scatter=scatter,
            weight_format=weight_format,
        )


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_fused_projection_matches_mm_plus_rs_on_every_rank(harness, tp_size, tokens):
    h = harness
    h.group.world_size = tp_size
    x = torch.arange(tokens * 256).remainder(4).reshape(tokens, 256).to(torch.bfloat16)
    for rank in range(tp_size):
        h.group.rank_in_group = rank
        expected = h.scatter(h.projection(h.layer, x, None))
        actual = h.op.run(x)
        torch.testing.assert_close(actual, expected)
        assert actual.shape == ((tokens + tp_size - 1) // tp_size, 32)
    h.reduce_scatter.assert_not_called()
    h.layer.quant_method.apply.assert_not_called()
    args, kwargs = h.fused.call_args
    # [K, N] view of the unquantized weight (no per-call copy).
    assert args[1].shape == (256, 32)
    assert args[1].stride() == (1, 256)
    assert args[1].untyped_storage().data_ptr() == h.layer.weight.untyped_storage().data_ptr()
    assert args[2:4] == ("original-tp", tp_size)
    assert "x1_scale" not in kwargs and "x2_scale" not in kwargs
    assert "bias" not in kwargs
    assert kwargs["reduce_op"] == "sum"
    # comm_mode is the device adaptor's per-platform choice (AIV on A2/A3,
    # AI CPU or CCU on A5); the layer op does not override it.
    assert "comm_mode" not in kwargs
    if tokens % tp_size:
        assert torch.count_nonzero(args[0][tokens:]) == 0


@pytest.mark.parametrize("prefills,decodes", [(0, 2), (1, 1)])
def test_decode_and_mixed_fallback_preserve_shards(harness, prefills, decodes):
    h = harness
    h.metadata.num_prefills = prefills
    h.metadata.num_decodes = decodes
    x = torch.ones(3, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(h.op.run(x), h.scatter(h.projection(h.layer, x, None)))
    h.fused.assert_not_called()
    h.layer.quant_method.apply.assert_called_once_with(h.layer, x, None)
    h.reduce_scatter.assert_called_once()


@pytest.mark.parametrize("metadata", [None, {}, {"another.attn": SimpleNamespace(num_prefills=1, num_decodes=0)}])
def test_missing_or_unrelated_metadata_falls_back(harness, metadata):
    h = harness
    h.context.attn_metadata = metadata
    assert h.op.run(torch.ones(1, 256, dtype=torch.bfloat16)).shape == (1, 32)
    h.fused.assert_not_called()
    h.reduce_scatter.assert_called_once()


def test_nz_weight_falls_back_without_fusion(harness):
    # Only ND weights fuse (weight_nz_mode=1); an NZ weight keeps the MM + RS
    # fallback instead of paying a per-layer format conversion.
    h = harness
    h.weight_format.return_value = ACL_FORMAT_FRACTAL_NZ
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(h.op.run(x), h.scatter(h.projection(h.layer, x, None)))
    h.fused.assert_not_called()
    h.reduce_scatter.assert_called_once()


def test_non_bf16_input_falls_back(harness):
    h = harness
    x = torch.ones(3, 256, dtype=torch.float16)
    torch.testing.assert_close(h.op.run(x), h.scatter(h.projection(h.layer, x, None)))
    h.fused.assert_not_called()
    h.reduce_scatter.assert_called_once()


@pytest.mark.parametrize("tokens,expected", [(255, False), (256, True), (16383, True), (16384, False)])
def test_model_token_limit_boundaries(harness, tokens, expected):
    # The model gates the token count (GEMM M) to its favorable range.
    h = harness
    op = make_op(h.layer, min_tokens=256, max_tokens=16384)
    x = torch.ones(tokens, 256, dtype=torch.bfloat16)
    assert op._can_fuse(x) is expected
    torch.testing.assert_close(op.run(x), h.scatter(h.projection(h.layer, x, None)))
    assert h.fused.called is expected
    assert h.reduce_scatter.called is not expected


@pytest.mark.parametrize(
    "min_tokens,max_tokens,expected",
    [(1, 16384, True), (3, 16384, False), (1, 2, False)],
)
def test_model_supplied_token_limits(harness, min_tokens, max_tokens, expected):
    h = harness
    op = make_op(h.layer, min_tokens=min_tokens, max_tokens=max_tokens)
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    assert op._can_fuse(x) is expected


@pytest.mark.parametrize("bad_case", ["weight_dtype", "weight_shape", "format", "small_k", "empty"])
def test_unsupported_runtime_layout_is_rejected_before_collective(harness, bad_case):
    h = harness
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    if bad_case == "weight_dtype":
        h.layer.weight = h.layer.weight.float()
    elif bad_case == "weight_shape":
        h.layer.weight = h.layer.weight.t().contiguous()
    elif bad_case == "format":
        h.weight_format.return_value = 3
    elif bad_case == "small_k":
        x = x[:, :128]
    elif bad_case == "empty":
        x = x[:0]
    assert not h.op._can_fuse(x)
    h.fused.assert_not_called()


def test_fused_failure_is_not_retried_as_another_collective(harness):
    h = harness
    h.fused.side_effect = RuntimeError("collective failed")
    with pytest.raises(RuntimeError, match="collective failed"):
        h.op.run(torch.ones(2, 256, dtype=torch.bfloat16))
    h.reduce_scatter.assert_not_called()


def test_apply_follows_return_bias_contract(harness):
    h = harness
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    expected = h.scatter(h.projection(h.layer, x, None))
    output, bias = h.op.apply(x)
    torch.testing.assert_close(output, expected)
    assert bias is None
    h.op.return_bias = False
    torch.testing.assert_close(h.op.apply(x), expected)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("bias", torch.ones(32)),
        ("reduce_results", True),
        ("input_is_parallel", False),
        ("disable_tp", True),
        ("quant_method", object()),
        ("weight", torch.ones(32, 256, dtype=torch.float32)),
        ("weight", torch.ones(32, 256, dtype=torch.float16)),
    ],
)
def test_static_layer_rejections_never_fuse(harness, attribute, value):
    # Installation always succeeds; an unsupported layer simply never leaves
    # the fallback path.
    h = harness
    setattr(h.layer, attribute, value)
    op = make_op(h.layer)
    assert op._static_reason is not None
    assert op._can_fuse(torch.ones(2, 256, dtype=torch.bfloat16)) is False
    h.fused.assert_not_called()


@pytest.mark.parametrize("tp_size", [1, 16])
def test_unsupported_tp_size_never_fuses(harness, tp_size):
    h = harness
    h.group.world_size = tp_size
    op = make_op(h.layer)
    assert op._static_reason is not None
    assert op._can_fuse(torch.ones(2, 256, dtype=torch.bfloat16)) is False
    h.fused.assert_not_called()


def test_existing_custom_op_keeps_ownership(harness):
    # A pre-existing custom op (e.g. fine-grained OTP) owns the layer's
    # communication; this op only delegates and never fuses.
    h = harness
    previous = Mock()
    sentinel = torch.ones(2, 32)
    previous.apply_impl.return_value = (sentinel, None)
    h.layer.custom_op = previous
    op = make_op(h.layer)
    assert op._static_reason is not None
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    assert op._can_fuse(x) is False
    output, bias = op.apply_impl(x)
    previous.apply_impl.assert_called_once_with(x)
    assert output is sentinel and bias is None
    h.fused.assert_not_called()
    h.layer.quant_method.apply.assert_not_called()


def test_non_parallel_input_is_split_before_fallback(harness):
    # A non-TP-partitioned input never fuses, and the fallback still splits
    # it along the last dim like the original forward.
    h = harness
    h.layer.input_is_parallel = False
    op = make_op(h.layer)
    assert op._static_reason is not None
    x = torch.arange(2 * 512, dtype=torch.float32).remainder(5).reshape(2, 512).to(torch.bfloat16)
    output = op.run(x)
    applied = h.layer.quant_method.apply.call_args.args[1]
    torch.testing.assert_close(applied, x[:, :256].contiguous())
    torch.testing.assert_close(output, h.scatter(h.projection(h.layer, x[:, :256], None)))
    h.fused.assert_not_called()


@pytest.mark.parametrize("gate", ["lora", "decode_node", "pcp", "dcp", "hardware"])
def test_config_and_hardware_gates_only_disable_fusion(harness, gate):
    # Every former install-time rejection now means: installed, but the
    # dynamic gate never leaves the fallback path.
    h = harness
    config = make_config()
    profile = Mock()
    profile.supports.return_value = True
    if gate == "lora":
        config.lora_config = object()
    elif gate == "decode_node":
        config.kv_transfer_config = SimpleNamespace(is_kv_consumer=True, is_kv_producer=False)
    elif gate == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    elif gate == "dcp":
        config.parallel_config.decode_context_parallel_size = 2
    elif gate == "hardware":
        profile.supports.side_effect = (
            lambda capability: capability is not mmrs.HardwareCapability.MM_REDUCE_SCATTER_BF16
        )
    with (
        patch.object(mmrs, "get_current_vllm_config", return_value=config),
        patch.object(mmrs, "get_current_hardware_profile", return_value=profile),
    ):
        op = make_op(h.layer)
    assert op._static_reason is not None
    x = torch.ones(3, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(op.run(x), h.scatter(h.projection(h.layer, x, None)))
    h.fused.assert_not_called()
    h.reduce_scatter.assert_called_once()


@pytest.mark.parametrize(
    "kv_role,expected",
    [
        (None, True),  # No PD split: the node serves prefill.
        ("kv_producer", True),  # Pure prefill node.
        ("kv_both", True),  # Producer and consumer.
        ("kv_consumer", False),  # Pure decode node: fusion can never fire.
    ],
)
def test_pd_node_role_only_gates_fusion(harness, kv_role, expected):
    h = harness
    transfer = None
    if kv_role is not None:
        transfer = SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    with patch.object(mmrs, "get_current_vllm_config", return_value=make_config(kv_transfer_config=transfer)):
        op = make_op(h.layer)
    assert (op._static_reason is None) is expected
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(op.run(x), h.scatter(h.projection(h.layer, x, None)))
    assert h.fused.called is expected
    assert h.reduce_scatter.called is not expected


def test_fusion_queries_the_bf16_mmrs_capability(harness):
    h = harness
    profile = Mock()
    profile.supports.return_value = True
    with patch.object(mmrs, "get_current_hardware_profile", return_value=profile):
        make_op(h.layer)
    assert profile.supports.call_args.args[0] is mmrs.HardwareCapability.MM_REDUCE_SCATTER_BF16


def test_weight_changes_are_picked_up_without_cache(harness):
    # Nothing about the weight is cached: a fresh transposed view is passed
    # per fused call, and the layout check runs per forward.
    h = harness
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    first = h.op.run(x)
    torch.testing.assert_close(first, h.scatter(h.projection(h.layer, x, None)))
    # Loader-style object replacement is visible on the very next call.
    h.layer.weight = torch.full((32, 256), 2, dtype=torch.bfloat16)
    second = h.op.run(x)
    torch.testing.assert_close(second, h.scatter(h.projection(h.layer, x, None)))
    torch.testing.assert_close(second, first * 2)
    # In-place mutation likewise.
    h.layer.weight.fill_(3)
    torch.testing.assert_close(h.op.run(x), h.scatter(h.projection(h.layer, x, None)))
    assert h.weight_format.call_count == 3  # validated on every fused forward
