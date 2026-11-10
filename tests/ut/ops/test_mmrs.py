# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ops import mmrs
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


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


@pytest.fixture
def harness():
    layer = make_layer()
    group = Mock(world_size=2, rank_in_group=0)
    group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "original-tp"
    metadata = SimpleNamespace(num_prefills=1, num_decodes=0)
    context = SimpleNamespace(attn_metadata={"model.layers.0.self_attn.attn": metadata})
    with (
        patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=group),
        patch.object(mmrs, "is_forward_context_available", return_value=True),
        patch.object(mmrs, "get_forward_context", return_value=context),
        patch.object(
            mmrs.torch_npu, "get_npu_format", return_value=mmrs.ACL_FORMAT_FRACTAL_ND, create=True
        ) as weight_format,
        patch.object(mmrs.torch_npu, "npu_mm_reduce_scatter_base", create=True) as fused,
        patch.object(mmrs, "sp_reduce_scatter") as reduce_scatter,
    ):
        op = mmrs.SequenceParallelMMReduceScatterOp(layer, "model.layers.0.self_attn.attn")

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
    assert "y_dtype" not in kwargs  # BF16 in, BF16 out.
    assert kwargs["comm_mode"] == "aiv"
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


def test_large_prefill_batches_are_not_gated(harness):
    # No token-count performance gate: large prefill batches still fuse so
    # the end-to-end benefit stays attributable to the fused operator.
    h = harness
    x = torch.ones(5000, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(h.op.run(x), h.scatter(h.projection(h.layer, x, None)))
    h.fused.assert_called_once()
    h.reduce_scatter.assert_not_called()


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


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("custom_op", object()),
        ("bias", torch.ones(32)),
        ("reduce_results", True),
        ("input_is_parallel", False),
        ("disable_tp", True),
        ("tp_size", 1),
        ("tp_size", 16),
        ("quant_method", object()),
        ("weight", torch.ones(32, 256, dtype=torch.float32)),
        ("weight", torch.ones(32, 256, dtype=torch.float16)),
    ],
)
def test_static_rejections_do_not_modify_layer(attribute, value):
    layer = make_layer()
    setattr(layer, attribute, value)
    original = layer.custom_op
    assert mmrs.SequenceParallelMMReduceScatterOp.unsupported_reason(layer) is not None
    assert layer.custom_op is original


@pytest.mark.parametrize("gate", ["enabled", "decode_node", "no_sp", "lora", "pcp", "dcp", "hardware", "conflict"])
def test_installation_and_static_output_contract(harness, gate):
    h = harness
    config = SimpleNamespace(
        lora_config=object() if gate == "lora" else None,
        kv_transfer_config=(
            # A pure PD decode node must not install the fused op.
            SimpleNamespace(is_kv_consumer=True, is_kv_producer=False) if gate == "decode_node" else None
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2 if gate == "pcp" else 1,
            decode_context_parallel_size=2 if gate == "dcp" else 1,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    if gate == "conflict":
        h.layer.custom_op = object()
    old_op = h.layer.custom_op
    profile = Mock()
    profile.supports.return_value = gate != "hardware"
    with (
        patch.object(mmrs, "get_current_vllm_config", return_value=config),
        patch.object(mmrs, "get_current_hardware_profile", return_value=profile),
    ):
        enabled = mmrs.install_sequence_parallel_mmrs(
            h.layer, sequence_parallel=gate != "no_sp", attention_layer_name=h.op.attention_layer_name
        )
    assert enabled == (gate == "enabled")
    if enabled:
        assert config.compilation_config.static_forward_context[h.layer.prefix] is h.layer
        assert isinstance(h.layer.custom_op, mmrs.SequenceParallelMMReduceScatterOp)
    else:
        assert h.layer.custom_op is old_op
        assert not config.compilation_config.static_forward_context


@pytest.mark.parametrize("supported", [True, False])
def test_install_requires_bf16_aiv_capability(harness, supported):
    h = harness
    config = SimpleNamespace(
        lora_config=None,
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    profile = Mock()
    profile.supports.side_effect = (
        lambda capability: capability is mmrs.HardwareCapability.MM_REDUCE_SCATTER_BF16 and supported
    )
    with (
        patch.object(mmrs, "get_current_vllm_config", return_value=config),
        patch.object(mmrs, "get_current_hardware_profile", return_value=profile),
    ):
        enabled = mmrs.install_sequence_parallel_mmrs(
            h.layer, sequence_parallel=True, attention_layer_name=h.op.attention_layer_name
        )
    assert enabled is supported
    assert isinstance(h.layer.custom_op, mmrs.SequenceParallelMMReduceScatterOp) is supported
    assert profile.supports.call_args.args[0] is mmrs.HardwareCapability.MM_REDUCE_SCATTER_BF16


@pytest.mark.parametrize(
    "kv_role,expected",
    [
        (None, True),  # No PD split: the node serves prefill.
        ("kv_producer", True),  # Pure prefill node.
        ("kv_both", True),  # Producer and consumer.
        ("kv_consumer", False),  # Pure decode node: fusion can never fire.
    ],
)
def test_install_follows_pd_node_role(harness, kv_role, expected):
    h = harness
    transfer = None
    if kv_role is not None:
        transfer = SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    config = SimpleNamespace(
        lora_config=None,
        kv_transfer_config=transfer,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    profile = Mock()
    profile.supports.return_value = True
    with (
        patch.object(mmrs, "get_current_vllm_config", return_value=config),
        patch.object(mmrs, "get_current_hardware_profile", return_value=profile),
    ):
        enabled = mmrs.install_sequence_parallel_mmrs(
            h.layer, sequence_parallel=True, attention_layer_name=h.op.attention_layer_name
        )
    assert enabled is expected
    assert isinstance(h.layer.custom_op, mmrs.SequenceParallelMMReduceScatterOp) is expected


def test_runtime_dispatch_uses_current_layer(harness):
    h = harness
    h.layer.custom_op = h.op
    h.context.no_compile_layers = {h.layer.prefix: h.layer}
    x = torch.ones(3, 256, dtype=torch.bfloat16)
    actual = mmrs._sequence_parallel_mmrs(x, h.layer.prefix, 32, 2)
    torch.testing.assert_close(actual, h.scatter(h.projection(h.layer, x, None)))


def test_fake_registration_returns_shard_without_communication():
    with FakeTensorMode():
        x = torch.empty(3, 256, dtype=torch.bfloat16)
        output = torch.ops.vllm.ascend_sequence_parallel_mmrs(x, "unregistered.layer", 32, 2)
        assert output.shape == (2, 32)
        assert output.dtype == x.dtype


def test_linear_tuple_and_tensor_return_contract(harness):
    h = harness
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    expected = torch.ones(1, 32, dtype=torch.bfloat16)
    with patch.object(torch.ops.vllm, "ascend_sequence_parallel_mmrs", return_value=expected):
        output, bias = h.op.apply(x)
        assert output is expected and bias is None
        h.op.return_bias = False
        assert h.op.apply(x) is expected


def test_static_weight_checks_are_cached_until_weights_change(harness):
    h = harness
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    h.op.run(x)
    assert h.weight_format.call_count == 1  # validated once; NZ flag is reused
    x2_first = h.op._x2_weight
    assert x2_first is not None
    h.op.run(x)
    assert h.weight_format.call_count == 1  # steady state: no re-validation
    assert h.fused.call_args.args[1] is x2_first
    # Loader-style replacement revalidates and refreshes the cached operand.
    h.layer.weight = torch.full((32, 256), 2, dtype=torch.bfloat16)
    h.op.run(x)
    assert h.weight_format.call_count == 2
    assert h.op._x2_weight is not x2_first
    # In-place mutation (version bump) also revalidates.
    h.layer.weight.fill_(3)
    h.op.run(x)
    assert h.weight_format.call_count == 3
