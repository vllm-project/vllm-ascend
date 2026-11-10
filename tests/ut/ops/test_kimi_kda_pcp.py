# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models.kimi_k3 import kimi_k3
from vllm_ascend.ops import kda_parallel, kda_pcp
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention


@contextmanager
def parallel_groups(tp, pcp):
    with (
        patch.object(kda_parallel, "get_tp_group", return_value=tp),
        patch.object(kda_parallel, "get_pcp_group", return_value=pcp),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=tp.rank_in_group),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=tp.world_size),
    ):
        yield


@pytest.mark.parametrize("pcp", [1, 2, 4])
@pytest.mark.parametrize(
    ("model", "upstream"),
    [
        (kimi_k3.AscendKimiLinearForCausalLM, kimi_k3.UpstreamKimiLinearForCausalLM),
        (kimi_k3.AscendKimiK3ForConditionalGeneration, kimi_k3.UpstreamKimiK3ForConditionalGeneration),
    ],
)
def test_kda_state_shapes_delegate_to_upstream_without_mutating_config(pcp, model, upstream):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2, prefill_context_parallel_size=pcp),
        model_config=object(),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        cache_config=SimpleNamespace(use_kda_recoverssm=True),
    )
    shapes = object()
    with patch.object(upstream, "get_mamba_state_shape_from_config", return_value=shapes) as get_shape:
        assert model.get_mamba_state_shape_from_config(config) is shapes
    get_shape.assert_called_once()
    effective = get_shape.call_args.args[0]
    assert effective.parallel_config.tensor_parallel_size == 2 * pcp
    assert effective.model_config is config.model_config
    assert effective.speculative_config is config.speculative_config
    assert effective.cache_config is config.cache_config
    assert config.parallel_config.tensor_parallel_size == 2
    assert (effective is config) == (pcp == 1)


@pytest.mark.parametrize(("tp", "pcp"), [(1, 2), (2, 2), (2, 4)])
def test_kda_projections_cover_global_heads_once(tp, pcp):
    size, width, hidden = tp * pcp, 64, 8
    weights = [torch.arange(n * hidden, dtype=torch.float32).view(n, hidden) for n in [width] * 4 + [4, 16]]
    output_weight = torch.arange(hidden * width, dtype=torch.float32).view(hidden, width)
    inputs = torch.arange(2 * hidden, dtype=torch.float32).view(2, hidden)
    contributions = []
    for rank in range(size):
        tp_rank, pcp_rank = divmod(rank, pcp)
        tp_group = SimpleNamespace(world_size=tp, rank_in_group=tp_rank, all_reduce=lambda x: x)
        pcp_group = SimpleNamespace(world_size=pcp, rank_in_group=pcp_rank, all_reduce=lambda x: x)
        with parallel_groups(tp_group, pcp_group):
            projection = kda_parallel.KDAInputParallelLinear(hidden, [width] * 4 + [4, 16], size, bias=False)
            out = kda_parallel.KDARowParallelLinear(width, hidden, bias=False)
        for sid, weight in enumerate(weights):
            projection.weight.weight_loader(projection.weight, weight, sid)
        expected = torch.cat(
            [weight if sid == 4 else weight.chunk(size, dim=0)[rank] for sid, weight in enumerate(weights)]
        )
        torch.testing.assert_close(projection.weight, expected)
        torch.testing.assert_close(projection(inputs)[0], inputs @ expected.T)
        out.weight.weight_loader(out.weight, output_weight)
        torch.testing.assert_close(out.weight, output_weight.chunk(size, dim=1)[rank])
        head_input = torch.arange(2 * width, dtype=torch.float32).view(2, width).chunk(size, dim=1)[rank]
        contributions.append(out(head_input)[0])
    torch.testing.assert_close(sum(contributions), torch.arange(2 * width).view(2, width).float() @ output_weight.T)


@pytest.mark.parametrize("loader", ["weight_loader", "weight_loader_v2"])
@pytest.mark.parametrize("padding", [0, 3])
def test_kda_input_projection_preserves_upstream_padding_and_replication(loader, padding):
    tp = SimpleNamespace(world_size=2, rank_in_group=1)
    pcp = SimpleNamespace(world_size=2, rank_in_group=1)
    output_sizes = [32] * 4 + [4, 8]
    if padding:
        output_sizes.append(padding * 4)
    with parallel_groups(tp, pcp):
        projection = kda_parallel.KDAInputParallelLinear(8, output_sizes, 4, padding_size=padding, bias=False)
    projection.weight.data.fill_(9)
    weight = torch.arange(32).float().view(4, 8)
    getattr(projection, loader)(projection.weight, weight, 4)
    torch.testing.assert_close(projection.weight[32:36], weight)
    assert projection.tp_rank == projection.weight.tp_rank == 3
    if padding:
        assert projection.weight.weight_loader_numel == projection.weight[:-padding].numel()
        torch.testing.assert_close(projection.weight[-padding:], torch.zeros(padding, 8))


@pytest.mark.parametrize("reduce_results", [False, True])
def test_kda_pcp_reduction_is_required_when_tp_reduction_is_deferred(reduce_results):
    calls = []

    def reduce_pcp(value):
        calls.append("pcp")
        return value * 2

    def reduce_tp(value):
        calls.append("tp")
        return value * 4

    tp = SimpleNamespace(world_size=4, rank_in_group=0, all_reduce=reduce_tp)
    pcp = SimpleNamespace(world_size=2, rank_in_group=0, all_reduce=reduce_pcp)
    with parallel_groups(tp, pcp):
        projection = kda_parallel.KDARowParallelLinear(64, 8, bias=False, reduce_results=reduce_results)
    projection.weight.data.fill_(1)
    output = projection(torch.ones(3, 8))[0]
    torch.testing.assert_close(output, torch.full((3, 8), 64.0 if reduce_results else 16.0))
    assert calls == (["pcp", "tp"] if reduce_results else ["pcp"])


def test_pcp_kda_output_projection_does_not_enter_tp_only_fusion(monkeypatch):
    tp = SimpleNamespace(world_size=4, rank_in_group=0)
    pcp = SimpleNamespace(world_size=2, rank_in_group=0)
    with parallel_groups(tp, pcp):
        projection = kda_parallel.KDARowParallelLinear(64, 8, bias=False)
    layer = kimi_k3.AscendKimiDecoderLayer.__new__(kimi_k3.AscendKimiDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.use_attn_residuals = True
    layer.self_attn = SimpleNamespace(o_proj=projection)
    monkeypatch.setattr(kimi_k3, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    monkeypatch.setattr(kimi_k3, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=0))
    assert not layer._enable_o_proj_mm_reduce_scatter(SimpleNamespace(lora_config=None))


def make_builder(pcp_rank, capacity=16):
    builder = object.__new__(kda_pcp.KimiKDAHostMetadataBuilder)
    builder.pcp_size = 2
    builder.pcp_rank = pcp_rank
    builder._pcp_restore_idx = torch.zeros(capacity * 2, dtype=torch.long)
    builder._pcp_restore_mask = torch.zeros(capacity * 2, dtype=torch.bool)
    builder._pcp_local_idx = torch.zeros(capacity, dtype=torch.long)
    builder._pcp_local_mask = torch.zeros(capacity, dtype=torch.bool)
    return builder


@pytest.mark.parametrize(
    ("rank_inputs", "restore", "local_indices"),
    [
        ([[0, 1, 6, -1], [2, 3, 4, 5]], [0, 1, 4, 5, 6, 7, 2], [[0, 1, 6], [2, 3, 4, 5]]),
        ([[0, 2], [1, -1]], [0, 2, 1], [[0, 2], [1]]),
        ([[0], [-1]], [0], [[0], []]),
        ([[0, 1], [0, 1]], [0, 1], [[0, 1], [0, 1]]),
    ],
)
def test_pcp_causal_order_decode_sharding_and_padding(rank_inputs, restore, local_indices):
    gathered = torch.tensor(rank_inputs, dtype=torch.float32).reshape(-1, 1)
    count = len(rank_inputs[0])
    for rank in range(2):
        builder = make_builder(rank)
        context = SimpleNamespace(
            local_num_tokens=count,
            hidden_restore_idx=torch.tensor(restore),
            local_token_indices=torch.tensor(local_indices[rank], dtype=torch.long),
        )
        metadata = builder._attach_pcp(SimpleNamespace(), context)
        group = SimpleNamespace(all_gather=lambda x, dim: gathered)
        with patch.object(
            kda_pcp,
            "get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata={"kda": metadata}, no_compile_layers={"kda": SimpleNamespace(pcp_group=group)}
            ),
        ):
            restored = kda_pcp.kda_pcp_gather(gathered[:count], 2, "kda")
            torch.testing.assert_close(restored[: len(restore), 0], torch.arange(len(restore)).float())
            local = kda_pcp.kda_pcp_scatter(restored * 10 + 1, 2, "kda")
        expected = torch.zeros(count)
        expected[: len(local_indices[rank])] = torch.tensor(local_indices[rank]).float() * 10 + 1
        torch.testing.assert_close(local[:, 0], expected)


def test_pcp_metadata_refresh_preserves_captured_addresses():
    builder = make_builder(0)
    context = SimpleNamespace(
        local_num_tokens=2, hidden_restore_idx=torch.tensor([0, 2, 1]), local_token_indices=torch.tensor([0, 2])
    )
    first = builder._attach_pcp(SimpleNamespace(), context)
    addresses = [
        getattr(first, name).data_ptr()
        for name in ("pcp_restore_idx", "pcp_restore_mask", "pcp_local_idx", "pcp_local_mask")
    ]
    context.hidden_restore_idx = torch.tensor([2, 0])
    context.local_token_indices = torch.tensor([1])
    second = builder._attach_pcp(SimpleNamespace(), context)
    assert addresses == [
        getattr(second, name).data_ptr()
        for name in ("pcp_restore_idx", "pcp_restore_mask", "pcp_local_idx", "pcp_local_mask")
    ]
    assert first.pcp_restore_idx.tolist() == [2, 0, 0, 0]
    assert first.pcp_restore_mask.tolist() == [True, True, False, False]
    assert first.pcp_local_idx.tolist() == [1, 0]
    assert first.pcp_local_mask.tolist() == [True, False]


@pytest.mark.parametrize("for_capture", [False, True])
def test_pcp_builder_consumes_common_metadata_context(for_capture):
    builder = make_builder(0)
    context = SimpleNamespace(
        local_num_tokens=2, hidden_restore_idx=torch.tensor([0, 2, 1]), local_token_indices=torch.tensor([0, 2])
    )
    common = SimpleNamespace(pcp_context=context)

    def build_base(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        return SimpleNamespace(pcp_context=common_attn_metadata.pcp_context)

    def capture_base(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)

    with (
        patch.object(kda_pcp.AscendGDNHostMetadataBuilder, "build", build_base),
        patch.object(kda_pcp.AscendGDNHostMetadataBuilder, "build_for_cudagraph_capture", capture_base),
    ):
        if for_capture:
            result = builder.build_for_cudagraph_capture(common)
        else:
            result = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert result.pcp_restore_idx.tolist() == [0, 2, 1, 0]
    assert result.pcp_local_idx.tolist() == [0, 2]


def test_pcp_fake_ops_keep_shapes_traceable():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        local = torch.empty(7, 32)
        global_tokens = torch.ops.vllm.kda_pcp_gather(local, 2, "kda")
        assert global_tokens.shape == (14, 32)
        assert torch.ops.vllm.kda_pcp_scatter(global_tokens, 2, "kda").shape == local.shape


def test_capture_capacity_counts_global_kda_tokens_without_changing_runner():
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=4),
    )

    def init_metadata(self, spec, layers, vllm_config, device):
        self.decode_cudagraph_max_bs = vllm_config.compilation_config.max_cudagraph_capture_size

    with (
        patch.object(kda_pcp.AscendGDNHostMetadataBuilder, "__init__", init_metadata),
        patch.object(kda_pcp, "get_pcp_group", return_value=SimpleNamespace(rank_in_group=0)),
    ):
        builder = kda_pcp.KimiKDAHostMetadataBuilder(None, [], config, torch.device("cpu"))
    assert builder.decode_cudagraph_max_bs == 8
    assert config.compilation_config.max_cudagraph_capture_size == 4


@pytest.mark.parametrize("rank", range(4))
def test_kimi_head_parameters_and_conv_load_the_same_combined_rank(rank):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    torch.nn.Module.__init__(attention)
    attention.num_heads, attention.head_dim = 8, 4
    attention.hidden_size, attention.projection_size = 8, 32
    attention.conv_size, attention.quant_config = 4, None
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2, prefill_context_parallel_size=2))
    tp = SimpleNamespace(world_size=2, rank_in_group=rank // 2)
    pcp = SimpleNamespace(world_size=2, rank_in_group=rank % 2)
    with parallel_groups(tp, pcp):
        attention._init_pcp_projections(config, "kda")
    attention.A_log.weight_loader(attention.A_log, torch.arange(8).view(1, 1, 8, 1))
    attention.dt_bias.weight_loader(attention.dt_bias, torch.arange(32))
    torch.testing.assert_close(attention.A_log, torch.arange(rank * 2, rank * 2 + 2).float())
    torch.testing.assert_close(attention.dt_bias, torch.arange(rank * 8, rank * 8 + 8).float())
    conv = torch.arange(3 * 32 * 4).float().view(96, 1, 4)
    for shard_id, projection in enumerate(conv.chunk(3)):
        attention.conv1d.weight.weight_loader(attention.conv1d.weight, projection, shard_id)
    expected = torch.cat([projection.chunk(4)[rank] for projection in conv.chunk(3)])
    torch.testing.assert_close(attention.conv1d.weight, expected)
