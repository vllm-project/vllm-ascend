# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor import custom_op, parameter
from vllm.model_executor.layers import linear

import vllm_ascend.models.glm5next.kda as model_kda
from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM


@pytest.fixture
def parallel_groups(monkeypatch):
    # Exercise inherited vLLM loaders without Ascend's OOT class replacement.
    for name in ("ColumnParallelLinear", "RowParallelLinear"):
        monkeypatch.delitem(custom_op.op_registry_oot, name, raising=False)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter, "get_tensor_model_parallel_world_size", lambda: linear.get_tensor_model_parallel_world_size()
    )
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: linear.get_tensor_model_parallel_rank())
    # Exercise shard loaders and collectives without platform GEMM setup.
    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda backend: lambda layer, x, weight, bias: torch.nn.functional.linear(x, weight, bias),
    )
    group = SimpleNamespace(world_size=4, rank_in_group=2, all_gather=Mock(), all_reduce=Mock())
    monkeypatch.setattr(model_kda, "get_pcp_group", lambda: group)
    return group


@pytest.mark.parametrize("loader_name", ["weight_loader", "weight_loader_v2"])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_pcp_merged_projection_loads_heads_and_replicates_low_rank_shards(parallel_groups, loader_name, rank):
    parallel_groups.rank_in_group = rank
    sizes = [8, 8, 8, 4, 2, 2]
    layer = model_kda.PCPGlm5NextMergedColumnParallelLinear(
        3, sizes, replicated_shard_ids=(4, 5), tp_size=4, bias=False, params_dtype=torch.float32
    )
    weights = [
        torch.arange(size * 3, dtype=torch.float32).reshape(size, 3) + 100 * sid for sid, size in enumerate(sizes)
    ]
    expected = []
    for sid, weight in enumerate(weights):
        getattr(layer, loader_name)(layer.weight, weight, sid)
        expected.append(weight if sid in (4, 5) else weight.chunk(4)[rank])
        assert layer.tp_rank == layer.weight.tp_rank == rank
    x = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(layer(x)[0], x @ torch.cat(expected).T)
    assert layer.tp_size == layer.weight.tp_size == 4
    assert sizes == [8, 8, 8, 4, 2, 2]
    parallel_groups.all_gather.assert_not_called()


@pytest.mark.parametrize("loader_name", ["weight_loader", "weight_loader_v2"])
@pytest.mark.parametrize("gather_output", [False, True])
def test_pcp_column_projection_gathers_with_pcp_group(parallel_groups, loader_name, gather_output):
    layer = model_kda.PCPColumnParallelLinear(3, 8, bias=False, gather_output=gather_output, params_dtype=torch.float32)
    weight = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    getattr(layer, loader_name)(layer.weight, weight)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    local = x @ weight[4:6].T
    gathered = x @ weight.T
    parallel_groups.all_gather.return_value = gathered
    torch.testing.assert_close(layer(x)[0], gathered if gather_output else local)
    if gather_output:
        torch.testing.assert_close(parallel_groups.all_gather.call_args.args[0], local)
        assert parallel_groups.all_gather.call_args.kwargs == {"dim": -1}
    else:
        parallel_groups.all_gather.assert_not_called()


@pytest.mark.parametrize("loader_name", ["weight_loader", "weight_loader_v2"])
@pytest.mark.parametrize("input_is_parallel", [False, True])
@pytest.mark.parametrize("reduce_results", [False, True])
def test_pcp_output_projection_loads_input_heads_and_reduces_over_pcp(
    parallel_groups, loader_name, input_is_parallel, reduce_results
):
    layer = model_kda.PCPRowParallelLinear(
        8, 3, bias=False, input_is_parallel=input_is_parallel, reduce_results=reduce_results, params_dtype=torch.float32
    )
    weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    getattr(layer, loader_name)(layer.weight, weight)
    torch.testing.assert_close(layer.weight, weight[:, 4:6])
    assert layer.tp_rank == layer.weight.tp_rank == 2
    assert layer.tp_size == layer.weight.tp_size == 4
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    local = x[:, 4:6] @ weight[:, 4:6].T
    full = x @ weight.T
    parallel_groups.all_reduce.return_value = full
    output = layer(x[:, 4:6] if input_is_parallel else x)[0]
    torch.testing.assert_close(output, full if reduce_results else local)
    if reduce_results:
        torch.testing.assert_close(parallel_groups.all_reduce.call_args.args[0], local)
    else:
        parallel_groups.all_reduce.assert_not_called()


@pytest.mark.parametrize(("tp_size", "pcp_size"), [(1, 1), (1, 2), (1, 4), (2, 1), (2, 4), (4, 2)])
def test_kda_projection_and_state_sharding_follow_pcp_only_with_tp_one(monkeypatch, parallel_groups, tp_size, pcp_size):
    # Construct the complete KDA layer to catch mismatched projection, gate,
    # convolution, and cache dimensions, including the TP>1 fallback.
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: tp_size - 1)
    parallel_groups.world_size = pcp_size
    parallel_groups.rank_in_group = pcp_size - 1
    config = SimpleNamespace(
        model_type="glm5_next",
        hidden_size=16,
        linear_head_dim=128,
        linear_num_heads=8,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-4,
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size, prefill_context_parallel_size=pcp_size),
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_config=config, hf_text_config=config),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto", use_replayssm=False),
        speculative_config=None,
        quant_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    def init_base(self, config, vllm_config, prefix):
        torch.nn.Module.__init__(self)
        self.tp_size, self.tp_rank = tp_size, tp_size - 1
        self.hidden_size = config.hidden_size
        self.quant_config = vllm_config.quant_config
        self.num_spec = 0

    monkeypatch.setattr(model_kda.GatedDeltaNetAttention, "__init__", init_base)
    monkeypatch.setattr(model_kda, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr("vllm_ascend.utils.register_ascend_fla_customop", lambda: None)
    monkeypatch.setattr(
        "vllm.third_party.flash_linear_attention.ops.kda.FusedRMSNormGated", lambda *args, **kwargs: torch.nn.Identity()
    )
    layer = model_kda.Glm5NextLinearAttention(config, vllm_config, prefix="model.layers.0.linear_attn")
    assert layer.pcp_group is parallel_groups
    assert layer.pcp_size == pcp_size
    assert layer.pcp_rank == pcp_size - 1
    size = pcp_size if tp_size == 1 else tp_size
    rank = pcp_size - 1 if tp_size == 1 else tp_size - 1
    assert layer.tp_size == size and layer.tp_rank == rank
    assert layer.local_num_heads == 8 // size
    assert layer.local_projection_size == 1024 // size
    assert layer.use_pcp_head_sharding == (tp_size == 1 and pcp_size > 1)
    assert (
        isinstance(layer.in_proj_qkvbfg_a, model_kda.PCPGlm5NextMergedColumnParallelLinear)
        == layer.use_pcp_head_sharding
    )
    assert isinstance(layer.o_proj, model_kda.PCPRowParallelLinear) == layer.use_pcp_head_sharding
    for name in ("f_b_proj", "g_b_proj", "q_conv1d", "k_conv1d", "v_conv1d"):
        projection = getattr(layer, name)
        assert isinstance(projection, model_kda.PCPColumnParallelLinear) == layer.use_pcp_head_sharding
        assert projection.weight.shape[0] == 1024 // size
    assert layer.in_proj_qkvbfg_a.weight.shape[0] == 3 * 1024 // size + 8 // size + 2 * 128
    assert layer.o_proj.weight.shape == (16, 1024 // size)
    assert layer.get_state_shape() == Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)
    assert layer.get_state_shape()[1] == (8 // size, 128, 128)

    dt_bias = torch.arange(1024, dtype=torch.float32)
    layer.dt_bias.weight_loader(layer.dt_bias, dt_bias)
    torch.testing.assert_close(layer.dt_bias, dt_bias.chunk(size)[rank])
    a_log = torch.arange(8, dtype=torch.float32)
    layer.A_log.weight_loader(layer.A_log, a_log)
    torch.testing.assert_close(layer.A_log.flatten(), a_log.chunk(size)[rank])


def test_pcp_head_count_must_be_divisible(monkeypatch, parallel_groups):
    with pytest.raises(AssertionError):
        model_kda.PCPColumnParallelLinear(3, 6, bias=False)
