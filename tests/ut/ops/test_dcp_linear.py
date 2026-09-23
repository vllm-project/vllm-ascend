# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops import dcp_linear
from vllm_ascend.ops.dcp_linear import AscendDCPGroupColumnParallelLinear, use_dcp_q_replicate
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.linear_op import DCPGroupColumnParallelOp


def make_config(requested=True, dcp=2, pcp=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            dcp_q_replicate=requested,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=4,
        ),
        model_config=SimpleNamespace(enforce_eager=True),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        attention_config=SimpleNamespace(indexer_kv_dtype="bf16"),
        speculative_config=None,
        lora_config=None,
    )


@pytest.mark.parametrize(
    "requested,legacy,expected", [(False, None, False), (True, None, True), (False, True, True), (True, False, False)]
)
@pytest.mark.parametrize("dcp,pcp", [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_request_precedence(requested, legacy, expected, dcp, pcp):
    with (
        patch.object(dcp_linear.envs, "is_set", return_value=legacy is not None),
        patch.object(dcp_linear.envs, "VLLM_DCP_Q_REPLICATE", legacy, create=True),
    ):
        assert use_dcp_q_replicate(make_config(requested, dcp, pcp), SimpleNamespace(), None) is (
            expected and dcp > 1 and pcp == 1
        )


@pytest.mark.parametrize("case", ["weights", "kv", "lora"])
def test_unsupported_requests_fail(case):
    cfg, model, quant = make_config(), SimpleNamespace(), None
    if case == "weights":
        quant = object()
    elif case == "kv":
        cfg.cache_config.cache_dtype = "fp8"
    elif case == "lora":
        cfg.lora_config = object()
    with (
        patch.object(dcp_linear.envs, "is_set", return_value=False),
        pytest.raises(ValueError, match="Ascend dcp_q_replicate"),
    ):
        use_dcp_q_replicate(cfg, model, quant)


@pytest.mark.parametrize("q_rank", [None, 4])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
def test_direct_q_graph_and_speculative_requests(q_rank, eager, speculative):
    cfg = make_config()
    cfg.model_config.enforce_eager = eager
    cfg.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=3) if speculative else None
    with (
        patch.object(dcp_linear.envs, "is_set", return_value=False),
    ):
        assert use_dcp_q_replicate(cfg, SimpleNamespace(q_lora_rank=q_rank), None)


@pytest.mark.parametrize("dcp", [2, 4])
@pytest.mark.parametrize("rank", range(4))
def test_group_projection_loads_actual_shards_and_uses_ascend_gemm(rank, dcp):
    cfg = make_config(dcp=dcp)
    group = SimpleNamespace(world_size=4, rank_in_group=rank)
    # Eight heads, three rows per head, distinct weights on every row.
    full_weight = torch.arange(24 * 5, dtype=torch.float32).view(24, 5) / 100
    x = torch.arange(15, dtype=torch.float32).view(3, 5)
    with (
        patch.object(dcp_linear, "get_current_vllm_config", return_value=cfg),
        patch.object(dcp_linear, "get_tensor_model_parallel_rank", return_value=rank),
        patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=group),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=group),
        patch("vllm_ascend.ops.linear._should_reshape_wo_a_to_3d", return_value=False),
        patch("torch.ops.vllm.unquantized_gemm", side_effect=torch.nn.functional.linear, create=True) as gemm,
    ):
        layer = AscendDCPGroupColumnParallelLinear(5, 24, prefix="model.layers.0.self_attn.q_b_proj")
        assert type(layer) is AscendDCPGroupColumnParallelLinear
        assert isinstance(layer.quant_method, AscendUnquantizedLinearMethod)
        assert isinstance(layer.custom_op, DCPGroupColumnParallelOp)
        assert (layer.tp_rank, layer.tp_size) == (rank // dcp, 4 // dcp)
        layer.weight.weight_loader(layer.weight, full_weight)
        start = (rank // dcp) * (6 * dcp)
        torch.testing.assert_close(layer.weight, full_weight[start : start + 6 * dcp])
        output = layer(x)[0].view(3, 2 * dcp, 3)
        expected = torch.nn.functional.linear(x, full_weight).view(3, 8, 3)
        torch.testing.assert_close(output, expected[:, (rank // dcp) * (2 * dcp) : (rank // dcp + 1) * (2 * dcp)])
        torch.testing.assert_close(layer._local_view(output), expected[:, rank * 2 : (rank + 1) * 2])
        gemm.assert_called_once()


@pytest.mark.parametrize("case", ["valid", "direct", "partial_dcp", "indexer_fp8", "indexer_int8"])
def test_sparse_q_replication_scope(case):
    cfg, model = make_config(dcp=4), SimpleNamespace(index_topk=8, q_lora_rank=4)
    if case == "direct":
        model.q_lora_rank = None
    if case == "partial_dcp":
        cfg.parallel_config.decode_context_parallel_size = 2
    if case.startswith("indexer_"):
        cfg.attention_config.indexer_kv_dtype = case.removeprefix("indexer_")
    with (
        patch.object(dcp_linear.envs, "is_set", return_value=False),
    ):
        # Existing native and platform checks own Q-LoRA and topology errors.
        if case in ("valid", "direct", "partial_dcp"):
            assert use_dcp_q_replicate(cfg, model, None)
        else:
            with pytest.raises(ValueError, match="Ascend dcp_q_replicate"):
                use_dcp_q_replicate(cfg, model, None)
