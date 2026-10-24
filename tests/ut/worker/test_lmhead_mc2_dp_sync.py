# SPDX-License-Identifier: Apache-2.0

"""Exercise synchronized DP lengths through V2 context and MC2 preparation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend import platform, utils
from vllm_ascend.ops.fused_moe import prepare_finalize


@pytest.fixture
def mc2_context(monkeypatch):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
    )
    reserved = torch.zeros(8, dtype=torch.bool)
    monkeypatch.setattr(utils, "lmhead_tp_enable", lambda: True)
    monkeypatch.setattr(platform, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "sync_v2_extra_kwargs", lambda _: None)
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", True)
    monkeypatch.setattr(afc, "get_mc2_mask", lambda: reserved)
    monkeypatch.setattr(afc, "get_mrv2_in_profile_run", lambda: False)
    monkeypatch.setattr(afc, "select_moe_comm_method", lambda *_, **__: afc.MoECommType.MC2)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)
    monkeypatch.setattr(afc, "_is_decode_only_node", lambda _: True)
    monkeypatch.setattr("vllm.distributed.get_dp_group", lambda: SimpleNamespace(world_size=2))
    monkeypatch.setattr("vllm.distributed.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_comm_method", lambda _: None)
    monkeypatch.setattr("vllm_ascend.quantization.utils.get_dynamic_mx_quant_scale_alg", lambda *_: 0)

    def build(boundaries, *, graph=False, live_gpu_field=False):
        # Eager keeps local input rows; a graph already has the global bucket.
        metadata = SimpleNamespace(query_start_loc=boundaries)
        if live_gpu_field:
            # FIA can expose an additional, dummy-padded query_start_loc.
            metadata.query_start_loc = torch.tensor([0, 8])
            metadata.query_start_loc_gpu = boundaries
        return platform.NPUPlatform.set_additional_forward_context(
            attn_metadata={"layer": metadata},
            vllm_config=config,
            dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([2, 8])),
            num_tokens=8 if graph else 2,
            cudagraph_runtime_mode=CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE,
        )

    return build


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("local_tokens", [0, 2, 8])
@pytest.mark.parametrize("live_gpu_field", [False, True])
def test_dp_max_length_and_mask_survive_tp_split(monkeypatch, mc2_context, graph, local_tokens, live_gpu_field):
    kwargs = mc2_context(torch.tensor([0, local_tokens]), graph=graph, live_gpu_field=live_gpu_field)
    expected = torch.arange(8) < local_tokens
    assert kwargs["padded_num_tokens"] == 8
    torch.testing.assert_close(kwargs["mc2_mask"], expected)
    monkeypatch.setattr(afc, "get_forward_context", lambda: SimpleNamespace(additional_kwargs=kwargs))
    monkeypatch.setattr(prepare_finalize, "get_dynamic_mx_quant_scale_alg", lambda: 0)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_world_size", lambda: 2)
    for rank in (0, 1):
        monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_rank", lambda rank=rank: rank)
        layer = prepare_finalize.PrepareAndFinalizeWithMC2(MagicMock())
        # In eager, the two ranks may start with different row counts.
        rows = 8 if graph else local_tokens
        hidden = torch.randn(rows, 4)
        prepared = layer.prepare(hidden, torch.zeros(rows, 2))
        assert prepared.hidden_states.shape == (4, 4)
        torch.testing.assert_close(prepared.mc2_mask, expected[rank * 4 : (rank + 1) * 4])
        assert int(prepared.mc2_mask.sum()) == max(0, min(4, local_tokens - rank * 4))


def test_graph_mask_depends_on_live_query_boundary(mc2_context):
    # CPU tracing is not an ACL-graph hardware check. It verifies the token
    # count remains a tensor dependency rather than a capture-time Python int.
    graph = torch.jit.trace(lambda boundary: mc2_context(boundary, graph=True)["mc2_mask"], torch.tensor([0, 8]))
    for actual in (2, 0, 8, 3):
        torch.testing.assert_close(graph(torch.tensor([0, actual])), torch.arange(8) < actual)
