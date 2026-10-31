# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.ops.fused_moe.prepare_finalize as prepare_module
import vllm_ascend.ops.fused_moe.routed_experts as routed_module
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("local_rows", [0, 1, 4])
@pytest.mark.parametrize("dp,pcp", [(1, 2), (1, 8), (2, 1), (2, 4)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_hidden_gather_preserves_rows_and_waits(monkeypatch, local_rows, dp, pcp, dtype):
    events = []

    def group(size, name):
        def gather(value, dim):
            events.append((name, value.shape[1]))
            return torch.cat([value] * size)

        return SimpleNamespace(all_gather=gather, reduce_scatter=lambda value, dim: value.chunk(size)[0], size=size)

    dp_group, pcp_group = group(dp, "dp"), group(pcp, "pcp")
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(dp_size=dp, pcp_size=pcp, dp_group=dp_group, is_sequence_parallel=False)
    pcp_rows = 4 * dp + 1 if dp > 1 else 4
    monkeypatch.setattr(
        prepare_module, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_dp=4, max_tokens_across_pcp=pcp_rows)
    )
    monkeypatch.setattr(prepare_module, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(prepare_module, "get_dp_group", lambda: dp_group)
    hidden = torch.arange(local_rows * 14, dtype=dtype).reshape(local_rows, 14)[:, ::2]
    logits = torch.randn(local_rows, 64)

    def launch(value, actual_group):
        assert actual_group is (dp_group if dp > 1 else pcp_group)
        assert value.is_contiguous()
        events.append(("launch", value.shape[1]))
        result = torch.full((value.shape[0] * actual_group.size, value.shape[1]), float("nan"), dtype=dtype)

        def wait():
            events.append(("wait", value.shape[1]))
            result.copy_(torch.cat([value] * actual_group.size))

        return result, SimpleNamespace(wait=wait)

    monkeypatch.setattr(prepare_module, "all_gather_async", launch)

    expected = layer.prepare(hidden, logits)
    events.clear()
    pending = layer.start_pcp_hidden_gather(hidden)
    output = pending(logits)
    assert torch.equal(output.router_logits, expected.router_logits)
    assert torch.equal(output.hidden_states, expected.hidden_states)
    assert torch.equal(layer.finalize(output.hidden_states, False), hidden)
    assert events[0] == ("launch", 7)
    wait_index = events.index(("wait", 7))
    assert all(name not in ("dp", "pcp") for name, _ in events[:wait_index])


@pytest.mark.parametrize("early,profiling", [(False, False), (True, False), (False, True), (True, True)])
def test_routing_consumes_full_gathered_logits(monkeypatch, early, profiling):
    hidden, logits = torch.zeros(1, 7), torch.zeros(1, 64)
    gathered_hidden, gathered_logits = hidden.repeat(4, 1), logits.repeat(4, 1)
    prepared = SimpleNamespace(
        hidden_states=gathered_hidden,
        router_logits=gathered_logits,
        mc2_mask=None,
        padded_hidden_states_shape=None,
        pertoken_scale=None,
    )
    events = []

    def prepare(*args, **kwargs):
        events.append("prepare")
        return prepared

    def select(**kwargs):
        assert events == ["prepare"]
        assert kwargs["hidden_states"] is gathered_hidden
        assert kwargs["router_logits"] is gathered_logits
        assert kwargs["enable_force_load_balance"] == profiling
        events.append("select")
        return torch.ones(4, 4), torch.zeros(4, 4, dtype=torch.int32)

    comm = routed_module.AllGatherCommImpl.__new__(routed_module.AllGatherCommImpl)
    comm.prepare = Mock(side_effect=prepare)
    comm.finalize = Mock(return_value=hidden)
    pending = Mock(side_effect=prepare)
    layer = SimpleNamespace(
        enable_npugraph_ex_static_kernel=False,
        dynamic_eplb=False,
        moe_config=SimpleNamespace(pcp_size=2, dp_size=2, is_sequence_parallel=False),
        router=SimpleNamespace(eplb_state=None),
        quant_type=QuantType.NONE,
        _select_experts=Mock(side_effect=select),
        quant_method=SimpleNamespace(
            apply=Mock(return_value=SimpleNamespace(routed_out=hidden, expert_tokens=None, group_list_type=1))
        ),
        _use_v2_model_runner=True,
        return_with_event=False,
    )
    monkeypatch.setattr(routed_module, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(routed_module, "get_moe_comm_method", lambda *_: comm)
    monkeypatch.setattr(routed_module, "_EXTRA_CTX", SimpleNamespace(in_profile_run=profiling, moe_comm_type=None))
    result = routed_module.AscendRoutedExperts.forward_impl(
        layer,
        hidden_states=hidden,
        router_logits=logits,
        pcp_hidden_prepare=(comm, pending) if early else None,
    )
    assert result is hidden and events == ["prepare", "select"]
    assert comm.prepare.call_count == int(not early)
    assert pending.call_count == int(early)
    layer._select_experts.assert_called_once()


@pytest.mark.parametrize(
    "mode,sp", [(CUDAGraphMode.NONE, False), (CUDAGraphMode.FULL, False), (CUDAGraphMode.NONE, True)]
)
def test_early_hidden_uses_runtime_graph_mode_not_model_allowlist(monkeypatch, mode, sp):
    layer = SimpleNamespace(moe_config=SimpleNamespace(dp_size=2, pcp_size=2, is_sequence_parallel=sp))
    comm = routed_module.AllGatherCommImpl.__new__(routed_module.AllGatherCommImpl)
    comm.prepare_finalize = SimpleNamespace(
        start_pcp_hidden_gather=Mock(return_value="pending"), is_large_payload=lambda byte_count: True
    )
    monkeypatch.setattr(routed_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    monkeypatch.setattr(routed_module, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=None))
    monkeypatch.setattr(routed_module, "get_moe_comm_method", lambda *_: comm)
    result = routed_module.AscendRoutedExperts.start_pcp_hidden_gather(layer, torch.zeros(1, 7))
    assert result == ((comm, "pending") if mode == CUDAGraphMode.NONE and not sp else None)


@pytest.mark.parametrize("dp,pcp", [(1, 8), (2, 2), (8, 1)])
@pytest.mark.parametrize(
    "rows,width,enabled", [(32, 4096, False), (96, 4096, True), (1024, 128, False), (3072, 128, True)]
)
def test_hidden_communication_budget_scales_with_payload(monkeypatch, dp, pcp, rows, width, enabled):
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(dp_size=dp, pcp_size=pcp, is_sequence_parallel=False)
    monkeypatch.setattr(
        prepare_module,
        "_EXTRA_CTX",
        SimpleNamespace(max_tokens_across_dp=rows, max_tokens_across_pcp=rows),
    )
    assert layer.is_large_payload(width * 2) is enabled


def test_communication_budget_boundary(monkeypatch):
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(dp_size=1, pcp_size=2, is_sequence_parallel=False)
    monkeypatch.setattr(prepare_module, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_pcp=768))
    assert layer.is_large_payload(1024)
    assert not layer.is_large_payload(1023)
