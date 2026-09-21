# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual preparation and graph counters with emulated TP/DP gathers."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.eplb.diagnostics.probe import ExpertLoadProbe
from vllm_ascend.ops import register_custom_ops as custom_ops
from vllm_ascend.ops.fused_moe import prepare_finalize as prepare


@pytest.mark.parametrize("backend", ["All2All", "MC2", "AllGather"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("sp", [False, True])
def test_prepared_source_routes(monkeypatch, backend, tp_size, sp):
    _check_routes(monkeypatch, backend, tp_size, sp, "local_sizes")


@pytest.mark.parametrize("layout", ["compact", "uniform"])
def test_allgather_sp_layouts(monkeypatch, layout):
    _check_routes(monkeypatch, "AllGather", 4, True, layout)


def _check_routes(monkeypatch, backend, tp_size, sp, layout):
    torch.npu.set_device(0)
    # First DP worker is idle. Unequal non-SP lengths exercise tensor_split;
    # SP shards include tail padding. No padding route may become real work.
    real_tokens = [0, 3, 7] if layout != "uniform" else [3]
    padded_tokens = [max(1, n) + 2 for n in real_tokens]
    if sp:
        padded_tokens = [(n + tp_size - 1) // tp_size * tp_size for n in padded_tokens]
    if layout == "compact":
        padded_tokens = [max(padded_tokens)] * len(real_tokens)
    inputs = [
        torch.nn.functional.one_hot((torch.arange(n, device="npu") + dp) % 4, 4).float()
        + torch.arange(4, device="npu") * 0.1
        for dp, n in enumerate(padded_tokens)
    ]
    shards = [chunk for value in inputs for chunk in torch.tensor_split(value, tp_size)]
    max_tokens = max(padded_tokens)
    max_shard = max(chunk.shape[0] for chunk in shards)
    gathered_dp = torch.cat([torch.nn.functional.pad(x, (0, 0, 0, max_tokens - x.shape[0])) for x in inputs])
    gathered_ep = torch.cat([torch.nn.functional.pad(x, (0, 0, 0, max_shard - x.shape[0])) for x in shards])
    local_sizes = [x.shape[0] for x in shards]
    metadata = SimpleNamespace(
        get_chunk_sizes_across_dp_rank=lambda: local_sizes if layout == "local_sizes" else None,
        num_tokens_across_dp_cpu=torch.tensor(real_tokens),
    )
    context = SimpleNamespace(dp_metadata=None if layout == "uniform" else metadata)
    extra = SimpleNamespace(max_tokens_across_dp=max_tokens, padded_num_tokens=max_tokens, padded_length=max_tokens)
    monkeypatch.setattr(prepare, "_EXTRA_CTX", extra)
    monkeypatch.setattr(custom_ops, "_EXTRA_CTX", extra)
    monkeypatch.setattr(prepare, "get_forward_context", lambda: context)
    monkeypatch.setattr(custom_ops, "get_forward_context", lambda: context)
    monkeypatch.setattr(prepare, "get_dynamic_mx_quant_scale_alg", lambda: 0)
    monkeypatch.setattr(prepare, "get_tensor_model_parallel_world_size", lambda: tp_size)
    # Use the real EP gather/unpad implementation, with only transport emulated.
    monkeypatch.setattr(
        torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", custom_ops._maybe_all_gather_and_maybe_unpad_impl
    )
    total = torch.zeros(4, dtype=torch.int64)
    for dp, count in enumerate(real_tokens):
        source_total = torch.zeros(4, dtype=torch.int64)
        dp_group = SimpleNamespace(rank_in_group=dp, world_size=len(real_tokens), all_gather=lambda x, dim: gathered_dp)
        monkeypatch.setattr(prepare, "get_dp_group", lambda group=dp_group: group)
        monkeypatch.setattr(custom_ops, "get_dp_group", lambda group=dp_group: group)
        config = SimpleNamespace(dp_size=len(real_tokens), pcp_size=1, dp_group=dp_group, is_sequence_parallel=sp)
        for tp in range(tp_size):
            rank = dp * tp_size + tp
            ep_group = SimpleNamespace(
                rank_in_group=rank, world_size=len(shards), all_gather=lambda x, dim: gathered_ep
            )
            monkeypatch.setattr(prepare, "get_ep_group", lambda group=ep_group: group)
            monkeypatch.setattr(custom_ops, "get_ep_group", lambda group=ep_group: group)
            monkeypatch.setattr(prepare, "get_tensor_model_parallel_rank", lambda rank=tp: rank)
            extra.mc2_mask = torch.arange(max_tokens, device="npu") < count
            impl = getattr(prepare, f"PrepareAndFinalizeWith{backend}")(config)
            value = shards[rank] if sp else inputs[dp]
            output = impl.prepare(value, value, replace_allreduce=sp)
            ids = output.router_logits.topk(2, dim=-1).indices
            routes = impl.diagnostic_source_routes(
                ids, value.shape[0], output.mc2_mask, output.padded_hidden_states_shape
            )
            assert routes is not None
            probe = ExpertLoadProbe(4, "npu")
            probe.source_positions = torch.arange(max_tokens, device="npu")
            probe.source_token_count = torch.tensor(count, device="npu")
            probe.record_routes(*routes)
            if tp_size == 2:
                # Capture source selection too. Replay must honor a changed
                # scheduled-token scalar without rerunning Python selection.
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    probe.record_routes(
                        *impl.diagnostic_source_routes(
                            ids, value.shape[0], output.mc2_mask, output.padded_hidden_states_shape
                        )
                    )
                probe.totals.zero_()
                graph.replay()
                probe.source_token_count.zero_()
                graph.replay()
                calls = 2
            else:
                calls = 1
            counts = probe.totals.cpu()
            assert counts[-2:].tolist() == [calls, 0]
            source_total += counts[:4]
        # Validate each independent DP source, not only a cancelling global sum.
        expected_ids = inputs[dp][:count].topk(2, dim=-1).indices.cpu().flatten()
        expected = torch.bincount(expected_ids, minlength=4)
        assert source_total.tolist() == expected.tolist()
        total += source_total
    assert total.sum().item() == sum(real_tokens) * 2
