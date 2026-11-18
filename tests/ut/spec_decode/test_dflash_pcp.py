# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regressions for replicated DFlash drafts under target PCP."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

import vllm_ascend.models.qwen3_dflash2 as model_module
import vllm_ascend.worker.v2.pcp_manager as pcp_module
import vllm_ascend.worker.v2.spec_decode.dflash.speculator as spec_module
from vllm_ascend.ops.vocab_parallel_embedding import VocabParallelMode
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize("pcp_enabled", [False, True])
@pytest.mark.parametrize("is_profile", [False, True])
def test_global_draft_batch_negotiates_its_own_capacity(monkeypatch, pcp_enabled, is_profile):
    spec = object.__new__(spec_module.AscendDFlashSpeculator)
    spec.pcp_manager = object.__new__(pcp_module.AscendPCPManager) if pcp_enabled else None
    spec.draft_logits = None
    monkeypatch.setattr(pcp_module, "get_pcp_group", lambda: SimpleNamespace(broadcast=lambda tensor, src: tensor))
    num_reqs = 32 if pcp_enabled else 4
    batch = SimpleNamespace(num_reqs=num_reqs, idx_mapping=torch.arange(num_reqs))
    kwargs = {} if vllm_version_is("0.33.0") else {"max_query_len": None}
    sync = DPSyncState(
        num_tokens_across_dp=torch.tensor([8, 8]), uniform_token_count=2, eager=False, num_reqs=4, **kwargs
    )
    monkeypatch.setattr(spec_module, "build_attn_metadata_wrapper", nullcontext)
    received = []

    def parent(self, *args, **kwargs):
        draft_sync = args[11]
        received.append(draft_sync)
        if draft_sync is not None:
            DraftModelSpeculator._build_uniform_batch_dp_sync(self, draft_sync, batch.num_reqs, 2)
        return torch.zeros(batch.num_reqs, 1, dtype=torch.int64)

    monkeypatch.setattr(DFlashSpeculator, "propose", parent)
    result = spec.propose(batch, *[None] * 10, dp_sync_state=sync, is_profile=is_profile)
    assert result.shape == (batch.num_reqs, 1)
    assert received == [None if pcp_enabled else sync]


@pytest.mark.parametrize("pcp_size,tp_size", [(1, 1), (1, 2), (2, 1), (2, 2), (8, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("vocab_range", ["small", "fp32_boundary", "above_fp32_boundary"])
def test_candidate_topk_covers_all_resident_vocab_shards(monkeypatch, pcp_size, tp_size, dtype, vocab_range):
    top_k, shard_size = 2, 4
    world_size = pcp_size * tp_size
    id_offset = {
        "small": 0,
        "fp32_boundary": (1 << 24) - world_size * shard_size,
        "above_fp32_boundary": (1 << 24) + 1,
    }[vocab_range]
    # Keep the final vocabulary row padded, even when it has the largest logit.
    logits = torch.arange(2 * world_size * shard_size, dtype=dtype).reshape(2, -1)
    logits[0] = logits[0].flip(0)
    logits[:, -1] = 1000
    local_rank = world_size - 1
    local_logits = logits[:, local_rank * shard_size :].clone()
    head = SimpleNamespace(
        weight=local_logits.T,
        quant_method=UnquantizedEmbeddingMethod(),
        parallel_mode=VocabParallelMode.PCP_X_TP if pcp_size > 1 else VocabParallelMode.STANDARD,
        num_embeddings_padded=id_offset + world_size * shard_size,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=1, org_vocab_start_index=id_offset + local_rank * shard_size
        ),
    )
    model = SimpleNamespace(
        lm_head=head,
        model=SimpleNamespace(candidate_selector=SimpleNamespace(top_k=top_k)),
        output_multiplier=1.0,
        final_logit_softcapping=None,
    )
    logits[:, -1] = -float("inf")
    shard_values, shard_ids = [], []
    for rank in range(world_size):
        values, ids = logits[:, rank * shard_size : (rank + 1) * shard_size].topk(top_k, dim=-1)
        shard_values.append(values)
        shard_ids.append(ids + id_offset + rank * shard_size)

    calls = []
    packed = dtype != torch.float64 and vocab_range != "above_fp32_boundary"

    def gather(tensor, dim, *, pcp):
        assert dim == (-2 if packed else -1)
        assert tensor.is_contiguous()
        calls.append((pcp, tensor.dtype))
        if packed:
            assert tensor.dtype == torch.float32
            candidates = [torch.stack((v.float(), i.float()), dim=-1) for v, i in zip(shard_values, shard_ids)]
        else:
            candidates = shard_ids if tensor.dtype == torch.int64 else shard_values
        expected_input = candidates[-1] if pcp else torch.cat(candidates[-pcp_size:], dim=dim)
        torch.testing.assert_close(tensor, expected_input, rtol=0, atol=0)
        selected = candidates[-pcp_size:] if pcp else candidates
        return torch.cat(selected, dim=dim)

    monkeypatch.setattr(
        model_module, "get_pcp_group", lambda: SimpleNamespace(all_gather=lambda x, dim: gather(x, dim, pcp=True))
    )
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(model_module, "tensor_model_parallel_all_gather", lambda x, dim: gather(x, dim, pcp=False))
    ids, values = model_module.DFlash2Qwen3ForCausalLM.compute_candidates(model, torch.eye(2, dtype=dtype))
    expected_values, expected_ids = logits.topk(top_k, dim=-1)
    torch.testing.assert_close(ids, expected_ids + id_offset)
    torch.testing.assert_close(values, expected_values.float())
    assert len(calls) == (1 if packed else 2) * (int(pcp_size > 1) + int(tp_size > 1))


@pytest.mark.parametrize("pcp_enabled", [False, True])
@pytest.mark.parametrize("dummy_run", [False, True])
@pytest.mark.parametrize("logits_kind", ["none", "tensor", "cache"] if vllm_version_is("0.33.0") else ["none", "cache"])
def test_completed_drafts_sync_once_across_runner_versions(monkeypatch, pcp_enabled, dummy_run, logits_kind):
    upstream_sync = not vllm_version_is("0.33.0")
    spec = object.__new__(spec_module.AscendDFlashSpeculator)
    spec.pcp_manager = object.__new__(pcp_module.AscendPCPManager) if pcp_enabled else None
    logits_buffer = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2) if logits_kind != "none" else None
    spec.draft_logits = SimpleNamespace(values=logits_buffer) if logits_kind == "cache" else logits_buffer
    initial_logits = logits_buffer.clone() if logits_buffer is not None else None
    batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([4, 1]))
    # Match the real drafter: a full-width row slice of the fixed-step buffer.
    token_buffer = torch.tensor([[9, 8], [7, 6], [5, 4]])
    tokens = token_buffer[:2]
    assert tokens.is_contiguous()
    shared_tokens = torch.tensor([[1, 2], [3, 4]])
    calls = []
    drafted = False

    def parent(self, *args, **kwargs):
        nonlocal drafted
        drafted = True
        return tokens

    def broadcast(tensor, src):
        assert drafted and src == 0
        assert tensor.is_contiguous()
        calls.append(tensor)
        tensor.copy_(shared_tokens if len(calls) == 1 else torch.full((2, 2, 2), 0.5))

    sync_method = pcp_module.AscendPCPManager.sync_drafts
    if upstream_sync:
        assert "sync_drafts" not in pcp_module.AscendPCPManager.__dict__
        assert sync_method is pcp_module.PCPManager.sync_drafts
    else:
        assert "sync_drafts" in pcp_module.AscendPCPManager.__dict__
    monkeypatch.setitem(sync_method.__globals__, "get_pcp_group", lambda: SimpleNamespace(broadcast=broadcast))
    monkeypatch.setattr(spec_module, "build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(DFlashSpeculator, "propose", parent)
    result = spec.propose(batch, *[None] * 10, dummy_run=dummy_run)
    broadcasts = 1 + int(logits_buffer is not None)
    fallback_sync = pcp_enabled and not upstream_sync and not dummy_run
    assert len(calls) == (broadcasts if fallback_sync else 0)
    if pcp_enabled and upstream_sync and not dummy_run:
        # ModelRunner ignores the return value and retains this proposal view.
        assert spec.pcp_manager.sync_drafts(result, batch.idx_mapping, spec.draft_logits) is None
    needs_sync = pcp_enabled and not dummy_run
    assert len(calls) == (broadcasts if needs_sync else 0)
    assert result is tokens
    assert result.is_contiguous()
    if needs_sync:
        assert calls[0] is tokens
    torch.testing.assert_close(token_buffer[2], torch.tensor([5, 4]))
    torch.testing.assert_close(result, shared_tokens if needs_sync else torch.tensor([[9, 8], [7, 6]]))
    if logits_buffer is not None:
        if needs_sync:
            initial_logits[batch.idx_mapping] = 0.5
        torch.testing.assert_close(logits_buffer, initial_logits)
