# SPDX-License-Identifier: Apache-2.0
"""DCP ownership and sparse-attention equivalence without NPU dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.models.qwen4_exp.test_gbsa_preparation import load_cpu_preparation
from vllm_ascend.models.qwen4_exp import qsa_cp

ROOT = Path(__file__).resolve().parents[4]


def local_winners(compact, lengths, world, rank, interleave):
    result = torch.full_like(compact, -1)
    result[:, -3:] = compact[:, -3:]  # Packed tail token IDs are global on every rank.
    for row, length in enumerate(lengths.tolist()):
        owned = [g for g in range(length // 4) if g * 4 // interleave % world == rank]
        local_ids = {g: i for i, g in enumerate(owned)}
        selected = [local_ids[g] for g in compact[row, :-3].tolist() if g in local_ids]
        result[row, : len(selected)] = torch.tensor(selected, dtype=result.dtype)
    return result


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [4, 16, 128, 768])
def test_indexer_counts_owned_complete_groups(world, interleave):
    lengths = torch.tensor([0, 1, 3, 4, 5, 127, 128, 129, 767, 768, 769, 2047, 2048, 4099])
    metadata = SimpleNamespace(
        num_actual_tokens=len(lengths),
        visible_blocks=lengths // 4,
        token_to_req=torch.zeros_like(lengths),
        block_table=torch.tensor([[1]], dtype=torch.int32),
        dcp_size=world,
        dcp_interleave=interleave,
    )
    for rank in range(world):
        metadata.dcp_rank = rank
        _, actual, _ = qsa_cp.qsa_dcp_indexer_inputs(metadata, 4)
        expected = [sum(g * 4 // interleave % world == rank for g in range(n // 4)) for n in lengths.tolist()]
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int32))


@pytest.mark.parametrize("world,interleave", [(2, 4), (2, 16), (2, 128), (3, 12), (4, 4), (4, 16), (4, 128)])
@pytest.mark.parametrize("groups", [2, 511, 512])
@pytest.mark.parametrize("tail", range(4))
def test_local_selections_cover_all_complete_groups_and_tail(world, interleave, groups, tail):
    lengths = torch.tensor([groups * 4 + tail, 3, 0, groups * 4 + tail], dtype=torch.int32)
    positions = lengths - 1
    requests = torch.arange(len(lengths))
    compact = torch.full((len(lengths), groups + 3), -1, dtype=torch.int32)
    compact[0, :groups] = torch.randperm(groups, generator=torch.Generator().manual_seed(7))
    for row, length in enumerate(lengths.tolist()):
        compact[row, groups : groups + length % 4] = torch.arange(length // 4 * 4, length)
    compact[3] = compact[0]
    compact[3, :groups:3] = -1
    compact[3, 1:groups:5] = groups + 1
    table = torch.tensor([[3, 7], [5, 9], [1, 4], [2, 6]], dtype=torch.int32)
    prepare = load_cpu_preparation()
    union: list[list[int]] = [[] for _ in lengths]
    for rank in range(world):
        winners = local_winners(compact, lengths, world, rank, interleave)
        indices, counts, visible, *_ = prepare(
            winners, positions, lengths, requests, table, 2, dcp_size=world, dcp_rank=rank, dcp_interleave=interleave
        )
        assert visible.tolist() == [sum(t // interleave % world == rank for t in range(n)) for n in lengths.tolist()]
        for row in range(len(lengths)):
            local = indices[0, row, : counts[0, row]]
            full_groups = winners[row, :-3]
            assert local.tolist() == full_groups[full_groups >= 0].tolist()
            local_tokens = [4 * g + t for g in local.tolist() for t in range(4)]
            end = int(visible[row])
            local_tokens.extend(range(end // 4 * 4, end))
            owned = [t for t in range(int(lengths[row])) if t // interleave % world == rank]
            union[row].extend(owned[t] for t in local_tokens)
            assert (indices[:, row, counts[0, row] :] == -1).all()
            torch.testing.assert_close(indices[0, row], indices[1, row])
    for row in range(len(lengths)):
        end = int(lengths[row])
        expected = [4 * g + t for g in compact[row, :-3].tolist() if 0 <= g < end // 4 for t in range(4)]
        assert sorted(union[row]) == sorted(expected + list(range(end // 4 * 4, end)))


@pytest.mark.parametrize("tail", [0, 2])
def test_dcp_mtp_reuses_step_zero_selection_and_ignores_padding(tail):
    # Draft positions advance through the next group boundary. The padded row aliases the same request.
    positions = torch.tensor([2047 + tail, 2050, 2051, -1], dtype=torch.int32)
    lengths = torch.tensor([2052], dtype=torch.int32)
    compact = torch.full((len(positions), 515), -1, dtype=torch.int32)
    compact[:, :512] = torch.arange(512)
    compact[:, 512 : 512 + tail] = torch.arange(2048, 2048 + tail)
    requests = torch.zeros_like(positions)
    table = torch.zeros((1, 9), dtype=torch.int32)
    prepare = load_cpu_preparation()
    for rank in range(2):
        winners = local_winners(compact, lengths.expand_as(positions), 2, rank, 4)
        indices, counts, visible, *_ = prepare(
            winners, positions, lengths, requests, table, 1, dcp_size=2, dcp_rank=rank, dcp_interleave=4
        )
        assert counts.tolist() == [[256, 256, 256, 0]]
        torch.testing.assert_close(indices[0, :3, :256], torch.arange(256, dtype=torch.int32).expand(3, -1))
        owned = sum(token // 4 % 2 == rank for token in range(2048 + tail))
        # An aligned step-0 has no residual. Its explicit group IDs still exclude the next complete group.
        expected = [owned] * 3 if tail else [1024, 1024, 1024 + (4 if rank == 0 else 0)]
        assert visible.tolist() == [*expected, 0]


def test_indexer_visibility_is_per_query_not_local_suffix():
    metadata = SimpleNamespace(
        num_actual_tokens=8,
        visible_blocks=torch.tensor([0, 1, 1, 2, 2, 4, 4, 0]),
        token_to_req=torch.tensor([0, 0, 0, 0, 0, 1, 1, 2]),
        block_table=torch.tensor([[3, 5], [7, 9], [0, 0]], dtype=torch.int32),
        dcp_size=2,
        dcp_rank=1,
        dcp_interleave=4,
    )
    cu, lengths, table = qsa_cp.qsa_dcp_indexer_inputs(metadata, 4)
    assert cu.tolist() == list(range(9))
    assert lengths.tolist() == [0, 0, 0, 1, 1, 2, 2, 0]
    torch.testing.assert_close(table, metadata.block_table[metadata.token_to_req])


@pytest.mark.parametrize(
    "tp,kv,world,interleave,pcp,valid",
    [
        (8, 2, 4, 4, 1, True),
        (16, 2, 4, 128, 1, True),
        (8, 2, 8, 4, 1, False),
        (8, 2, 4, 1, 1, False),
        (8, 2, 4, 6, 1, False),
        (8, 2, 4, 4, 2, False),
    ],
)
def test_dcp_config_preserves_kv_replication_groups(tp, kv, world, interleave, pcp, valid):
    parallel = SimpleNamespace(
        tensor_parallel_size=tp,
        decode_context_parallel_size=world,
        cp_kv_cache_interleave_size=interleave,
        prefill_context_parallel_size=pcp,
        _allow_auto_resolve_cp_interleave_size=False,
    )
    if valid:
        qsa_cp.configure_qsa_dcp(parallel, kv, 4)
    else:
        with pytest.raises((ValueError, NotImplementedError)):
            qsa_cp.configure_qsa_dcp(parallel, kv, 4)


def test_dcp_rejects_compression_incompatible_with_gbsa():
    parallel = SimpleNamespace(decode_context_parallel_size=2)
    with pytest.raises(ValueError, match="compression ratio 4"):
        qsa_cp.configure_qsa_dcp(parallel, 1, 2)


@pytest.mark.parametrize("world,interleave,length", [(2, 4, 9), (4, 16, 3), (4, 128, 2051)])
def test_partial_attention_lse_merge_matches_unsharded_qsa(world, interleave, length):
    source = ROOT / "vllm_ascend/models/qwen4_exp/gbsa.py"
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    generator = torch.Generator().manual_seed(91)
    full_key = torch.randn(length, 1, 128, generator=generator).bfloat16()
    full_value = torch.randn(length, 1, 128, generator=generator).bfloat16()
    query = torch.randn(1, 4, 128, generator=generator).bfloat16()
    groups = min(512, length // 4)
    compact = torch.full((1, 515), -1, dtype=torch.int32)
    compact[0, :groups] = torch.randperm(groups, generator=generator)
    compact[0, 512 : 512 + length % 4] = torch.arange(length // 4 * 4, length)
    lengths = torch.tensor([length], dtype=torch.int32)
    positions = lengths - 1
    requests = torch.tensor([0], dtype=torch.int32)
    tokens = [4 * g + t for g in compact[0, :groups].tolist() for t in range(4)] + list(range(length // 4 * 4, length))
    logits = torch.einsum("hd,shd->hs", query[0].float(), full_key[tokens].float().expand(-1, 4, -1)) / 128**0.5
    expected = torch.einsum("hs,shd->hd", logits.softmax(-1), full_value[tokens].float().expand(-1, 4, -1))

    kernel_output = torch.empty_like(query)

    def attention(q, k, v, indices, count, block_shape, **kwargs):
        assert kwargs["return_softmax_lse"]
        assert kwargs["residual_block_mode"] == 1
        assert block_shape == [1, 4]
        visible = int(kwargs["seqused_kv"][0])
        local_tokens = [t for g in indices[0, 0, : count[0, 0]].tolist() for t in range(g * 4, min(g * 4 + 4, visible))]
        local_tokens.extend(range(visible // 4 * 4, visible))
        if not local_tokens:
            # Exercise explicit neutralization of unspecified empty outputs.
            return kernel_output.fill_(torch.nan), torch.full((*q.shape[:2], 1), torch.nan)
        t = torch.tensor(local_tokens)
        pages = kwargs["block_table"][0, t // 128].long()
        local_k = k[pages, t % 128].float().expand(-1, q.shape[1], -1)
        local_v = v[pages, t % 128].float().expand_as(local_k)
        scores = torch.einsum("hd,shd->hs", q[0].float(), local_k) * kwargs["softmax_scale"]
        result = torch.einsum("hs,shd->hd", scores.softmax(-1), local_v).unsqueeze(0).to(q.dtype)
        return kernel_output.copy_(result), scores.logsumexp(-1).reshape(1, -1, 1)

    namespace = {
        "torch": torch,
        "prepare_qsa_gbsa_inputs": load_cpu_preparation(),
        "generic_block_sparse_attention_metadata": lambda *args, **kwargs: torch.empty(0),
        "generic_block_sparse_attention": attention,
    }
    exec(compile(module, str(source), "exec"), namespace)
    partials, lses = [], []
    for rank in range(world):
        owned = torch.tensor([t for t in range(length) if t // interleave % world == rank], dtype=torch.long)
        pages = max(1, (owned.numel() + 127) // 128)
        table = torch.arange(pages - 1, -1, -1, dtype=torch.int32).view(1, -1)
        # Descriptor-backed pages need not be adjacent in storage.
        key = torch.zeros(pages * 2, 128, 1, 128, dtype=torch.bfloat16)[::2]
        value = torch.zeros_like(key)
        local = torch.arange(owned.numel())
        key[table[0, local // 128].long(), local % 128] = full_key[owned]
        value[table[0, local // 128].long(), local % 128] = full_value[owned]
        partial, lse = namespace["qsa_gbsa_local_attention"](
            query,
            key,
            value,
            local_winners(compact, lengths, world, rank, interleave),
            table,
            requests,
            positions,
            lengths,
            dcp_size=world,
            dcp_rank=rank,
            dcp_interleave=interleave,
        )
        assert partial is kernel_output
        partials.append(partial.float())
        lses.append(lse)
    lse = torch.stack(lses)
    merged = (torch.stack(partials) * (lse - lse.logsumexp(0)).exp()).sum(0)
    torch.testing.assert_close(merged[0], expected, atol=0.012, rtol=0.012)
