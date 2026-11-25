# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise replay coordination with real DP, PCP and TP collectives on CPU."""

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import vllm.v1.utils as vllm_v1_utils
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import decoder_replay_layers as layers_mod
from vllm_ascend.models.deepseek_v41 import decoder_replay_state as replay_mod
from vllm_ascend.models.deepseek_v41.decoder_replay_layers import AscendDecoderReplayLayers

PCP_SIZE = 2
SEGMENT = 4
LOCAL_WIDTH = 2 * SEGMENT


@dataclass
class ReplayBatch:
    num_reqs: int
    num_tokens: int
    num_tokens_after_padding: int
    num_scheduled_tokens: np.ndarray
    num_computed_tokens_np: np.ndarray
    query_start_loc_np: np.ndarray
    query_start_loc: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    seq_lens_np: np.ndarray
    is_prefilling_np: np.ndarray
    idx_mapping_np: np.ndarray
    max_query_len: int
    is_dummy: bool = False

    @property
    def seq_lens(self):
        return torch.from_numpy(self.seq_lens_np)

    @property
    def is_padding(self):
        return torch.arange(self.num_tokens_after_padding) >= self.num_tokens


@dataclass
class ReplayPCPContext:
    global_batch: ReplayBatch
    global_slot_mappings: torch.Tensor
    local_token_indices: torch.Tensor
    hidden_restore_idx: torch.Tensor | None = None
    padded_gather_idx: torch.Tensor | None = None
    gathered_kv_write_mask: torch.Tensor | None = None


class PCPGroup:
    world_size = PCP_SIZE

    def __init__(self, group, rank):
        self.group = group
        self.rank_in_group = rank

    def all_gather(self, tensor, dim=0):
        outputs = [torch.empty_like(tensor) for _ in range(PCP_SIZE)]
        dist.all_gather(outputs, tensor, group=self.group)
        return torch.cat(outputs, dim=dim)


def _batch(segments, width, prefill, idle=False):
    # In PCP prefill a request can have two non-adjacent local segments that
    # map to the same request state. Offsets index the packed local batch.
    starts, ends = np.array(segments, dtype=np.int32).T
    lens = ends - starts
    offsets = np.insert(np.cumsum(lens, dtype=np.int32), 0, 0)
    positions = torch.zeros(width, dtype=torch.long)
    positions[: offsets[-1]] = torch.cat([torch.arange(begin, end) for begin, end in segments])
    return ReplayBatch(
        num_reqs=len(lens),
        num_tokens=int(offsets[-1]),
        num_tokens_after_padding=width,
        num_scheduled_tokens=lens,
        num_computed_tokens_np=starts,
        query_start_loc_np=offsets,
        query_start_loc=torch.from_numpy(offsets),
        input_ids=positions + 10,
        positions=positions,
        seq_lens_np=ends,
        is_prefilling_np=np.full(len(lens), prefill),
        idx_mapping_np=np.zeros(len(lens), dtype=np.int32),
        max_query_len=int(lens.max()),
        is_dummy=idle,
    )


def _gather_cpu(
    query, dropped, seq_lens, floor, pos, slots, rows, offsets, out_pos, out_slots, starts, width, window, **_
):
    kept_offsets = query - dropped
    kept = kept_offsets.diff()
    valid_rows = torch.cat([torch.arange(end - count, end) for end, count in zip(query[1:], kept)])
    rows[:width].zero_()
    rows[: len(valid_rows)].copy_(valid_rows)
    offsets[: len(query)].copy_(kept_offsets)
    out_pos[:width].copy_(pos[rows[:width]])
    out_slots[:, :width].fill_(-1)
    out_slots[:, : len(valid_rows)].copy_(slots[:, valid_rows])
    history = torch.where(kept < query.diff(), 0, window - 1)
    starts[: len(kept)].copy_(torch.maximum((seq_lens - kept - history).clamp(min=0), floor))


def _exercise_batch(state, pcp, dp, tp, window, scenario):
    prefill = dp.rank_in_group == 0 or scenario == "both-trim"
    idle = not prefill and scenario == "idle-peer"
    n = PCP_SIZE * LOCAL_WIDTH
    if prefill:
        first = pcp.rank_in_group * SEGMENT
        segments = [(first, first + SEGMENT), (n - first - SEGMENT, n - first)]
    else:
        segments = [(0, int(not idle and pcp.rank_in_group == 0))]
    batch = _batch(segments, LOCAL_WIDTH, prefill, idle)
    global_batch = _batch([(0, n if prefill else 1)], n, prefill, idle)
    slots = pcp.all_gather(batch.positions).view(1, -1)
    ctx = ReplayPCPContext(global_batch, torch.arange(n).view(1, -1), batch.positions[: batch.num_tokens])
    width = ((window + tp.world_size - 1) // tp.world_size) * tp.world_size
    compact = prefill and width < LOCAL_WIDTH
    state.layers.window = window
    state.prepare(
        replay_mod.DecoderReplayInputs(batch, (), slots, [], torch.zeros(batch.num_reqs, dtype=torch.int32), ctx),
        CUDAGraphMode.NONE,
    )
    counts = [width if d == 0 or scenario == "both-trim" else LOCAL_WIDTH for d in range(dp.world_size)]
    assert (state.layers.context_factory is not None) == (width < LOCAL_WIDTH)
    assert (state.layers.rows is not None) == compact
    if dp.world_size > 1:
        assert state.dp_tokens.tolist() == counts
    rows = torch.where(batch.positions >= n - window)[0] if compact else torch.arange(LOCAL_WIDTH)
    if compact:
        assert torch.equal(state.layers.rows, rows)
        assert state.num_actual_tokens == len(rows)
        restored = pcp.all_gather(state.positions[:width])[state.replay_pcp_context.hidden_restore_idx]
        assert restored.tolist() == list(range(n - window, n))
        assert torch.equal(state.slots[0, : len(rows)], batch.positions[rows])
        assert (state.slots[0, len(rows) : width] == -1).all()
    else:
        assert state.replay_pcp_context is None

    # Compare the actual decoder's inputs and restored SP outputs with a
    # reference in the original PCP-local token coordinates.
    shard = slice(
        tp.rank_in_group * (LOCAL_WIDTH // tp.world_size), (tp.rank_in_group + 1) * (LOCAL_WIDTH // tp.world_size)
    )
    ids = batch.input_ids[shard]
    hidden, mix = ids.view(-1, 1, 1).float(), ids.view(-1, 1).float()
    outer = SimpleNamespace(
        is_padding=batch.is_padding[shard], attn_metadata={}, slot_mapping={}, additional_kwargs={}, moe_layer_index=0
    )
    install_context = Mock(side_effect=lambda *a, **kw: nullcontext())
    exchange = Mock(wraps=dist.all_to_all_single)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dist, "all_to_all_single", exchange)
        patch.setattr(replay_mod, "get_forward_context", lambda: outer)
        patch.setattr(replay_mod, "set_forward_context", install_context)
        patch.setattr(replay_mod, "get_mc2_mask", lambda: None)
        # The non-SP adapter works on full PCP-local rows and pads with zeros.
        result, result_mix = state.layers(hidden, mix, batch.positions, ids)
    assert exchange.call_count == (2 if compact and tp.world_size > 1 and rows.numel() else 0)
    seen_hidden, seen_mix, seen_pos, seen_ids = state.layers.run_layers.call_args.args
    if compact:
        expected_ids = torch.zeros(width, dtype=ids.dtype)
        expected_ids[: len(rows)] = batch.input_ids[rows]
        expected_pos = torch.zeros(width, dtype=batch.positions.dtype)
        if state.layers.sequence_parallel:
            expected_pos.fill_(batch.positions[0])
        expected_pos[: len(rows)] = batch.positions[rows]
        expected_ids = expected_ids.chunk(tp.world_size)[tp.rank_in_group]
        assert torch.equal(seen_pos, expected_pos)
        assert torch.equal(seen_ids, expected_ids)
        assert torch.equal(seen_hidden.flatten(), expected_ids.float())
        assert torch.equal(seen_mix.flatten(), expected_ids.float())
        assert torch.equal(
            install_context.call_args.kwargs["is_padding"],
            (torch.arange(width) >= len(rows)).chunk(tp.world_size)[tp.rank_in_group],
        )
    else:
        assert torch.equal(seen_ids, ids)
        assert torch.equal(seen_pos, batch.positions)
    selected = torch.isin(torch.arange(LOCAL_WIDTH)[shard], rows)
    assert torch.equal(result.flatten(), (hidden.flatten() + 1) * selected)
    assert torch.equal(result_mix.flatten(), (mix.flatten() + 2) * selected)
    # AllGather MoE consumes DP then PCP gathers after padding to the largest replica.
    values = torch.nn.functional.pad(seen_ids, (0, max(counts) // tp.world_size - seen_ids.numel()))
    dp_outputs = [torch.empty_like(values) for _ in range(dp.world_size)]
    dist.all_gather(dp_outputs, values, group=dp.cpu_group)
    assert pcp.all_gather(torch.cat(dp_outputs)).numel() == len(values) * dp.world_size * PCP_SIZE


def _distributed_replay_worker(rank, init_method, dp_size, tp_size):
    world_size = dp_size * PCP_SIZE * tp_size
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=45)
    )
    try:
        ranks = np.arange(world_size).reshape(dp_size, PCP_SIZE, tp_size)
        d, p, t = np.unravel_index(rank, ranks.shape)
        tp_groups = [[dist.new_group(group.tolist()) for group in replica] for replica in ranks]
        pcp_groups = [[dist.new_group(group.tolist()) for group in replica.T] for replica in ranks]
        dp_groups = [[dist.new_group(group.tolist()) for group in replica.T] for replica in ranks.swapaxes(0, 1)]
        tp = SimpleNamespace(world_size=tp_size, rank_in_group=int(t), device_group=tp_groups[d][p])
        pcp = PCPGroup(pcp_groups[d][t], int(p))
        dp = SimpleNamespace(world_size=dp_size, cpu_group=dp_groups[p][t], rank_in_group=int(d))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(vllm_v1_utils, "PIN_MEMORY", False)
            patch.setattr(
                torch.Tensor,
                "pin_memory",
                Mock(side_effect=AssertionError("CPU replay must not pin accelerator memory")),
            )
            patch.setattr(replay_mod, "get_dp_group", lambda: dp)
            patch.setattr(replay_mod, "get_pcp_group", lambda: pcp)
            patch.setattr(replay_mod, "get_tp_group", lambda: tp)
            patch.setattr(layers_mod, "get_tp_group", lambda: tp)
            patch.setattr(replay_mod, "should_skip_allreduce_across_dp_group", lambda _: False)
            patch.setattr(replay_mod, "should_skip_dp_coordination", lambda: False)
            patch.setattr(replay_mod, "get_mrv2_in_profile_run", lambda: False)
            patch.setattr(replay_mod.envs, "VLLM_MOE_SKIP_PADDING", True)
            patch.setattr(replay_mod, "gather_replay_batch", _gather_cpu)
            patch.setattr(replay_mod, "get_cos_and_sin_dsa", lambda *args, **kwargs: None)
            config = SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=16),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=tp_size, prefill_context_parallel_size=PCP_SIZE, data_parallel_size=dp_size
                ),
            )
            layers = AscendDecoderReplayLayers(3, Mock(side_effect=lambda h, m, p, i: (h + 1, m + 2)), [])
            layers.sequence_parallel = tp_size > 1
            state = replay_mod.DecoderReplayState(config, layers, torch.device("cpu"))
            state.slots = torch.empty(1, 16, dtype=torch.int64)
            # Three kept rows all belong to PCP0. Five span both PCPs. With
            # TP2 seven rounds to eight, so neither replica should compact.
            for window, scenario in (
                (3, "both-trim"),
                (5, "decode-peer"),
                (3, "idle-peer"),
                (7, "both-trim"),
                (8, "both-trim"),
            ):
                _exercise_batch(state, pcp, dp, tp, window, scenario)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="fork inherits the CPU test dependency mocks")
@pytest.mark.parametrize(
    "dp_size,tp_size,pin_memory_available", [(2, 1, False), (2, 1, True), (1, 2, False), (2, 2, True)]
)
def test_pcp_dp_replay_collectives(tmp_path, monkeypatch, dp_size, tp_size, pin_memory_available):
    monkeypatch.setattr(vllm_v1_utils, "PIN_MEMORY", pin_memory_available)
    mp.start_processes(
        _distributed_replay_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", dp_size, tp_size),
        nprocs=dp_size * PCP_SIZE * tp_size,
        start_method="fork",
    )
