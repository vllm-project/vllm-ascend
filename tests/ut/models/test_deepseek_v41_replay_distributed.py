# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise replay coordination with real DP/PCP collectives on four CPUs."""

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import decoder_replay_layers as layers_mod
from vllm_ascend.models.deepseek_v41 import decoder_replay_state as replay_mod
from vllm_ascend.models.deepseek_v41.decoder_replay_layers import AscendDecoderReplayLayers

PCP_SIZE = 2
DP_SIZE = 2
LOCAL_WIDTH = 4
WINDOW = 2


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


def _distributed_replay_worker(rank, init_method):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=DP_SIZE * PCP_SIZE,
        timeout=timedelta(seconds=30),
    )
    dp_rank, pcp_rank = divmod(rank, PCP_SIZE)
    pcp_groups = [dist.new_group([d * PCP_SIZE + p for p in range(PCP_SIZE)]) for d in range(DP_SIZE)]
    dp_groups = [dist.new_group([d * PCP_SIZE + p for d in range(DP_SIZE)]) for p in range(PCP_SIZE)]
    pcp = PCPGroup(pcp_groups[dp_rank], pcp_rank)
    dp = SimpleNamespace(cpu_group=dp_groups[pcp_rank], rank_in_group=dp_rank)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(replay_mod, "get_dp_group", lambda: dp)
            patch.setattr(replay_mod, "get_pcp_group", lambda: pcp)
            patch.setattr(replay_mod, "should_skip_allreduce_across_dp_group", lambda _: False)
            patch.setattr(replay_mod, "should_skip_dp_coordination", lambda: False)
            patch.setattr(replay_mod, "get_mrv2_in_profile_run", lambda: False)
            patch.setattr(replay_mod, "enable_dsa_cp", lambda: False)
            config = SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=16),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=1,
                    prefill_context_parallel_size=PCP_SIZE,
                    data_parallel_size=DP_SIZE,
                ),
            )
            layers = AscendDecoderReplayLayers(WINDOW, lambda *states: states, [])
            state = replay_mod.DecoderReplayState(config, layers, torch.device("cpu"))
            state.slots = torch.empty(1, 16, dtype=torch.int64)

            def gather(inputs, **kwargs):
                # Only replace the device kernel. Decisions, counts, ownership
                # restoration and collectives run through the production code.
                kept = LOCAL_WIDTH - int(state.dropped_before.np[1])
                state.rows[:kept] = torch.arange(LOCAL_WIDTH - kept, LOCAL_WIDTH)

            patch.setattr(state, "_gather", gather)
            patch.setattr(state, "_build_metadata", lambda *args: {})
            for scenario in ("both-trim", "decode-peer", "idle-peer", "no-trim"):
                compact = scenario != "no-trim" and (dp_rank == 0 or scenario == "both-trim")
                idle = dp_rank == 1 and scenario == "idle-peer"
                actual = LOCAL_WIDTH if compact else (1 if pcp_rank == 0 and not idle else 0)
                start = pcp_rank * LOCAL_WIDTH
                batch = ReplayBatch(
                    num_reqs=1,
                    num_tokens=actual,
                    num_tokens_after_padding=LOCAL_WIDTH,
                    num_scheduled_tokens=np.array([actual]),
                    num_computed_tokens_np=np.array([start]),
                    query_start_loc_np=np.array([0, actual], dtype=np.int32),
                    query_start_loc=torch.tensor([0, actual], dtype=torch.int32),
                    input_ids=torch.arange(LOCAL_WIDTH),
                    positions=torch.arange(start, start + LOCAL_WIDTH),
                    seq_lens_np=np.array([start + actual]),
                    is_prefilling_np=np.array([compact]),
                    idx_mapping_np=np.array([0]),
                    max_query_len=actual,
                    is_dummy=idle,
                )
                global_batch = ReplayBatch(
                    num_reqs=1,
                    num_tokens=8 if compact else 1,
                    num_tokens_after_padding=8,
                    num_scheduled_tokens=np.array([8 if compact else 1]),
                    num_computed_tokens_np=np.array([0]),
                    query_start_loc_np=np.array([0, 8 if compact else 1], dtype=np.int32),
                    query_start_loc=torch.tensor([0, 8 if compact else 1], dtype=torch.int32),
                    input_ids=torch.arange(8),
                    positions=torch.arange(8),
                    seq_lens_np=np.array([8]),
                    is_prefilling_np=np.array([compact]),
                    idx_mapping_np=np.array([0]),
                    max_query_len=8,
                )
                context = ReplayPCPContext(global_batch, torch.arange(8).reshape(1, -1), batch.positions)
                state.prepare(
                    replay_mod.DecoderReplayInputs(
                        batch, (), torch.arange(8).reshape(1, -1), [], torch.zeros(1, dtype=torch.int32), context
                    ),
                    CUDAGraphMode.NONE,
                )
                expected_counts = [WINDOW, WINDOW if scenario == "both-trim" else LOCAL_WIDTH]
                if scenario == "no-trim":
                    assert layers.context_factory is None
                    assert layers.rows is None
                    dist.monitored_barrier(timeout=timedelta(seconds=30))
                    continue
                assert state.dp_tokens.tolist() == expected_counts
                assert layers.context_factory is not None
                values = batch.positions.clone()
                if compact:
                    assert layers.rows.numel() == (WINDOW if pcp_rank == 1 else 0)
                    padded = torch.full((WINDOW,), -1, dtype=torch.int64)
                    padded[: state.num_actual_tokens] = values.index_select(0, layers.rows)
                    restored = pcp.all_gather(padded)[state.replay_pcp_context.hidden_restore_idx]
                    assert restored.tolist() == [6, 7]
                    values = padded
                else:
                    assert layers.rows is None
                    assert state.replay_pcp_context is None
                # Exercise the DP-then-PCP gather order used by AllGather MoE.
                width = max(expected_counts)
                padded = torch.nn.functional.pad(values, (0, width - values.numel()))
                dp_outputs = [torch.empty_like(padded) for _ in range(DP_SIZE)]
                dist.all_gather(dp_outputs, padded, group=dp.cpu_group)
                gathered = pcp.all_gather(torch.cat(dp_outputs))
                assert gathered.numel() == width * DP_SIZE * PCP_SIZE
                dist.monitored_barrier(timeout=timedelta(seconds=30))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform != "linux", reason="fork inherits the CPU test dependency mocks")
def test_pcp_dp_replay_collectives(tmp_path):
    mp.start_processes(
        _distributed_replay_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=DP_SIZE * PCP_SIZE,
        start_method="fork",
    )


def _sp_replay_worker(rank, init_method):
    tp_size = 4
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=tp_size, timeout=timedelta(seconds=30)
    )

    def gather(tensor, dim=0):
        outputs = [torch.empty_like(tensor) for _ in range(tp_size)]
        dist.all_gather(outputs, tensor.contiguous())
        return torch.cat(outputs, dim=dim)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(layers_mod, "get_tp_group", lambda: SimpleNamespace(world_size=tp_size, rank_in_group=rank))
            patch.setattr(layers_mod, "tensor_model_parallel_all_gather", gather)
            patch.setattr(layers_mod, "enable_dsa_cp", lambda: False)
            # The original seven rows and the retained suffix both need TP padding.
            original = torch.arange(8).reshape(8, 1)
            for retained in ([], [4, 5, 6], [0, 3, 4, 5, 6]):
                rows = torch.tensor(retained, dtype=torch.int64)
                width = max(tp_size, ((len(retained) + tp_size - 1) // tp_size) * tp_size)
                expected = torch.zeros(width, 1, dtype=torch.int64)
                expected[: len(retained)] = original.index_select(0, rows)

                def run_layers(hidden, mix, positions, ids, expected=expected):
                    assert torch.equal(hidden, expected.chunk(tp_size)[rank])
                    assert torch.equal(mix, hidden)
                    assert torch.equal(ids, hidden[:, 0])
                    assert torch.equal(positions, expected[:, 0])
                    return hidden + 10, mix + 20, gather(hidden) + 30

                buffer = original.clone()
                replay = AscendDecoderReplayLayers(3, run_layers, [buffer])
                replay.sequence_parallel = True
                replay.rows = rows
                replay.num_actual_rows = len(retained)
                replay.num_padded_rows = width
                replay.context_factory = nullcontext
                local = original.chunk(tp_size)[rank]
                output, mix, aux = replay(local, local, torch.arange(7), local[:, 0])
                for result, offset in ((gather(output), 10), (gather(mix), 20), (aux, 30)):
                    restored = torch.zeros_like(result)
                    restored.index_copy_(0, rows, original.index_select(0, rows) + offset)
                    assert torch.equal(result, restored)
                assert aux.shape[0] == 7
                assert torch.equal(buffer[:width], expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform != "linux", reason="fork inherits the CPU test dependency mocks")
def test_sp_replay_gather_pad_shard_and_scatter(tmp_path):
    mp.start_processes(
        _sp_replay_worker,
        args=(f"file://{tmp_path / 'sp-rendezvous'}",),
        nprocs=4,
        start_method="fork",
    )
