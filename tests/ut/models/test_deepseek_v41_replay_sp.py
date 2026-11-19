# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm.config import CUDAGraphMode
from vllm.v1 import utils as vllm_v1_utils

from vllm_ascend.models.deepseek_v41 import decoder_replay_layers as layers_mod
from vllm_ascend.models.deepseek_v41 import decoder_replay_state as replay_mod
from vllm_ascend.models.deepseek_v41.decoder_replay_sp import SPReplayPlan, _pack_states, _unpack_states


def _kept_rows(lengths, kept):
    rows: list[int] = []
    start = 0
    for length, count in zip(lengths, kept):
        rows.extend(range(start + length - count, start + length))
        start += length
    return torch.tensor(rows, dtype=torch.int64)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens", [0, 1, 3])
@pytest.mark.parametrize("hc_mult,hidden_size", [(1, 3), (4, 8)])
def test_packed_states_preserve_bits_and_contiguous_layout(dtype, num_tokens, hc_mult, hidden_size):
    # Include special FP32 values and low mantissa bits that a BF16 cast loses.
    mix_bits = torch.tensor([0, -2147483648, 0x7FC01234, 0x7F800000, 0x3F800001, 1], dtype=torch.int32)
    mix = mix_bits.repeat(num_tokens * hc_mult)[: num_tokens * hc_mult].view(torch.float32)
    mix = mix.reshape(num_tokens, hc_mult)
    num_bytes = num_tokens * hc_mult * hidden_size * torch.empty((), dtype=dtype).element_size()
    hidden = (torch.arange(num_bytes) * 37).to(torch.uint8).view(dtype)
    hidden = hidden.reshape(num_tokens, hc_mult, hidden_size)
    packed = _pack_states(hidden, mix)
    actual_hidden, actual_mix = _unpack_states(packed, hidden, mix)
    assert packed.dtype == torch.int8
    assert packed.numel() == num_bytes + mix.numel() * mix.element_size()
    for actual, expected in ((actual_hidden, hidden), (actual_mix, mix)):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.is_contiguous()
        assert torch.equal(actual.view(torch.int8), expected.view(torch.int8))


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_sp_replay_plan_matches_global_selection(tp_size):
    rng = np.random.default_rng(42)
    for _ in range(100):
        lengths = rng.integers(1, 150, size=7)
        kept = np.minimum(lengths, rng.integers(1, 130, size=7))
        offsets = np.insert(np.cumsum(lengths), 0, 0)
        original_tokens = int(lengths.sum()) + int(rng.integers(0, tp_size * 3))
        original_width = (original_tokens + tp_size - 1) // tp_size
        replay_width = (int(kept.sum()) + tp_size - 1) // tp_size + int(rng.integers(0, 4))
        selected = _kept_rows(lengths, kept)
        padded_rows = torch.nn.functional.pad(selected, (0, replay_width * tp_size - len(selected)))
        plans = [
            SPReplayPlan.build(offsets, kept, original_tokens, replay_width * tp_size, tp_size, rank, padded_rows)
            for rank in range(tp_size)
        ]
        source_values = [plan.local_rows + rank * original_width for rank, plan in enumerate(plans)]
        received: list[list[torch.Tensor]] = [[] for _ in plans]
        for source, (plan, values) in enumerate(zip(plans, source_values)):
            for destination, part in enumerate(values.split(plan.send_counts)):
                assert part.numel() == plans[destination].recv_counts[source]
                received[destination].append(part)
        for destination, parts in enumerate(received):
            assert torch.equal(
                torch.cat(parts), selected[destination * replay_width : (destination + 1) * replay_width]
            )
        # Every retained row is sent exactly once; no discarded or padded rows.
        assert sum(sum(plan.send_counts) for plan in plans) == selected.numel()


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("original_padding", [0, 3])
@pytest.mark.parametrize(
    "lengths,kept",
    [([0, 8, 0, 7, 0], [0, 2, 0, 3, 0]), ([8, 8], [8, 8]), ([16], [1])],
)
def test_sp_plan_interval_boundaries_and_padding(tp_size, lengths, kept, original_padding):
    offsets = np.insert(np.cumsum(lengths), 0, 0)
    original_tokens = sum(lengths) + original_padding
    original_width = (original_tokens + tp_size - 1) // tp_size
    replay_width = (sum(kept) + tp_size - 1) // tp_size + 2
    rows = _kept_rows(lengths, kept)
    padded_rows = torch.nn.functional.pad(rows, (0, replay_width * tp_size - len(rows)))
    for rank in range(tp_size):
        plan = SPReplayPlan.build(
            offsets, np.array(kept), original_tokens, replay_width * tp_size, tp_size, rank, padded_rows
        )
        owned = (rows >= rank * original_width) & (rows < (rank + 1) * original_width)
        torch.testing.assert_close(plan.local_rows, rows[owned] - rank * original_width)
        expected_send = torch.bincount(torch.arange(len(rows))[owned] // replay_width, minlength=tp_size)
        received = rows[rank * replay_width : (rank + 1) * replay_width]
        expected_recv = torch.bincount(received // original_width, minlength=tp_size)
        assert plan.send_counts == expected_send.tolist()
        assert plan.recv_counts == expected_recv.tolist()
        assert plan.received_tokens == len(received)


def test_sp_plan_reuses_device_rows_without_host_token_expansion(monkeypatch):
    # A meta tensor represents an existing, very large device index buffer.
    # Planning must not materialize those tokens on CPU or read device values.
    offsets = np.array([0, 2**35, 2**35 + 3], dtype=np.int64)
    kept = np.array([2**34, 3], dtype=np.int64)
    tp_size = 8
    replay_tokens = ((int(kept.sum()) + tp_size - 1) // tp_size) * tp_size
    rows = torch.empty(replay_tokens, dtype=torch.int64, device="meta")
    forbidden = Mock(side_effect=AssertionError("SP planning must reuse device rows without token expansion"))
    for name in ("arange", "concatenate", "bincount"):
        monkeypatch.setattr(np, name, forbidden)
    for name in ("as_tensor", "tensor", "arange"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("cpu", "numpy", "tolist", "item"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    plans = [
        SPReplayPlan.build(offsets, kept, int(offsets[-1]), replay_tokens, tp_size, rank, rows)
        for rank in range(tp_size)
    ]
    assert sum(sum(plan.send_counts) for plan in plans) == int(kept.sum())
    for rank, plan in enumerate(plans):
        assert plan.local_rows.device.type == "meta"
        assert plan.local_rows.numel() == sum(plan.send_counts)
        assert plan.recv_counts == [source.send_counts[rank] for source in plans]
        assert plan.received_tokens == sum(plan.recv_counts)
    forbidden.assert_not_called()


def _sp_replay_worker(rank, world_size, hidden_dtype, init_method):
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=45)
    )
    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl("dsv41_decoder_replay_forward", layers_mod.dsv41_decoder_replay_forward)
    try:
        tp = SimpleNamespace(world_size=world_size, rank_in_group=rank, device_group=dist.group.WORLD)
        cases = [
            ([4096], [128]),  # all retained rows start on the last source rank
            ([4096, 1], [128, 1]),  # 129 valid rows need TP padding
            ([1, 19, 2, 17], [1, 3, 2, 4]),  # decode and short prefill coexist
            ([33, 35, 2], [30, 2, 2]),  # a retained interval crosses SP shards
            ([16], [1]),  # decoder ranks with zero valid rows
        ]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(layers_mod, "get_tp_group", lambda: tp)
            for lengths, kept in cases:
                offsets = np.insert(np.cumsum(lengths), 0, 0)
                original_width = (sum(lengths) + world_size - 1) // world_size
                replay_width = (sum(kept) + world_size - 1) // world_size
                rows = _kept_rows(lengths, kept)
                padded_rows = torch.nn.functional.pad(rows, (0, replay_width * world_size - len(rows)))
                plan = SPReplayPlan.build(
                    offsets,
                    np.array(kept),
                    sum(lengths),
                    replay_width * world_size,
                    world_size,
                    rank,
                    padded_rows,
                )
                full_positions = torch.arange(original_width * world_size, dtype=torch.int64)
                full_ids = full_positions * 7 + 5
                ids = full_ids[rank * original_width : (rank + 1) * original_width]
                hidden = (ids % 31 + 1).to(hidden_dtype).reshape(-1, 1, 1).expand(-1, 2, 3).clone()
                mix = (ids.float() / 7).reshape(-1, 1).expand(-1, 2).clone()
                observed_calls: list[int] = []
                all_to_all = dist.all_to_all_single
                row_bytes = 6 * hidden.element_size() + 2 * mix.element_size()

                def exchange(
                    output,
                    source,
                    calls=observed_calls,
                    collective=all_to_all,
                    row_bytes=row_bytes,
                    **kwargs,
                ):
                    assert source.dtype == output.dtype == torch.int8
                    assert source.shape[1] == output.shape[1] == row_bytes
                    calls.append(source.shape[0])
                    return collective(output, source, **kwargs)

                def run_layers(
                    local_hidden,
                    local_mix,
                    positions,
                    local_ids,
                    expected_positions=padded_rows,
                    expected_ids=full_ids[rows[rank * replay_width : (rank + 1) * replay_width]],
                    width=replay_width,
                ):
                    assert positions.tolist() == expected_positions.tolist()
                    assert torch.equal(local_ids[: len(expected_ids)], expected_ids)
                    assert not local_ids[len(expected_ids) :].any()
                    assert local_hidden.shape == (width, 2, 3)
                    assert local_hidden.dtype == hidden_dtype and local_mix.dtype == torch.float32
                    assert local_hidden.is_contiguous() and local_mix.is_contiguous()
                    outputs = [torch.empty_like(local_ids) for _ in range(world_size)]
                    dist.all_gather(outputs, local_ids)
                    aux = torch.cat(outputs).float().reshape(-1, 1) + 7
                    return local_hidden * 2, local_mix + 3, aux

                replay = layers_mod.AscendDecoderReplayLayers(128, run_layers, [])
                replay.sequence_parallel = True
                replay.sp_plan = plan
                replay.sp_replay_input_ids = torch.zeros(replay_width, dtype=full_ids.dtype)
                replay.sp_replay_input_ids[: plan.received_tokens] = full_ids[
                    rows[rank * replay_width : (rank + 1) * replay_width]
                ]
                replay.rows = padded_rows
                replay.num_actual_rows = len(rows)
                replay.context_factory = nullcontext
                context = SimpleNamespace(no_compile_layers={"decoder_replay": replay})
                patch.setattr(layers_mod, "get_forward_context", lambda context=context: context)
                with pytest.MonkeyPatch.context() as exchange_patch:
                    exchange_patch.setattr(dist, "all_to_all_single", exchange)
                    output, output_mix, aux = torch.ops.vllm.dsv41_decoder_replay_forward(
                        hidden, mix, full_positions, ids, 1, "decoder_replay"
                    )
                expected = torch.zeros_like(hidden)
                expected_mix = torch.zeros_like(mix)
                local_selected = rows[(rows >= rank * original_width) & (rows < (rank + 1) * original_width)]
                local_selected = local_selected - rank * original_width
                expected[local_selected] = hidden[local_selected] * 2
                expected_mix[local_selected] = mix[local_selected] + 3
                assert torch.equal(output, expected)
                assert torch.equal(output_mix, expected_mix)
                expected_aux = torch.zeros((len(full_ids), 1))
                expected_aux[rows, 0] = full_ids[rows].float() + 7
                assert torch.equal(aux, expected_aux)
                assert observed_calls == [len(local_selected), plan.received_tokens]
                total_sent = torch.tensor(observed_calls)
                dist.all_reduce(total_sent)
                assert total_sent.tolist() == [len(rows)] * 2
    finally:
        library._destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("hidden_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_sp_replay_collectives(tp_size, hidden_dtype, tmp_path):
    # Match the existing replay Gloo tests: inherit CPU dependency mocks.
    torch.set_num_threads(1)
    mp.start_processes(
        _sp_replay_worker,
        args=(tp_size, hidden_dtype, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=tp_size,
        start_method="fork",
    )


def _sp_dp_replay_worker(rank, init_method):
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=4, timeout=timedelta(seconds=45))
    dp_rank, tp_rank = divmod(rank, 2)
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    tp = SimpleNamespace(device_group=tp_groups[dp_rank], rank_in_group=tp_rank, world_size=2)
    dp = SimpleNamespace(cpu_group=dp_groups[tp_rank], rank_in_group=dp_rank)

    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl("dsv41_decoder_replay_forward", layers_mod.dsv41_decoder_replay_forward)
    try:
        with pytest.MonkeyPatch.context() as patch:
            # copy_to_gpu() reads PIN_MEMORY independently of the buffer's
            # pin_memory=False constructor argument. Forked CPU workers must
            # not enter the accelerator's pinned-memory allocator.
            patch.setattr(vllm_v1_utils, "PIN_MEMORY", False)
            patch.setattr(
                torch.Tensor,
                "pin_memory",
                Mock(side_effect=AssertionError("CPU replay must not pin accelerator memory")),
            )
            patch.setattr(layers_mod, "get_tp_group", lambda: tp)
            patch.setattr(replay_mod, "get_tp_group", lambda: tp)
            patch.setattr(replay_mod, "get_dp_group", lambda: dp)
            no_gather = Mock(side_effect=AssertionError("replay metadata must not all-gather"))
            patch.setattr(dist, "all_gather", no_gather)
            exchanges = Mock(wraps=dist.all_to_all_single)
            patch.setattr(dist, "all_to_all_single", exchanges)
            patch.setattr(replay_mod, "should_skip_allreduce_across_dp_group", lambda _: False)
            patch.setattr(replay_mod, "should_skip_dp_coordination", lambda: False)
            patch.setattr(replay_mod, "get_mrv2_in_profile_run", lambda: False)
            config = SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=32),
                parallel_config=SimpleNamespace(tensor_parallel_size=2, data_parallel_size=2),
            )
            layers = layers_mod.AscendDecoderReplayLayers(2, lambda hidden, mix, pos, ids: (hidden + 1, mix + 2), [])
            layers.sequence_parallel = True
            op_context = SimpleNamespace(no_compile_layers={"decoder_replay": layers})
            patch.setattr(layers_mod, "get_forward_context", lambda: op_context)
            state = replay_mod.DecoderReplayState(config, layers, torch.device("cpu"))
            state.slots = torch.empty(1, 32, dtype=torch.int64)

            def gather(inputs, **kwargs):
                # Replace only the metadata kernel; planning and all collective
                # paths below execute production code with real Gloo groups.
                offsets = inputs.batch.query_start_loc_np
                kept = np.diff(offsets - state.dropped_before.np[: len(offsets)])
                rows = _kept_rows(np.diff(offsets), kept)
                state.rows[: state.num_tokens].zero_()
                state.rows[: len(rows)].copy_(rows)

            patch.setattr(state, "_gather", gather)
            patch.setattr(state, "_build_metadata", lambda *args: {})
            for skip_padding in (False, True):
                patch.setattr(replay_mod.envs, "VLLM_MOE_SKIP_PADDING", skip_padding)
                for scenario in ("both-trim", "decode-peer", "idle-peer", "no-trim"):
                    compact = scenario != "no-trim" and (dp_rank == 0 or scenario == "both-trim")
                    idle = dp_rank == 1 and scenario == "idle-peer"
                    lengths = [17, 1] if compact else [1, 1]
                    offsets = np.insert(np.cumsum(lengths, dtype=np.int32), 0, 0)
                    full_ids = torch.arange(18, dtype=torch.int32) * 7 + 5
                    full_padding = torch.arange(18) >= (0 if idle else sum(lengths))
                    full_padding[sum(lengths) - 1] = True  # a masked row inside the query span
                    batch = SimpleNamespace(
                        num_reqs=2,
                        num_tokens_after_padding=18,
                        query_start_loc_np=offsets,
                        is_prefilling_np=np.array([compact, False]),
                        idx_mapping_np=np.array([0, 1]),
                        is_dummy=idle,
                        input_ids=full_ids,
                        is_padding=full_padding,
                    )
                    exchanges.reset_mock()
                    state.prepare(
                        replay_mod.DecoderReplayInputs(batch, (), torch.arange(18).reshape(1, -1), [], torch.zeros(2)),
                        CUDAGraphMode.NONE,
                    )
                    counts = [18, 18] if scenario == "no-trim" else [4, 4 if scenario == "both-trim" else 18]
                    assert state.dp_tokens.tolist() == counts
                    assert (layers.sp_plan is not None) == compact
                    assert (layers.context_factory is not None) == (scenario != "no-trim")
                    outer = SimpleNamespace(
                        is_padding=full_padding.chunk(2)[tp_rank] if skip_padding else full_padding,
                        attn_metadata={},
                        slot_mapping={},
                        additional_kwargs={},
                    )
                    current = [outer]
                    mask = torch.arange(32) % 2 == 0
                    saved_mask = mask.clone()
                    patch.setattr(replay_mod, "get_forward_context", lambda current=current: current[0])
                    patch.setattr(replay_mod, "get_mc2_mask", lambda mask=mask: mask)
                    rows = _kept_rows(lengths, [2, 1]) if compact else torch.arange(18)
                    expected_ids = full_ids.chunk(2)[tp_rank]
                    if compact:
                        expected_ids = torch.zeros(counts[dp_rank] // 2, dtype=full_ids.dtype)
                        rank_rows = rows[tp_rank * len(expected_ids) : (tp_rank + 1) * len(expected_ids)]
                        expected_ids[: len(rank_rows)] = full_ids[rank_rows]

                    def run_layers(hidden, mix, positions, ids, expected=expected_ids):
                        assert torch.equal(ids, expected)
                        return hidden + 1, mix + 2

                    layers.run_layers = run_layers
                    expected_padding = torch.ones(counts[dp_rank], dtype=torch.bool)
                    expected_padding[: len(rows)] = full_padding[rows]

                    @contextmanager
                    def forward_context(
                        metadata,
                        config,
                        expected=expected_padding,
                        counts=counts,
                        current=current,
                        outer=outer,
                        mask=mask,
                        skip_padding=skip_padding,
                        **kwargs,
                    ):
                        assert kwargs["num_tokens_across_dp"].tolist() == counts
                        assert kwargs["num_tokens"] == counts[dp_rank]
                        assert torch.equal(
                            kwargs["is_padding"], expected.chunk(2)[tp_rank] if skip_padding else expected
                        )
                        mask.fill_(True)
                        current[0] = SimpleNamespace(additional_kwargs={"mc2_mask": mask})
                        try:
                            yield
                            assert torch.equal(mask[: counts[dp_rank]], ~expected)
                            assert not mask[counts[dp_rank] :].any()
                        finally:
                            current[0] = outer

                    patch.setattr(replay_mod, "set_forward_context", forward_context)
                    ids = torch.arange(tp_rank * 9, (tp_rank + 1) * 9)
                    hidden = ids.float().reshape(-1, 1, 1)
                    mix = ids.float().reshape(-1, 1)
                    output, output_mix = torch.ops.vllm.dsv41_decoder_replay_forward(
                        hidden, mix, torch.arange(18), full_ids.chunk(2)[tp_rank], 0, "decoder_replay"
                    )
                    local_kept = torch.isin(ids, rows)
                    expected_hidden = torch.where(local_kept[:, None, None], hidden + 1, 0)
                    expected_mix = torch.where(local_kept[:, None], mix + 2, 0)
                    assert torch.equal(output, expected_hidden)
                    assert torch.equal(output_mix, expected_mix)
                    assert current[0] is outer
                    assert torch.equal(mask, saved_mask)
                    assert exchanges.call_count == (2 if compact else 0)
                    no_gather.assert_not_called()
    finally:
        library._destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("pin_memory_available", [False, True])
def test_sp_replay_with_dp_trimmed_decode_and_idle_peers(tmp_path, monkeypatch, pin_memory_available):
    monkeypatch.setattr(vllm_v1_utils, "PIN_MEMORY", pin_memory_available)
    torch.set_num_threads(1)
    mp.start_processes(
        _sp_dp_replay_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=4,
        start_method="fork",
    )
