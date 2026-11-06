# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV1 replay extent regression tests with CPU tensors and Gloo collectives.

Load the production method without importing the NPU worker dependency tree,
following test_spec_pp_cpu_counts.py. Only metadata construction and NPU state
queries are substituted; row selection and DP coordination execute unchanged.
"""

import ast
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WINDOW = 128
CHUNK = 4092
DP_SIZE = 4


def _runner(*, rank=0, dp_size=DP_SIZE, query_lens=(), prefills=(), logprobs=()):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v1/model_runner.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in cls.body if getattr(node, "name", None) == "_prepare_decoder_replay")
    context = SimpleNamespace(cudagraph_runtime_mode="NONE", in_profile_run=False, capturing=False, attn_metadata={})
    group = SimpleNamespace(rank_in_group=rank, cpu_group=None)
    collective = Mock()
    skip_allreduce = Mock(return_value=False)
    stream_capture = Mock(return_value=False)
    namespace = {
        "np": np,
        "torch": SimpleNamespace(
            zeros=torch.zeros,
            tensor=torch.tensor,
            arange=torch.arange,
            int32=torch.int32,
            npu=SimpleNamespace(is_current_stream_capturing=stream_capture),
        ),
        "dist": SimpleNamespace(all_reduce=collective),
        "get_dp_group": lambda: group,
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(NONE="NONE"),
        "should_skip_allreduce_across_dp_group": skip_allreduce,
        "should_skip_dp_coordination": lambda: False,
        "logger": Mock(),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    offsets = np.concatenate(([0], np.cumsum(query_lens))).astype(np.int32)
    runner = SimpleNamespace(
        decoder_replay=SimpleNamespace(window=WINDOW, rows=None, context_factory=None),
        parallel_config=SimpleNamespace(data_parallel_size=dp_size),
        vllm_config=SimpleNamespace(),
        input_batch=SimpleNamespace(num_reqs=len(query_lens), req_ids=list(range(len(query_lens)))),
        num_prompt_logprobs=set(logprobs),
        _decoder_replay_common_metadata=SimpleNamespace(
            query_start_loc_cpu=torch.from_numpy(offsets),
            is_prefilling=torch.tensor(prefills, dtype=torch.bool),
        ),
        _decoder_replay_offsets=SimpleNamespace(np=np.zeros(len(offsets), dtype=np.int32)),
        _decoder_replay_dropped_before=SimpleNamespace(np=np.zeros(len(offsets), dtype=np.int32)),
        _decoder_replay_rows=torch.full((CHUNK,), -1, dtype=torch.int64),
        _decoder_replay_context=lambda: None,
    )

    def metadata(kept_lens):
        # Stand in for the device metadata kernel, retaining exact source rows.
        rows = np.concatenate([np.arange(end - kept, end) for end, kept in zip(offsets[1:], kept_lens)])
        runner._decoder_replay_rows[: len(rows)] = torch.from_numpy(rows)
        return {"kept_lens": kept_lens.tolist()}

    runner._build_decoder_replay_metadata = Mock(side_effect=metadata)
    runner.prepare = namespace["_prepare_decoder_replay"].__get__(runner)
    return runner, context, collective, skip_allreduce, stream_capture


def _peer_reduction(collective, *, rank=0, peer_trim=True, peer_tokens=WINDOW):
    def reduce(values, group):
        # Preserve the local contribution so a wrong dummy extent is observable.
        for peer in range(values.shape[1]):
            if peer != rank:
                values[:, peer] = torch.tensor([peer_trim, peer_tokens])

    collective.side_effect = reduce


@pytest.mark.parametrize("num_tokens", [0, 1, WINDOW - 1, WINDOW, WINDOW + 1, CHUNK])
def test_idle_extent_and_rows(num_tokens):
    runner, context, collective, _, _ = _runner()
    _peer_reduction(collective)
    runner.prepare(num_tokens, is_dummy=True)
    expected = min(num_tokens, WINDOW)
    assert runner._decoder_replay_counts.tolist() == [expected, WINDOW, WINDOW, WINDOW]
    assert runner._decoder_replay_num_actual_tokens == expected
    assert runner.decoder_replay.num_actual_rows == expected
    assert runner.decoder_replay.rows.tolist() == list(range(expected))
    assert runner.decoder_replay.context_factory is runner._decoder_replay_context
    assert runner._decoder_replay_metadata is context.attn_metadata
    runner._build_decoder_replay_metadata.assert_not_called()
    collective.assert_called_once()


@pytest.mark.parametrize("is_dummy", [False, True])
def test_no_replica_trims_keeps_replay_disabled(is_dummy):
    runner, _, collective, _, _ = _runner(query_lens=[1], prefills=[False])
    _peer_reduction(collective, peer_trim=False, peer_tokens=1)
    runner.decoder_replay.rows = torch.tensor([0])
    runner.decoder_replay.context_factory = runner._decoder_replay_context
    runner.prepare(CHUNK if is_dummy else 1, is_dummy=is_dummy)
    assert runner.decoder_replay.rows is None
    assert runner.decoder_replay.context_factory is None
    runner._build_decoder_replay_metadata.assert_not_called()
    collective.assert_called_once()


@pytest.mark.parametrize("dp_size", [1, DP_SIZE])
def test_real_prefill_decode_and_prompt_logprobs_keep_exact_rows(dp_size):
    lens = [256, 6, 200, 7]
    runner, _, collective, _, _ = _runner(
        dp_size=dp_size, query_lens=lens, prefills=[True, False, True, True], logprobs=[2]
    )
    _peer_reduction(collective)
    runner.prepare(sum(lens), is_dummy=False)
    expected_rows = list(range(128, sum(lens)))
    assert runner.decoder_replay.rows.tolist() == expected_rows
    assert runner._decoder_replay_num_actual_tokens == len(expected_rows)
    assert runner._decoder_replay_metadata == {"kept_lens": [128, 6, 200, 7]}
    assert runner._decoder_replay_offsets.np.tolist() == [0, 128, 134, 334, 341]
    assert runner._decoder_replay_dropped_before.np.tolist() == [0, 128, 128, 128, 128]
    assert collective.call_count == (1 if dp_size > 1 else 0)


@pytest.mark.parametrize("prefill,logprobs", [(False, ()), (True, (0,))])
def test_untrimmed_real_peer_is_not_bounded(prefill, logprobs):
    runner, _, collective, _, _ = _runner(query_lens=[256], prefills=[prefill], logprobs=logprobs)
    _peer_reduction(collective)
    runner.prepare(256, is_dummy=False)
    assert runner._decoder_replay_counts.tolist() == [256, WINDOW, WINDOW, WINDOW]
    assert runner.decoder_replay.rows.tolist() == list(range(256))


@pytest.mark.parametrize("guard", ["FULL", "PIECEWISE", "profile", "capture", "stream", "unsynchronized"])
def test_guarded_execution_clears_replay_without_collectives(guard):
    runner, context, collective, skip_allreduce, stream_capture = _runner()
    runner.decoder_replay.rows = torch.tensor([0])
    runner.decoder_replay.context_factory = runner._decoder_replay_context
    if guard in ("FULL", "PIECEWISE"):
        context.cudagraph_runtime_mode = guard
    elif guard == "profile":
        context.in_profile_run = True
    elif guard == "capture":
        context.capturing = True
    elif guard == "stream":
        stream_capture.return_value = True
    else:
        skip_allreduce.return_value = True
    runner.prepare(CHUNK, is_dummy=True)
    assert runner.decoder_replay.rows is None
    assert runner.decoder_replay.context_factory is None
    collective.assert_not_called()


def test_coordination_failure_does_not_enable_replay():
    runner, _, collective, _, _ = _runner()
    runner.decoder_replay.rows = torch.tensor([0])
    runner.decoder_replay.context_factory = runner._decoder_replay_context
    collective.side_effect = RuntimeError("coordination failed")
    with pytest.raises(RuntimeError, match="coordination failed"):
        runner.prepare(CHUNK, is_dummy=True)
    assert runner.decoder_replay.rows is None
    assert runner.decoder_replay.context_factory is None


def _distributed_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=DP_SIZE, timeout=timedelta(seconds=30)
    )
    try:
        for active in (1, 2, DP_SIZE):
            real = rank < active
            runner, _, collective, _, _ = _runner(
                rank=rank, query_lens=[CHUNK] if real else [], prefills=[True] if real else []
            )
            collective.side_effect = dist.all_reduce
            runner.prepare(CHUNK, is_dummy=not real)
            # Every rank observes the same bounded widths, even with idle peers.
            assert runner._decoder_replay_counts.tolist() == [WINDOW] * DP_SIZE
            assert runner.decoder_replay.rows.numel() == WINDOW
            expected = list(range(CHUNK - WINDOW, CHUNK)) if real else list(range(WINDOW))
            assert runner.decoder_replay.rows.tolist() == expected
            # Exercise actual gather/reduce-scatter semantics after coordination.
            payload = torch.full((WINDOW, 2), rank, dtype=torch.float32)
            gathered = [torch.empty_like(payload) for _ in range(DP_SIZE)]
            dist.all_gather(gathered, payload)
            assert torch.cat(gathered).shape == (WINDOW * DP_SIZE, 2)
            output = torch.empty_like(payload)
            dist.reduce_scatter_tensor(output, torch.cat(gathered))
            torch.testing.assert_close(output, payload * DP_SIZE)
            collective.assert_called_once()
            # A subsequent non-trimming step must retire the previous replay.
            runner._decoder_replay_common_metadata.query_start_loc_cpu = torch.tensor([0, 1], dtype=torch.int32)
            runner.prepare(1, is_dummy=not real)
            assert runner.decoder_replay.rows is None
            assert runner.decoder_replay.context_factory is None
            assert runner._decoder_replay_counts.tolist() == [1] * DP_SIZE
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires the CPU Gloo backend")
def test_idle_and_active_dp_collectives(tmp_path):
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'replay-rendezvous'}",), nprocs=DP_SIZE)
