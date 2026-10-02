# SPDX-License-Identifier: Apache-2.0
"""Regression tests for native PP P2P warmup ordering."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from vllm_ascend.worker import worker as worker_module


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8])
def test_pipeline_warmup_adjacent_links(size):
    # Noncontiguous global ranks must not be confused with local PP stages.
    ranks = [3 + stage * 8 for stage in range(size)]
    for stage in range(size):
        events = []
        pp = SimpleNamespace(
            world_size=size,
            rank_in_group=stage,
            ranks=ranks,
            device_group=object(),
            cpu_group=object(),
        )
        probe = object()
        worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)

        def transfer(kind):
            def invoke(tensor, **kwargs):
                peer = kwargs.get("dst", kwargs.get("src"))
                assert tensor is probe
                assert kwargs["group"] is pp.device_group
                events.append((kind, peer))
                handle = MagicMock()
                handle.wait.side_effect = lambda: events.append(("wait", peer))
                return handle

            return invoke

        with (
            patch.object(worker_module, "get_pp_group", return_value=pp),
            patch.object(worker_module.torch, "ones", return_value=probe) as allocate,
            patch.object(worker_module.dist, "isend", side_effect=transfer("send")),
            patch.object(worker_module.dist, "irecv", side_effect=transfer("recv")),
            patch.object(worker_module.torch.npu, "synchronize", side_effect=lambda: events.append(("sync",))),
            patch.object(
                worker_module.dist, "barrier", side_effect=lambda **kw: events.append(("barrier", kw["group"]))
            ),
        ):
            worker._warmup_pipeline_parallel_p2p()

        if size == 1:
            allocate.assert_not_called()
            assert events == []
            continue
        allocate.assert_called_once_with(1, dtype=worker_module.torch.float32, device="npu")
        expected = []
        for src in range(size - 1):
            if stage == src:
                expected.extend([("send", ranks[src + 1]), ("wait", ranks[src + 1]), ("sync",)])
            elif stage == src + 1:
                expected.extend([("recv", ranks[src]), ("wait", ranks[src]), ("sync",)])
            expected.append(("barrier", pp.cpu_group))
        assert events == expected


def test_pipeline_warmup_runs_before_transfer_initialization():
    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.parallel_config = SimpleNamespace(
        world_size=4,
        tensor_parallel_size=1,
        pipeline_parallel_size=4,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    worker.rank = 0
    worker.local_rank = 0
    worker.distributed_init_method = "tcp://test:1234"
    worker.vllm_config = object()
    order = MagicMock()
    with (
        patch.object(worker_module, "init_batch_invariance"),
        patch.object(worker_module, "init_distributed_environment"),
        patch.object(worker_module, "ensure_model_parallel_initialized", side_effect=order.model_parallel),
        patch.object(worker_module, "init_ascend_model_parallel", side_effect=order.ascend_parallel),
        patch.object(worker, "_warmup_pipeline_parallel_p2p", side_effect=order.warmup),
        patch.object(worker_module, "ensure_ec_transfer_initialized", side_effect=order.ec_transfer),
    ):
        worker._init_worker_distributed_environment()
    assert order.mock_calls == [
        call.model_parallel(1, 4, 1, 1),
        call.ascend_parallel(worker.parallel_config),
        call.warmup(),
        call.ec_transfer(worker.vllm_config),
    ]


def test_pipeline_warmup_propagates_communication_failure():
    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    pp = SimpleNamespace(world_size=2, rank_in_group=0, ranks=[2, 10], device_group=object(), cpu_group=object())
    with (
        patch.object(worker_module, "get_pp_group", return_value=pp),
        patch.object(worker_module.torch, "ones"),
        patch.object(worker_module.dist, "isend", side_effect=RuntimeError("HCCL init failed")),
        patch.object(worker_module.dist, "barrier") as barrier,
    ):
        with pytest.raises(RuntimeError, match="HCCL init failed"):
            worker._warmup_pipeline_parallel_p2p()
    barrier.assert_not_called()
