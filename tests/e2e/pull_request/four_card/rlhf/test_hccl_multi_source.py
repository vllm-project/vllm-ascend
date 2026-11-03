# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two partitioned HCCL sources -> TP2, checked against startup-loaded weights."""

import multiprocessing
import os
import time
import traceback
from dataclasses import replace
from functools import cached_property
from queue import Empty

import pytest
import requests
import torch
import torch_npu  # noqa: F401
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerWeightTransferEngine
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    PACKED_LANE_SKIP_REASON,
    FixedRandomWeightSource,
    WeightUpdateModelCase,
    generation_signature,
    live_update_serve_args,
    reference_signature,
    wait_for_free_device_memory,
)

UPDATE_TIMEOUT = 300
MEMORY_UTILIZATION = 0.7
CASE = WeightUpdateModelCase("qwen3-multi-source", "Qwen/Qwen3-0.6B", {"num_hidden_layers": 2})
pytestmark = [
    pytest.mark.skipif(torch.npu.device_count() < 4, reason="requires four NPUs"),
    pytest.mark.skipif(
        not hasattr(NCCLTrainerWeightTransferEngine, "_broadcast_plan"),
        reason="requires vLLM with multi-source ownership planning (#62557)",
    ),
]


class _RoundSource(FixedRandomWeightSource):
    update_round = 0

    def _parameter_seed(self, name):
        return super()._parameter_seed(f"round-{self.update_round}:{name}")


class _StageSource(WeightSource):
    """Expose global ownership metadata but materialize only this stage."""

    def __init__(self, case, device, rank):
        self.full = _RoundSource(case, device)
        self.rank = rank

    @cached_property
    def metadata(self):
        return [
            ParamMeta(
                m.name, m.dtype, m.shape, owners=(0 if "embed_tokens" in m.name or ".layers.0." in m.name else 4,)
            )
            for m in self.full.metadata
        ]

    def iter_full(self):
        for m in self.metadata:
            if self.rank in m.owners:
                yield m.name, self.full._make_tensor(m.name, m.shape, m.dtype)

    def __iter__(self):
        yield from self.iter_full()


def _trainer_process(rank, device_index, address, port, base_url, packed, commands, results):
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerInitInfo,
        HCCLTrainerWeightTransferEngine,
    )

    engine = None
    try:
        torch.npu.set_device(device_index)
        source = _StageSource(CASE, torch.device("npu", device_index), rank)
        engine = HCCLTrainerWeightTransferEngine.trainer_init(
            HCCLTrainerInitInfo(
                rank=rank,
                master_address=address,
                master_port=port,
                world_size=4,
                packed=packed,
                packed_buffer_size_bytes=1024 * 1024,
            ),
            client=HTTPVLLMWeightSyncClient(base_url, timeout=UPDATE_TIMEOUT),
            source=source,
        )
        results.put((rank, "ready", None))
        for update_round in range(2):
            assert commands.get(timeout=UPDATE_TIMEOUT) == update_round
            source.full.update_round = update_round
            engine.send_weights()
            results.put((rank, "done", update_round))
    except BaseException:
        results.put((rank, "error", traceback.format_exc()))
        raise
    finally:
        if engine is not None:
            engine.shutdown()


def _wait_sources(results, phase, update_round=None):
    pending = {0, 4}
    deadline = time.monotonic() + UPDATE_TIMEOUT
    while pending:
        try:
            rank, actual_phase, detail = results.get(timeout=max(0.01, deadline - time.monotonic()))
        except Empty:
            pytest.fail(f"timed out waiting for sources {pending} in {phase}")
        assert actual_phase != "error", detail
        assert rank in pending and actual_phase == phase and detail == update_round
        pending.remove(rank)


@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_hccl_multiple_sources_match_independent_startup_load(packed):
    if packed:
        pytest.skip(PACKED_LANE_SKIP_REASON)
    # Existing packed-lane skip stays in place until its cross-stream hardware
    # regression is resolved. The CPU protocol test covers both packing modes.
    base_device = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
    if torch.npu.device_count() < base_device + 4:
        pytest.skip("requires four contiguous visible NPUs")
    torch.npu.set_device(base_device + 2)
    env = {"VLLM_SERVER_DEV_MODE": "1", "ASCEND_RT_VISIBLE_DEVICES": f"{base_device},{base_device + 1}"}
    references = []
    for update_round in range(2):
        case = replace(CASE, id=f"{CASE.id}-round-{update_round}")
        source = _RoundSource(case, torch.device("npu", base_device + 2))
        source.update_round = update_round
        references.append(
            reference_signature(
                source,
                case,
                port=get_open_port(),
                gpu_memory_utilization=MEMORY_UTILIZATION,
                tensor_parallel_size=2,
                device_index=base_device,
                env_dict=env,
            )
        )
    assert references[0] != references[1], "the second payload must observably change the policy"
    torch.npu.empty_cache()
    wait_for_free_device_memory(base_device, MEMORY_UTILIZATION)
    port = get_open_port()
    with RemoteOpenAIServer(
        CASE.model,
        vllm_serve_args=live_update_serve_args(
            CASE,
            backend="hccl",
            port=port,
            gpu_memory_utilization=MEMORY_UTILIZATION,
            tensor_parallel_size=2,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env,
        auto_port=False,
    ) as server:
        client = server.get_client()
        assert generation_signature(client, CASE.model) != references[0]
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        commands = [context.Queue(), context.Queue()]
        # Both sources must use the same rendezvous address and port.
        transfer_address, transfer_port = get_ip(), get_open_port()
        processes = [
            context.Process(
                target=_trainer_process,
                args=(
                    rank,
                    base_device + 2 + index,
                    transfer_address,
                    transfer_port,
                    server.url_root,
                    packed,
                    commands[index],
                    results,
                ),
            )
            for index, rank in enumerate((0, 4))
        ]
        try:
            for process in processes:
                process.start()
            _wait_sources(results, "ready")
            for update_round, reference in enumerate(references):
                response = requests.post(server.url_for("pause"), timeout=UPDATE_TIMEOUT)
                response.raise_for_status()
                for queue in commands:
                    queue.put(update_round)
                _wait_sources(results, "done", update_round)
                # Resume only after both source processes report success.
                response = requests.post(server.url_for("resume"), timeout=UPDATE_TIMEOUT)
                response.raise_for_status()
                assert generation_signature(client, CASE.model) == reference
            for process in processes:
                process.join(timeout=30)
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=10)
            for queue in [results, *commands]:
                queue.close()
                queue.cancel_join_thread()
