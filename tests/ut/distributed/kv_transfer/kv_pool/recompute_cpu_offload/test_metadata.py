# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import (  # noqa: E402
    INVALID_JOB_ID,
    RecomputeCPUOffloadMetadata,
    RecomputeCPUOffloadWorkerMetadata,
)


def test_recompute_cpu_offload_worker_metadata_aggregate():
    metadata = RecomputeCPUOffloadWorkerMetadata(completed_store_events={1: 1, 2: 2})
    other = RecomputeCPUOffloadWorkerMetadata(completed_store_events={2: 3, 4: 1})

    merged = metadata.aggregate(other)

    assert isinstance(merged, RecomputeCPUOffloadWorkerMetadata)
    assert merged.completed_store_events == {1: 1, 2: 5, 4: 1}


def test_recompute_cpu_offload_metadata_defaults_are_empty():
    metadata = RecomputeCPUOffloadMetadata()

    assert metadata.need_flush is False
    assert metadata.preempt_store_event == INVALID_JOB_ID
    assert metadata.preempt_store_gpu_blocks == []
    assert metadata.preempt_store_cpu_blocks == []
    assert metadata.preempt_load_event == INVALID_JOB_ID
    assert metadata.preempt_load_gpu_blocks == []
    assert metadata.preempt_load_cpu_blocks == []
    assert metadata.preempt_load_event_to_reqs == {}
