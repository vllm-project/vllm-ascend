# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import MambaSpec

import vllm_ascend.worker.v2.model_states.mamba_hybrid as hybrid
from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata, AscendSFAIndexerMetadataBuilder
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.worker.device_metadata import TargetDeviceMetadata


@pytest.fixture
def state_env(monkeypatch):
    class Stream:
        def wait_event(self, event):
            pass

    class Event:
        def record(self, stream):
            pass

    main = Stream()
    monkeypatch.setattr(torch.npu, "Stream", Stream)
    monkeypatch.setattr(torch.npu, "Event", Event)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: main)
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    producer = Mock(return_value=torch.zeros(1024, dtype=torch.int32))
    monkeypatch.setattr(DeviceOperator, "build_indexer_local_metadata", producer)

    config = SimpleNamespace(
        num_speculative_tokens=0,
        parallel_config=SimpleNamespace(decode_context_parallel_size=8),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_n_heads=32, index_head_dim=128, index_topk=2048)
        ),
    )
    builder = AscendSFAIndexerMetadataBuilder.__new__(AscendSFAIndexerMetadataBuilder)
    builder.vllm_config = config
    builder.use_sharded_dcp = True
    builder._device_metadata_enabled = False
    builder._device_metadata_tasks = ()
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: builder)]]
    builds = []

    def build(**kwargs):
        builds.append(kwargs)
        count = kwargs["num_reqs"]
        starts = kwargs["query_start_loc_gpu"][: count + 1]
        lengths = kwargs["seq_lens"][:count]
        empty = torch.empty(0)
        metadata = AscendSFAIndexerMetadata(
            num_actual_tokens=kwargs["num_actual_tokens"],
            slot_mapping=empty,
            seq_lens=lengths,
            cum_query_lens=starts[1:],
            block_table=empty,
            sin=empty,
            cos=empty,
            query_start_loc=starts,
            actual_seq_lengths_key=lengths,
        )
        if builder._device_metadata_enabled:
            builder._defer_local_qli_metadata(metadata)
        return {"indexer": metadata}

    monkeypatch.setattr(hybrid, "build_attn_metadata", build)
    state = hybrid.AscendMambaHybridModelState.__new__(hybrid.AscendMambaHybridModelState)
    state.vllm_config = config
    state.max_model_len = 151552
    state.device_metadata = TargetDeviceMetadata()
    state.recoverssm = None
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=2,
        num_reqs_after_padding=4,
        num_tokens_after_padding=4,
        is_prefilling_np=np.array([False, False]),
        prefill_runs_as_decode_np=None,
        num_scheduled_tokens=np.array([1, 1]),
        query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
        query_start_loc_np=np.array([0, 1, 2, 2, 2], dtype=np.int32),
        seq_lens=torch.tensor([16896, 16896, 0, 0], dtype=torch.int32),
        seq_lens_np=np.array([16896, 16896, 0, 0], dtype=np.int32),
        dcp_local_seq_lens=torch.tensor([16896, 16896, 0, 0], dtype=torch.int32),
        positions=torch.arange(4),
        attn_state=None,
    )

    def prepare(mode, *, for_capture=False):
        result = state.prepare_attn(batch, mode, (), torch.empty(0), groups, SimpleNamespace(), for_capture=for_capture)
        assert result is state.attn_metadata
        assert isinstance(builds[-1]["model_specific_attn_metadata"], hybrid.MambaHybridAttnMetadata)
        assert not builder._device_metadata_enabled
        return result["indexer"]

    return SimpleNamespace(
        state=state, builder=builder, producer=producer, builds=builds, prepare=prepare, batch=batch, groups=groups
    )


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("for_capture", [False, True])
@pytest.mark.parametrize("is_dummy", [False, True])
def test_pcp_local_and_global_builds_share_one_qli_producer_lifecycle(
    state_env, monkeypatch, mode, for_capture, is_dummy
):
    env = state_env
    text_config = env.state.vllm_config.model_config.hf_text_config
    text_config.model_type = "kimi_linear"
    text_config.linear_attn_config = {"num_heads": 96}
    env.batch.is_dummy = is_dummy
    context = SimpleNamespace(global_batch=env.batch, global_block_tables=(), global_slot_mappings=torch.empty(0))
    env.state.pcp_manager = SimpleNamespace(build_attention_context=lambda *args: context)
    env.groups[0][0].kv_cache_spec = object()
    spec = MambaSpec(block_size=16, shapes=((2, 3), (2, 2)), dtypes=(torch.float16, torch.float32))
    env.groups.append([SimpleNamespace(kv_cache_spec=spec, get_metadata_builder=lambda _: SimpleNamespace())])
    build_indexer = hybrid.build_attn_metadata

    def build(**kwargs):
        metadata = {}
        if kwargs["attn_groups"][0]:
            metadata.update(build_indexer(**kwargs))
        if kwargs["attn_groups"][1]:
            metadata["kda"] = SimpleNamespace()
        return metadata

    monkeypatch.setattr(hybrid, "build_attn_metadata", build)
    run_build = Mock(wraps=env.state.device_metadata.run_build)
    monkeypatch.setattr(env.state.device_metadata, "run_build", run_build)
    metadata = env.prepare(mode, for_capture=for_capture)
    run_build.assert_called_once()
    assert set(env.state.attn_metadata) == {"indexer", "kda"}
    executor = env.state.device_metadata.executor
    assert metadata.device_metadata_executor is executor
    assert env.state.attn_metadata["kda"].device_metadata_executor is executor
    if mode == CUDAGraphMode.FULL or for_capture:
        env.producer.assert_not_called()
        env.state.device_metadata.begin_forward()
    env.producer.assert_called_once()
    assert metadata.qli_metadata is env.producer.return_value
    assert executor.submission_in_flight
    env.state.device_metadata.finish()
    assert not env.state.device_metadata._tasks
    assert not executor.submission_in_flight


def test_hybrid_eager_submits_qli_producer(state_env):
    env = state_env
    metadata = env.prepare(CUDAGraphMode.NONE)
    env.producer.assert_called_once()
    assert metadata.qli_metadata is env.producer.return_value
    assert metadata.qli_metadata_group_id == id(env.builder)
    assert metadata.device_metadata_executor is env.state.device_metadata.executor
    assert env.builds[-1]["full_graph_mode"] is False
    assert env.state.device_metadata.executor.submission_in_flight
    env.state.device_metadata.finish()
    assert not env.state.device_metadata.executor.submission_in_flight


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_hybrid_capture_submits_only_at_graph_entry(state_env, mode):
    env = state_env
    metadata = env.prepare(mode, for_capture=True)
    env.producer.assert_not_called()
    assert len(env.state.device_metadata._tasks) == 1
    assert env.builds[-1]["for_cudagraph_capture"] is True
    env.state.device_metadata.begin_forward()
    env.producer.assert_called_once()
    assert metadata.qli_metadata is env.producer.return_value
    env.state.device_metadata.finish()


def test_hybrid_full_replay_prepare_does_not_resubmit_producer(state_env):
    env = state_env
    env.prepare(CUDAGraphMode.FULL, for_capture=True)
    env.state.device_metadata.begin_forward()
    env.state.device_metadata.finish()
    env.producer.assert_called_once()

    metadata = env.prepare(CUDAGraphMode.FULL)
    assert env.builds[-1]["full_graph_mode"] is True
    assert env.builds[-1]["num_reqs"] == 4
    assert env.builds[-1]["num_actual_reqs"] == 2
    assert metadata.device_metadata_executor is env.state.device_metadata.executor
    assert len(env.state.device_metadata._tasks) == 1
    assert not env.state.device_metadata.executor.submission_in_flight
    env.producer.assert_called_once()
    env.state.device_metadata.finish_replay()
    assert not env.state.device_metadata._tasks
    env.producer.assert_called_once()


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("for_capture", [False, True])
def test_hybrid_without_device_metadata_keeps_direct_build(state_env, mode, for_capture):
    env = state_env
    del env.state.device_metadata
    assert env.state.device_metadata is None
    metadata = env.prepare(mode, for_capture=for_capture)
    env.producer.assert_not_called()
    assert metadata.qli_metadata_group_id is None
    assert not hasattr(metadata, "device_metadata_executor")
    assert env.builds[-1]["full_graph_mode"] is (mode == CUDAGraphMode.FULL)
    assert env.builds[-1]["for_cudagraph_capture"] is for_capture
