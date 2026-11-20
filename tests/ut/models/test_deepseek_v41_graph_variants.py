# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("legacy", [False, True])
def test_v41_full_decode_replays_the_graph_matching_indexer_metadata(monkeypatch, legacy):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.attention.dsa_v41_indexer_shard import V41_FULL_INDEXER_SHARD
    from vllm_ascend.models.deepseek_v41.aclgraph import DeepSeekV41AclGraphManager
    from vllm_ascend.models.deepseek_v41.model_state import DeepSeekV41ModelState, EngramModelState
    from vllm_ascend.worker.v2 import aclgraph_utils

    short = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 8, uniform_token_count=1)
    dspark = BatchExecutionDescriptor(CUDAGraphMode.FULL, 24, 4, uniform_token_count=6)
    small = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 4, uniform_token_count=1)

    def init(self, vllm_config, *args, **kwargs):
        self.vllm_config = vllm_config
        self.graphs = {}
        self._capture_descs = {CUDAGraphMode.FULL: [dspark, short, small]}

    def capture(self, *_args, **_kwargs):
        assert ("has_lora" in _kwargs) is legacy
        for desc in self._capture_descs[CUDAGraphMode.FULL]:
            assert desc not in self.graphs
            self.graphs[desc] = V41_FULL_INDEXER_SHARD.get()

    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "__init__", init)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", capture)
    monkeypatch.setattr(aclgraph_utils, "vllm_version_is", lambda _version: legacy)
    monkeypatch.setattr(aclgraph_utils, "set_graph_params", lambda _sizes: None)
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils, "set_current_vllm_config", lambda _config: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "_get_graph_update_backend", lambda _groups: None)
    monkeypatch.setattr(aclgraph_utils, "use_updatable_graph", lambda _backend: False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, decode_context_parallel_size=1, prefill_context_parallel_size=2
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        lora_config=None,
    )
    config.model_config.hf_text_config = SimpleNamespace(index_topk=512)
    config.model_config.max_model_len = 8192
    model_state = DeepSeekV41ModelState.__new__(DeepSeekV41ModelState)
    model_state.vllm_config = config
    model_state.model_config = config.model_config
    model_state.decoder_replay = None
    model_state.prepare_engram = None
    global_batch = SimpleNamespace(num_reqs=2, is_dummy=False)
    pcp_context = SimpleNamespace(global_batch=global_batch, min_local_num_tokens=0)
    model_state.pcp_manager = SimpleNamespace(
        global_batch=global_batch,
        min_local_num_tokens=0,
        is_target_attn_groups=lambda _groups: True,
        build_attention_context=lambda *args: pcp_context,
    )

    def prepare_metadata(self, *args, **kwargs):
        self.attn_metadata = {"sharded": V41_FULL_INDEXER_SHARD.get()}
        self.pcp_context = pcp_context
        return self.attn_metadata

    monkeypatch.setattr(EngramModelState, "prepare_attn", prepare_metadata)
    runner = SimpleNamespace(model_state=model_state, attn_groups=[], update_stream=object())
    manager = DeepSeekV41AclGraphManager(config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, 6, runner)
    manager.capture(torch.nn.Identity(), model_state, None, None, None, [], None)
    assert manager.capture_sizes == [4, 8, 24]
    assert len(manager.graphs) == 5
    assert V41_FULL_INDEXER_SHARD.get() is None

    replayed = []
    monkeypatch.setattr(manager, "_graph_relay", lambda _backend, desc, _num_tokens, _metadata: replayed.append(desc))
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=0,
        num_computed_tokens_np=np.zeros(1, dtype=np.int32),
        query_start_loc_np=np.zeros(2, dtype=np.int32),
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        is_prefilling_np=np.zeros(1, dtype=np.bool_),
        num_scheduled_tokens=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.zeros(1, dtype=torch.int32),
        seq_lens_np=np.zeros(1, dtype=np.int32),
        positions=torch.empty(0, dtype=torch.int64),
        dcp_local_seq_lens=None,
        attn_state=None,
        is_dummy=False,
    )
    for desc in (short, dspark, small):
        batch.num_tokens_after_padding = desc.num_tokens
        batch.num_reqs_after_padding = desc.num_reqs
        for hist, min_rows in ((128, desc.num_tokens), (8192, desc.num_tokens), (8192, 0)):
            global_batch.num_computed_tokens_np = np.array([hist, hist], dtype=np.int32)
            pcp_context.min_local_num_tokens = min_rows
            model_state.pcp_manager.min_local_num_tokens = min_rows
            batch.num_tokens = min_rows
            expected = hist == 8192 and min_rows > 0 and desc is not small
            model_state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, [], None)
            manager.run_fullgraph(desc)
            replay_desc = replayed[-1]
            assert replay_desc.num_tokens == desc.num_tokens
            assert replay_desc.uniform_token_count == desc.uniform_token_count
            assert manager.graphs[replay_desc] is model_state.attn_metadata["sharded"] is expected


@pytest.mark.parametrize("fail_variant", [False, True, None])
def test_capture_restores_variant_and_descriptors(monkeypatch, fail_variant):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.attention.dsa_v41_indexer_shard import V41_FULL_INDEXER_SHARD
    from vllm_ascend.models.deepseek_v41.aclgraph import (
        DeepSeekV41AclGraphManager,
        _ShardedV41DecodeDescriptor,
    )
    from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

    replicated = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 8, uniform_token_count=1)
    sharded = _ShardedV41DecodeDescriptor(CUDAGraphMode.FULL, 8, 8, uniform_token_count=1)
    manager = DeepSeekV41AclGraphManager.__new__(DeepSeekV41AclGraphManager)
    manager._v41_sharded_descs = {replicated: sharded}
    descs = {CUDAGraphMode.FULL: [replicated, sharded]}
    manager._capture_descs = descs
    captured = []

    def capture(self, *args, **kwargs):
        variant = V41_FULL_INDEXER_SHARD.get()
        captured.append(variant)
        assert self._capture_descs[CUDAGraphMode.FULL] == [sharded if variant else replicated]
        if variant is fail_variant:
            raise RuntimeError("capture failed")

    monkeypatch.setattr(ModelAclGraphManager, "capture", capture)
    context = pytest.raises(RuntimeError, match="capture failed") if fail_variant is not None else nullcontext()
    with context:
        manager.capture(None, None, None, None, None, [], None)
    assert manager._capture_descs is descs
    assert V41_FULL_INDEXER_SHARD.get() is None
    assert captured == ([False] if fail_variant is False else [False, True])


@pytest.mark.parametrize("pcp", [1, 2])
def test_model_state_capture_override_and_eager_reset(monkeypatch, pcp):
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.attention.dsa_v41_indexer_shard import V41_FULL_INDEXER_SHARD, v41_full_indexer_sharding
    from vllm_ascend.models.deepseek_v41.aclgraph import DeepSeekV41AclGraphManager
    from vllm_ascend.models.deepseek_v41.model_state import DeepSeekV41ModelState, EngramModelState

    state = DeepSeekV41ModelState.__new__(DeepSeekV41ModelState)
    state.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            hf_text_config=SimpleNamespace(index_topk=512),
            max_model_len=16384,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, prefill_context_parallel_size=pcp, decode_context_parallel_size=1
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        lora_config=None,
    )
    state.pcp_manager = None
    batch = SimpleNamespace(
        is_dummy=True, num_tokens=8, num_tokens_after_padding=8, num_reqs=1, num_computed_tokens_np=np.array([8192])
    )
    observed = []

    def prepare(self, *args, **kwargs):
        observed.append(V41_FULL_INDEXER_SHARD.get())
        return {"variant": observed[-1]}

    monkeypatch.setattr(EngramModelState, "prepare_attn", prepare)
    assert state.get_graph_manager_cls() is DeepSeekV41AclGraphManager
    for variant in (True, False):
        with v41_full_indexer_sharding(variant):
            assert state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, [], None)["variant"] is variant
        assert state.v41_full_indexer_sharded is variant
        assert V41_FULL_INDEXER_SHARD.get() is None
    # An ordinary dummy run must not reuse the last captured variant.
    state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, [], None)
    assert observed[-1] is False
    batch.is_dummy = False
    state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, [], None)
    assert observed[-1] is (pcp == 1)  # PCP without a current layout falls back.
    state.prepare_attn(batch, CUDAGraphMode.NONE, (), None, [], None)
    assert observed[-1] is None
    assert state.v41_full_indexer_sharded is False


@pytest.mark.parametrize("max_model_len", [3072, 4096, 4097])
def test_unreachable_history_skips_sharded_capture_and_replay(monkeypatch, max_model_len):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.attention.dsa_v41_indexer_shard import V41_FULL_INDEXER_SHARD
    from vllm_ascend.models.deepseek_v41.aclgraph import DeepSeekV41AclGraphManager
    from vllm_ascend.models.deepseek_v41.model_state import DeepSeekV41ModelState, EngramModelState
    from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            hf_text_config=SimpleNamespace(index_topk=2048),
            max_model_len=max_model_len,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        lora_config=None,
    )
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 8, uniform_token_count=1)

    def init(self, *args, **kwargs):
        self._capture_descs = {CUDAGraphMode.FULL: [desc]}
        self.model_runner = args[4]

    variants = []
    monkeypatch.setattr(ModelAclGraphManager, "__init__", init)
    monkeypatch.setattr(
        ModelAclGraphManager, "capture", lambda *args, **kwargs: variants.append(V41_FULL_INDEXER_SHARD.get())
    )
    monkeypatch.setattr(ModelAclGraphManager, "run_fullgraph", lambda self, selected: selected)
    monkeypatch.setattr(
        EngramModelState, "prepare_attn", lambda *args, **kwargs: {"variant": V41_FULL_INDEXER_SHARD.get()}
    )
    state = DeepSeekV41ModelState.__new__(DeepSeekV41ModelState)
    state.vllm_config = config
    manager = DeepSeekV41AclGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, 1, SimpleNamespace(model_state=state)
    )
    manager.capture(None, state, None, None, None, [], None)
    reachable = max_model_len > 4096
    assert bool(manager._v41_sharded_descs) is reachable
    assert variants == ([False, True] if reachable else [False])
    assert V41_FULL_INDEXER_SHARD.get() is None
    # The largest possible history is one token below max_model_len.
    batch = SimpleNamespace(
        is_dummy=False,
        num_tokens=8,
        num_tokens_after_padding=8,
        num_reqs=1,
        num_computed_tokens_np=np.array([max_model_len - 1]),
    )
    assert state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, [], None)["variant"] is reachable
    assert manager.run_fullgraph(desc) == (manager._v41_sharded_descs[desc] if reachable else desc)


@pytest.mark.parametrize("live_rows", [0, 1, 7, 8])
@pytest.mark.parametrize("mixed_mode", [False, True])
def test_full_selection_uses_live_work_and_supported_graph_mode(monkeypatch, live_rows, mixed_mode):
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.attention.dsa_v41_indexer_shard import V41_FULL_INDEXER_SHARD
    from vllm_ascend.models.deepseek_v41.model_state import DeepSeekV41ModelState, EngramModelState

    state = DeepSeekV41ModelState.__new__(DeepSeekV41ModelState)
    state.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            hf_text_config=SimpleNamespace(index_topk=512),
            max_model_len=8192,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE if mixed_mode else CUDAGraphMode.FULL_DECODE_ONLY
        ),
        lora_config=None,
    )
    batch = SimpleNamespace(
        is_dummy=False,
        num_tokens=live_rows,
        num_tokens_after_padding=8,
        num_reqs=1,
        num_computed_tokens_np=np.array([4096]),
    )
    monkeypatch.setattr(
        EngramModelState, "prepare_attn", lambda *args, **kwargs: {"variant": V41_FULL_INDEXER_SHARD.get()}
    )
    expected = live_rows == 8 and not mixed_mode
    assert state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, [], None)["variant"] is expected
    assert state.v41_full_indexer_sharded is expected
