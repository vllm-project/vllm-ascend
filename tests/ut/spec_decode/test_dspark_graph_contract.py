# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as aclgraph_module
import vllm_ascend.worker.v2.spec_decode.dspark.speculator as speculator_module
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


class _BackendA:
    pass


def _speculator(**attributes):
    speculator = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    for name, value in attributes.items():
        setattr(speculator, name, value)
    return speculator


def test_set_attn_preserves_cache_group_order(monkeypatch):
    draft_config = object()
    active_context = []

    @contextmanager
    def config_context(config):
        assert config is draft_config
        active_context.append(config)
        try:
            yield
        finally:
            active_context.pop()

    def parent_set_attn(self, *_args):
        assert active_context == [draft_config]
        self._context_slot_mappings = torch.zeros(2, dtype=torch.int64)

    def get_layers(config, layer_type, names):
        assert active_context == [draft_config]
        return {name: SimpleNamespace(get_attn_backend=lambda: _BackendA) for name in names}

    monkeypatch.setattr(speculator_module, "set_current_vllm_config", config_context)
    monkeypatch.setattr(speculator_module.DSparkSpeculator, "set_attn", parent_set_attn)
    monkeypatch.setattr(speculator_module, "get_layers_from_vllm_config", get_layers)
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: draft_config))
    speculator = _speculator(vllm_config=object(), draft_attn_layer_names={"draft.2", "draft.0"})
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["draft.2", "target.0", "draft.0"])])

    speculator.set_attn(None, cache, None, None, None)

    assert list(speculator.attn_backends) == ["draft.2", "draft.0"]
    assert speculator._context_slot_mappings.dtype == torch.int32
    assert active_context == []


@pytest.mark.parametrize("query_count", [1, 7, 8])
@pytest.mark.parametrize("num_reqs_padded", [1, 2, 4, 16])
def test_update_draft_metadata_refreshes_all_padded_query_lengths(query_count, num_reqs_padded):
    speculator = _speculator(num_query_per_req=query_count)
    metadata = {
        name: SimpleNamespace(actual_seq_lengths_q=[query_count], seq_lens=object()) for name in ("draft.2", "draft.0")
    }
    seq_lens = {name: value.seq_lens for name, value in metadata.items()}

    assert speculator._update_draft_attn_metadata(metadata, num_reqs_padded) is metadata
    for name, value in metadata.items():
        assert value.actual_seq_lengths_q == [query_count * (i + 1) for i in range(num_reqs_padded)]
        assert value.seq_lens is seq_lens[name]


def test_replay_installs_forward_context_before_accessing_extra_ctx(monkeypatch):
    class _ContextProxy:
        active = False

        def __getattr__(self, name):
            if not self.active:
                raise AssertionError("forward context accessed before installation")
            return self.__dict__.get(name, False)

        def __setattr__(self, name, value):
            if name != "active" and not self.active:
                raise AssertionError("forward context accessed before installation")
            object.__setattr__(self, name, value)

    proxy = _ContextProxy()

    @contextmanager
    def _set_forward_context(*args, **kwargs):
        counts = kwargs["num_tokens_across_dp"]
        assert counts.device.type == "cpu"
        assert counts.tolist() == [7]
        proxy.active = True
        try:
            yield
        finally:
            proxy.active = False

    speculator = SimpleNamespace(
        num_query_per_req=7,
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
        build_draft_attn_metadatas=lambda *args: {"draft.0": object()},
        attn_backends={"draft.0": _BackendA},
        dp_size=1,
        model_state=SimpleNamespace(attn_metadata={}),
        speculative_config=object(),
    )
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = speculator
    manager.update_stream = SimpleNamespace(wait_stream=lambda stream: None)
    manager.device = torch.device("cpu")
    manager.vllm_config = object()

    monkeypatch.setattr(aclgraph_module, "_EXTRA_CTX", proxy)
    monkeypatch.setattr(aclgraph_module, "set_forward_context", _set_forward_context)
    monkeypatch.setattr(aclgraph_module, "get_forward_context", lambda: object())
    monkeypatch.setattr(aclgraph_module, "update_full_graph_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(DFlashCudaGraphManager, "run_fullgraph", lambda self, desc: "replayed")

    desc = SimpleNamespace(num_tokens=7, num_reqs=1, cg_mode=object())
    assert manager.run_fullgraph(desc) == "replayed"
    assert proxy.active is False


@pytest.mark.parametrize("query_count", [7, 8])
def test_dispatcher_pads_uniform_draft_descriptors(query_count):
    manager = DFlashCudaGraphManager.__new__(DFlashCudaGraphManager)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[16, 32, 48, 64, 80, 96, 112, 128],
        max_cudagraph_capture_size=128,
    )
    manager.vllm_config = SimpleNamespace(speculative_config=None)
    manager.max_num_reqs = 16
    manager.decode_query_len = query_count
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.varlen_decode = False
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {}
    manager._candidates = {}
    manager._capture_descs = {}
    manager._graphs_captured = True
    manager._init_candidates()

    for num_reqs in (1, 2, 3, 4, 8, 16):
        desc = manager.dispatch(num_reqs, num_reqs * query_count, query_count, 0)
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.num_reqs >= num_reqs
        assert desc.num_tokens == desc.num_reqs * query_count
        assert desc.uniform_token_count == query_count
