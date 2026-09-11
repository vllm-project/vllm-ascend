# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as aclgraph_module
import vllm_ascend.worker.v2.spec_decode.dspark.speculator as speculator_module
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


class _BackendA:
    pass


class _BackendB:
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


def test_single_draft_graph_backend_is_selected_explicitly():
    speculator = _speculator(attn_backends={"draft.0": _BackendA, "draft.1": _BackendA})
    assert speculator.get_draft_graph_backend() is _BackendA


@pytest.mark.parametrize("attn_backends", [{}, {"draft.0": _BackendA, "draft.1": _BackendB}])
def test_unsupported_draft_graph_backend_mapping_fails(attn_backends):
    with pytest.raises((RuntimeError, NotImplementedError), match="DSpark ACL graph"):
        _speculator(attn_backends=attn_backends).get_draft_graph_backend()


def test_draft_metadata_uses_padded_query_token_contract():
    speculator = _speculator(num_query_per_req=7)
    metadata = {"draft.0": SimpleNamespace(actual_seq_lengths_q=[7, 14, 21, 28])}
    speculator._validate_draft_attn_metadata(metadata, num_reqs_padded=4)


def test_draft_metadata_rejects_stale_unpadded_lengths():
    speculator = _speculator(num_query_per_req=7)
    metadata = {"draft.0": SimpleNamespace(actual_seq_lengths_q=[7, 14, 14, 14])}
    with pytest.raises(RuntimeError, match="query-length mismatch"):
        speculator._validate_draft_attn_metadata(metadata, num_reqs_padded=4)


def test_graph_param_cardinality_accepts_complete_capture():
    graph_params = SimpleNamespace(
        attn_params={28: [object(), object()]},
        handles={28: [object(), object()]},
        events={28: [object(), object()]},
    )
    assert DFlashAclGraphManager._validate_graph_param_cardinality(graph_params, 28) == 2


@pytest.mark.parametrize(
    "attn_count,handle_count,event_count",
    [(0, 0, 0), (2, 1, 2), (2, 2, 1)],
)
def test_graph_param_cardinality_rejects_incomplete_capture(attn_count, handle_count, event_count):
    graph_params = SimpleNamespace(
        attn_params={28: [object()] * attn_count},
        handles={28: [object()] * handle_count},
        events={28: [object()] * event_count},
    )
    with pytest.raises(RuntimeError, match="captured no|cardinality mismatch"):
        DFlashAclGraphManager._validate_graph_param_cardinality(graph_params, 28)


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
        proxy.active = True
        try:
            yield
        finally:
            proxy.active = False

    graph_params = SimpleNamespace(
        attn_params={7: [object()]},
        handles={7: [object()]},
        events={7: [object()]},
    )
    speculator = SimpleNamespace(
        num_query_per_req=7,
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
        build_draft_attn_metadatas=lambda *args: {"draft.0": object()},
        get_draft_graph_backend=lambda: object(),
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
    monkeypatch.setattr(aclgraph_module, "get_draft_graph_params", lambda: graph_params)
    monkeypatch.setattr(aclgraph_module, "update_full_graph_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(DFlashCudaGraphManager, "run_fullgraph", lambda self, desc: "replayed")

    desc = SimpleNamespace(num_tokens=7, num_reqs=1, cg_mode=object())
    assert manager.run_fullgraph(desc) == "replayed"
    assert proxy.active is False
