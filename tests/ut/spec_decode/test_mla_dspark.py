# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSABackend, AscendDSAMetadata
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAMetadata
from vllm_ascend.models import kimi_k3_dspark
from vllm_ascend.models.kimi_k3 import AscendKimiLinearModel
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as graph
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as shared
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def make_speculator():
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.attn_architecture = "MLA"
    spec.vllm_config = SimpleNamespace()
    spec.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(target_layer_ids=[0, 2], target_hidden_size=4, num_target_layers=2)
    )
    spec.vllm_config.speculative_config = SimpleNamespace(draft_model_config=spec.draft_model_config)
    spec.num_query_per_req = 5
    spec.input_buffers = SimpleNamespace(positions=torch.arange(20))
    return spec


def make_target():
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=4, hidden_size=4),
            aux_hidden_state_layers=(1, 3),
        ),
        set_dspark_aux_capture_materialized=MagicMock(),
    )


def make_draft(config):
    draft = kimi_k3_dspark.AscendK3DSparkForCausalLM.__new__(kimi_k3_dspark.AscendK3DSparkForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.config = config
    return draft


class DerivedMLABackend(AscendMLABackend):
    pass


@pytest.mark.parametrize("target_backend", [AscendMLABackend, AscendAttentionBackend])
@pytest.mark.parametrize(
    "draft_backend,expected",
    [
        (AscendMLABackend, "MLA"),
        (DerivedMLABackend, "MLA"),
        (AscendAttentionBackend, "GQA"),
        (AscendDSABackend, None),
        (AscendSFABackend, None),
    ],
)
def test_shared_speculator_selects_draft_backend(monkeypatch, target_backend, draft_backend, expected):
    spec = initialize_attention(monkeypatch, draft_backend, target_backend)
    assert spec.attn_architecture == expected
    assert spec.attn_backends == {"draft": draft_backend}


def initialize_attention(monkeypatch, draft_backend, target_backend=AscendMLABackend):
    monkeypatch.setattr(draft_backend, "get_impl_cls", staticmethod(lambda: object))
    monkeypatch.setattr(DSparkSpeculator, "__init__", lambda self, *args: None)
    monkeypatch.setattr(shared, "prepare_replicated_pcp_config", lambda config: (config, False))
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="dspark", use_dspark=lambda: True))
    spec = init_speculator(config, torch.device("cpu"))
    assert type(spec) is AscendDSparkSpeculator
    assert spec.attn_architecture is None
    spec.vllm_config = config
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: config))
    spec.draft_attn_layer_names = {"draft"}
    spec._context_slot_mappings = torch.zeros(1, dtype=torch.int64)
    target_groups = [[SimpleNamespace(backend=target_backend)]]
    draft_groups = [[], [SimpleNamespace(backend=draft_backend)]]

    def set_attn(self, model_state, kv_cache_config, block_tables, input_buffers, target_attn_groups):
        assert target_attn_groups is target_groups
        self.attn_groups = draft_groups

    monkeypatch.setattr(DSparkSpeculator, "set_attn", set_attn)
    monkeypatch.setattr(shared, "set_current_vllm_config", lambda _: nullcontext())

    def get_layers(config, layer_type, layer_names):
        assert layer_names == ["draft"]
        return {"draft": SimpleNamespace(get_attn_backend=lambda: draft_backend)}

    monkeypatch.setattr(shared, "get_layers_from_vllm_config", get_layers)
    cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["target", "draft"])])
    spec.set_attn(None, cache_config, None, None, target_groups)
    assert spec._context_slot_mappings.dtype == torch.int32
    return spec


@pytest.mark.parametrize(
    "backend,metadata_cls", [(AscendDSABackend, AscendDSAMetadata), (AscendSFABackend, AscendSFAMetadata)]
)
def test_sparse_mla_metadata_keeps_shared_update(monkeypatch, backend, metadata_cls):
    spec = initialize_attention(monkeypatch, backend)
    assert spec.attn_architecture is None
    spec.num_query_per_req = 5
    metadata = metadata_cls.__new__(metadata_cls)
    assert not hasattr(metadata, "decode")
    layers = {"draft": metadata}
    assert spec._update_draft_attn_metadata(layers, 2) is layers
    assert metadata.actual_seq_lengths_q == [5, 10]


@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_loader_configures_mla_model(monkeypatch, wrapped):
    spec, target = make_speculator(), make_target()
    config = spec.draft_model_config.hf_config
    draft = make_draft(config)
    draft.post_process = MagicMock()
    monkeypatch.setattr(shared, "set_current_vllm_config", lambda _: nullcontext())

    def load(*args):
        return draft

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", load)
    outer = SimpleNamespace(get_language_model=lambda: target) if wrapped else target
    assert spec.load_draft_model(outer, set()) is draft
    target.set_dspark_aux_capture_materialized.assert_called_once_with(False)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("target_layer_ids", [], "incompatible"),
        ("target_layer_ids", [0, 0], "incompatible"),
        ("target_layer_ids", [-1, 2], "incompatible"),
        ("target_layer_ids", [0, 9], "incompatible"),
        ("target_layer_ids", [1, 2], "incompatible"),
        ("target_hidden_size", 8, "incompatible"),
    ],
)
def test_rejects_invalid_raw_contract(field, value, message):
    spec = make_speculator()
    setattr(spec.draft_model_config.hf_config, field, value)
    draft = make_draft(spec.draft_model_config.hf_config)
    with pytest.raises(ValueError, match=message):
        draft.configure_target_aux_hidden_capture(make_target())


@pytest.mark.parametrize("layer_idx", [1, 2, 3])
@pytest.mark.parametrize("has_residual", [False, True])
def test_raw_prefix_capture_does_not_add_attnres_bank(layer_idx, has_residual):
    state = torch.arange(8).view(2, 4)
    residual = torch.ones(2, 3, 4) if has_residual else None
    target = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    torch.nn.Module.__init__(target)
    target.config = SimpleNamespace(attn_res_block_size=2)
    target.aux_hidden_state_layers = (1, 3)

    captured = target._maybe_add_hidden_state([], layer_idx, state, residual)
    if layer_idx in target.aux_hidden_state_layers:
        assert len(captured) == 1
        assert captured[0] is state
        assert captured[0].ndim == 2
    else:
        assert captured == []
    if residual is not None:
        torch.testing.assert_close(residual, torch.ones(2, 3, 4))


def test_padded_mla_query_lengths_are_nested():
    spec = make_speculator()
    metadata = {"draft.0": SimpleNamespace(decode=SimpleNamespace(actual_seq_lengths_q=[5, 5]))}
    assert spec._update_draft_attn_metadata(metadata, 2) is metadata
    assert metadata["draft.0"].decode.actual_seq_lengths_q == [5, 10]
    assert not hasattr(metadata["draft.0"], "actual_seq_lengths_q")


@pytest.mark.parametrize("architecture", [None, "GQA", "MLA"])
def test_empty_metadata_is_a_noop(architecture):
    spec = make_speculator()
    spec.attn_architecture = architecture
    metadata: dict[str, SimpleNamespace] = {}
    assert spec._update_draft_attn_metadata(metadata, 1) is metadata


@pytest.mark.parametrize("architecture", [None, "GQA", "MLA"])
@pytest.mark.parametrize("fail", [False, True])
def test_capture_delegates_and_restores_contexts(monkeypatch, architecture, fail):
    manager = graph.DFlashAclGraphManager.__new__(graph.DFlashAclGraphManager)
    manager.speculator = SimpleNamespace(attn_architecture=architecture)
    events = []

    @contextmanager
    def context(name):
        events.append(f"enter {name}")
        try:
            yield
        finally:
            events.append(f"exit {name}")

    monkeypatch.setattr(graph, "communicator_switch", lambda: context("communicator"))

    def model_context(speculator, is_prefill):
        assert speculator is manager.speculator
        assert is_prefill is False
        return context("model")

    monkeypatch.setattr(graph, "model_capture_wrapper", model_context)
    args: tuple[Any, ...] = (
        MagicMock(),
        SimpleNamespace(positions=torch.arange(20)),
        object(),
        [],
        object(),
        128,
        False,
        "capture",
    )

    def capture(self, *received):
        assert self is manager
        assert received == args
        assert events == ["enter communicator", "enter model"]
        events.append("capture")
        if fail:
            raise RuntimeError("capture failed")

    monkeypatch.setattr(graph.DFlashCudaGraphManager, "capture", capture)
    with pytest.raises(RuntimeError, match="capture failed") if fail else nullcontext():
        manager.capture(*args)
    assert events == ["enter communicator", "enter model", "capture", "exit model", "exit communicator"]


@pytest.mark.parametrize("architecture", [None, "GQA", "MLA"])
def test_replay_metadata_preserves_architecture_behavior(monkeypatch, architecture):
    spec = make_speculator()
    spec.attn_architecture = architecture
    spec.input_batch = SimpleNamespace(num_reqs=1, is_prefilling_np=np.array([True, True]))
    spec._group_causal = {0: False}
    query_metadata = SimpleNamespace(actual_seq_lengths_q=[5, 5])
    metadata = {"draft": SimpleNamespace(decode=query_metadata) if architecture == "MLA" else query_metadata}
    builder = MagicMock(return_value=metadata)
    monkeypatch.setattr(DSparkSpeculator, "_build_draft_attn_metadata", builder)
    update = MagicMock(wraps=spec._update_draft_attn_metadata)
    monkeypatch.setattr(spec, "_update_draft_attn_metadata", update)
    captured: dict[str, Any] = {}

    @contextmanager
    def factory(positions, pad, is_prefilling):
        captured.update(pad=pad, is_prefilling=is_prefilling)
        yield

    monkeypatch.setattr(shared, "build_draft_attn_metadata_factory", factory)
    result = spec.build_draft_attn_metadatas(2, torch.tensor([128]))
    assert captured["pad"] == 10
    assert result == [metadata]
    assert query_metadata.actual_seq_lengths_q == [5, 10]
    if architecture == "MLA":
        assert captured["is_prefilling"].tolist() == [False, False]
        assert result[0]["draft"].attn_state == AscendAttentionState.ChunkedPrefill
    else:
        assert captured["is_prefilling"].tolist() == [True, True]
        assert np.shares_memory(captured["is_prefilling"].numpy(), spec.input_batch.is_prefilling_np)
        assert not hasattr(result[0]["draft"], "attn_state")
    builder.assert_called_once()
    update.assert_called_once_with(metadata, 2)
    kwargs = builder.call_args.kwargs
    assert "update_query_lengths" not in kwargs
    assert kwargs["num_reqs"] == 1
    assert kwargs["num_reqs_padded"] == 2
    assert kwargs["causal"] == {0: False}
    assert spec.input_batch.is_prefilling_np.tolist() == [True, True]


@pytest.mark.parametrize("missing", [False, True])
def test_missing_target_aux_layers_reports_incompatibility(missing):
    target = make_target()
    if missing:
        del target.model.aux_hidden_state_layers
    else:
        target.model.aux_hidden_state_layers = None
    with pytest.raises(ValueError, match="incompatible"):
        make_draft(make_speculator().draft_model_config.hf_config).configure_target_aux_hidden_capture(target)


@pytest.mark.parametrize("fail", [False, True])
def test_default_metadata_factory_preserves_caller_state(monkeypatch, fail):
    module = attn_utils._BUILD_ATTN_METADATA_MODULE
    original = module.build_attn_metadata
    builder = MagicMock()
    flags = torch.tensor([True, False])
    monkeypatch.setattr(attn_utils, "build_attn_metadata", builder)
    with (
        pytest.raises(RuntimeError, match="build failed") if fail else nullcontext(),
        attn_utils.build_attn_metadata_wrapper(),
        attn_utils.build_draft_attn_metadata_factory(torch.arange(10), 6, flags),
    ):
        module.build_attn_metadata(num_tokens=6, attn_state=AscendAttentionState.DecodeOnly)
        if fail:
            raise RuntimeError("build failed")
    assert module.build_attn_metadata is original
    assert builder.call_args.kwargs["is_prefilling"] is flags
    assert builder.call_args.kwargs["attn_state"] == AscendAttentionState.DecodeOnly
    torch.testing.assert_close(builder.call_args.kwargs["positions"], torch.arange(6))
