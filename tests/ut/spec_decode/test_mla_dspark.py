# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadata
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata
from vllm_ascend.models import kimi_k3_dspark
from vllm_ascend.models.kimi_k3 import AscendKimiLinearModel
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as shared
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def make_speculator():
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.attn_architecture = "MLA"
    spec.vllm_config = SimpleNamespace()
    spec.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(target_layer_ids=[0, 2], target_hidden_size=4, num_target_layers=2)
    )
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


@pytest.mark.parametrize("target_use_mla", [False, True])
@pytest.mark.parametrize(
    "architecture,draft_use_mla,text_fields,outer_fields,expect_dense_mla",
    [
        ("K3DSparkModel", True, {}, {}, True),
        ("OtherMLADraftModel", True, {}, {}, True),
        ("K3DSparkModel", False, {}, {}, False),
        ("Qwen3DSparkModel", False, {}, {}, False),
        ("DSparkDraftModel", False, {}, {}, False),
        ("DSparkDraftModel", True, {"compress_ratios": [0, 4, 128], "index_topk": 2048}, {}, False),
        ("CompressedMLADraftModel", True, {"compress_ratios": [0]}, {}, False),
        ("WrappedMLADraftModel", True, {}, {"compress_ratios": [0, 4]}, False),
        ("SparseMLADraftModel", True, {"index_topk": 2048}, {}, False),
        ("KpoolMLADraftModel", True, {"index_topk": 2048, "index_kpool": 4}, {}, True),
    ],
)
def test_shared_speculator_selects_draft_mla_metadata(
    monkeypatch, architecture, draft_use_mla, target_use_mla, text_fields, outer_fields, expect_dense_mla
):
    patch_upstream_init(monkeypatch)
    text_config = SimpleNamespace(**text_fields)
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=target_use_mla),
        speculative_config=SimpleNamespace(
            method="dspark",
            use_dspark=lambda: True,
            draft_model_config=SimpleNamespace(
                use_mla=draft_use_mla,
                hf_config=SimpleNamespace(architectures=[architecture], **outer_fields),
                hf_text_config=text_config,
            ),
        ),
    )
    device = torch.device("cpu")
    result = init_speculator(config, device)
    assert type(result) is AscendDSparkSpeculator
    assert result.attn_architecture == ("MLA" if expect_dense_mla else None)


def patch_upstream_init(monkeypatch):
    def init(self, config, device):
        self.draft_model_config = config.speculative_config.draft_model_config

    monkeypatch.setattr(DSparkSpeculator, "__init__", init)
    monkeypatch.setattr(shared, "prepare_replicated_pcp_config", lambda config: (config, False))


@pytest.mark.parametrize(
    "metadata_cls,config_fields",
    [
        (AscendDSAMetadata, {"compress_ratios": [0, 4, 128], "index_topk": 2048}),
        (AscendSFAMetadata, {"index_topk": 2048}),
    ],
)
def test_sparse_mla_metadata_keeps_shared_update(monkeypatch, metadata_cls, config_fields):
    patch_upstream_init(monkeypatch)
    hf_config = SimpleNamespace(architectures=["DSparkDraftModel"], **config_fields)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            use_dspark=lambda: True,
            draft_model_config=SimpleNamespace(use_mla=True, hf_config=hf_config, hf_text_config=hf_config),
        )
    )
    selected = init_speculator(config, torch.device("cpu"))
    assert type(selected) is AscendDSparkSpeculator
    assert selected.attn_architecture is None
    selected.num_query_per_req = 5
    # Use the backend's actual type: neither DSA nor SFA has a dense decode
    # object. Only the fields touched by the shared update are needed here.
    metadata = metadata_cls.__new__(metadata_cls)
    assert not hasattr(metadata, "decode")
    layers = {"draft": metadata}
    assert selected._update_draft_attn_metadata(layers, 2) is layers
    assert metadata.actual_seq_lengths_q == [5, 10]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("rotation_path", [None, "/rotation"])
def test_shared_loader_configures_mla_model(monkeypatch, wrapped, rotation_path):
    spec, target = make_speculator(), make_target()
    config = spec.draft_model_config.hf_config
    monkeypatch.setattr(shared, "get_rotation_path", lambda _: rotation_path)
    draft = make_draft(config)

    def load(*args):
        assert config._ascend_target_rotation_path == rotation_path
        return draft

    # Keep the actual shared loader: only skip upstream weight construction.
    # This exercises shared rotation injection and its model configuration hook.
    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", load)
    outer = SimpleNamespace(get_language_model=lambda: target) if wrapped else target
    assert spec.load_draft_model(outer, set()) is draft
    target.set_dspark_aux_capture_materialized.assert_called_once_with(False)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("target_layer_ids", [], "requires target_layer_ids"),
        ("target_layer_ids", [0, 0], "Invalid"),
        ("target_layer_ids", [-1, 2], "Invalid"),
        ("target_layer_ids", [0, 9], "Invalid"),
        ("target_layer_ids", [1, 2], "boundaries do not match"),
        ("target_hidden_size", 8, "hidden sizes"),
        ("num_target_layers", 3, "num_target_layers"),
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

    # Exercise the inherited Kimi implementation, including its AttnRes guard.
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


@pytest.mark.parametrize("architecture", [None, "MLA"])
def test_empty_metadata_is_a_noop(architecture):
    spec = make_speculator()
    spec.attn_architecture = architecture
    metadata = {}
    assert spec._update_draft_attn_metadata(metadata, 1) is metadata


def test_capture_uses_descriptor_positions_and_restores_on_error(monkeypatch):
    spec = make_speculator()
    original = shared.dflash_cudagraph.build_attn_metadata
    builder = MagicMock()
    monkeypatch.setattr(shared, "build_attn_metadata", builder)
    with pytest.raises(RuntimeError, match="capture failed"), spec.draft_capture_context():
        shared.dflash_cudagraph.build_attn_metadata(num_tokens=10, num_reqs=2, causal=False)
        kwargs = builder.call_args.kwargs
        torch.testing.assert_close(kwargs["positions"], torch.arange(10))
        assert kwargs["is_prefilling"].tolist() == [False, False]
        assert kwargs["attn_state"] == AscendAttentionState.SpecDecoding
        assert kwargs["causal"] is False
        raise RuntimeError("capture failed")
    assert shared.dflash_cudagraph.build_attn_metadata is original


def test_non_mla_capture_keeps_shared_builder():
    spec = make_speculator()
    spec.attn_architecture = None
    original = shared.dflash_cudagraph.build_attn_metadata
    with spec.draft_capture_context():
        assert shared.dflash_cudagraph.build_attn_metadata is original
    assert shared.dflash_cudagraph.build_attn_metadata is original


@pytest.mark.parametrize("architecture", [None, "MLA"])
def test_replay_metadata_preserves_architecture_behavior(monkeypatch, architecture):
    spec = make_speculator()
    spec.attn_architecture = architecture
    spec.input_batch = SimpleNamespace(num_reqs=1, is_prefilling_np=np.array([True, True]))
    spec._group_causal = {0: False}
    query_metadata = SimpleNamespace(actual_seq_lengths_q=[5, 5])
    metadata = {"draft": SimpleNamespace(decode=query_metadata) if architecture == "MLA" else query_metadata}
    spec._build_draft_attn_metadata = MagicMock(return_value=metadata)
    captured = {}

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
        assert result[0]["draft"].attn_state == AscendAttentionState.SpecDecoding
    else:
        assert captured["is_prefilling"].tolist() == [True, True]
        assert np.shares_memory(captured["is_prefilling"].numpy(), spec.input_batch.is_prefilling_np)
        assert not hasattr(result[0]["draft"], "attn_state")
    kwargs = spec._build_draft_attn_metadata.call_args.kwargs
    assert kwargs["num_reqs"] == 1
    assert kwargs["num_reqs_padded"] == 2
    assert kwargs["causal"] == {0: False}
    assert spec.input_batch.is_prefilling_np.tolist() == [True, True]


@pytest.mark.parametrize(
    "direct,preserved,expected", [("target", "fallback", "target"), (None, "fallback", "fallback"), (None, None, None)]
)
def test_mla_model_recovers_target_rotation(monkeypatch, direct, preserved, expected):
    draft_config = SimpleNamespace(hf_config=SimpleNamespace(_ascend_target_rotation_path=preserved))
    config = SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft_config))
    monkeypatch.setattr(kimi_k3_dspark, "get_rotation_path", lambda _: direct)
    assert kimi_k3_dspark._get_target_rotation_path(config) == expected
