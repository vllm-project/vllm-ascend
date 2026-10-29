# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# ruff: noqa: E402

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip(
    "vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator",
    reason="Requires vLLM H-Spec support (PR #62780)",
)

from vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.speculator import MambaAttnHybridSpeculator

import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as graph_module
import vllm_ascend.worker.v2.spec_decode.mamba_attn_hybrid.speculator as hspec_module
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.models.mamba_attn_hybrid import AscendMambaAttnHybridForCausalLM
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.mamba_attn_hybrid.speculator import AscendMambaAttnHybridSpeculator


def _config():
    draft = SimpleNamespace(
        get_hidden_size=lambda: 8,
        hf_config=SimpleNamespace(
            sample_from_anchor=False,
            eagle_aux_hidden_state_layer_ids=[1, 3],
            block_pattern=["mamba", "attention", "mlp"],
        ),
    )
    return SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=2,
        ),
        model_config=SimpleNamespace(get_hidden_size=lambda: 8),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=SimpleNamespace(
            method="mamba_attn_hybrid",
            enforce_eager=False,
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=2),
            draft_model_config=draft,
            kv_cache_dtype=None,
        ),
    )


def test_factory_selects_hspec_before_dflash(monkeypatch):
    config = _config()
    constructor = MagicMock()
    monkeypatch.setattr(hspec_module, "AscendMambaAttnHybridSpeculator", constructor)
    assert init_speculator(config, torch.device("cpu")) is constructor.return_value


@pytest.mark.parametrize(
    "field", ["pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"]
)
def test_rejects_sharded_target_cache(field):
    config = _config()
    setattr(config.parallel_config, field, 2)
    with pytest.raises(ValueError, match="PP=PCP=DCP=1"):
        AscendMambaAttnHybridSpeculator._validate_config(config)


def test_requires_matching_cache_shards_and_latent_dimensions():
    config = _config()
    AscendMambaAttnHybridSpeculator._validate_config(config)
    config.speculative_config.draft_parallel_config.tensor_parallel_size = 1
    with pytest.raises(ValueError, match="tensor parallel"):
        AscendMambaAttnHybridSpeculator._validate_config(config)
    config.speculative_config.draft_parallel_config.tensor_parallel_size = 2
    config.model_config.get_hidden_size = lambda: 16
    with pytest.raises(ValueError, match="hidden sizes"):
        AscendMambaAttnHybridSpeculator._validate_config(config)


@pytest.mark.parametrize("sample_from_anchor", [False, True, None])
def test_query_layout(sample_from_anchor):
    config = _config()
    config.speculative_config.draft_model_config.hf_config.sample_from_anchor = sample_from_anchor
    if sample_from_anchor is not False:
        with pytest.raises(ValueError, match="1\\+K"):
            AscendMambaAttnHybridSpeculator._validate_config(config)
    else:
        AscendMambaAttnHybridSpeculator._validate_config(config)


def test_mro_keeps_upstream_latent_anchor_and_model_forward():
    mro = AscendMambaAttnHybridSpeculator.__mro__
    assert mro.index(AscendDSparkSpeculator) < mro.index(MambaAttnHybridSpeculator)
    assert AscendMambaAttnHybridSpeculator.prepare_context_anchor is MambaAttnHybridSpeculator.prepare_context_anchor
    assert AscendMambaAttnHybridSpeculator._sample_sequential is AscendDSparkSpeculator._sample_sequential


def test_markov_sampling_uses_bonus_then_previous_draft():
    spec = object.__new__(AscendMambaAttnHybridSpeculator)
    spec.num_speculative_steps = 3
    spec._draft_topk = None
    spec._step_cols = torch.arange(3)
    spec.acceptance_estimator = None
    spec.use_confidence_head = False
    spec.draft_logits = None
    spec.draft_tokens = torch.full((2, 3), -1, dtype=torch.int64)
    spec.sample_indices = torch.tensor([1, 2, 3, 5, 6, 7])
    spec.sample_idx_mapping = torch.tensor([0, 0, 0, 1, 1, 1])
    spec.sample_pos = torch.tensor([11, 12, 13, 21, 22, 23])
    spec._anchor_idx = torch.tensor([0, 4])
    spec.input_buffers = SimpleNamespace(input_ids=torch.tensor([1, 0, 0, 0, 2, 0, 0, 0]))
    spec.model = SimpleNamespace(
        compute_draft_logits=lambda hidden: hidden,
        markov_embed=lambda ids: ids,
        markov_bias=lambda ids: torch.nn.functional.one_hot((ids + 1) % 4, 4).float() * 10,
        map_draft_to_target=lambda ids: ids,
    )
    spec._sample_sequential(2, torch.zeros(8, 4))
    assert spec.draft_tokens.tolist() == [[2, 3, 0], [3, 0, 1]]


@pytest.mark.parametrize("rejected", [[0, 0], [1, 2]])
def test_latent_anchor_uses_last_accepted_auxiliary_row(rejected):
    spec = object.__new__(AscendMambaAttnHybridSpeculator)
    spec.hidden_states = torch.arange(32, dtype=torch.float32).view(8, 4)
    spec.latent_seed = torch.zeros(3, 4)
    starts = torch.tensor([0, 3, 8])
    rejected_tensor = torch.tensor(rejected)
    spec.prepare_context_anchor(2, starts, rejected_tensor)
    expected_rows = starts[1:] - rejected_tensor - 1
    torch.testing.assert_close(spec.latent_seed[:2], spec.hidden_states[expected_rows])
    assert torch.count_nonzero(spec.latent_seed[2]) == 0


def test_exact_kv_lengths_are_reused_and_padding_is_inert():
    spec = object.__new__(AscendMambaAttnHybridSpeculator)
    spec._exact_seq_lens_cpu = None
    spec.input_buffers = SimpleNamespace(seq_lens=torch.tensor([8, 13, 99], dtype=torch.int32))
    first, is_prefilling = spec._prepare_draft_dcp_metadata_inputs(2, 3, 4)
    assert first.tolist() == [8, 13, 0]
    assert not is_prefilling.any()
    cached = spec._exact_seq_lens_cpu
    second, _ = spec._prepare_draft_dcp_metadata_inputs(2, 4, 4)
    assert spec._exact_seq_lens_cpu is cached
    assert second.tolist() == [8, 13, 0, 0]
    spec.input_buffers.seq_lens[0] = 100
    assert cached.tolist() == [8, 13]


@pytest.mark.parametrize("fail", [False, True])
def test_proposal_clears_exact_length_cache(monkeypatch, fail):
    spec = object.__new__(AscendMambaAttnHybridSpeculator)
    spec._exact_seq_lens_cpu = torch.tensor([99])

    def propose(self):
        assert self._exact_seq_lens_cpu is None
        self._exact_seq_lens_cpu = torch.tensor([8])
        if fail:
            raise RuntimeError("model failed")
        return "drafts"

    monkeypatch.setattr(AscendDSparkSpeculator, "propose", propose)
    if fail:
        with pytest.raises(RuntimeError, match="model failed"):
            spec.propose()
    else:
        assert spec.propose() == "drafts"
    assert spec._exact_seq_lens_cpu is None


def _attention(target_name=None, **updates):
    fields = dict(
        num_kv_heads=2,
        head_size=4,
        head_size_v=4,
        kv_cache_dtype="auto",
        kv_sharing_target_layer_name=target_name,
        impl=SimpleNamespace(kv_sharing_target_layer_name=target_name, sliding_window=None),
        get_attn_backend=lambda: AscendAttentionBackend,
    )
    fields.update(updates)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "mismatch", [None, "num_kv_heads", "head_size", "head_size_v", "kv_cache_dtype", "backend", "sliding_window"]
)
def test_shared_cache_validation_and_candidate_writes(monkeypatch, mismatch):
    target_name = "model.layers.2.self_attn.attn"
    target, draft = _attention(), _attention(target_name)
    if mismatch == "backend":
        target.get_attn_backend = lambda: object
    elif mismatch == "sliding_window":
        target.impl.sliding_window = 128
    elif mismatch:
        setattr(target, mismatch, "different")
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=draft))]))
    monkeypatch.setattr(AscendDSparkSpeculator, "load_draft_model", lambda *args: model)
    monkeypatch.setattr(hspec_module, "get_layers_from_vllm_config", lambda *args: {target_name: target})
    spec = object.__new__(AscendMambaAttnHybridSpeculator)
    spec.vllm_config = object()
    if mismatch:
        with pytest.raises(ValueError):
            spec.load_draft_model(object(), {target_name})
    else:
        assert spec.load_draft_model(object(), {target_name}) is model
        assert draft.kv_sharing_target_layer_name == target_name
        assert draft.impl.kv_sharing_target_layer_name is None


def test_model_reports_attention_order_and_causality():
    model = object.__new__(AscendMambaAttnHybridForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(
        layers=[
            SimpleNamespace(),
            SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(layer_name="draft.2"), causal=False)),
            SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(layer_name="draft.4"), causal=True)),
        ]
    )
    assert model.get_draft_kv_cache_layer_names() == ["draft.2", "draft.4"]
    assert model.get_draft_attn_causal() == [False, True]


@pytest.mark.parametrize("release", [False, True])
def test_capture_forwards_versioned_preparation_callback(monkeypatch, release):
    parent = DFlashAclGraphManager.__mro__[1]
    capture = MagicMock()
    monkeypatch.setattr(parent, "capture", capture)
    monkeypatch.setattr(graph_module, "vllm_version_is", lambda version: release)
    monkeypatch.setattr(graph_module, "communicator_switch", lambda: nullcontext())
    monkeypatch.setattr(graph_module, "model_capture_wrapper", lambda *args: nullcontext())
    manager = object.__new__(DFlashAclGraphManager)
    manager.speculator = object()
    prepare = MagicMock()
    callback = {"precompute_context_kv": prepare} if release else {"prepare_forward": prepare}
    manager.capture(MagicMock(), object(), object(), [], object(), 128, **callback)
    assert capture.call_args.kwargs[next(iter(callback))] is prepare
