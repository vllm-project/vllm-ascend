# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MRV2 GQA DSpark target/draft contract."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.model_loader.utils import get_model_cls
from vllm.model_executor.models import ModelRegistry
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

import vllm_ascend.worker.v2.spec_decode.dspark.speculator as dspark_module
from vllm_ascend.models import register_model
from vllm_ascend.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM
from vllm_ascend.models.qwen3_dspark import (
    AscendQwen3DSparkForCausalLM,
    process_weight,
)
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)

_HIDDEN = 8


def _spec(vllm_config, draft_hf_config) -> AscendDSparkSpeculator:
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.vllm_config = vllm_config
    spec.draft_model_config = SimpleNamespace(hf_config=draft_hf_config)
    vllm_config.speculative_config = SimpleNamespace(draft_model_config=spec.draft_model_config)
    return spec


def _target() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(embed_tokens=object()),
        lm_head=object(),
        set_dspark_aux_capture_materialized=MagicMock(),
    )


def _gqa_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["Qwen3DSparkModel"],
        model_type="qwen3",
        dspark_aux_hidden_state_format="materialized",
    )


def _vllm_config(*, quarot: bool) -> SimpleNamespace:
    quant_config = None
    if quarot:
        quant_config = SimpleNamespace(
            quant_description={"optional": {"quarot": {"rotation_map": {"global_rotation": "rotation.safetensors"}}}}
        )
    return SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(model="/target"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )


def _draft():
    draft = AscendQwen3DSparkForCausalLM.__new__(AscendQwen3DSparkForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.config = _gqa_config()
    return draft


@pytest.mark.parametrize("architecture", ["Qwen3DSparkModel", "Qwen3OmniDSparkModel", "DSparkDraftModel"])
def test_registered_draft_class_declares_capabilities(architecture, monkeypatch):
    monkeypatch.setattr(ModelRegistry, "models", ModelRegistry.models.copy())
    register_model()
    config = SimpleNamespace(
        model=f"/test/{architecture}",
        convert_type="none",
        runner_type="generate",
        trust_remote_code=False,
        model_impl="vllm",
        hf_config=SimpleNamespace(architectures=[architecture]),
        registry=ModelRegistry,
        _get_transformers_backend_cls=lambda: "TransformersForCausalLM",
    )
    draft_cls = get_model_cls(config)
    if architecture == "DSparkDraftModel":
        assert not hasattr(draft_cls, "configure_target_aux_hidden_capture")
    else:
        assert draft_cls is AscendQwen3DSparkForCausalLM


def test_draft_without_hook_preserves_target_capture(monkeypatch):
    config = SimpleNamespace(architectures=["DSparkDraftModel"])
    target = _target()
    draft = object()

    def load(*args):
        return draft

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", load)
    assert _spec(_vllm_config(quarot=True), config).load_draft_model(target, set()) is draft
    target.set_dspark_aux_capture_materialized.assert_not_called()
    assert vars(config) == {"architectures": ["DSparkDraftModel"]}


def test_qwen3_class_selects_materialized_target_capture():
    target = _target()
    _draft().configure_target_aux_hidden_capture(target)
    target.set_dspark_aux_capture_materialized.assert_called_once_with(True)


def test_undeclared_format_uses_materialized_capture():
    target = _target()
    draft = _draft()
    del draft.config.dspark_aux_hidden_state_format
    draft.configure_target_aux_hidden_capture(target)
    target.set_dspark_aux_capture_materialized.assert_called_once_with(True)


def test_wrapped_target_capture():
    target = _target()
    _draft().configure_target_aux_hidden_capture(SimpleNamespace(get_language_model=lambda: target))
    target.set_dspark_aux_capture_materialized.assert_called_once_with(True)


@pytest.mark.parametrize("wrapped", [False, True])
def test_missing_target_setter_preserves_native_behavior(wrapped):
    target = SimpleNamespace()
    model = SimpleNamespace(get_language_model=lambda: target) if wrapped else target
    _draft().configure_target_aux_hidden_capture(model)
    assert vars(target) == {}


def test_configures_capture_after_loading_draft(monkeypatch):
    events = []
    target = _target()
    target.set_dspark_aux_capture_materialized = lambda enabled: events.append(("capture", enabled))
    draft = _draft()
    draft.post_process = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.set_current_vllm_config", lambda _: nullcontext()
    )

    def _load(self, target_model, target_attn_layer_names):
        events.append(("load", None))
        return draft

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _load)
    spec = _spec(_vllm_config(quarot=False), _gqa_config())

    assert spec.load_draft_model(target, set()) is draft
    assert events == [("load", None), ("capture", True)]


def test_process_weight_preserves_the_unrotated_projection():
    generator = torch.Generator().manual_seed(7)
    rotation, _ = torch.linalg.qr(torch.randn(_HIDDEN, _HIDDEN, dtype=torch.float64, generator=generator))
    inputs = torch.randn(3, 5, _HIDDEN, dtype=torch.float64, generator=generator)
    weight = torch.randn(_HIDDEN, 5 * _HIDDEN, dtype=torch.float64, generator=generator)

    expected = torch.nn.functional.linear(inputs.reshape(3, -1), weight)
    actual = torch.nn.functional.linear((inputs @ rotation).reshape(3, -1), process_weight(weight, rotation))

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-6)


@pytest.mark.parametrize("num_rows", [0, 3, 8, 9])
def test_dspark_lmhead_aligns_collective_rows_and_trims_logits(monkeypatch, num_rows):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.max_num_reqs, spec.num_speculative_steps = 4, 2
    inputs_seen = []
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def compute_logits(hidden):
        inputs_seen.append(hidden.clone())
        return hidden @ weight

    model = SimpleNamespace(compute_draft_logits=compute_logits)
    monkeypatch.setattr(dspark_module, "lmhead_tp_enable", lambda: True)
    spec._lmhead_tp_wrap_draft_logits(model)
    hidden = torch.arange(num_rows * 2, dtype=torch.float32).reshape(num_rows, 2)

    if num_rows > 8:
        with pytest.raises(ValueError, match="exceed the group-agreed capacity"):
            model.compute_draft_logits(hidden)
        assert inputs_seen == []
    else:
        torch.testing.assert_close(model.compute_draft_logits(hidden), hidden @ weight)
        assert inputs_seen[0].shape == (8, 2)
        torch.testing.assert_close(inputs_seen[0][:num_rows], hidden)
        assert torch.count_nonzero(inputs_seen[0][num_rows:]) == 0


def _deepseek_v4_local_argmax_model(
    *,
    tp_size: int = 2,
    vocab_size: int = 6,
    comm_group=None,
):
    model = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=vocab_size)
    model.draft_id_to_target_id = None
    lm_head_fields = {
        "tp_size": tp_size,
        "shard_indices": SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=vocab_size // tp_size,
        ),
    }
    if comm_group is not None:
        lm_head_fields["comm_group"] = comm_group
    model.lm_head = SimpleNamespace(**lm_head_fields)
    return model


def _local_argmax_spec(model) -> AscendDSparkSpeculator:
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.use_local_argmax_reduction = True
    spec.speculative_config = SimpleNamespace(draft_sample_method="greedy")
    spec._draft_topk = None
    spec.model = model
    return spec


def test_local_argmax_validation_accepts_wrapped_head_without_comm_group(monkeypatch):
    model = _deepseek_v4_local_argmax_model()
    spec = _local_argmax_spec(ModelWithContext(model))
    tp_group = SimpleNamespace(world_size=2, unique_name="tp")
    monkeypatch.setattr(dspark_module, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(dspark_module, "lmhead_tp_enable", lambda: False)

    spec._validate_local_argmax_reduction()
    assert spec._unwrap_draft_model(spec.model) is model


@pytest.mark.parametrize(
    "mutate,message",
    [
        (
            lambda spec, model: setattr(spec.speculative_config, "draft_sample_method", "probabilistic"),
            "requires draft_sample_method='greedy'",
        ),
        (
            lambda spec, model: setattr(spec, "_draft_topk", 16),
            "cannot be combined with dspark_draft_topk",
        ),
        (
            lambda spec, model: setattr(model, "draft_id_to_target_id", torch.arange(6)),
            "does not support draft-to-target vocabulary remapping",
        ),
        (
            lambda spec, model: setattr(model.lm_head, "tp_size", 1),
            "LM head TP size",
        ),
    ],
)
def test_local_argmax_validation_rejects_unsupported_config(monkeypatch, mutate, message):
    model = _deepseek_v4_local_argmax_model()
    spec = _local_argmax_spec(model)
    mutate(spec, model)
    monkeypatch.setattr(
        dspark_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, unique_name="tp"),
    )
    monkeypatch.setattr(dspark_module, "lmhead_tp_enable", lambda: False)

    with pytest.raises(ValueError, match=message):
        spec._validate_local_argmax_reduction()


def test_local_argmax_disabled_uses_upstream_sampling(monkeypatch):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.use_local_argmax_reduction = False
    hidden_states = torch.ones(2, 3)
    upstream = MagicMock()
    monkeypatch.setattr(DSparkSpeculator, "_sample_sequential", upstream)

    spec._sample_sequential(2, hidden_states)

    upstream.assert_called_once_with(2, hidden_states)


def test_local_argmax_sampling_matches_full_vocab_markov_reference(monkeypatch):
    batch_size, num_steps, vocab_size = 2, 3, 6
    shard_size = vocab_size // 2
    full_logits = torch.tensor(
        [
            [
                [4.0, 1.0, 0.0, 2.0, 3.0, 0.0],
                [0.0, 3.0, 2.0, 4.0, 1.0, 0.0],
                [1.0, 0.0, 4.0, 3.0, 2.0, 0.0],
            ],
            [
                [0.0, 2.0, 3.0, 1.0, 4.0, 0.0],
                [3.0, 0.0, 1.0, 2.0, 4.0, 0.0],
                [0.0, 4.0, 1.0, 3.0, 2.0, 0.0],
            ],
        ]
    )
    anchors = torch.tensor([0, 1])

    def markov_bias(previous: torch.Tensor) -> torch.Tensor:
        bias = torch.zeros(previous.shape[0], vocab_size)
        favored = (previous + 1) % vocab_size
        return bias.scatter_(1, favored.unsqueeze(-1), 5.0)

    expected_steps = []
    remote_pairs = []
    previous = anchors
    for step in range(num_steps):
        bias = markov_bias(previous)
        scores = full_logits[:, step] + bias
        sampled = scores.argmax(dim=-1)
        expected_steps.append(sampled)
        remote_scores = scores[:, shard_size:]
        remote_values, remote_indices = remote_scores.max(dim=-1)
        remote_pairs.append(
            torch.stack(
                (remote_values, (remote_indices + shard_size).float()),
                dim=-1,
            ).float()
        )
        previous = sampled
    expected = torch.stack(expected_steps, dim=1)

    class DraftModel(DSparkDeepseekV4ForCausalLM):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(vocab_size=vocab_size)
            self.draft_id_to_target_id = None
            self.lm_head = SimpleNamespace(
                tp_size=2,
                shard_indices=SimpleNamespace(
                    org_vocab_start_index=0,
                    org_vocab_end_index=shard_size,
                ),
            )
            self.previous_tokens = []
            self.seen_hidden_states = None

        def compute_local_draft_logits(self, hidden_states):
            self.seen_hidden_states = hidden_states
            return full_logits[:, :, :shard_size].flatten(0, 1)

        def markov_embed(self, token_ids):
            self.previous_tokens.append(token_ids.clone())
            return token_ids

        def markov_bias(self, markov_embed):
            return markov_bias(markov_embed)

        def map_draft_to_target(self, draft_ids):
            return draft_ids

    class TPGroup:
        world_size = 2
        unique_name = "tp"

        def __init__(self) -> None:
            self.calls = 0

        def all_gather(self, local_pair, dim=-1):
            assert dim == -1
            remote_pair = remote_pairs[self.calls]
            self.calls += 1
            return torch.cat((local_pair, remote_pair), dim=-1)

    base_model = DraftModel()
    runtime_model = ModelWithContext(base_model)
    tp_group = TPGroup()
    monkeypatch.setattr(dspark_module, "get_tp_group", lambda: tp_group)

    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.use_local_argmax_reduction = True
    spec.model = runtime_model
    spec.num_speculative_steps = num_steps
    spec.sample_indices = torch.arange(batch_size * num_steps)
    spec.input_buffers = SimpleNamespace(input_ids=anchors)
    spec._anchor_idx = torch.arange(batch_size)
    spec.draft_tokens = torch.full((batch_size, num_steps), -1, dtype=torch.int64)
    spec.use_confidence_head = False
    head_hidden = torch.arange(batch_size * num_steps * 2, dtype=torch.float32).view(-1, 2)

    spec._sample_sequential(batch_size, head_hidden)

    torch.testing.assert_close(spec.draft_tokens, expected)
    torch.testing.assert_close(base_model.seen_hidden_states, head_hidden)
    assert tp_group.calls == num_steps
    assert len(base_model.previous_tokens) == num_steps
    torch.testing.assert_close(base_model.previous_tokens[0], anchors)
    for step in range(1, num_steps):
        torch.testing.assert_close(base_model.previous_tokens[step], expected[:, step - 1])


def _context_speculator(monkeypatch, group_ids=(0, 1), layer_map=(1, 0)):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: ("draft.0", "draft.1"))
    spec.draft_kv_cache_group_ids = group_ids
    spec._layer_group_idx = layer_map
    spec.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=size)) for size in (64, 128)]
    )
    spec.attn_architecture, spec.use_dcp = "MLA", False
    spec.device = torch.device("cpu")
    spec.max_model_len = 16
    spec.speculative_config = object()
    spec.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 4),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
    )
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    return spec


@pytest.mark.parametrize("group_ids,layer_map,expected", [((0,), None, (0, 0)), ((0, 1), (1, 0), (1, 0))])
def test_dspark_context_layout_uses_loaded_layer_cache_groups(monkeypatch, group_ids, layer_map, expected):
    spec = _context_speculator(monkeypatch, group_ids, layer_map)
    groups, layers, sizes = spec.get_draft_context_group_layout()
    assert groups == group_ids
    assert layers == expected
    assert sizes == {group: (64, 128)[group] for group in group_ids}


@pytest.mark.parametrize(
    "group_ids,layer_map,empty_layers,message",
    [
        ((0, 1), None, False, "explicit cache-group map"),
        ((0,), (0,), False, "does not match"),
        ((), (), True, "requires loaded draft attention layers"),
        ((0,), None, True, "requires loaded draft attention layers"),
    ],
)
def test_dspark_context_layout_rejects_incomplete_cache_maps(monkeypatch, group_ids, layer_map, empty_layers, message):
    spec = _context_speculator(monkeypatch, group_ids, layer_map)
    if empty_layers:
        spec.model.get_draft_kv_cache_layer_names = lambda: ()
    with pytest.raises(ValueError, match=message):
        spec.get_draft_context_group_layout()


@pytest.mark.parametrize(
    "invalid",
    [None, "gqa", "dcp", "pcp", "aux_layers", "hidden_size", "prompt_length", "dtype", "shape"],
)
def test_dspark_local_context_validates_before_writing_kv(monkeypatch, invalid):
    spec = _context_speculator(monkeypatch)
    descriptor = SimpleNamespace(aux_layer_ids=(1, 3), hidden_size=4, prompt_tokens=8, feature_width=8)
    chunk = SimpleNamespace(descriptor=descriptor, num_tokens=2)
    features = torch.ones(2, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(dspark_module, "get_eagle3_aux_layers_from_config", lambda _: [1, 3])
    monkeypatch.setattr(dspark_module, "set_current_vllm_config", lambda _: nullcontext())
    write_kv = MagicMock()
    monkeypatch.setattr(dspark_module, "initialize_draft_context_chunk", write_kv)
    if invalid == "gqa":
        spec.attn_architecture = "GQA"
    elif invalid == "dcp":
        spec.use_dcp = True
    elif invalid == "pcp":
        spec.vllm_config.parallel_config.prefill_context_parallel_size = 2
    elif invalid == "aux_layers":
        descriptor.aux_layer_ids = (3, 1)
    elif invalid == "hidden_size":
        descriptor.hidden_size = 8
    elif invalid == "prompt_length":
        descriptor.prompt_tokens = 17
    elif invalid == "dtype":
        features = features.float()
    elif invalid == "shape":
        features = features[:, :4]

    if invalid is not None:
        with pytest.raises(ValueError):
            spec.initialize_local_context(chunk, features, {0: [4], 1: [7]})
        write_kv.assert_not_called()
    else:
        spec.initialize_local_context(chunk, features, {0: [4], 1: [7]})
        args, kwargs = write_kv.call_args
        assert args[:2] == (spec.model, chunk)
        assert args[2] is features
        assert kwargs["draft_block_ids_by_group"] == {0: (4,), 1: (7,)}
        assert kwargs["layer_group_ids"] == (1, 0)
        assert kwargs["block_sizes_by_group"] == {0: 64, 1: 128}


@pytest.mark.parametrize("replicated_pcp", [False, True])
def test_dspark_sampling_uses_replicated_backbone_output(monkeypatch, replicated_pcp):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.replicated_pcp = replicated_pcp
    local, replicated = torch.ones(3, 2), torch.zeros(3, 2)
    monkeypatch.setattr(DSparkSpeculator, "_run_model", lambda *args: local)
    broadcast = MagicMock(return_value=(replicated, replicated))
    monkeypatch.setattr(dspark_module.AscendPCPManager, "broadcast_replicated_hidden_states", broadcast)

    assert spec._run_model(3, None, None, None) is replicated
    broadcast.assert_called_once_with(local, local, 3, replicated_pcp=replicated_pcp)
