# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU contracts for the NPU EAGLE-4 boundary, without importing GPU/NPU runners.

Can also run with --noconftest on a non-NPU developer host.
The numerical tests execute the production functions and official core helpers.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("eagle_harness")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def modules():
    root = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode"
    loaded = []
    for name in ("eagle4", "eagle4_head"):
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def draft_config(**extra):
    return transformers.LlamaConfig(
        **{
            "architectures": ["LlamaForCausalLMEagle4"],
            "num_hidden_layers": 1,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 128,
            "efficient_lm_head_config": {"type": "low_rank", "rank": 8, "num_candidates": 16},
            "eagle_config": {"eagle_aux_hidden_state_layer_ids": [0, 3, 6]},
            **extra,
        }
    )


def spec_config(module, **extra):
    draft = module.normalize_eagle4_config(draft_config())
    return SimpleNamespace(
        **{
            "method_config": {},
            "num_speculative_tokens": 3,
            "num_speculative_tokens_per_batch_size": None,
            "parallel_drafting": False,
            "rejection_sample_method": "standard",
            "use_local_argmax_reduction": False,
            "enable_adaptive_verification": False,
            "use_heterogeneous_vocab": False,
            "quantization": None,
            "draft_lm_head_quantization": None,
            "draft_model_config": SimpleNamespace(hf_config=draft),
            "target_model_config": SimpleNamespace(
                get_total_num_hidden_layers=lambda: 8, get_vocab_size=lambda: 128, get_hidden_size=lambda: 32
            ),
            **extra,
        }
    )


def test_config_uses_checkpoint_output_layers_once(modules):
    module, _ = modules
    spec = spec_config(module)
    module.configure_eagle4(spec)
    module.configure_eagle4(spec)
    assert spec.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids == (1, 4, 7)
    assert spec.num_speculative_tokens == 3
    assert spec.method_config == {
        "mode": "static",
        "intensity": "chain_intensity",
        "node_top_k": 1,
        "max_verify_width": 4,
    }
    assert spec.num_speculative_tokens_per_batch_size is None


def test_verify_width_can_set_the_draft_count(modules):
    module, _ = modules
    spec = spec_config(module, num_speculative_tokens=None, method_config={"max_verify_width": 6})
    module.configure_eagle4(spec)
    assert spec.num_speculative_tokens == 5


@pytest.mark.parametrize(
    "extra",
    [
        {"method_config": {"node_top_k": 4, "intensity": "high_intensity"}},
        {"method_config": {"mode": "serve"}},
        {"method_config": {"record_rounds": True}},
        {"method_config": {"data_path": "/cuda/costs"}},
        {"method_config": {"max_verify_width": 6}},
        {"parallel_drafting": True},
        {"use_local_argmax_reduction": True},
        {"enable_adaptive_verification": True},
        {"quantization": "ascend"},
        {"rejection_sample_method": "synthetic"},
        {"num_speculative_tokens_per_batch_size": [(1, 4, 2)]},
        {"use_heterogeneous_vocab": True},
        {"draft_lm_head_quantization": "fp8"},
    ],
)
def test_unsupported_config_fails_before_execution(modules, extra):
    module, _ = modules
    with pytest.raises((ValueError, NotImplementedError)):
        module.configure_eagle4(spec_config(module, **extra))


@pytest.mark.parametrize("layers", [[], [0, 0, 6], [0, 3, 7], [-1, 3, 6], [True, 3, 6], [0, 3.0, 6], [6, 3, 0]])
def test_bad_auxiliary_layers_cannot_silently_use_defaults(modules, layers):
    module, _ = modules
    spec = spec_config(module)
    spec.draft_model_config.hf_config.eagle_config = {"eagle_aux_hidden_state_layer_ids": layers}
    with pytest.raises(ValueError):
        module.configure_eagle4(spec)


@pytest.mark.parametrize(
    "extra",
    [
        {"use_rope": False},
        {"attn_norm": True},
        {"enable_moe": True},
        {"share_draft_embedding_from_target": False},
        {"draft_vocab_size": 64},
        {"num_hidden_layers": 2},
        {"efficient_lm_head_config": {"type": "low_rank", "rank": 0, "num_candidates": 16}},
    ],
)
def test_unbuilt_checkpoint_structures_are_rejected(modules, extra):
    module, _ = modules
    with pytest.raises((ValueError, NotImplementedError)):
        module.validate_draft_config(module.normalize_eagle4_config(draft_config(**extra)))


def test_validation_does_not_mutate_checkpoint_config(modules):
    module, _ = modules
    config = module.normalize_eagle4_config(draft_config(enable_fc_norm=True))
    before = config.to_dict()
    module.validate_draft_config(config)
    assert config.to_dict() == before
    assert config.fc_norm is True


def test_official_qwen38_draft_window_and_auxiliary_layer_contract(modules):
    module, _ = modules
    # Structural fields of EAGLE-Corp/Qwen3.8-27B-EAGLE4 config.json.
    draft = module.normalize_eagle4_config(
        draft_config(
            sliding_window=4096,
            use_sliding_window=True,
            layer_types=["sliding_attention"],
            eagle_config={"eagle_aux_hidden_state_layer_ids": [3, 31, 59]},
        )
    )
    spec = spec_config(
        module,
        draft_model_config=SimpleNamespace(hf_config=draft),
        target_model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: 64, get_vocab_size=lambda: 128, get_hidden_size=lambda: 32
        ),
    )
    module.configure_eagle4(spec)
    assert module.validate_draft_config(draft) == 4096
    assert draft.layer_types == ["sliding_attention"]
    assert draft.eagle_aux_hidden_state_layer_ids == (4, 32, 60)


@pytest.mark.parametrize("window", [1, 32])
def test_window_without_layer_types_is_normalized_for_upstream_llama(modules, window):
    module, _ = modules
    config = module.normalize_eagle4_config(draft_config(sliding_window=window))
    assert config.layer_types == ["sliding_attention"]
    assert module.validate_draft_config(config) == window


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [0, 1, 5])
def test_candidate_head_matches_low_rank_reference_with_tp_padding(modules, dtype, rows):
    _, head = modules
    generator = torch.Generator().manual_seed(91)
    hidden = torch.randn(rows, 8, generator=generator).to(dtype)
    down = torch.randn(4, 8, generator=generator).to(dtype)
    # Two TP shards with a padded tail. Different rows must retain their own
    # candidates after gathering; padding must never become a token.
    pieces = []
    for start, real_rows in ((0, 13), (13, 11)):
        weight = torch.randn(16, 8, generator=generator).to(dtype)
        score = torch.randn(16, 4, generator=generator).to(dtype)
        logits, ids = head.local_candidates(hidden, down, score, weight, 5, start, real_rows)
        reference_scores = (hidden @ down.T) @ score[:real_rows].T
        expected_ids = reference_scores.topk(5, dim=-1, sorted=False).indices
        expected_logits = torch.einsum("bkh,bh->bk", weight[expected_ids].float(), hidden.float()).to(dtype).float()
        assert torch.equal(ids, expected_ids + start)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        pieces.append((logits, ids))
    logits = torch.cat([piece[0] for piece in pieces], dim=1)
    ids = torch.cat([piece[1] for piece in pieces], dim=1)
    full = head.candidates_to_logits(logits, ids, 24)
    assert full.shape == (rows, 24)
    assert torch.equal(full.gather(1, ids), logits)
    assert torch.equal(torch.isfinite(full).sum(dim=1), torch.full((rows,), 10))


def test_rescoring_tiles_rows_and_candidates_without_changing_values(modules, monkeypatch):
    _, head = modules
    monkeypatch.setattr(head, "_MAX_GATHER_ELEMENTS", 128)
    monkeypatch.setattr(head, "_CANDIDATE_TILE", 3)
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(30, 8, generator=generator)
    hidden = torch.randn(11, 8, generator=generator)
    ids = torch.randint(0, 30, (11, 9), generator=generator)
    result = head.rescore_candidates(weight, hidden, ids)
    expected = torch.einsum("bkh,bh->bk", weight[ids], hidden)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 1, 5])
def test_packed_tp_gather_preserves_token_ids_and_logit_bits(modules, rows):
    _, head = modules
    logits = [torch.arange(rows * 3, dtype=torch.float32).view(rows, 3) + rank for rank in range(2)]
    ids = [torch.arange(rows * 3).view(rows, 3) + rank * 248320 for rank in range(2)]
    packed = torch.cat([head.pack_candidates(values, tokens) for values, tokens in zip(logits, ids)], dim=1)
    actual_logits, actual_ids = head.unpack_candidates(packed, 2, 3)
    assert torch.equal(actual_logits, torch.cat(logits, dim=1))
    assert torch.equal(actual_ids, torch.cat(ids, dim=1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("enable_dbo", True),
    ],
)
def test_unvalidated_parallel_modes_fail_closed(modules, field, value):
    module, _ = modules
    parallel = dict(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_dbo=False,
    )
    parallel[field] = value
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="eagle4"),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(**parallel),
    )
    with pytest.raises(NotImplementedError, match="TP only"):
        module.validate_eagle4_runtime(config)


def runtime_config():
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="eagle4"),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
        ),
        lora_config=None,
        model_config=SimpleNamespace(is_multimodal_model=False),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


def test_standard_runtime_is_accepted_without_touching_device_state(modules):
    module, _ = modules
    module.validate_eagle4_runtime(runtime_config())


@pytest.mark.parametrize(
    "field,value", [("use_v2_model_runner", False), ("lora_config", object()), ("kv_transfer_config", object())]
)
def test_unvalidated_runtime_features_are_rejected(modules, field, value):
    module, _ = modules
    config = runtime_config()
    setattr(config, field, value)
    with pytest.raises(NotImplementedError):
        module.validate_eagle4_runtime(config)


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
def test_quantized_kv_cache_is_rejected(modules, dtype):
    module, _ = modules
    config = runtime_config()
    config.cache_config.cache_dtype = dtype
    with pytest.raises(NotImplementedError, match="unquantized KV"):
        module.validate_eagle4_runtime(config)
