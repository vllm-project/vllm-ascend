# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the UNO config patch: the two gates and the refusals.

Runs without an NPU: it only touches pydantic schemas and the config-level
validators, never a device.
"""

from __future__ import annotations

import typing
from types import SimpleNamespace

import pydantic
import pytest

from vllm_ascend.patch.platform import patch_uno_config


@pytest.fixture(scope="module", autouse=True)
def _apply_patch():
    patch_uno_config.apply()


def _cfg(**kwargs):
    from vllm.config.speculative import SpeculativeConfig

    payload = {
        "method": "uno",
        "num_speculative_tokens": 4,
        "target_model_config": None,
        "target_parallel_config": None,
    }
    payload.update(kwargs)
    return SpeculativeConfig(**payload)


# ---- Gate 1: pydantic ------------------------------------------------------


def test_method_literal_contains_uno():
    from vllm.config import speculative as sc

    members = typing.get_args(sc.SpeculativeMethod)
    assert "uno" in members
    assert members.count("uno") == 1
    assert len(members) == len(set(members)), "duplicate Literal members"


def test_all_upstream_methods_survive_the_rebuild():
    from vllm.config import speculative as sc

    members = typing.get_args(sc.SpeculativeMethod)
    for method in (
        "ngram",
        "ngram_gpu",
        "draft_model",
        "suffix",
        "custom_class",
        "eagle",
        "eagle3",
        "mtp",
        "dflash",
        "dspark",
        "extract_hidden_states",
    ):
        assert method in members, method


def test_uno_passes_pydantic_validation():
    assert _cfg().method == "uno"


def test_unknown_method_is_still_rejected():
    with pytest.raises(pydantic.ValidationError, match="literal_error"):
        _cfg(method="definitely_not_a_method")


# ---- Gate 2: __post_init__ -------------------------------------------------


def test_post_init_populates_uno_fields():
    cfg = _cfg()
    assert cfg.model == "uno"
    assert cfg.parallel_drafting is True
    assert cfg.draft_sample_method == "probabilistic"
    assert cfg.prompt_lookup_max == 0
    assert cfg.prompt_lookup_min == 0


def test_post_init_keeps_the_requested_token_count():
    assert _cfg(num_speculative_tokens=8).num_speculative_tokens == 8


def test_upstream_post_init_still_runs_for_other_methods():
    from vllm.config.speculative import SpeculativeConfig

    cfg = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=2,
        target_model_config=None,
        target_parallel_config=None,
    )
    # Upstream's own branch sets model="ngram"; this proves the chain call works
    # rather than the UNO branch swallowing every method.
    assert cfg.model == "ngram"


# ---- idempotency -----------------------------------------------------------


def test_reapplying_the_patch_is_a_noop():
    before = typing.get_args(__import__("vllm.config.speculative", fromlist=["SpeculativeMethod"]).SpeculativeMethod)
    for _ in range(3):
        patch_uno_config.apply()
    after = typing.get_args(__import__("vllm.config.speculative", fromlist=["SpeculativeMethod"]).SpeculativeMethod)
    assert after == before
    assert after.count("uno") == 1


def test_post_init_wrapper_is_marked():
    from vllm.config.speculative import SpeculativeConfig

    assert getattr(SpeculativeConfig.__post_init__, "_uno_patched", False)


# ---- num_lookahead_tokens --------------------------------------------------


def _bare_vllm_config(spec_config):
    from vllm.config.vllm import VllmConfig

    cfg = object.__new__(VllmConfig)
    object.__setattr__(cfg, "speculative_config", spec_config)
    return cfg


def test_uno_reserves_k_lookahead_slots():
    from vllm.config.vllm import VllmConfig

    spec = _cfg(num_speculative_tokens=6)
    cfg = _bare_vllm_config(spec)
    assert VllmConfig.num_lookahead_tokens.fget(cfg) == 6


def test_lookahead_is_zero_without_spec_config():
    from vllm.config.vllm import VllmConfig

    cfg = _bare_vllm_config(None)
    assert VllmConfig.num_lookahead_tokens.fget(cfg) == 0


# ---- V1 refusal ------------------------------------------------------------


def _stub_vllm_config_for_v1(spec_config):
    cfg = _bare_vllm_config(spec_config)
    object.__setattr__(
        cfg,
        "parallel_config",
        SimpleNamespace(
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            enable_batch_sharded_sampling=False,
        ),
    )
    object.__setattr__(cfg, "scheduler_config", SimpleNamespace(async_scheduling=False))
    object.__setattr__(cfg, "model_config", None)
    object.__setattr__(cfg, "_dflash_needs_multi_kv_group", lambda: False)
    object.__setattr__(cfg, "_is_dflash_candidate_draft", lambda: False)
    return cfg


def test_v1_refuses_uno():
    from vllm.config.vllm import VllmConfig

    unsupported = VllmConfig._get_v1_model_runner_unsupported_features(_stub_vllm_config_for_v1(_cfg()))
    assert "uno speculative decoding" in unsupported


def test_v1_does_not_add_uno_for_other_methods():
    from vllm.config.speculative import SpeculativeConfig
    from vllm.config.vllm import VllmConfig

    spec = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=2,
        target_model_config=None,
        target_parallel_config=None,
    )
    unsupported = VllmConfig._get_v1_model_runner_unsupported_features(_stub_vllm_config_for_v1(spec))
    assert "uno speculative decoding" not in unsupported


# ---- per-request LoRA refusal ---------------------------------------------


def test_input_processor_refuses_request_lora_under_uno():
    from vllm.v1.engine.input_processor import InputProcessor

    proc = object.__new__(InputProcessor)
    object.__setattr__(proc, "speculative_config", _cfg())
    object.__setattr__(proc, "lora_config", object())
    object.__setattr__(proc, "renderer", SimpleNamespace(tokenizer=None))
    with pytest.raises(ValueError, match="UNO speculative decoding reserves"):
        InputProcessor._validate_lora(proc, SimpleNamespace(lora_name="user-adapter"))


def test_input_processor_allows_no_lora_request():
    from vllm.v1.engine.input_processor import InputProcessor

    proc = object.__new__(InputProcessor)
    object.__setattr__(proc, "speculative_config", _cfg())
    object.__setattr__(proc, "lora_config", object())
    object.__setattr__(proc, "renderer", SimpleNamespace(tokenizer=None))
    InputProcessor._validate_lora(proc, None)  # must not raise


# ---- CLI choices -----------------------------------------------------------


def test_cli_choices_include_uno():
    from vllm.config.speculative import SpeculativeConfig
    from vllm.engine.arg_utils import _compute_kwargs

    _compute_kwargs.cache_clear()
    choices = _compute_kwargs(SpeculativeConfig)["method"].get("choices") or []
    assert "uno" in choices
