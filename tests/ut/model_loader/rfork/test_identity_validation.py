# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
from dataclasses import dataclass
from enum import Enum, auto
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def identity_module(monkeypatch):
    _stub(monkeypatch, "vllm.config", ModelConfig=object, VllmConfig=object)
    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.types", RFORK_PROTOCOL_VERSION=1)
    _stub(monkeypatch, "vllm_ascend.ascend_config", get_ascend_config=lambda: SimpleNamespace(weight_nz_mode=None))
    _stub(
        monkeypatch,
        "vllm_ascend.device.hardware_profile",
        get_current_hardware_profile=lambda: SimpleNamespace(weight_layout_policy=None, _device_type="A2"),
    )
    return _load_module(monkeypatch, "rfork_test_identity", "identity.py")


@pytest.fixture
def manifest_module(monkeypatch):
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-manifest-validation-test"))
    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.types", SeedTransferInfo=object)
    return _load_module(monkeypatch, "rfork_test_manifest", "manifest.py")


@pytest.fixture
def seed_module(monkeypatch):
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-seed-validation-test"))

    @dataclass(frozen=True)
    class SeedTransferInfo:
        session_id: str
        weights: dict
        shared_names: list[str] | None = None
        formats: dict | None = None
        load_state: dict | None = None

    _stub(
        monkeypatch,
        "vllm_ascend.model_loader.rfork.types",
        RFORK_PROTOCOL_VERSION=1,
        SeedTransferInfo=SeedTransferInfo,
    )
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.manifest", "manifest.py")
    return _load_module(monkeypatch, "rfork_test_seed_client", "seed_client.py")


@pytest.fixture
def planner_module(monkeypatch):
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-planner-validation-test"))
    _stub(monkeypatch, "vllm.utils.network_utils", get_ip=lambda: "127.0.0.1")
    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.config", RForkConfig=object)
    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.identity", build_seed_key=lambda **kwargs: "seed-key")

    class LeaseReleaseResult(Enum):
        RELEASED = auto()
        RETRYABLE = auto()
        REJECTED = auto()

    class SeedReportStatus(Enum):
        ACCEPTED = auto()
        RETRYABLE = auto()
        REJECTED = auto()

    @dataclass(frozen=True)
    class SeedReportResult:
        status: SeedReportStatus
        reason: str = ""

    @dataclass(frozen=True)
    class SeedLease:
        seed_ip: str
        seed_port: int
        user_id: str
        seed_rank: int
        seed_key: str
        lease_ttl_sec: float = 60.0

    _stub(
        monkeypatch,
        "vllm_ascend.model_loader.rfork.types",
        LeaseReleaseResult=LeaseReleaseResult,
        SeedLease=SeedLease,
        RForkIdentity=object,
        SeedAdvertisement=object,
        SeedReportResult=SeedReportResult,
        SeedReportStatus=SeedReportStatus,
    )
    return _load_module(monkeypatch, "rfork_test_planner_client", "planner_client.py")


def _model_config(
    *,
    dtype=torch.float32,
    quantization=None,
    revision="rev-a",
    max_model_len=8192,
    max_seq_len_to_capture=8192,
    runner_type="generate",
    task="generate",
    convert="auto",
    use_mla=False,
    enforce_eager=False,
    multimodal_config=None,
):
    return SimpleNamespace(
        dtype=dtype,
        quantization=quantization,
        revision=revision,
        max_model_len=max_model_len,
        max_seq_len_to_capture=max_seq_len_to_capture,
        runner_type=runner_type,
        task=task,
        convert=convert,
        use_mla=use_mla,
        enforce_eager=enforce_eager,
        multimodal_config=multimodal_config,
        hf_config=SimpleNamespace(
            model_type="test-model",
            architectures=["TestModel"],
            quantization_config=None,
        ),
    )


def _vllm_config(
    tp_size=1,
    *,
    kv_role=None,
    max_num_batched_tokens=4096,
    max_num_seqs=8,
    cache_block_size=128,
    cache_dtype="auto",
    mamba_cache_dtype="auto",
    speculative_config=None,
    quant_config=None,
    compilation_config=None,
    lora_config=None,
    prompt_adapter_config=None,
    pooler_config=None,
    use_v2_model_runner=False,
    prefill_context_parallel_size=1,
    decode_context_parallel_size=1,
    use_sequence_parallel_moe=False,
    cp_kv_cache_interleave_size=1,
    data_parallel_rank=0,
):
    kv_transfer_config = None
    if kv_role is not None:
        kv_transfer_config = SimpleNamespace(
            kv_role=kv_role,
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    if compilation_config is None:
        compilation_config = SimpleNamespace(
            cudagraph_mode="NONE",
            cudagraph_capture_sizes=[],
            max_cudagraph_capture_size=0,
        )
    return SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
        cache_config=SimpleNamespace(
            block_size=cache_block_size,
            cache_dtype=cache_dtype,
            mamba_cache_dtype=mamba_cache_dtype,
        ),
        speculative_config=speculative_config,
        quant_config=quant_config,
        compilation_config=compilation_config,
        lora_config=lora_config,
        prompt_adapter_config=prompt_adapter_config,
        pooler_config=pooler_config,
        use_v2_model_runner=use_v2_model_runner,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            expert_parallel_size=None,
            data_parallel_size=None,
            enable_expert_parallel=False,
            is_moe_model=False,
            prefill_context_parallel_size=prefill_context_parallel_size,
            decode_context_parallel_size=decode_context_parallel_size,
            use_sequence_parallel_moe=use_sequence_parallel_moe,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            data_parallel_rank=data_parallel_rank,
        ),
    )


def _build_fingerprint(
    identity_module,
    model_config,
    *,
    tp_size=1,
    kv_role=None,
    max_num_batched_tokens=4096,
    **vllm_config_kwargs,
):
    return identity_module.build_compatibility_fingerprint(
        _vllm_config(
            tp_size,
            kv_role=kv_role,
            max_num_batched_tokens=max_num_batched_tokens,
            **vllm_config_kwargs,
        ),
        model_config,
        model_url="model",
        model_deploy_strategy_name="strategy",
    )


def _fingerprint(identity_module, *, dtype=torch.float32, quantization=None, revision="rev-a", tp_size=1):
    return _build_fingerprint(
        identity_module,
        _model_config(dtype=dtype, quantization=quantization, revision=revision),
        tp_size=tp_size,
    )


def _ascend_config(**overrides):
    values = {
        "weight_nz_mode": 1,
        "enable_kv_nz": False,
        "enable_transpose_kv_cache_by_block": True,
        "enable_fused_mc2": 0,
        "enable_mlapo": True,
        "mlapo_keep_prefill_weights": False,
        "enable_sparse_sfa_c8": False,
        "enable_sparse_li_c8": False,
        "c8_reshape_optim_enabled": False,
        "enable_dsa_cp": False,
        "mix_placement": False,
        "enable_shared_expert_dp": False,
        "enable_sp_by_pass": False,
        "pd_tp_ratio": 1,
        "pd_head_ratio": 1,
        "num_head_replica": 1,
        "draft_window_size": None,
        "finegrained_tp_config": SimpleNamespace(
            oproj_tensor_parallel_size=0,
            lmhead_tensor_parallel_size=0,
            embedding_tensor_parallel_size=0,
            mlp_tensor_parallel_size=0,
            olora_tensor_parallel_size=0,
        ),
        "dynamic_spec_config": SimpleNamespace(method=None, method_params={}),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonicalization_is_deterministic_for_dicts_sets_and_dtype(identity_module):
    first = {"set": {3, 1, 2}, "mapping": {"b": 2, "a": 1}, "dtype": torch.float16}
    second = {"dtype": torch.float16, "mapping": {"a": 1, "b": 2}, "set": {2, 3, 1}}

    assert identity_module._canonicalize_fingerprint_value(first) == identity_module._canonicalize_fingerprint_value(
        second
    )


def test_fingerprint_changes_when_seed_compatibility_changes(identity_module):
    changes = (
        {"dtype": torch.float16},
        {"quantization": "compressed"},
        {"revision": "rev-b"},
        {"tp_size": 2},
    )
    base = _fingerprint(identity_module)
    for change in changes:
        assert _fingerprint(identity_module, **change) != base, change


def test_fingerprint_changes_when_fused_mc2_changes(identity_module, monkeypatch):
    # Fused MC2 rewrites MoE storage, so differently configured peers must not share a seed.
    monkeypatch.setattr(
        identity_module, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1, enable_fused_mc2=0)
    )
    base = _fingerprint(identity_module)
    monkeypatch.setattr(
        identity_module, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1, enable_fused_mc2=1)
    )
    assert _fingerprint(identity_module) != base


def test_fingerprint_quantization_config_changes(identity_module):
    base_config = _model_config()
    changed_config = _model_config()
    base_config.hf_config.quantization_config = {"bits": 8, "group_size": 128}
    changed_config.hf_config.quantization_config = {"bits": 4, "group_size": 128}

    assert _build_fingerprint(identity_module, base_config) != _build_fingerprint(identity_module, changed_config)


def test_effective_quantization_description_isolates_weight_scheme(identity_module):
    base_model = _model_config(quantization="ascend")
    producer = SimpleNamespace(
        get_name=lambda: "ascend",
        quant_description={"group_size": 32, "model.layers.0.self_attn.q_proj.weight": "W4A8"},
    )
    consumer = SimpleNamespace(
        get_name=lambda: "ascend",
        quant_description={"group_size": 128, "model.layers.0.self_attn.q_proj.weight": "W4A8"},
    )
    assert _build_fingerprint(identity_module, base_model, quant_config=producer) != _build_fingerprint(
        identity_module, base_model, quant_config=consumer
    )


def test_fingerprint_changes_when_rope_configuration_changes(identity_module):
    changes = (
        ("rope_theta", None, 1e6),
        (
            "rope_scaling",
            None,
            {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
        ),
        (
            "rope_parameters",
            {
                "rope_type": "deepseek_yarn",
                "rope_theta": 10000.0,
                "factor": 2.0,
                "beta_fast": 32,
                "mscale_all_dim": 0.0,
            },
            {
                "rope_type": "deepseek_yarn",
                "rope_theta": 10000.0,
                "factor": 4.0,
                "beta_fast": 32,
                "mscale_all_dim": 0.0,
            },
        ),
    )
    for field, base_value, changed_value in changes:
        # RoPE values can alter cache contents without changing tensor shapes.
        base_config = _model_config()
        changed_config = _model_config()
        setattr(base_config.hf_config, field, base_value)
        setattr(changed_config.hf_config, field, changed_value)

        base_fingerprint = _build_fingerprint(identity_module, base_config)
        changed_fingerprint = _build_fingerprint(identity_module, changed_config)
        assert base_fingerprint != changed_fingerprint, field
        assert identity_module.build_seed_key(0, "model", "strategy", base_fingerprint) != (
            identity_module.build_seed_key(0, "model", "strategy", changed_fingerprint)
        ), field


def test_fingerprint_changes_for_rope_overrides_on_text_config(identity_module):
    first = _model_config()
    second = _model_config()
    for config, rope_theta in ((first, 1e4), (second, 1e6)):
        config.hf_text_config = SimpleNamespace(
            model_type="test-model-text",
            architectures=["TestModelText"],
            rope_theta=rope_theta,
            rope_scaling=None,
            max_position_embeddings=8192,
        )

    assert _build_fingerprint(identity_module, first) != _build_fingerprint(identity_module, second)


def test_resolved_commits_isolate_moving_revision_seeds(identity_module):
    for config_field, commit_field in (("hf_config", "_commit_hash"), ("hf_text_config", "commit_hash")):
        first = _model_config(revision="main")
        second = _model_config(revision="main")
        for config, commit in ((first, "a" * 40), (second, "b" * 40)):
            if config_field == "hf_text_config":
                config.hf_text_config = SimpleNamespace()
                config.hf_config.revision = "main"
            setattr(getattr(config, config_field), commit_field, commit)
        first_fingerprint = _build_fingerprint(identity_module, first)
        second_fingerprint = _build_fingerprint(identity_module, second)
        assert first_fingerprint != second_fingerprint, (config_field, commit_field)
        assert identity_module.build_seed_key(0, "model", "strategy", first_fingerprint) != (
            identity_module.build_seed_key(0, "model", "strategy", second_fingerprint)
        ), (config_field, commit_field)


def test_revision_aliases_share_the_same_resolved_commit(identity_module):
    first = _model_config(revision="main")
    second = _model_config(revision="release")
    first.hf_config._commit_hash = second.hf_config._commit_hash = "a" * 40
    assert _build_fingerprint(identity_module, first) == _build_fingerprint(identity_module, second)


def test_unresolved_revision_keeps_explicit_version_fallback(identity_module):
    config = _model_config(revision="local-version")
    config.hf_config._commit_hash = ""
    config.hf_config.commit_hash = None
    assert identity_module._get_model_revision(config) == "local-version"
    config.revision = None
    config.model_revision = "model-version"
    assert identity_module._get_model_revision(config) == "model-version"
    config.model_revision = None
    config.hf_config.revision = "config-version"
    assert identity_module._get_model_revision(config) == "config-version"
    config.hf_config.revision = None
    assert identity_module._get_model_revision(config) is None


def test_fingerprint_changes_when_kv_role_changes(identity_module):
    # Role isolation is unconditional, including checkpoint-layout transfer.
    for quantization in (None, "ascend"):
        model_config = _model_config(quantization=quantization)
        producer = _build_fingerprint(identity_module, model_config, kv_role="kv_producer")
        consumer = _build_fingerprint(identity_module, model_config, kv_role="kv_consumer")
        assert producer != consumer, quantization


def test_effective_kv_role_supports_legacy_boolean_flags(identity_module):
    producer_config = _vllm_config()
    producer_config.kv_transfer_config = SimpleNamespace(is_kv_producer=True, is_kv_consumer=False)
    consumer_config = _vllm_config()
    consumer_config.kv_transfer_config = SimpleNamespace(is_kv_producer=False, is_kv_consumer=True)

    assert identity_module._get_effective_kv_role(producer_config) == "kv_producer"
    assert identity_module._get_effective_kv_role(consumer_config) == "kv_consumer"


def test_fingerprint_changes_when_ascend_layout_config_changes(identity_module, monkeypatch):
    changes = (
        ("enable_mlapo", False, True),
        ("mlapo_keep_prefill_weights", False, True),
        ("enable_sparse_sfa_c8", False, True),
        ("enable_sparse_li_c8", False, True),
        ("c8_reshape_optim_enabled", False, True),
        ("enable_dsa_cp", False, True),
        ("mix_placement", False, True),
        ("enable_shared_expert_dp", False, True),
        ("enable_sp_by_pass", False, True),
        ("pd_tp_ratio", 1, 2),
        ("pd_head_ratio", 1, 2),
        ("num_head_replica", 1, 2),
        ("draft_window_size", None, 8),
    )
    for field, base_value, changed_value in changes:
        monkeypatch.setattr(
            identity_module,
            "get_ascend_config",
            lambda field=field, base_value=base_value: _ascend_config(**{field: base_value}),
        )
        base = _fingerprint(identity_module)
        monkeypatch.setattr(
            identity_module,
            "get_ascend_config",
            lambda field=field, changed_value=changed_value: _ascend_config(**{field: changed_value}),
        )
        assert _fingerprint(identity_module) != base, field


def test_fingerprint_changes_when_max_num_batched_tokens_changes(identity_module):
    model_config = _model_config(quantization="ascend")
    small_batch = _build_fingerprint(identity_module, model_config, max_num_batched_tokens=1024)
    large_batch = _build_fingerprint(identity_module, model_config, max_num_batched_tokens=4096)
    assert small_batch != large_batch


def test_fingerprint_changes_when_runtime_tensor_layout_changes(identity_module):
    changes: tuple[tuple[dict[str, object], dict[str, object]], ...] = (
        ({"cache_block_size": 64}, {}),
        ({"cache_dtype": "int8"}, {}),
        ({"mamba_cache_dtype": "float32"}, {}),
        ({"use_v2_model_runner": True}, {}),
        ({"prefill_context_parallel_size": 2}, {}),
        ({"decode_context_parallel_size": 2}, {}),
        ({"use_sequence_parallel_moe": True}, {}),
        ({}, {"max_model_len": 16384}),
        ({}, {"runner_type": "draft"}),
        ({}, {"task": "embed"}),
        ({}, {"convert": "embed"}),
        ({}, {"use_mla": True}),
        ({}, {"enforce_eager": True}),
    )
    base = _build_fingerprint(identity_module, _model_config())
    for config_kwargs, model_kwargs in changes:
        changed = _build_fingerprint(identity_module, _model_config(**model_kwargs), **config_kwargs)
        assert changed != base, (config_kwargs, model_kwargs)


def test_runtime_sizing_does_not_partition_weight_seeds(identity_module):
    changes: tuple[tuple[dict[str, object], dict[str, object]], ...] = (
        ({"max_num_seqs": 16}, {}),
        ({"cp_kv_cache_interleave_size": 2}, {}),
        ({}, {"max_seq_len_to_capture": 16384}),
    )
    base = _build_fingerprint(identity_module, _model_config())
    for config_kwargs, model_kwargs in changes:
        changed = _build_fingerprint(identity_module, _model_config(**model_kwargs), **config_kwargs)
        assert changed == base, (config_kwargs, model_kwargs)


def test_kv_cache_layout_does_not_partition_weight_seeds(identity_module, monkeypatch):
    changes = (
        ("enable_kv_nz", True),
        ("enable_transpose_kv_cache_by_block", False),
    )
    for field, changed_value in changes:
        monkeypatch.setattr(identity_module, "get_ascend_config", lambda: _ascend_config())
        base = _fingerprint(identity_module)
        monkeypatch.setattr(
            identity_module,
            "get_ascend_config",
            lambda field=field, changed_value=changed_value: _ascend_config(**{field: changed_value}),
        )
        assert _fingerprint(identity_module) == base, field


def test_fingerprint_changes_for_every_finegrained_tp_size(identity_module, monkeypatch):
    field_names = (
        "oproj_tensor_parallel_size",
        "lmhead_tensor_parallel_size",
        "embedding_tensor_parallel_size",
        "mlp_tensor_parallel_size",
        "olora_tensor_parallel_size",
    )
    for field_name in field_names:
        monkeypatch.setattr(identity_module, "get_ascend_config", lambda: _ascend_config())
        base = _fingerprint(identity_module)

        finegrained_values = {
            name: (2 if name == field_name else 0) for name in identity_module._FINEGRAINED_TP_LAYOUT_FIELDS
        }
        monkeypatch.setattr(
            identity_module,
            "get_ascend_config",
            lambda finegrained_values=finegrained_values: _ascend_config(
                finegrained_tp_config=SimpleNamespace(**finegrained_values)
            ),
        )
        assert _fingerprint(identity_module) != base, field_name


def test_finegrained_tp_isolates_data_parallel_layout_rank(identity_module, monkeypatch):
    finegrained = SimpleNamespace(
        oproj_tensor_parallel_size=2,
        lmhead_tensor_parallel_size=0,
        embedding_tensor_parallel_size=0,
        mlp_tensor_parallel_size=0,
        olora_tensor_parallel_size=0,
    )
    monkeypatch.setattr(
        identity_module,
        "get_ascend_config",
        lambda: _ascend_config(finegrained_tp_config=finegrained),
    )
    model_config = _model_config()
    first_rank = _build_fingerprint(identity_module, model_config, data_parallel_rank=0)
    second_rank = _build_fingerprint(identity_module, model_config, data_parallel_rank=1)
    assert first_rank != second_rank


def test_replicated_data_parallel_ranks_keep_sharing_fingerprint(identity_module, monkeypatch):
    monkeypatch.setattr(identity_module, "get_ascend_config", lambda: _ascend_config())
    model_config = _model_config()
    first_rank = _build_fingerprint(identity_module, model_config, data_parallel_rank=0)
    second_rank = _build_fingerprint(identity_module, model_config, data_parallel_rank=1)
    assert first_rank == second_rank


def test_fingerprint_changes_when_speculative_layout_changes(identity_module):
    changes = (
        ("method", "eagle3"),
        ("num_speculative_tokens", 3),
        ("num_speculative_tokens_per_batch_size", {1: 3, 8: 1}),
        ("draft_tensor_parallel_size", 2),
        ("parallel_drafting", True),
        ("disable_padded_drafter_batch", True),
        ("enforce_eager", True),
        ("draft_sample_method", "probabilistic"),
        ("speculative_token_tree", "[(0,), (0, 0)]"),
        ("quantization", "ascend"),
        ("use_local_argmax_reduction", True),
    )
    base = _build_fingerprint(identity_module, _model_config())
    for field, changed_value in changes:
        speculative_config = SimpleNamespace(**{field: changed_value})
        changed = _build_fingerprint(identity_module, _model_config(), speculative_config=speculative_config)
        assert changed != base, field


def test_fingerprint_changes_when_draft_model_or_parallel_layout_changes(identity_module):
    base_draft = _model_config()
    base_draft.hf_config.hidden_size = 1024
    changed_draft = _model_config()
    changed_draft.hf_config.hidden_size = 2048
    base_spec = SimpleNamespace(
        method="eagle3",
        draft_model_config=base_draft,
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    changed_model_spec = SimpleNamespace(
        method="eagle3",
        draft_model_config=changed_draft,
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    changed_parallel_spec = SimpleNamespace(
        method="eagle3",
        draft_model_config=base_draft,
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )

    base = _build_fingerprint(identity_module, _model_config(), speculative_config=base_spec)
    model_changed = _build_fingerprint(identity_module, _model_config(), speculative_config=changed_model_spec)
    parallel_changed = _build_fingerprint(identity_module, _model_config(), speculative_config=changed_parallel_spec)
    assert model_changed != base
    assert parallel_changed != base


def test_fingerprint_changes_when_dynamic_spec_layout_changes(identity_module, monkeypatch):
    monkeypatch.setattr(identity_module, "get_ascend_config", lambda: _ascend_config())
    base = _fingerprint(identity_module)
    monkeypatch.setattr(
        identity_module,
        "get_ascend_config",
        lambda: _ascend_config(dynamic_spec_config=SimpleNamespace(method="mtp", method_params={"window": 4})),
    )
    assert _fingerprint(identity_module) != base


def test_fingerprint_changes_when_hardware_device_type_changes(identity_module, monkeypatch):
    monkeypatch.setattr(
        identity_module,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(weight_layout_policy="CONFIGURABLE", _device_type="A2"),
    )
    a2_fingerprint = _fingerprint(identity_module)
    monkeypatch.setattr(
        identity_module,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(weight_layout_policy="CONFIGURABLE", _device_type="A5"),
    )
    assert _fingerprint(identity_module) != a2_fingerprint


def test_compilation_capture_sizes_do_not_partition_weight_seeds(identity_module):
    changes = (
        ("cudagraph_mode", "FULL"),
        ("cudagraph_capture_sizes", [1, 2, 16]),
        ("max_cudagraph_capture_size", 16),
    )
    base = _build_fingerprint(identity_module, _model_config())
    for field, changed_value in changes:
        values = {
            "cudagraph_mode": "NONE",
            "cudagraph_capture_sizes": [],
            "max_cudagraph_capture_size": 0,
        }
        values[field] = changed_value
        changed = _build_fingerprint(
            identity_module,
            _model_config(),
            compilation_config=SimpleNamespace(**values),
        )
        assert changed == base, field


def test_compilation_runtime_registry_is_not_part_of_fingerprint(identity_module):
    base = _build_fingerprint(identity_module, _model_config())
    compilation_config = SimpleNamespace(
        cudagraph_mode="NONE",
        cudagraph_capture_sizes=[],
        max_cudagraph_capture_size=0,
        static_forward_context={"current": object()},
    )
    assert _build_fingerprint(identity_module, _model_config(), compilation_config=compilation_config) == base


def test_multimodal_layout_isolates_tower_and_shard_choices(identity_module):
    changes = (
        ("limit_per_prompt", {"image": 0}),
        ("mm_encoder_tp_mode", "data"),
        ("enable_multimodal_pruning", True),
    )
    base = _build_fingerprint(identity_module, _model_config())
    for field, changed_value in changes:
        multimodal_config = SimpleNamespace(**{field: changed_value})
        changed = _build_fingerprint(identity_module, _model_config(multimodal_config=multimodal_config))
        assert changed != base, field


def test_multimodal_processor_kwargs_do_not_partition_weight_seeds(identity_module):
    base_multimodal = SimpleNamespace(limit_per_prompt={"image": 1}, mm_encoder_tp_mode="tensor")
    runtime_multimodal = SimpleNamespace(
        limit_per_prompt={"image": 1},
        mm_encoder_tp_mode="tensor",
        mm_processor_kwargs={"resize": 1024},
    )
    base = _build_fingerprint(identity_module, _model_config(multimodal_config=base_multimodal))
    changed = _build_fingerprint(identity_module, _model_config(multimodal_config=runtime_multimodal))
    assert changed == base


def test_effective_multimodal_limit_and_pruning_isolate_model_layout(identity_module):
    class MultimodalConfig:
        limit_per_prompt = {"image": 1}
        mm_encoder_tp_mode = "tensor"

        def __init__(self, image_limit, pruning):
            self.image_limit = image_limit
            self.pruning = pruning

        def get_limit_per_prompt(self, modality):
            return self.image_limit if modality == "image" else 0

        def is_multimodal_pruning_enabled(self):
            return self.pruning

    base = _build_fingerprint(identity_module, _model_config(multimodal_config=MultimodalConfig(1, False)))
    limit_changed = _build_fingerprint(identity_module, _model_config(multimodal_config=MultimodalConfig(0, False)))
    pruning_changed = _build_fingerprint(identity_module, _model_config(multimodal_config=MultimodalConfig(1, True)))
    assert limit_changed != base
    assert pruning_changed != base


def test_optional_model_features_isolate_tensor_layout(identity_module):
    changes = (
        ("lora_config", "max_lora_rank", 128),
        ("prompt_adapter_config", "max_prompt_adapter_token", 16),
        ("pooler_config", "pooling_type", "LAST"),
    )
    base = _build_fingerprint(identity_module, _model_config())
    for feature, field, value in changes:
        changed = _build_fingerprint(
            identity_module,
            _model_config(),
            **{feature: SimpleNamespace(**{field: value})},
        )
        assert changed != base, feature


def test_optional_runtime_fields_do_not_partition_weight_seeds(identity_module):
    changes = (
        ("lora_config", "max_lora_rank", "max_cpu_loras", 8),
        ("prompt_adapter_config", "max_prompt_adapter_token", "max_prompt_adapters", 8),
        ("pooler_config", "pooling_type", "normalize", True),
    )
    for feature, layout_field, runtime_field, runtime_value in changes:
        base_feature = SimpleNamespace(**{layout_field: 16})
        changed_feature = SimpleNamespace(**{layout_field: 16, runtime_field: runtime_value})
        base = _build_fingerprint(identity_module, _model_config(), **{feature: base_feature})
        changed = _build_fingerprint(identity_module, _model_config(), **{feature: changed_feature})
        assert changed == base, feature


def test_derived_hardware_capabilities_do_not_partition_weight_seeds(identity_module, monkeypatch):
    fields = (
        "attention_backend_family",
        "device_adaptor_family",
        "moe_comm_policy",
        "quantization_backend_family",
        "capabilities",
    )
    for field in fields:
        base_profile = SimpleNamespace(weight_layout_policy="CONFIGURABLE", _device_type="A2")
        monkeypatch.setattr(
            identity_module,
            "get_current_hardware_profile",
            lambda base_profile=base_profile: base_profile,
        )
        base = _fingerprint(identity_module)
        setattr(base_profile, field, {"MLAPO_NATIVE_WEIGHTS"} if field == "capabilities" else "COMPATIBILITY")
        assert _fingerprint(identity_module) == base, field


def test_hardware_weight_policy_still_isolates_layout(identity_module, monkeypatch):
    profile = SimpleNamespace(weight_layout_policy="CONFIGURABLE", _device_type="A2")
    monkeypatch.setattr(identity_module, "get_current_hardware_profile", lambda: profile)
    base = _fingerprint(identity_module)
    profile.weight_layout_policy = "FORCE_NZ"
    assert _fingerprint(identity_module) != base


def test_multimodal_layout_descriptor_reads_dataclass_fields(identity_module):
    @dataclass(slots=True)
    class MultimodalConfig:
        limit_per_prompt: dict[str, int]
        mm_encoder_tp_mode: str

    producer = _model_config(multimodal_config=MultimodalConfig({"image": 0}, "data"))
    consumer = _model_config(multimodal_config=MultimodalConfig({"image": 1}, "data"))
    assert _build_fingerprint(identity_module, producer) != _build_fingerprint(identity_module, consumer)


def test_recursive_fingerprint_values_fail_with_actionable_error(identity_module):
    recursive_dict: dict[str, object] = {}
    recursive_dict["self"] = recursive_dict
    recursive_list: list[object] = []
    recursive_list.append(recursive_list)
    recursive_object = SimpleNamespace()
    recursive_object.child = recursive_object

    for value in (recursive_dict, recursive_list, recursive_object):
        with pytest.raises(ValueError, match="recursive reference"):
            identity_module._canonicalize_fingerprint_value(value)


@pytest.mark.parametrize("raw", [[], "model", 1])
def test_extra_config_requires_json_object(monkeypatch, raw):
    config = _load_module(monkeypatch, "rfork_test_config", "config.py")
    with pytest.raises(RuntimeError, match="JSON object"):
        config.RForkConfig.from_extra_config(raw)


@pytest.mark.parametrize("value", ["8080", 8080.5, True, -1, 65536])
def test_seed_port_base_rejects_non_integer_extra_config(monkeypatch, value):
    config = _load_module(monkeypatch, "rfork_test_config", "config.py")
    with pytest.raises(ValueError, match="rfork_seed_port_base must be a JSON integer in range"):
        config.RForkConfig.from_extra_config({"rfork_seed_port_base": value})


def test_build_seed_key_rejects_missing_identity_values(identity_module):
    build = identity_module.build_seed_key
    with pytest.raises(RuntimeError, match="model_url"):
        build(0, "", "strategy", "fingerprint")
    with pytest.raises(RuntimeError, match="model_deploy_strategy_name"):
        build(0, "model", "", "fingerprint")
    with pytest.raises(RuntimeError, match="compatibility fingerprint"):
        build(0, "model", "strategy", "")


def test_manifest_accepts_only_the_current_five_field_format(manifest_module):
    assert manifest_module.parse_weight_info((1, 1, 4)) is None
    assert manifest_module.parse_weight_info((1, 1, 4, [1])) is None
    assert manifest_module.parse_weight_info({"ptr": 1, "numel": 1, "element_size": 4}) is None
    parsed = manifest_module.parse_weight_info((1, 1, 4, [1], torch.float32))
    assert parsed == (1, 1, 4, (1,), "float32")


def test_seed_url_appends_independent_port_and_rejects_conflicts(seed_module):
    assert seed_module.build_seed_url("http://seed-host", 1234) == "http://seed-host:1234"
    assert seed_module.build_seed_url("https://seed-host:1234", 1234) == "https://seed-host:1234"
    assert seed_module.build_seed_url("[::1]", 1234) == "http://[::1]:1234"
    assert seed_module.build_seed_url("[::1]:1234", 1234) == "http://[::1]:1234"
    with pytest.raises(ValueError, match="conflicts"):
        seed_module.build_seed_url("http://seed-host:4321", 1234)
    with pytest.raises(ValueError, match="conflicts"):
        seed_module.build_seed_url("[::1]:4321", 1234)


@pytest.mark.parametrize("protocol_version", [None, 0, True, 1.0, 2])
def test_seed_metadata_rejects_other_protocol_versions(seed_module, monkeypatch, protocol_version):
    response = SimpleNamespace(
        status_code=200,
        json=lambda: {
            "rfork_protocol_version": protocol_version,
            "rfork_transfer_engine_info": ["session", {"weight": [1, 1, 4, [1], "float32"]}],
            "rfork_transfer_engine_format_info": {"weight": 2},
        },
    )
    monkeypatch.setattr(seed_module.requests, "get", Mock(return_value=response))
    assert seed_module.fetch_seed_transfer_info("http://seed", "key", 1.0) is None


@pytest.mark.parametrize(
    ("seed_port", "seed_rank", "message"),
    [("not-a-port", "0", "malformed"), ("0", "-1", "invalid")],
)
def test_planner_rejects_invalid_lease_headers_with_error(
    planner_module, monkeypatch, caplog, seed_port, seed_rank, message
):
    config = SimpleNamespace(
        request_timeout_sec=1.0,
        planner_url="http://planner",
        model_url="model",
        model_deploy_strategy_name="strategy",
        lease_release_max_attempts=1,
        lease_release_retry_interval_sec=1.0,
        heartbeat_interval_sec=1.0,
    )
    identity = SimpleNamespace(
        tp_rank=0, pp_rank=None, ep_rank=None, is_draft_model=False, compatibility_fingerprint="fp"
    )
    client = planner_module.RForkPlannerClient(config, identity)
    response = SimpleNamespace(
        status_code=200,
        headers={
            "SEED_IP": "127.0.0.1",
            "SEED_PORT": seed_port,
            "USER_ID": "lease",
            "SEED_RANK": seed_rank,
            "LEASE_TTL_SEC": "60",
        },
    )
    monkeypatch.setattr(planner_module.requests, "get", Mock(return_value=response))

    with caplog.at_level(logging.WARNING):
        assert client.acquire_seed() is None
    assert message in caplog.text


def test_planner_acquires_and_renews_a_ttl_lease(planner_module, monkeypatch):
    config = SimpleNamespace(
        request_timeout_sec=1.0,
        planner_url="http://planner",
        model_url="model",
        model_deploy_strategy_name="strategy",
    )
    identity = SimpleNamespace(
        tp_rank=0, pp_rank=None, ep_rank=None, is_draft_model=False, compatibility_fingerprint="fp"
    )
    get_response = SimpleNamespace(
        status_code=200,
        headers={
            "SEED_IP": "127.0.0.1",
            "SEED_PORT": "1234",
            "USER_ID": "lease",
            "SEED_RANK": "0",
            "LEASE_TTL_SEC": "90",
        },
    )
    monkeypatch.setattr(planner_module.requests, "get", Mock(return_value=get_response))
    post = Mock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(planner_module.requests, "post", post)

    client = planner_module.RForkPlannerClient(config, identity)
    lease = client.acquire_seed()
    assert lease.lease_ttl_sec == 90
    assert client.renew_seed_once(lease)
    assert post.call_args.args[0].endswith("/renew_seed_lease")
