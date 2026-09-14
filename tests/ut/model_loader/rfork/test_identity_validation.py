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
        get_current_hardware_profile=lambda: SimpleNamespace(weight_layout_policy=None),
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
    )
    return _load_module(monkeypatch, "rfork_test_planner_client", "planner_client.py")


def _model_config(*, dtype=torch.float32, quantization=None, revision="rev-a"):
    return SimpleNamespace(
        dtype=dtype,
        quantization=quantization,
        revision=revision,
        hf_config=SimpleNamespace(
            model_type="test-model",
            architectures=["TestModel"],
            quantization_config=None,
        ),
    )


def _vllm_config(tp_size=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            expert_parallel_size=None,
            data_parallel_size=None,
            enable_expert_parallel=False,
            is_moe_model=False,
        )
    )


def _build_fingerprint(identity_module, model_config, *, tp_size=1):
    return identity_module.build_compatibility_fingerprint(
        _vllm_config(tp_size),
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


def test_canonicalization_is_deterministic_for_dicts_sets_and_dtype(identity_module):
    first = {"set": {3, 1, 2}, "mapping": {"b": 2, "a": 1}, "dtype": torch.float16}
    second = {"dtype": torch.float16, "mapping": {"a": 1, "b": 2}, "set": {2, 3, 1}}

    assert identity_module._canonicalize_fingerprint_value(first) == identity_module._canonicalize_fingerprint_value(
        second
    )


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": torch.float16},
        {"quantization": "compressed"},
        {"revision": "rev-b"},
        {"tp_size": 2},
    ],
)
def test_fingerprint_changes_when_seed_compatibility_changes(identity_module, change):
    base = _fingerprint(identity_module)
    assert _fingerprint(identity_module, **change) != base


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


def test_fingerprint_changes_for_other_hf_behavior_fields(identity_module):
    first = _model_config()
    second = _model_config()
    first.hf_config.hidden_act = "silu"
    second.hf_config.hidden_act = "gelu"
    assert _build_fingerprint(identity_module, first) != _build_fingerprint(identity_module, second)


@pytest.mark.parametrize(
    ("field", "base_value", "changed_value"),
    [
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
    ],
)
def test_fingerprint_changes_when_rope_configuration_changes(identity_module, field, base_value, changed_value):
    # RoPE values can alter cache contents without changing tensor shapes.
    base_config = _model_config()
    changed_config = _model_config()
    setattr(base_config.hf_config, field, base_value)
    setattr(changed_config.hf_config, field, changed_value)

    base_fingerprint = _build_fingerprint(identity_module, base_config)
    changed_fingerprint = _build_fingerprint(identity_module, changed_config)
    assert base_fingerprint != changed_fingerprint
    assert identity_module.build_seed_key(0, "model", "strategy", base_fingerprint) != (
        identity_module.build_seed_key(0, "model", "strategy", changed_fingerprint)
    )


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


@pytest.mark.parametrize(
    ("config_field", "commit_field"),
    [("hf_config", "_commit_hash"), ("hf_text_config", "commit_hash")],
)
def test_resolved_commits_isolate_moving_revision_seeds(identity_module, config_field, commit_field):
    first = _model_config(revision="main")
    second = _model_config(revision="main")
    for config, commit in ((first, "a" * 40), (second, "b" * 40)):
        if config_field == "hf_text_config":
            config.hf_text_config = SimpleNamespace()
            config.hf_config.revision = "main"
        setattr(getattr(config, config_field), commit_field, commit)
    first_fingerprint = _build_fingerprint(identity_module, first)
    second_fingerprint = _build_fingerprint(identity_module, second)
    assert first_fingerprint != second_fingerprint
    assert identity_module.build_seed_key(0, "model", "strategy", first_fingerprint) != (
        identity_module.build_seed_key(0, "model", "strategy", second_fingerprint)
    )


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


def test_recursive_fingerprint_values_fail_with_actionable_error(identity_module):
    recursive_dict = {}
    recursive_dict["self"] = recursive_dict
    recursive_list = []
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


def test_build_seed_key_rejects_missing_identity_values(identity_module):
    build = identity_module.build_seed_key
    with pytest.raises(RuntimeError, match="model_url"):
        build(0, "", "strategy", "fingerprint")
    with pytest.raises(RuntimeError, match="model_deploy_strategy_name"):
        build(0, "model", "", "fingerprint")
    with pytest.raises(RuntimeError, match="compatibility fingerprint"):
        build(0, "model", "strategy", "")
    with pytest.raises(TypeError):
        build(0, "model", "strategy")


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
