# SPDX-License-Identifier: Apache-2.0
"""The V2/SFA combination has one Host owner and an explicit runner/backend."""

from types import SimpleNamespace

import pytest
import vllm.envs

from vllm_ascend import ascend_config
from vllm_ascend.ascend_config import SparseKVOffloadConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.config import validate_mooncake_sfa_config


def config():
    return SimpleNamespace(
        additional_config={},
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeConnectorV2",
            kv_connector_extra_config={},
            is_kv_consumer=True,
            has_connector=lambda _name: False,
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        speculative_config=None,
        use_v2_model_runner=True,
    )


def resolve_config(vllm_config, settings):
    sparse = SparseKVOffloadConfig.from_additional_config(vllm_config, settings)
    validate_mooncake_sfa_config(vllm_config, sparse)
    return sparse


@pytest.fixture(autouse=True)
def v2_a3(monkeypatch):
    monkeypatch.setattr(ascend_config, "is_950", lambda: False)


def test_v2_consumer_selects_existing_lim_copy_backend():
    resolved = resolve_config(config(), {"enabled": True})
    assert resolved.enabled and resolved.use_fused_copy_sfa


@pytest.mark.parametrize("consumer", [False, True])
def test_pull_alias_rejects_sparse_offload_with_v2_migration_message(consumer):
    cfg = config()
    cfg.kv_transfer_config.kv_connector = "MooncakePullConnector"
    cfg.kv_transfer_config.is_kv_consumer = consumer
    with pytest.raises(ValueError, match="kv_connector='MooncakeConnectorV2'"):
        resolve_config(cfg, {"enabled": True, "keep_device_kv_cache": not consumer})


@pytest.mark.parametrize("settings", [{}, {"enabled": False}])
def test_pull_alias_without_sparse_offload_remains_supported(settings):
    cfg = config()
    cfg.kv_transfer_config.kv_connector = "MooncakePullConnector"
    assert not resolve_config(cfg, settings).enabled


def test_rejects_model_runner_v1():
    cfg = config()
    cfg.use_v2_model_runner = False
    with pytest.raises(ValueError, match="ModelRunner V2"):
        resolve_config(cfg, {"enabled": True})


def test_accepts_v2_after_upstream_removes_runner_environment_entry(monkeypatch):
    monkeypatch.delitem(vllm.envs.environment_variables, "VLLM_USE_V2_MODEL_RUNNER", raising=False)
    resolved = resolve_config(config(), {"enabled": True})
    assert resolved.enabled and resolved.use_fused_copy_sfa


def test_rejects_unadapted_overlap_backend():
    with pytest.raises(ValueError, match="LIM \\+ Copy-SFA"):
        resolve_config(config(), {"enabled": True, "use_fused_overlap": True})


def test_rejects_dspark_before_uninitialized_remote_prompt_can_be_used():
    cfg = config()
    cfg.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=1)
    with pytest.raises(ValueError, match="DSpark remote prompt protocol"):
        resolve_config(cfg, {"enabled": True})


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
def test_rejects_multiconnector_without_sparse_handoff_forwarding(connector):
    cfg = config()
    cfg.kv_transfer_config.kv_connector = "MultiConnector"
    cfg.kv_transfer_config.has_connector = lambda name: name == connector
    with pytest.raises(ValueError, match="direct connector"):
        resolve_config(cfg, {"enabled": True})


@pytest.mark.parametrize("keep_device_kv_cache", [False, True])
def test_consumer_rejects_recompute_without_a_failed_pd_recovery_handoff(keep_device_kv_cache):
    cfg = config()
    cfg.kv_transfer_config.kv_load_failure_policy = "recompute"
    with pytest.raises(ValueError, match="kv_load_failure_policy='fail'"):
        resolve_config(cfg, {"enabled": True, "keep_device_kv_cache": keep_device_kv_cache})


@pytest.mark.parametrize("keep_device_kv_cache", [False, True])
@pytest.mark.parametrize("failure_policy", [None, "fail"])
def test_consumer_accepts_default_and_explicit_fail_policy(keep_device_kv_cache, failure_policy):
    cfg = config()
    if failure_policy is not None:
        cfg.kv_transfer_config.kv_load_failure_policy = failure_policy
    resolved = resolve_config(cfg, {"enabled": True, "keep_device_kv_cache": keep_device_kv_cache})
    assert resolved.keep_device_kv_cache == keep_device_kv_cache


def test_producer_colocate_debug_does_not_require_a_receive_failure_policy():
    cfg = config()
    cfg.kv_transfer_config.is_kv_consumer = False
    cfg.kv_transfer_config.kv_load_failure_policy = "recompute"
    resolved = resolve_config(cfg, {"enabled": True, "keep_device_kv_cache": True})
    assert resolved.keep_device_kv_cache


@pytest.mark.parametrize("connector", ["SfaRemoteD2HConnector", "MultiConnector"])
def test_existing_sfa_configuration_is_unchanged(connector):
    cfg = config()
    cfg.kv_transfer_config.kv_connector = connector
    cfg.kv_transfer_config.kv_load_failure_policy = "recompute"
    cfg.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=1)
    cfg.use_v2_model_runner = True
    resolved = resolve_config(cfg, {"enabled": True})
    assert resolved.enabled
