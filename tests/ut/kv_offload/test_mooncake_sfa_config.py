# SPDX-License-Identifier: Apache-2.0
"""The V2/SFA combination has one Host owner and an explicit runner/backend."""

from types import SimpleNamespace

import pytest
import vllm.envs

from vllm_ascend import ascend_config
from vllm_ascend.ascend_config import SparseKVOffloadConfig


def config(dsa=None):
    return SimpleNamespace(
        additional_config={"dsa_sparse_attention_config": dsa or {}},
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeConnectorV2",
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


@pytest.fixture(autouse=True, params=["MooncakeConnectorV2", "MooncakePullConnector"])
def v2_a3(monkeypatch, request):
    factory = config

    def selected_config(dsa=None):
        resolved = factory(dsa)
        resolved.kv_transfer_config.kv_connector = request.param
        return resolved

    monkeypatch.setattr(__import__(__name__, fromlist=["config"]), "config", selected_config)
    monkeypatch.setattr(ascend_config, "is_950", lambda: False)


def test_v2_consumer_selects_existing_lim_copy_backend():
    resolved = SparseKVOffloadConfig.from_additional_config(config(), {"enabled": True})
    assert resolved.enabled and resolved.use_fused_copy_sfa


@pytest.mark.parametrize("legacy", [{"enable_cpu_kv_store": True}, {"tp_shared_host_kv": True}])
def test_rejects_two_host_pool_owners(legacy):
    with pytest.raises(ValueError, match="SparseKVOffloadManager"):
        SparseKVOffloadConfig.from_additional_config(config(legacy), {"enabled": True})


def test_rejects_model_runner_v1():
    cfg = config()
    cfg.use_v2_model_runner = False
    with pytest.raises(ValueError, match="ModelRunner V2"):
        SparseKVOffloadConfig.from_additional_config(cfg, {"enabled": True})


def test_accepts_v2_after_upstream_removes_runner_environment_entry(monkeypatch):
    monkeypatch.delitem(vllm.envs.environment_variables, "VLLM_USE_V2_MODEL_RUNNER", raising=False)
    resolved = SparseKVOffloadConfig.from_additional_config(config(), {"enabled": True})
    assert resolved.enabled and resolved.use_fused_copy_sfa


def test_rejects_unadapted_overlap_backend():
    with pytest.raises(ValueError, match="LIM \\+ Copy-SFA"):
        SparseKVOffloadConfig.from_additional_config(config(), {"enabled": True, "use_fused_overlap": True})


def test_rejects_dspark_before_uninitialized_remote_prompt_can_be_used():
    cfg = config()
    cfg.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=1)
    with pytest.raises(ValueError, match="DSpark remote prompt protocol"):
        SparseKVOffloadConfig.from_additional_config(cfg, {"enabled": True})


def test_rejects_multiconnector_without_sparse_handoff_forwarding():
    cfg = config()
    cfg.kv_transfer_config.kv_connector = "MultiConnector"
    cfg.kv_transfer_config.has_connector = lambda name: name == "MooncakeConnectorV2"
    with pytest.raises(ValueError, match="direct connector"):
        SparseKVOffloadConfig.from_additional_config(cfg, {"enabled": True})


@pytest.mark.parametrize("keep_device_kv_cache", [False, True])
def test_consumer_rejects_recompute_without_a_failed_pd_recovery_handoff(keep_device_kv_cache):
    cfg = config()
    cfg.kv_transfer_config.kv_load_failure_policy = "recompute"
    with pytest.raises(ValueError, match="kv_load_failure_policy='fail'"):
        SparseKVOffloadConfig.from_additional_config(
            cfg, {"enabled": True, "keep_device_kv_cache": keep_device_kv_cache}
        )


@pytest.mark.parametrize("keep_device_kv_cache", [False, True])
@pytest.mark.parametrize("failure_policy", [None, "fail"])
def test_consumer_accepts_default_and_explicit_fail_policy(keep_device_kv_cache, failure_policy):
    cfg = config()
    if failure_policy is not None:
        cfg.kv_transfer_config.kv_load_failure_policy = failure_policy
    resolved = SparseKVOffloadConfig.from_additional_config(
        cfg, {"enabled": True, "keep_device_kv_cache": keep_device_kv_cache}
    )
    assert resolved.keep_device_kv_cache == keep_device_kv_cache


def test_producer_colocate_debug_does_not_require_a_receive_failure_policy():
    cfg = config()
    cfg.kv_transfer_config.is_kv_consumer = False
    cfg.kv_transfer_config.kv_load_failure_policy = "recompute"
    resolved = SparseKVOffloadConfig.from_additional_config(cfg, {"enabled": True, "keep_device_kv_cache": True})
    assert resolved.keep_device_kv_cache


@pytest.mark.parametrize("connector", ["SfaRemoteD2HConnector", "MultiConnector"])
def test_existing_sfa_configuration_is_unchanged(connector):
    cfg = config()
    cfg.kv_transfer_config.kv_connector = connector
    cfg.kv_transfer_config.kv_load_failure_policy = "recompute"
    cfg.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=1)
    cfg.use_v2_model_runner = True
    resolved = SparseKVOffloadConfig.from_additional_config(cfg, {"enabled": True})
    assert resolved.enabled
