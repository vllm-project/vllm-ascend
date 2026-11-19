# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Owner-sharded PCP KV placement selection and topology validation."""

import pytest

from vllm_ascend.attention.context_parallel import pcp_owner_kv as owner_kv


def _config(*, pcp=4, dcp=1, tp=1, kv_transfer=None, additional_config=None):
    return type(
        "Config",
        (),
        {
            "parallel_config": type(
                "Parallel",
                (),
                {
                    "prefill_context_parallel_size": pcp,
                    "decode_context_parallel_size": dcp,
                    "tensor_parallel_size": tp,
                },
            )(),
            "kv_transfer_config": kv_transfer,
            "additional_config": additional_config or {},
        },
    )()


def test_owner_sharing_mode_flag_defaults_to_replicated(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_PCP_KV_SHARING", raising=False)
    assert owner_kv.get_pcp_kv_sharing_mode() == owner_kv.PCP_KV_SHARING_REPLICATED
    monkeypatch.setenv("VLLM_ASCEND_PCP_KV_SHARING", "owner")
    assert owner_kv.owner_sharding_enabled()
    monkeypatch.setenv("VLLM_ASCEND_PCP_KV_SHARING", "invalid")
    with pytest.raises(ValueError, match="must be one of"):
        owner_kv.get_pcp_kv_sharing_mode()


def test_additional_config_takes_precedence_over_the_env_flag(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PCP_KV_SHARING", "replicated")
    config = _config(additional_config={"pcp_kv_sharing": "owner"})
    assert owner_kv.pcp_kv_sharing_mode_from_config(config) == owner_kv.PCP_KV_SHARING_OWNER
    assert owner_kv.owner_sharding_enabled(config)
    monkeypatch.setenv("VLLM_ASCEND_PCP_KV_SHARING", "owner")
    config = _config(additional_config={"pcp_kv_sharing": "replicated"})
    assert owner_kv.pcp_kv_sharing_mode_from_config(config) == owner_kv.PCP_KV_SHARING_REPLICATED


def test_validate_owner_sharing_rejects_incompatible_topologies():
    assert owner_kv.validate_owner_sharing(_config(pcp=1))[0] is False
    assert owner_kv.validate_owner_sharing(_config(dcp=2))[0] is False
    assert owner_kv.validate_owner_sharing(_config(tp=2))[0] is False
    # KV-transfer connectors are out of scope for this placement.
    assert owner_kv.validate_owner_sharing(_config(kv_transfer=object()))[0] is False
    assert owner_kv.validate_owner_sharing(_config())[0] is True


def test_resolve_owner_sharing_uses_the_device_backend_probe(monkeypatch):
    from vllm_ascend.attention.context_parallel import pcp_owner_vmm

    monkeypatch.setattr(pcp_owner_vmm, "owner_slab_support", lambda: (True, ""))
    assert owner_kv.resolve_owner_sharing(_config()) == (True, "")
    monkeypatch.setattr(pcp_owner_vmm, "owner_slab_support", lambda: (False, "missing symbol"))
    supported, reason = owner_kv.resolve_owner_sharing(_config())
    assert supported is False
    assert "not installed" in reason
