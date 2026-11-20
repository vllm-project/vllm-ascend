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


def test_owner_sharing_mode_defaults_to_replicated():
    assert owner_kv.pcp_kv_sharing_mode_from_config(_config()) == owner_kv.PCP_KV_SHARING_REPLICATED
    assert owner_kv.owner_sharding_enabled(_config()) is False
    # No vllm_config means the replicated defaults; nothing reads the process
    # environment anymore.
    assert owner_kv.owner_sharding_enabled() is False


def test_typed_pcp_config_selects_owner_and_rejects_unknown_mode():
    config = _config(additional_config={"pcp_config": {"kv_sharing": "owner"}})
    assert owner_kv.pcp_kv_sharing_mode_from_config(config) == owner_kv.PCP_KV_SHARING_OWNER
    assert owner_kv.owner_sharding_enabled(config)

    invalid = _config(additional_config={"pcp_config": {"kv_sharing": "invalid"}})
    with pytest.raises(ValueError):
        owner_kv.pcp_kv_sharing_mode_from_config(invalid)


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
