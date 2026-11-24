# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Owner-sharded PCP KV placement selection and topology validation."""

import pytest

from vllm_ascend.attention.context_parallel import pcp_owner_kv as owner_kv


def _kv_transfer(name, extra=None):
    return type(
        "KVTransfer",
        (),
        {
            "kv_connector": name,
            "kv_connector_extra_config": extra or {},
        },
    )()


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
    assert owner_kv.validate_owner_sharing(_config())[0] is True


def test_owner_connector_gate_allows_owner_aware_store():
    store = _kv_transfer("AscendStoreConnector", {"backend": "mooncake"})
    assert owner_kv.validate_owner_connector(_config(kv_transfer=store)) == (True, "")
    assert owner_kv.validate_owner_sharing(_config(kv_transfer=store))[0] is True

    old_name = _kv_transfer("MooncakeConnectorStoreV1", {"backend": "mooncake"})
    assert owner_kv.validate_owner_connector(_config(kv_transfer=old_name)) == (True, "")


def test_owner_connector_gate_accepts_multi_connector_store_children():
    multi = _kv_transfer(
        "MultiConnector",
        {
            "connectors": [
                {"kv_connector": "AscendStoreConnector", "kv_connector_extra_config": {"backend": "mooncake"}},
                {"kv_connector": "AscendStoreConnector", "kv_connector_extra_config": {"backend": "mooncake"}},
            ]
        },
    )
    assert owner_kv.validate_owner_connector(_config(kv_transfer=multi)) == (True, "")


def test_owner_connector_gate_accepts_staged_direct_p2p_connectors():
    direct = _kv_transfer("MooncakeConnectorV1", {})
    assert owner_kv.validate_owner_connector(_config(kv_transfer=direct)) == (True, "")
    assert owner_kv.validate_owner_sharing(_config(kv_transfer=direct))[0] is True

    hybrid = _kv_transfer("MooncakeHybridConnector", {"backend": "mooncake"})
    assert owner_kv.validate_owner_connector(_config(kv_transfer=hybrid)) == (True, "")

    multi = _kv_transfer(
        "MultiConnector",
        {
            "connectors": [
                {"kv_connector": "AscendStoreConnector", "kv_connector_extra_config": {}},
                {"kv_connector": "MooncakeConnectorV1", "kv_connector_extra_config": {}},
            ]
        },
    )
    assert owner_kv.validate_owner_connector(_config(kv_transfer=multi)) == (True, "")

    mixed = _kv_transfer(
        "MultiConnector",
        {
            "connectors": [
                {"kv_connector": "AscendStoreConnector", "kv_connector_extra_config": {}},
                {"kv_connector": "MooncakeHybridConnector", "kv_connector_extra_config": {}},
            ]
        },
    )
    assert owner_kv.validate_owner_connector(_config(kv_transfer=mixed)) == (True, "")


def test_owner_connector_gate_rejects_layerwise_direct_p2p():
    for name in ("MooncakeConnectorV1", "MooncakeHybridConnector"):
        layerwise = _kv_transfer(name, {"use_layerwise": True})
        supported, reason = owner_kv.validate_owner_connector(_config(kv_transfer=layerwise))
        assert supported is False
        assert "layerwise" in reason

    # Options are not normalized across connector groups; a nested truthy
    # layerwise switch must also fail closed.
    nested = _kv_transfer(
        "MultiConnector",
        {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_connector_extra_config": {"transfer": {"use_layerwise": True}},
                }
            ]
        },
    )
    supported, reason = owner_kv.validate_owner_connector(_config(kv_transfer=nested))
    assert supported is False
    assert "layerwise" in reason

    top_level = _kv_transfer(
        "MultiConnector",
        {
            "use_layerwise": True,
            "connectors": [
                {"kv_connector": "MooncakeHybridConnector", "kv_connector_extra_config": {}},
            ],
        },
    )
    supported, reason = owner_kv.validate_owner_connector(_config(kv_transfer=top_level))
    assert supported is False
    assert "layerwise" in reason


def test_owner_connector_gate_rejects_unvalidated_connectors():
    layerwise = _kv_transfer("AscendStoreConnector", {"use_layerwise": True})
    supported, reason = owner_kv.validate_owner_connector(_config(kv_transfer=layerwise))
    assert supported is False
    assert "layerwise" in reason

    unsupported = _kv_transfer("MooncakeD2RHConnectorV1", {})
    supported, reason = owner_kv.validate_owner_connector(_config(kv_transfer=unsupported))
    assert supported is False
    assert "not owner-aware" in reason

    unnamed = _kv_transfer("MultiConnector", {"connectors": [{"kv_role": "kv_producer"}]})
    assert owner_kv.validate_owner_connector(_config(kv_transfer=unnamed))[0] is False


def test_resolve_owner_sharing_uses_the_device_backend_probe(monkeypatch):
    from vllm_ascend.attention.context_parallel import pcp_owner_vmm

    monkeypatch.setattr(pcp_owner_vmm, "owner_slab_support", lambda: (True, ""))
    assert owner_kv.resolve_owner_sharing(_config()) == (True, "")
    monkeypatch.setattr(pcp_owner_vmm, "owner_slab_support", lambda: (False, "missing symbol"))
    supported, reason = owner_kv.resolve_owner_sharing(_config())
    assert supported is False
    assert "not installed" in reason
