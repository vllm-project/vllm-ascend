# SPDX-License-Identifier: Apache-2.0
"""P-side DSpark configuration belongs to the context backend, not the runner."""

from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    get_pd_dspark_aux_layer_ids,
    uses_sfa_dspark_kv_transfer,
)


def _config(
    ids=(2, 22, 38, 58, 74), *, producer=True, consumer=False, speculative=None, pcp=1, dcp=1, eager=True, prefix=False
):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="SfaRemoteD2HConnector",
            kv_connector_extra_config={"dspark_aux_hidden_state_layer_ids": ids},
            is_kv_producer=producer,
            is_kv_consumer=consumer,
        ),
        speculative_config=speculative
        or SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=ids)),
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp),
        model_config=SimpleNamespace(enforce_eager=eager, hf_text_config=SimpleNamespace(num_hidden_layers=78)),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix),
    )


@pytest.mark.parametrize("ids", [[2, 22, 38, 58, 74], (2, 22, 38, 58, 74), [0, 78], (0, 78)])
def test_backend_returns_immutable_boundaries_without_mutating_config(ids):
    config = _config(ids)
    assert get_pd_dspark_aux_layer_ids(config) == tuple(ids)
    assert config.kv_transfer_config.kv_connector_extra_config["dspark_aux_hidden_state_layer_ids"] is ids


@pytest.mark.parametrize("ids", [[], [38, 22], [2, 2], [79], [-1], [True], [1.0], "2,22,38", {2: 22}, [[2]]])
def test_backend_rejects_invalid_auxiliary_schema(ids):
    with pytest.raises(ValueError, match="boundaries"):
        get_pd_dspark_aux_layer_ids(_config(ids))


@pytest.mark.parametrize(
    "options,message",
    [
        ({"pcp": 2}, "context parallelism"),
        ({"dcp": 2}, "context parallelism"),
        ({"eager": False}, "eager prefill"),
        ({"prefix": True}, "prefix caching disabled"),
    ],
)
def test_backend_preserves_capture_configuration_constraints(options, message):
    with pytest.raises(ValueError, match=message):
        get_pd_dspark_aux_layer_ids(_config(**options))


@pytest.mark.parametrize(
    "options", [{"producer": False}, {"consumer": True}, {"speculative": SimpleNamespace(method="mtp")}]
)
def test_other_roles_and_drafters_do_not_enable_p_dspark_generation(options):
    assert get_pd_dspark_aux_layer_ids(_config(**options)) == ()


@pytest.mark.parametrize("extra", [None, {}, {"dspark_aux_hidden_state_layer_ids": None}])
def test_unconfigured_backend_does_not_apply_dspark_constraints(extra):
    # Other fields deliberately omitted: opting out must not inspect them.
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra))
    assert get_pd_dspark_aux_layer_ids(config) == ()


@pytest.mark.parametrize("config", [SimpleNamespace(), SimpleNamespace(kv_transfer_config=None)])
def test_without_kv_transfer_does_not_enable_capture(config):
    assert get_pd_dspark_aux_layer_ids(config) == ()


def test_d_checkpoint_without_p_option_does_not_enable_prefill_capture():
    config = _config(None, producer=False, consumer=True, speculative=SimpleNamespace(method="dspark"), eager=False)
    assert get_pd_dspark_aux_layer_ids(config) == ()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("sfa", [False, True])
def test_backend_opt_in_follows_exact_connector_in_nested_multi(nested, sfa):
    config = _config()
    child: dict[str, object] = {"kv_connector": "SfaRemoteD2HConnector" if sfa else "MooncakeConnector"}
    if nested:
        child = {"kv_connector": "MultiConnector", "kv_connector_extra_config": {"connectors": [child]}}
    config.kv_transfer_config.kv_connector = "MultiConnector"
    config.kv_transfer_config.kv_connector_extra_config = {
        "connectors": [{"kv_connector": "AscendStoreConnector"}, child]
    }
    assert uses_sfa_dspark_kv_transfer(config) is sfa
    assert get_pd_dspark_aux_layer_ids(config) == ((2, 22, 38, 58, 74) if sfa else ())


@pytest.mark.parametrize("connector", ["MooncakeConnector", "AscendStoreConnector", "MultiConnector"])
def test_other_pd_backends_do_not_apply_sfa_prefill_constraints(connector):
    config = _config(eager=False, prefix=True, pcp=2)
    config.kv_transfer_config.kv_connector = connector
    assert not uses_sfa_dspark_kv_transfer(config)
    assert get_pd_dspark_aux_layer_ids(config) == ()
