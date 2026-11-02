# SPDX-License-Identifier: Apache-2.0
"""The transported prefix must keep target/draft ownership and graph bindings."""

import importlib
from types import SimpleNamespace as NS

import pytest


def config(**changes):
    spec = dict(
        method="mtp",
        num_speculative_tokens=3,
        draft_model_config=NS(model="checkpoint"),
        enforce_eager=False,
    )
    spec.update(changes)
    return NS(
        speculative_config=NS(**spec),
        model_config=NS(model="checkpoint", enforce_eager=False),
        compilation_config=NS(
            cudagraph_mode=NS(name="FULL_DECODE_ONLY"),
            static_forward_context={"target": NS(), "draft": NS()},
        ),
        kv_transfer_config=NS(
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "HiSparseConnector"},
                    {"kv_connector": "HiSparsePDConnector"},
                ]
            },
        ),
    )


def policy(modules):
    return importlib.import_module("pd_under_test.execution")


def test_mtp_graph_handoff_exports_only_completed_prefix(modules):
    execution = policy(modules)
    cfg = config()
    assert execution.validate_execution(cfg) == "decode_graph"
    assert execution.execution_contract(cfg) == execution.MTP_CONTRACT
    state = execution.draft_state(512)
    assert state["prefix_tokens"] == 512 and state["lookahead_tokens"] == 0
    execution.validate_draft_state(state, 512, True)
    with pytest.raises(ValueError):
        execution.validate_draft_state({**state, "lookahead_tokens": 1}, 512, True)
    cfg.speculative_config = None
    assert execution.execution_contract(cfg) == execution.TARGET_CONTRACT
    with pytest.raises(ValueError):
        execution.validate_draft_state(state, 512, False)


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "eagle"},
        {"num_speculative_tokens": 2},
        {"draft_model_config": NS(model="other-checkpoint")},
        {"enforce_eager": True},
    ],
)
def test_unimplemented_draft_policies_remain_rejected(modules, changes):
    with pytest.raises(ValueError):
        policy(modules).validate_execution(config(**changes))


def test_loaded_layer_names_do_not_change_common_group_planning(modules):
    owners = importlib.import_module("pd_under_test.owners")
    cfg = config()
    group = NS(
        enable_kv_transfer=True,
        is_eagle_group=False,
        layer_names=["target", "draft"],
    )
    cache = NS(kv_cache_groups=[group])
    loaded = NS(draft_attn_layer_names={"draft"})
    owners.bind_loaded_owners(cfg, loaded, cache)
    context = cfg.compilation_config.static_forward_context
    assert context["target"].hisparse_pd_model_role == "target"
    assert context["draft"].hisparse_pd_model_role == "draft"
    assert group.is_eagle_group is False
    loaded.draft_attn_layer_names.add("missing-cache")
    with pytest.raises(ValueError):
        owners.bind_loaded_owners(cfg, loaded, cache)


def test_export_waits_for_the_common_draft_mirror(modules):
    execution = policy(modules)
    worker = NS(_draft_mirror_pending=True)
    assert execution.export_mirror_ready(worker, True) is False
    worker._draft_mirror_pending = False
    assert execution.export_mirror_ready(worker, True) is True
    with pytest.raises(RuntimeError):
        execution.export_mirror_ready(None, True)


def test_transfer_rejects_capture_and_changed_registered_binding(modules):
    execution = policy(modules)
    layer = modules.protocol.LayerLayout("target", 0, "hisparse_source", 128, 656, 4096, 83968, 2)
    owner = object()
    guard = execution.CacheBindingGuard([layer], {4096: 167936}, [owner])
    guard.validate([layer], {4096: 167936}, [owner])
    with pytest.raises(RuntimeError):
        guard.validate([layer], {4096: 167936}, [object()])
    with pytest.raises(RuntimeError):
        execution.assert_transfer_boundary(capturing=True, closing=False, closed=False)
