"""Tests for Ascend-specific MultiConnector allocation fan-out."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (  # noqa: E402
    AscendMultiConnector,
    MultiConnector,
)


@pytest.mark.parametrize(
    "producer,consumer,method,full_prompt",
    [
        (True, False, "dspark", True),
        (True, False, "mtp", False),
        (True, False, None, False),
        (False, True, "dspark", False),
        (True, True, "dspark", False),
    ],
)
@pytest.mark.parametrize("pd_connector", ["SfaRemoteD2HConnector", "MooncakeConnector"])
def test_dspark_full_prompt_policy_is_sfa_producer_only(producer, consumer, method, full_prompt, pd_connector):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MultiConnector",
            is_kv_producer=producer,
            is_kv_consumer=consumer,
            kv_connector_extra_config={"connectors": [{"kv_connector": pd_connector}]},
        ),
        speculative_config=SimpleNamespace(method=method) if method is not None else None,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )
    with patch.object(MultiConnector, "__init__", return_value=None):
        connector = AscendMultiConnector.__new__(AscendMultiConnector)
        connector._connectors = []
        connector.__init__(config, object(), None)
    assert connector._requires_full_dspark_prompt is (full_prompt and pd_connector == "SfaRemoteD2HConnector")


@pytest.mark.parametrize("cached_prefix", [17920, 19042])
def test_dspark_producer_does_not_skip_aux_states_on_repeated_prefix(cached_prefix):
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._requires_full_dspark_prompt = True
    connector._connectors = [SimpleNamespace(has_preempted_request=MagicMock(return_value=False))]
    connector._requests_to_connector = {}
    request = SimpleNamespace(request_id="repeated-prefix", num_tokens=19043)
    with patch.object(MultiConnector, "get_num_new_matched_tokens", return_value=(cached_prefix, True)) as parent:
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    parent.assert_not_called()
    connector._connectors[0].has_preempted_request.assert_not_called()
    assert connector._requests_to_connector == {}


def _dspark_producer_config():
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MultiConnector",
            is_kv_producer=True,
            is_kv_consumer=False,
            kv_connector_extra_config={"connectors": [{"kv_connector": "SfaRemoteD2HConnector"}]},
        ),
        speculative_config=SimpleNamespace(method="dspark"),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )


def test_dspark_complete_prefix_provider_enables_only_its_lookup():
    config = _dspark_producer_config()
    target_only = SimpleNamespace(
        configure_dspark_prefix_cache=MagicMock(return_value=False),
        get_num_new_matched_tokens=MagicMock(return_value=(1024, True)),
        has_preempted_request=MagicMock(return_value=True),
    )
    joint_store = SimpleNamespace(
        configure_dspark_prefix_cache=MagicMock(return_value=True),
        get_num_new_matched_tokens=MagicMock(return_value=(128, False)),
    )
    with patch.object(MultiConnector, "__init__", return_value=None):
        connector = AscendMultiConnector.__new__(AscendMultiConnector)
        connector._connectors = [target_only, joint_store]
        connector._requests_to_connector = {}
        connector.__init__(config, object(), None)

    assert connector._requires_full_dspark_prompt is False
    assert connector._dspark_prefix_provider_index == 1
    request = SimpleNamespace(request_id="paired-prefix")
    with patch.object(MultiConnector, "get_num_new_matched_tokens") as parent:
        assert connector.get_num_new_matched_tokens(request, 0) == (128, False)
    joint_store.get_num_new_matched_tokens.assert_called_once_with(request, 0)
    target_only.get_num_new_matched_tokens.assert_not_called()
    target_only.has_preempted_request.assert_not_called()
    parent.assert_not_called()
    assert connector._requests_to_connector == {request.request_id: 1}


@pytest.mark.parametrize("hit", [(0, False), (None, False)])
def test_dspark_missing_joint_prefix_does_not_fall_back_to_target_only_hit(hit):
    target_only = SimpleNamespace(get_num_new_matched_tokens=MagicMock(return_value=(1024, True)))
    joint_store = SimpleNamespace(get_num_new_matched_tokens=MagicMock(return_value=hit))
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._requires_full_dspark_prompt = False
    connector._dspark_prefix_provider_index = 1
    connector._connectors = [target_only, joint_store]
    connector._requests_to_connector = {}
    request = SimpleNamespace(request_id="incomplete-prefix")
    with patch.object(MultiConnector, "get_num_new_matched_tokens") as parent:
        assert connector.get_num_new_matched_tokens(request, 0) == hit
    target_only.get_num_new_matched_tokens.assert_not_called()
    parent.assert_not_called()
    assert connector._requests_to_connector == {}


def test_nested_dspark_prefix_provider_keeps_exact_metadata_route():
    config = _dspark_producer_config()
    store = SimpleNamespace(
        configure_dspark_prefix_cache=MagicMock(return_value=True),
        get_num_new_matched_tokens=MagicMock(return_value=(128, False)),
    )
    nested = AscendMultiConnector.__new__(AscendMultiConnector)
    nested._connectors = [SimpleNamespace(), store]
    nested._requests_to_connector = {}
    nested._requires_full_dspark_prompt = True
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [nested, SimpleNamespace()]
    connector._requests_to_connector = {}
    connector._requires_full_dspark_prompt = True

    assert connector.configure_dspark_prefix_cache(config)
    assert connector._dspark_prefix_provider_index == 0
    assert nested._dspark_prefix_provider_index == 1
    # Configuration is recursive and lookup stays with the matching child;
    # normal MultiConnector metadata nesting need not be flattened.
    connector._requires_full_dspark_prompt = nested._requires_full_dspark_prompt = False
    request = SimpleNamespace(request_id="nested-prefix")
    assert connector.get_num_new_matched_tokens(request, 0) == (128, False)
    assert connector._requests_to_connector == {request.request_id: 0}
    assert nested._requests_to_connector == {request.request_id: 1}


def test_dspark_prefix_provider_must_be_unique():
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [
        SimpleNamespace(configure_dspark_prefix_cache=MagicMock(return_value=True)) for _ in range(2)
    ]
    with pytest.raises(ValueError, match="exactly one"):
        connector.configure_dspark_prefix_cache(_dspark_producer_config())


def test_legacy_prefix_lookup_and_preempted_connector_priority_are_unchanged():
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._requires_full_dspark_prompt = False
    child = SimpleNamespace(has_preempted_request=MagicMock(return_value=False))
    connector._connectors = [child]
    connector._requests_to_connector = {}
    request = SimpleNamespace(request_id="legacy")
    with patch.object(MultiConnector, "get_num_new_matched_tokens", return_value=(17920, True)) as parent:
        assert connector.get_num_new_matched_tokens(request, 0) == (17920, True)
    parent.assert_called_once_with(request, 0)
    child.has_preempted_request.return_value = True
    child.get_num_new_matched_tokens = MagicMock(return_value=(128, False))
    with patch.object(MultiConnector, "get_num_new_matched_tokens") as parent:
        assert connector.get_num_new_matched_tokens(request, 0) == (128, False)
    parent.assert_not_called()
    assert connector._requests_to_connector == {"legacy": 0}


@pytest.mark.parametrize("num_connectors", [0, 2])
def test_kv_cache_events_without_events(num_connectors):
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [
        SimpleNamespace(get_kv_connector_kv_cache_events=MagicMock(return_value=None)) for _ in range(num_connectors)
    ]

    assert connector.get_kv_connector_kv_cache_events() is None
    for child in connector._connectors:
        child.get_kv_connector_kv_cache_events.assert_called_once_with()


@pytest.mark.parametrize("num_event_sources", [1, 3])
def test_kv_cache_events_combines_child_events_and_workers(num_event_sources):
    event_batches = [MagicMock() for _ in range(num_event_sources)]
    for index, batch in enumerate(event_batches):
        batch.get_all_events.return_value = [object()]
        batch.get_number_of_workers.return_value = index + 1
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [
        SimpleNamespace(get_kv_connector_kv_cache_events=MagicMock(return_value=batch))
        for batch in [None, *event_batches, None]
    ]

    combined = connector.get_kv_connector_kv_cache_events()

    assert combined is event_batches[0]
    assert combined.add_events.call_args_list == [
        call(batch.get_all_events.return_value) for batch in event_batches[1:]
    ]
    assert combined.increment_workers.call_args_list == [
        call(batch.get_number_of_workers.return_value) for batch in event_batches[1:]
    ]
    for child in connector._connectors:
        child.get_kv_connector_kv_cache_events.assert_called_once_with()


class _FakeBlocks:
    def __init__(self) -> None:
        self.empty = object()

    def new_empty(self):
        return self.empty


def _make_connector(*, requires_full_blocks: bool = False):
    return SimpleNamespace(
        requires_full_blocks_on_update_after_alloc=requires_full_blocks,
        update_state_after_alloc=MagicMock(),
    )


def test_update_state_after_alloc_forwards_full_blocks_to_observer():
    chosen = _make_connector()
    full_blocks_observer = _make_connector(requires_full_blocks=True)
    unrelated = _make_connector()
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [chosen, full_blocks_observer, unrelated]
    connector._requests_to_connector = {"req-0": 0}
    request = SimpleNamespace(request_id="req-0")
    blocks = _FakeBlocks()

    connector.update_state_after_alloc(request, blocks, num_external_tokens=16)

    chosen.update_state_after_alloc.assert_called_once_with(request, blocks, 16)
    full_blocks_observer.update_state_after_alloc.assert_called_once_with(
        request,
        blocks,
        16,
    )
    unrelated.update_state_after_alloc.assert_called_once_with(request, blocks.empty, 0)


def test_update_state_after_alloc_forwards_observer_without_chosen_connector():
    full_blocks_observer = _make_connector(requires_full_blocks=True)
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [full_blocks_observer]
    connector._requests_to_connector = {}
    connector._requires_full_dspark_prompt = True
    request = SimpleNamespace(request_id="req-0")
    blocks = _FakeBlocks()

    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)

    full_blocks_observer.update_state_after_alloc.assert_called_once_with(
        request,
        blocks,
        0,
    )


def test_layerwise_reuse_completion_is_wired_and_provider_hooks_run_first():
    call_order = []
    provider = SimpleNamespace(
        is_producer=True,
        connector_worker=object(),
        supports_layerwise_buffer_reuse=True,
        wait_for_layer_reuse=MagicMock(),
        wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("pd-load")),
        save_kv_layer=MagicMock(side_effect=lambda *_args, **_kwargs: call_order.append("pd-save")),
        on_kv_cache_written=MagicMock(side_effect=lambda *_: call_order.append("pd-written")),
    )
    store = SimpleNamespace(
        set_external_slot_release_waiter=MagicMock(return_value=True),
        wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("store-load")),
        save_kv_layer=MagicMock(side_effect=lambda *_args, **_kwargs: call_order.append("store-save")),
        on_kv_cache_written=MagicMock(side_effect=lambda *_: call_order.append("store-written")),
    )
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    # Put the store first to verify the dependency does not rely on config order.
    connector._connectors = [store, provider]

    connector._configure_layerwise_reuse_completion()

    waiter = store.set_external_slot_release_waiter.call_args.args[0]
    waiter(7)
    provider.wait_for_layer_reuse.assert_called_once_with(7)

    connector.wait_for_layer_load("model.layers.7.self_attn")
    connector.save_kv_layer("model.layers.7.self_attn", object(), object())
    connector.on_kv_cache_written("model.layers.7.self_attn")
    assert call_order == [
        "store-load",
        "pd-save",
        "store-save",
        "pd-written",
        "store-written",
    ]
    provider.wait_for_layer_load.assert_not_called()


def test_layerwise_reuse_without_sink_keeps_provider_layer_entry_wait():
    call_order = []
    provider = SimpleNamespace(
        is_producer=True,
        connector_worker=object(),
        supports_layerwise_buffer_reuse=True,
        wait_for_layer_reuse=MagicMock(),
        wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("provider")),
    )
    sibling = SimpleNamespace(
        wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("sibling")),
    )
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [sibling, provider]

    connector._configure_layerwise_reuse_completion()
    connector.wait_for_layer_load("model.layers.7.self_attn")

    assert call_order == ["provider", "sibling"]


def test_mamba_state_copy_runs_after_all_connector_loads():
    call_order = []
    first = SimpleNamespace(wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("first-load")))
    second = SimpleNamespace(wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("second-load")))
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [first, second]
    connector._layerwise_slot_release_providers = []
    connector._non_slot_release_connectors = [first, second]
    connector._external_slot_release_sink_configured = False
    connector._mamba_copy_bufs = object()

    with patch(
        "vllm_ascend.distributed.kv_transfer.ascend_multi_connector.mamba_utils.do_mamba_copy_block_for_layer",
        side_effect=lambda *_: call_order.append("copy"),
        create=True,
    ):
        connector.wait_for_layer_load("model.layers.7.linear_attn")

    assert call_order == ["first-load", "second-load", "copy"]


def test_v2_mamba_state_copy_runs_after_all_connector_loads():
    call_order = []
    first = SimpleNamespace(wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("first-load")))
    second = SimpleNamespace(wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("second-load")))
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [first, second]
    connector._layerwise_slot_release_providers = []
    connector._non_slot_release_connectors = [first, second]
    connector._external_slot_release_sink_configured = False
    connector._mamba_state = SimpleNamespace(
        do_mamba_copy_for_layer=MagicMock(side_effect=lambda layer: call_order.append("copy:" + layer))
    )

    connector.wait_for_layer_load("model.layers.7.linear_attn")

    assert call_order == ["first-load", "second-load", "copy:model.layers.7.linear_attn"]


def test_mamba_state_copy_skipped_without_deferral():
    first = SimpleNamespace(wait_for_layer_load=MagicMock())
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [first]
    connector._layerwise_slot_release_providers = []
    connector._non_slot_release_connectors = [first]
    connector._external_slot_release_sink_configured = False
    connector._mamba_state = None

    connector.wait_for_layer_load("model.layers.7.linear_attn")

    first.wait_for_layer_load.assert_called_once_with("model.layers.7.linear_attn")
