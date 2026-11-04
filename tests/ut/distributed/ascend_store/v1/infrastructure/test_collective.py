"""Independent communication groups and retryable cleanup."""

from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import collective
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.collective import GroupCompletion


def test_collective_group_owns_a_dedicated_cpu_channel(monkeypatch):
    group = object()
    create, destroy = Mock(return_value=group), Mock()
    monkeypatch.setattr(collective.dist, "new_group", create)
    monkeypatch.setattr(collective.dist, "destroy_process_group", destroy)
    gather = Mock(side_effect=lambda results, value, **_kw: results.__setitem__(slice(None), [value, "peer"]))
    monkeypatch.setattr(collective.dist, "all_gather_object", gather)
    channel = collective.CollectiveGroup((4, 5))
    assert create.call_args.kwargs["ranks"] == [4, 5]
    assert create.call_args.kwargs["backend"] == "gloo"
    assert create.call_args.kwargs["use_local_synchronization"] is True
    assert channel.all_gather("local") == {4: "local", 5: "peer"}
    assert gather.call_args.kwargs["group"] is group
    channel.close()
    channel.close()
    destroy.assert_called_once_with(group)
    with pytest.raises(RuntimeError, match="closed"):
        channel.all_gather("late")


def test_failed_channel_close_retains_group_for_retry(monkeypatch):
    group = object()
    monkeypatch.setattr(collective.dist, "new_group", Mock(return_value=group))
    destroy = Mock(side_effect=[RuntimeError("destroy failed"), None])
    monkeypatch.setattr(collective.dist, "destroy_process_group", destroy)
    channel = collective.CollectiveGroup((4, 5))
    with pytest.raises(RuntimeError, match="destroy failed"):
        channel.close()
    channel.close()
    channel.close()
    assert destroy.call_count == 2
    assert all(call.args == (group,) for call in destroy.call_args_list)


@pytest.mark.parametrize(
    "peer,expected",
    (
        ({10: ("first", 1), 20: ("second", 2)}, {10, 20}),
        ({10: ("first", 1), 30: ("third", 3)}, {10}),
        ({10: ("first", 2), 20: ("second", 2)}, {20}),
        ({10: ("first", 1), 20: ("other", 2)}, {10}),
        ({}, set()),
    ),
)
def test_every_required_participant_must_report_the_same_identity(peer, expected):
    local = {10: ("first", 1), 20: ("second", 2)}
    # An unrelated participant neither supplies a missing contribution nor vetoes a completed item.
    collective = Mock(all_gather=Mock(return_value={8: local, 13: peer, 21: {10: "different"}}))
    completion = GroupCompletion((8, 13), collective)
    assert completion.resolve(local) == expected
    collective.all_gather.assert_called_once_with(local)


def test_empty_local_report_still_enters_the_collective_once():
    collective = Mock(all_gather=Mock(return_value={8: {}, 13: {10: "instance"}}))
    assert GroupCompletion((8, 13), collective).resolve({}) == set()
    collective.all_gather.assert_called_once_with({})


def test_single_required_participant_still_preserves_collective_order():
    local = {10: "instance"}
    collective = Mock(all_gather=Mock(return_value={8: local, 13: {10: "different"}}))
    assert GroupCompletion((8,), collective).resolve(local) == {10}
    collective.all_gather.assert_called_once_with(local)


def test_identities_need_equality_without_hashing_or_field_inspection():
    item_id = (7, "stage")
    local = {item_id: {"instance": [1, 2]}}
    peer = {item_id: {"instance": [1, 2]}}
    collective = Mock(all_gather=Mock(return_value={8: local, 13: peer}))
    assert GroupCompletion((8, 13), collective).resolve(local) == {item_id}


@pytest.mark.parametrize("peer,expected", (({"item": None}, {"item"}), ({}, set())))
def test_none_identity_is_distinct_from_a_missing_item(peer, expected):
    local = {"item": None}
    collective = Mock(all_gather=Mock(return_value={8: local, 13: peer}))
    assert GroupCompletion((8, 13), collective).resolve(local) == expected


def test_each_round_uses_only_its_own_reports():
    local = {10: "instance"}
    collective = Mock(all_gather=Mock(side_effect=[{8: local, 13: dict(local)}, {8: local, 13: {}}]))
    completion = GroupCompletion((8, 13), collective)
    assert completion.resolve(local) == {10}
    assert completion.resolve(local) == set()
    assert local == {10: "instance"}


def test_missing_required_participant_is_an_error():
    collective = Mock(all_gather=Mock(return_value={8: {10: "instance"}}))
    with pytest.raises(KeyError):
        GroupCompletion((8, 13), collective).resolve({10: "instance"})


def test_collective_exceptions_propagate_to_the_caller():
    error = RuntimeError("collective failed")
    collective = Mock(all_gather=Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        GroupCompletion((8, 13), collective).resolve({})
    assert caught.value is error


def test_at_least_one_required_participant_is_needed():
    collective = Mock()
    with pytest.raises(ValueError, match="requires participants"):
        GroupCompletion((), collective)
    collective.all_gather.assert_not_called()
