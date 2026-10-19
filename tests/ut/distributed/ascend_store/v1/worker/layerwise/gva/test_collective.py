"""Store communication is isolated from foreground model collective ordering."""

from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.gva import collective


def test_store_collective_owns_a_dedicated_cpu_channel(monkeypatch):
    group = object()
    create, destroy = Mock(return_value=group), Mock()
    monkeypatch.setattr(collective.dist, "new_group", create)
    monkeypatch.setattr(collective.dist, "destroy_process_group", destroy)
    gather = Mock(side_effect=lambda results, value, **_kw: results.__setitem__(slice(None), [value, "peer"]))
    monkeypatch.setattr(collective.dist, "all_gather_object", gather)
    channel = collective.StoreCollective((4, 5))
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
    channel = collective.StoreCollective((4, 5))
    with pytest.raises(RuntimeError, match="destroy failed"):
        channel.close()
    channel.close()
    channel.close()
    assert destroy.call_count == 2
    assert all(call.args == (group,) for call in destroy.call_args_list)
