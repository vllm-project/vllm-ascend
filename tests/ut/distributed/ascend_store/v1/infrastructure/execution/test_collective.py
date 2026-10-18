"""Store communication is isolated from foreground model collective ordering."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.execution import collective


def test_store_collective_owns_a_dedicated_cpu_channel(monkeypatch):
    group = object()
    create, destroy = Mock(return_value=group), Mock()
    monkeypatch.setattr(collective, "get_tp_group", lambda: SimpleNamespace(ranks=[4, 5]))
    monkeypatch.setattr(collective.dist, "new_group", create)
    monkeypatch.setattr(collective.dist, "destroy_process_group", destroy)
    gather = Mock(side_effect=lambda results, value, **_kw: results.__setitem__(slice(None), [value, "peer"]))
    monkeypatch.setattr(collective.dist, "all_gather_object", gather)
    channel = collective.StoreCollective()
    assert create.call_args.kwargs["ranks"] == [4, 5]
    assert create.call_args.kwargs["backend"] == "gloo"
    assert create.call_args.kwargs["use_local_synchronization"] is True
    assert channel.all_gather("local") == ("local", "peer")
    assert gather.call_args.kwargs["group"] is group
    channel.close()
    channel.close()
    destroy.assert_called_once_with(group)
    with pytest.raises(RuntimeError, match="closed"):
        channel.all_gather("late")
