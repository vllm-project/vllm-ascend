# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    SizedDict,
    as_kv_cache_tensors,
    collect_configured_register_regions,
    ensure_zmq_recv,
    ensure_zmq_send,
    group_concurrent_contiguous,
    split_if_not_byte_contiguous,
    string_to_int64_hash,
    zmq_ctx,
)


def test_as_kv_cache_tensors_normalizes_supported_inputs() -> None:
    first = torch.empty(1)
    second = torch.empty(1)

    assert as_kv_cache_tensors(first) == (first,)
    assert as_kv_cache_tensors([first, second]) == (first, second)
    with pytest.raises(TypeError, match="must be a tensor"):
        as_kv_cache_tensors([first, object()])


def test_sized_dict_evicts_oldest_entry_and_initializes_missing_value() -> None:
    values: SizedDict[str, dict[int, list[int]]] = SizedDict(max_size=2)
    values["a"] = {0: [1]}
    values["b"] = {0: [2]}
    values["c"] = {0: [3]}

    assert list(values) == ["b", "c"]
    assert values["missing"] == {}
    assert list(values) == ["c", "missing"]


def test_collect_register_regions_deduplicates_shared_storage() -> None:
    backing = torch.empty(128, dtype=torch.uint8)
    config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(shared_by=["layer.0", "layer.1"], offset=0, size=backing.nbytes)]
    )

    regions = collect_configured_register_regions(
        config,
        {"layer.0": backing[:64], "layer.1": backing[64:]},
    )

    assert regions.ptrs == [backing.data_ptr()]
    assert regions.lengths == [backing.nbytes]
    assert regions.logical_tensor_count == 1


def test_collect_register_regions_handles_independent_storages() -> None:
    first = torch.empty(32, dtype=torch.uint8)
    second = torch.empty(48, dtype=torch.uint8)
    config = SimpleNamespace(kv_cache_tensors=[SimpleNamespace(shared_by=["layer.0", "layer.1"], offset=0, size=80)])

    regions = collect_configured_register_regions(config, {"layer.0": first, "layer.1": second})

    assert set(zip(regions.ptrs, regions.lengths)) == {
        (first.data_ptr(), first.nbytes),
        (second.data_ptr(), second.nbytes),
    }


def test_group_concurrent_contiguous_honors_byte_strides() -> None:
    src_groups, dst_groups = group_concurrent_contiguous(
        [1, 2, 4],
        [10, 11, 13],
        src_block_stride=64,
        dst_block_stride=64,
        block_len=64,
    )

    assert src_groups == [[1, 2], [4]]
    assert dst_groups == [[10, 11], [13]]


def test_split_if_not_byte_contiguous_preserves_fast_path_identity() -> None:
    src = [[1, 2]]
    dst = [[3, 4]]

    result_src, result_dst = split_if_not_byte_contiguous(src, dst, 64, 64, 64)

    assert result_src is src
    assert result_dst is dst


def test_string_hash_is_stable_and_distinguishes_inputs() -> None:
    assert string_to_int64_hash("request") == string_to_int64_hash("request")
    assert string_to_int64_hash("request") != string_to_int64_hash("other")


def test_zmq_ctx_rejects_unknown_socket_type() -> None:
    with pytest.raises(ValueError, match="Unexpected socket type"), zmq_ctx(-1, "tcp://127.0.0.1:1"):
        pass


def test_zmq_ctx_destroys_context(monkeypatch: pytest.MonkeyPatch) -> None:
    context = MagicMock()
    socket = MagicMock()
    monkeypatch.setattr(zmq, "Context", MagicMock(return_value=context))
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils.make_zmq_socket",
        MagicMock(return_value=socket),
    )

    with zmq_ctx(zmq.REQ, "tcp://127.0.0.1:1") as result:  # type: ignore[attr-defined]
        assert result is socket

    context.destroy.assert_called_once_with(linger=0)


def test_zmq_retry_helpers_succeed_after_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils.time.sleep", MagicMock())
    socket = MagicMock()
    socket.send.side_effect = [zmq.ZMQError("transient"), None]  # type: ignore[attr-defined]
    socket.recv.side_effect = [zmq.ZMQError("transient"), b"response"]  # type: ignore[attr-defined]

    ensure_zmq_send(socket, b"request", "tcp://peer", max_retries=2)
    assert ensure_zmq_recv(socket, "tcp://peer", max_retries=2) == b"response"
    assert socket.send.call_count == 2
    assert socket.recv.call_count == 2


def test_zmq_retry_helpers_raise_after_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils.time.sleep", MagicMock())
    socket = MagicMock()
    socket.send.side_effect = zmq.ZMQError("failed")  # type: ignore[attr-defined]
    socket.recv.side_effect = zmq.ZMQError("failed")  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="Failed to send"):
        ensure_zmq_send(socket, b"request", "tcp://peer", max_retries=2)
    with pytest.raises(RuntimeError, match="Failed to receive"):
        ensure_zmq_recv(socket, "tcp://peer", max_retries=2)
