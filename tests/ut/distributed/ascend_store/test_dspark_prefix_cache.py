# SPDX-License-Identifier: Apache-2.0
"""Draft prefix pages keep independent storage and exact publication contracts."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_cache import (
    DSparkPrefixCache,
    DSparkPrefixKeys,
)


class _KeyInfo:
    def __init__(self, buffer, readable):
        self.buffer = buffer
        self.readable = readable

    def size(self):
        return self.buffer.nbytes

    def gva_list(self):
        return [self.buffer.ctypes.data if self.readable else 0]


class _ByteBackend:
    """External Store leaf: real CPU bytes, explicit readable state and leases."""

    def __init__(self):
        self.buffers = {}
        self.readable = set()
        self.leases = set()
        self.calls = []
        self.lease_error = False
        self.on_lease = None
        self.store: _ByteBackend | SimpleNamespace = self

    def ensure_initialized(self):
        return None

    def exists(self, keys):
        return [int(key in self.buffers) for key in keys]

    def batch_is_readable(self, keys):
        return [key in self.readable for key in keys]

    def batch_get_key_info(self, keys, for_load=False):
        self.calls.append(("info", tuple(keys), for_load))
        return [_KeyInfo(self.buffers[key], key in self.readable) if key in self.buffers else None for key in keys]

    def batch_alloc(self, keys, sizes, ttl):
        self.calls.append(("alloc", tuple(keys)))
        addresses = []
        for key, size in zip(keys, sizes, strict=True):
            if key in self.buffers:
                addresses.append(-1)
                continue
            self.buffers[key] = np.zeros(size, dtype=np.uint8)
            self.leases.add(key)
            addresses.append(self.buffers[key].ctypes.data)
        return addresses

    def batch_add_lease(self, keys, ttl):
        self.calls.append(("lease", tuple(keys)))
        if self.lease_error:
            return [-1] * len(keys)
        assert all(key in self.readable for key in keys)
        self.leases.update(keys)
        if self.on_lease is not None:
            self.on_lease(keys)
        return [0] * len(keys)

    def batch_write_finish(self, keys, statuses):
        self.calls.append(("finish", tuple(statuses)))
        for key, status in zip(keys, statuses, strict=True):
            assert key in self.leases
            if status == 0:
                self.readable.add(key)
            else:
                self.readable.discard(key)
        return [0] * len(keys)

    def batch_remove_lease(self, keys):
        self.calls.append(("release", tuple(keys)))
        assert set(keys) <= self.leases
        self.leases.difference_update(keys)
        return 0

    def copy(self, remote, local, sizes, direction):
        self.calls.append(("copy", direction))
        for gva, address, size in zip(remote, local, sizes, strict=True):
            assert any(
                key in self.leases
                and buffer.ctypes.data <= int(gva) < int(gva) + int(size) <= buffer.ctypes.data + buffer.nbytes
                for key, buffer in self.buffers.items()
            )
            source, destination = (address, gva) if direction == 0 else (gva, address)
            ctypes.memmove(int(destination), int(source), int(size))
        return 0


def _cache_case():
    backend = _ByteBackend()
    keys = DSparkPrefixKeys("checkpoint-identity", pp_size=2, tp_size=4)
    cache = DSparkPrefixCache(keys, backend, tp_rank=2, block_size=4, hash_block_size=4, copy_fn=backend.copy)
    tensors = {
        "draft.a": (
            torch.arange(48, dtype=torch.float32).reshape(6, 4, 2),
            torch.arange(48, dtype=torch.float32).reshape(6, 4, 2) + 1000,
        ),
        "draft.b": torch.arange(24, dtype=torch.uint8).reshape(6, 4, 1),
    }
    cache.register(tensors, {"draft.a": 0, "draft.b": 1}, [4, 4], num_blocks=6)
    request = SimpleNamespace(
        num_prompt_tokens=12,
        target_token_len=12,
        block_hashes=["h0", "h1", "h2"],
        load_spec=SimpleNamespace(can_load=True, kvpool_cached_tokens=8),
    )
    return cache, backend, tensors, request


def test_draft_prefix_roundtrip_remaps_independent_group_blocks_and_keeps_partial_tail_out():
    cache, backend, tensors, request = _cache_case()
    k, v = tensors["draft.a"]
    extra = tensors["draft.b"]
    expected = (k[[2, 3]].clone(), v[[2, 3]].clone(), extra[[4, 2]].clone())
    addresses, sizes = cache.registered_regions()
    assert addresses == [k.data_ptr(), v.data_ptr(), extra.data_ptr()]
    assert sizes == [k.numel() * k.element_size(), v.numel() * v.element_size(), extra.numel()]
    assert cache.page_bytes == 68  # Two FP32 pages and one independent UINT8 page.

    assert cache.save(request, 10, {0: (2, 3, 4), 1: (4, 2, 1)}) is None
    assert backend.readable == {cache.keys.make_key("h0", 2), cache.keys.make_key("h1", 2)}
    assert not backend.leases
    k.zero_()
    v.zero_()
    extra.zero_()

    assert cache.restore(request, 8, {0: (5, 0), 1: (1, 0)}) == 8
    torch.testing.assert_close(k[[5, 0]], expected[0])
    torch.testing.assert_close(v[[5, 0]], expected[1])
    torch.testing.assert_close(extra[[1, 0]], expected[2])
    assert not torch.count_nonzero(k[2:5])
    assert not torch.count_nonzero(extra[2:])
    assert not backend.leases
    assert [call[0] for call in backend.calls] == [
        "alloc",
        "copy",
        "finish",
        "release",
        "info",
        "lease",
        "info",
        "copy",
        "release",
    ]


def test_complete_draft_prefix_is_not_overwritten_by_a_second_request():
    cache, backend, tensors, request = _cache_case()
    blocks = {0: (0, 1), 1: (0, 1)}
    cache.save(request, 8, blocks)
    expected = {key: buffer.copy() for key, buffer in backend.buffers.items()}
    for value in tensors.values():
        for tensor in value if isinstance(value, tuple) else (value,):
            tensor.zero_()
    backend.calls.clear()
    assert cache.save(request, 8, blocks) is None
    assert backend.calls == []
    for key, value in expected.items():
        np.testing.assert_array_equal(backend.buffers[key], value)


def test_readable_prefix_evicted_after_probe_does_not_fail_save_or_reallocate():
    cache, backend, _, request = _cache_case()
    blocks = {0: (0, 1), 1: (0, 1)}
    cache.save(request, 8, blocks)
    probe = backend.batch_is_readable

    def probe_then_evict(keys):
        readable = probe(keys)
        assert all(readable)
        for key in keys:
            del backend.buffers[key]
            backend.readable.remove(key)
        return readable

    backend.batch_is_readable = probe_then_evict
    backend.calls.clear()
    assert cache.save(request, 8, blocks) is None
    assert backend.calls == []  # No metadata query, new allocation or copy.
    assert not backend.buffers
    assert not backend.readable
    assert not backend.leases


def test_failed_draft_copy_is_not_published_and_releases_only_owned_write_lease():
    cache, backend, _, request = _cache_case()
    cache.copy_fn = Mock(return_value=-1)
    with pytest.raises(RuntimeError, match="copy failed"):
        cache.save(request, 4, {0: (0,), 1: (0,)})
    assert not backend.readable
    assert not backend.leases
    assert ("finish", (-1,)) in backend.calls
    assert ("finish", (0,)) not in backend.calls


def test_incomplete_other_writer_object_is_not_overwritten_or_published():
    cache, backend, _, request = _cache_case()
    key = cache.keys.make_key("h0", 2)
    original = np.full(cache.page_bytes, 99, dtype=np.uint8)
    backend.buffers[key] = original
    assert cache.save(request, 4, {0: (0,), 1: (0,)}) is None
    assert backend.buffers[key] is original
    assert key not in backend.readable
    assert not backend.leases
    assert not any(call[0] in ("copy", "finish", "release") for call in backend.calls)


@pytest.mark.parametrize("failure", ["missing", "size", "lease"])
def test_failed_prefix_restore_does_not_copy_unvalidated_pages(failure):
    cache, backend, _, request = _cache_case()
    cache.save(request, 4, {0: (0,), 1: (0,)})
    key = cache.keys.make_key("h0", 2)
    if failure == "missing":
        del backend.buffers[key]
        backend.readable.remove(key)
    elif failure == "size":
        backend.buffers[key] = np.zeros(cache.page_bytes + 1, dtype=np.uint8)
    else:
        backend.lease_error = True
    backend.calls.clear()
    cache.copy_fn = Mock(return_value=0)
    with pytest.raises(RuntimeError, match="missing|layout|lease"):
        cache.restore(request, 4, {0: (1,), 1: (1,)})
    cache.copy_fn.assert_not_called()
    assert not backend.leases


@pytest.mark.parametrize("statuses", [None, [0], [0, -1]])
def test_partial_or_unknown_read_lease_never_releases_another_owners_keys(statuses):
    cache, backend, _, request = _cache_case()
    blocks = {0: (0, 1), 1: (0, 1)}
    cache.save(request, 8, blocks)
    keys = [cache.keys.make_key(f"h{index}", 2) for index in range(2)]
    other_owner_key = keys[1]
    backend.leases.add(other_owner_key)
    backend.calls.clear()

    def add_partial_lease(requested, ttl):
        assert requested == keys
        if statuses is not None:
            backend.leases.update(key for key, status in zip(requested, statuses) if status == 0)
        return statuses

    backend.batch_add_lease = add_partial_lease
    cache.copy_fn = Mock(return_value=0)
    with pytest.raises(RuntimeError, match="lease"):
        cache.restore(request, 8, blocks)
    cache.copy_fn.assert_not_called()
    assert backend.leases == {other_owner_key}
    releases = [call[1] for call in backend.calls if call[0] == "release"]
    assert releases == ([(keys[0],)] if statuses is not None else [])


def test_prefix_register_rejects_sdk_without_atomic_publication():
    backend = _ByteBackend()
    backend.store = SimpleNamespace()
    cache = DSparkPrefixCache(
        DSparkPrefixKeys("checkpoint", 1, 1), backend, 0, block_size=4, hash_block_size=4, copy_fn=backend.copy
    )
    with pytest.raises((ValueError, RuntimeError), match="publication|batch_write_finish"):
        cache.register({"draft": torch.zeros(2, 4, 1)}, {"draft": 0}, [4], num_blocks=2)


def test_writing_prefix_object_is_copied_from_allocation_address_before_becoming_readable():
    cache, backend, tensors, request = _cache_case()
    backend.calls.clear()
    cache.save(request, 4, {0: (1,), 1: (2,)})
    key = cache.keys.make_key("h0", 2)
    assert key in backend.readable
    assert [call[0] for call in backend.calls] == ["alloc", "copy", "finish", "release"]
    expected_k = tensors["draft.a"][0][1].clone()
    tensors["draft.a"][0].zero_()
    assert cache.restore(request, 4, {0: (0,), 1: (0,)}) == 4
    torch.testing.assert_close(tensors["draft.a"][0][0], expected_k)


def test_prefix_restore_refreshes_addresses_under_lease_before_copy():
    cache, backend, tensors, request = _cache_case()
    cache.save(request, 4, {0: (2,), 1: (3,)})
    expected_k = tensors["draft.a"][0][2].clone()
    previous_addresses = {key: buffer.ctypes.data for key, buffer in backend.buffers.items()}
    previous_buffers = list(backend.buffers.values())  # Keep old GVAs alive to catch stale-address use.

    def relocate(keys):
        for key in keys:
            backend.buffers[key] = backend.buffers[key].copy()
            assert backend.buffers[key].ctypes.data != previous_addresses[key]
        for buffer in previous_buffers:
            buffer.fill(0)

    backend.on_lease = relocate
    tensors["draft.a"][0].zero_()
    assert cache.restore(request, 4, {0: (0,), 1: (0,)}) == 4
    torch.testing.assert_close(tensors["draft.a"][0][0], expected_k)
    assert not backend.leases


@pytest.mark.parametrize("can_load,extent", [(False, 8), (True, 0), (True, 3)])
def test_prefix_restore_requires_matching_scheduler_hit(can_load, extent):
    cache, backend, _, request = _cache_case()
    request.load_spec = SimpleNamespace(can_load=can_load, kvpool_cached_tokens=extent)
    with pytest.raises(RuntimeError, match="validated pool hit"):
        cache.restore(request, 4, {0: (0,), 1: (0,)})
    assert backend.calls == []


@pytest.mark.parametrize("offset", [3, 16])
def test_prefix_restore_requires_complete_hashed_prompt_blocks(offset):
    cache, backend, _, request = _cache_case()
    with pytest.raises(ValueError, match="complete hashed|extent"):
        cache.restore(request, offset, {0: (0,), 1: (0,)})
    assert backend.calls == []


def test_draft_hit_keys_require_all_tp_replicas_on_only_final_pp_stage():
    keys = DSparkPrefixKeys("same-target-draft", pp_size=2, tp_size=4)
    assert keys.make_hit_check_keys("hash") == [
        f"dspark-prefix-v1@same-target-draft@pp1@tp{rank}@hash" for rank in range(4)
    ]


def _model_config(path):
    return SimpleNamespace(model=path, dtype=torch.bfloat16, hf_config=SimpleNamespace(hidden_size=4))


@pytest.mark.parametrize(
    "change", ["target", "draft", "revision", "dtype", "width", "aux_layers", "tp", "block_size", "hash_block_size"]
)
def test_draft_prefix_identity_changes_with_checkpoint_and_projection_contract(change):
    config = SimpleNamespace(
        model_config=_model_config("target-a"),
        speculative_config=SimpleNamespace(
            method="dspark", draft_model_config=_model_config("draft-a"), num_speculative_tokens=8
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2, tensor_parallel_size=4),
        cache_config=SimpleNamespace(block_size=128, prefix_match_unit=128),
    )
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_cache"
    with patch(f"{module}.get_dspark_aux_layer_ids", return_value=(2, 4)) as aux_layers:
        original = DSparkPrefixKeys.from_config(config)
        if change == "target":
            config.model_config.model = "target-b"
        elif change == "draft":
            config.speculative_config.draft_model_config.model = "draft-b"
        elif change == "revision":
            config.speculative_config.draft_model_config.revision = "new-revision"
        elif change == "dtype":
            config.speculative_config.draft_model_config.dtype = torch.float32
        elif change == "width":
            config.speculative_config.num_speculative_tokens = 3
        elif change == "aux_layers":
            aux_layers.return_value = (2, 5)
        elif change == "tp":
            config.parallel_config.tensor_parallel_size = 8
        elif change == "block_size":
            config.cache_config.block_size = 256
        else:
            config.cache_config.prefix_match_unit = 64
        changed = DSparkPrefixKeys.from_config(config)
    assert original is not None and changed is not None
    assert original.identity != changed.identity


@pytest.mark.parametrize("speculative", [None, SimpleNamespace(method="mtp"), SimpleNamespace(method="eagle3")])
def test_other_speculative_methods_do_not_enable_draft_prefix_objects(speculative):
    assert DSparkPrefixKeys.from_config(SimpleNamespace(speculative_config=speculative)) is None
