"""CPU byte-copy SDK stand-in; production Backend adapters and sessions stay real."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

from tests.ut.distributed.ascend_store.v1.backend.test_memcache import _make_memcache_registration_backend
from tests.ut.distributed.ascend_store.v1.backend.test_mooncake import _make_mooncake_registration_backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.backend import MmcDirect


class ByteCopySDK:
    """Model native publication and copies over registered CPU memory, not NPU transport."""

    def __init__(self, objects):
        self.objects = objects
        self.readable = set(objects)
        self.regions = {}
        self.calls = []
        self.closed = False

    def register_buffer(self, address, size):
        self.regions[address] = size
        self.calls.append(("register", address, size))
        return 0

    def unregister_buffer(self, address, *_size):
        del self.regions[address]
        self.calls.append(("unregister", address))
        return 0

    def close(self):
        assert not self.regions
        self.closed = True
        return 0

    def batch_is_exist(self, keys):
        return [int(key in self.readable) for key in keys]

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, _replicate_config):
        self.batch_put_session_start(keys, [sum(row) for row in sizes], None)
        self.batch_put_from_multi_buffer_ranges(keys, addresses, sizes, self._offsets(sizes))
        return self.batch_put_session_end(keys)

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        return self.batch_get_into_multi_buffer_ranges(keys, addresses, sizes, self._offsets(sizes))

    def batch_put_from_layers(self, keys, addresses, sizes, direction):
        assert direction == MmcDirect.COPY_L2G.value
        return self.batch_put_from_multi_buffers(keys, addresses, sizes, None)

    def batch_get_into_layers(self, keys, addresses, sizes, direction):
        assert direction == MmcDirect.COPY_G2L.value
        self.batch_get_into_multi_buffers(keys, addresses, sizes)
        return [0] * len(keys)

    def batch_put_session_start(self, keys, sizes, _replicate_config):
        for key, size in zip(keys, sizes, strict=True):
            assert key not in self.objects
            self.objects[key] = ctypes.create_string_buffer(size)
        self.calls.append(("allocate", tuple(keys), tuple(sizes)))
        return [0] * len(keys)

    def batch_get_session_start(self, keys):
        return [0 if key in self.readable else -1 for key in keys]

    def batch_get_session_end(self, keys):
        self.calls.append(("release", tuple(keys)))
        return 0

    def batch_put_session_end(self, keys):
        self.readable.update(keys)
        self.calls.append(("publish", tuple(keys)))
        return [0] * len(keys)

    def batch_put_session_revoke(self, keys):
        for key in keys:
            del self.objects[key]
        return [0] * len(keys)

    def batch_put_from_multi_buffer_ranges(self, keys, addresses, sizes, offsets):
        return self._copy_ranges(keys, addresses, sizes, offsets, to_remote=True)

    def batch_get_into_multi_buffer_ranges(self, keys, addresses, sizes, offsets):
        assert all(key in self.readable for key in keys)
        return self._copy_ranges(keys, addresses, sizes, offsets, to_remote=False)

    def batch_get_key_info(self, keys, _query):
        return [
            SimpleNamespace(
                size=lambda key=key: ctypes.sizeof(self.objects[key]) if key in self.readable else 0,
                gva_list=lambda key=key: [ctypes.addressof(self.objects[key])] if key in self.readable else [],
            )
            for key in keys
        ]

    def batch_add_lease(self, keys):
        return self.batch_get_session_start(keys)

    def batch_remove_lease(self, keys):
        return self.batch_get_session_end(keys)

    def batch_alloc(self, keys, sizes):
        self.batch_put_session_start(keys, sizes, None)
        return [ctypes.addressof(self.objects[key]) for key in keys]

    def batch_copy(self, remote, local, sizes, direction):
        to_remote = direction == MmcDirect.COPY_L2G.value
        self.calls.append(("gva_copy", tuple(remote), tuple(sizes), to_remote))
        for remote_address, local_address, size in zip(remote, local, sizes, strict=True):
            assert any(
                ctypes.addressof(value) <= remote_address
                and remote_address + size <= ctypes.addressof(value) + ctypes.sizeof(value)
                for value in self.objects.values()
            )
            self._copy(remote_address, local_address, size, to_remote)
        return 0

    def batch_write_finish(self, keys, results):
        assert results == [0] * len(keys)
        return self.batch_put_session_end(keys)

    def _copy_ranges(self, keys, addresses, sizes, offsets, *, to_remote):
        self.calls.append(("ranges", tuple(keys), offsets, sizes, to_remote))
        for key, local_row, size_row, offset_row in zip(keys, addresses, sizes, offsets, strict=True):
            value = self.objects[key]
            for address, size, offset in zip(local_row, size_row, offset_row, strict=True):
                assert 0 <= offset < offset + size <= ctypes.sizeof(value)
                self._copy(ctypes.addressof(value) + offset, address, size, to_remote)
        return [sum(row) for row in sizes]

    def _copy(self, remote, local, size, to_remote):
        assert any(base <= local and local + size <= base + length for base, length in self.regions.items())
        destination, source = (remote, local) if to_remote else (local, remote)
        ctypes.memmove(destination, source, size)

    @staticmethod
    def _offsets(sizes):
        rows = []
        for row in sizes:
            offset = 0
            offsets = []
            for size in row:
                offsets.append(offset)
                offset += size
            rows.append(offsets)
        return rows


def native_backend(name, sdk):
    if name == "mooncake":
        backend = _make_mooncake_registration_backend(sdk)
        backend.config = SimpleNamespace(preferred_segment=False, prefer_alloc_in_same_node=False)
    else:
        backend = _make_memcache_registration_backend(sdk, initialized=True)
        backend._lazy_init = False
    backend.device_id = 0
    return backend
