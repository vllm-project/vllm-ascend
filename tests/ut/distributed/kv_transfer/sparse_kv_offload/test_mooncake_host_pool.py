import sys
import types
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    mooncake_host_pool as host_pool_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
    HostMemoryRegion,
    HostPoolTopology,
    MooncakeHostPool,
)

_shared_segment_stub: Any = types.ModuleType("mooncake.shared_segment")
_shared_segment_stub.create_shared_segment = MagicMock()
_shared_segment_stub.shared_segment_supported = MagicMock(return_value=True)
sys.modules.setdefault("mooncake.shared_segment", _shared_segment_stub)


class TestMooncakeHostPool(unittest.TestCase):
    def _allocate_region_for_mode(self, *, host_register: bool) -> HostMemoryRegion:
        raw = MagicMock()
        raw.reshape.return_value = raw
        raw.data_ptr.return_value = 0x1000
        aligned = MagicMock()
        aligned.device = SimpleNamespace(type="npu")
        raw.narrow.return_value = aligned
        segment = MagicMock()
        segment.tensors.return_value = [raw]

        with (
            patch.object(
                host_pool_module,
                "_select_shared_segment_mode",
                return_value=(host_register, host_register),
            ),
            patch(
                "mooncake.shared_segment.create_shared_segment",
                return_value=segment,
            ),
        ):
            return host_pool_module.allocate_mooncake_host_region(
                size_bytes=32,
                alignment=16,
                topology=HostPoolTopology(tp_rank=0, tp_size=1),
            )

    def test_allocates_aligned_views_from_one_region(self):
        raw = torch.empty(256, dtype=torch.int8)
        pool = MooncakeHostPool(
            HostMemoryRegion(raw),
            HostPoolTopology(tp_rank=0, tp_size=1),
        )

        first, second = pool.allocate_tensors([17, 23], alignment=32)

        self.assertEqual(first.numel(), 17)
        self.assertEqual(second.numel(), 23)
        self.assertEqual(first.data_ptr() % 32, 0)
        self.assertEqual(second.data_ptr() % 32, 0)
        self.assertGreaterEqual(second.data_ptr(), first.data_ptr() + first.numel())
        self.assertEqual(first.untyped_storage().data_ptr(), raw.untyped_storage().data_ptr())
        self.assertEqual(second.untyped_storage().data_ptr(), raw.untyped_storage().data_ptr())

    def test_exhausted_pool_is_rejected(self):
        pool = MooncakeHostPool(
            HostMemoryRegion(torch.empty(32, dtype=torch.int8)),
            HostPoolTopology(tp_rank=0, tp_size=1),
        )

        with self.assertRaisesRegex(MemoryError, "exhausted"):
            pool.allocate_tensors([64], alignment=16)

    def test_region_preserves_requested_capacity_after_alignment(self):
        requested_size = 32
        alignment = 16
        raw = MagicMock()
        raw.reshape.return_value = raw
        raw.data_ptr.return_value = 0x1001
        aligned = MagicMock()
        aligned.device = SimpleNamespace(type="npu")
        aligned.numel.return_value = requested_size
        raw.narrow.return_value = aligned
        segment = MagicMock()
        segment.tensors.return_value = [raw]

        with (
            patch.object(
                host_pool_module,
                "_select_shared_segment_mode",
                return_value=(False, False),
            ),
            patch(
                "mooncake.shared_segment.create_shared_segment",
                return_value=segment,
            ) as create_segment,
        ):
            region = host_pool_module.allocate_mooncake_host_region(
                size_bytes=requested_size,
                alignment=alignment,
                topology=HostPoolTopology(
                    tp_rank=0,
                    tp_size=1,
                    device_id=7,
                ),
            )

        block = create_segment.call_args.kwargs["blocks"]["pool"]
        self.assertEqual(block["shape"], (requested_size + alignment - 1,))
        raw.narrow.assert_called_once_with(0, alignment - 1, requested_size)
        self.assertIs(region.tensor, aligned)
        self.assertEqual(region.tensor.numel(), requested_size)

    def test_host_register_sets_migratepages_guard(self):
        key = "VLLM_ASCEND_SKIP_MIGRATEPAGES"
        with patch.dict(host_pool_module.os.environ, {}, clear=False):
            host_pool_module.os.environ.pop(key, None)

            self._allocate_region_for_mode(host_register=True)

            self.assertEqual(host_pool_module.os.environ[key], "1")

    def test_host_register_preserves_explicit_migratepages_setting(self):
        key = "VLLM_ASCEND_SKIP_MIGRATEPAGES"
        with patch.dict(
            host_pool_module.os.environ,
            {key: "0"},
            clear=False,
        ):
            self._allocate_region_for_mode(host_register=True)

            self.assertEqual(host_pool_module.os.environ[key], "0")

    def test_non_host_register_mode_does_not_set_migratepages_guard(self):
        key = "VLLM_ASCEND_SKIP_MIGRATEPAGES"
        with patch.dict(host_pool_module.os.environ, {}, clear=False):
            host_pool_module.os.environ.pop(key, None)

            self._allocate_region_for_mode(host_register=False)

            self.assertNotIn(key, host_pool_module.os.environ)

    def test_no_supported_mode_raises(self):
        with (
            patch(
                "mooncake.shared_segment.shared_segment_supported",
                return_value=False,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "cannot expose an NPU-addressable",
            ),
        ):
            host_pool_module._select_shared_segment_mode()

    def test_failed_construction_releases_region(self):
        release = MagicMock()
        region = HostMemoryRegion(
            torch.empty(8, dtype=torch.float32),
            handle="segment",
            release_callback=release,
        )

        with (
            patch.object(
                host_pool_module,
                "allocate_mooncake_host_region",
                return_value=region,
            ),
            self.assertRaisesRegex(TypeError, "int8 byte tensor"),
        ):
            MooncakeHostPool.allocate(
                size_bytes=8,
                alignment=8,
                topology=HostPoolTopology(tp_rank=0, tp_size=1),
            )

        release.assert_called_once_with("segment")


if __name__ == "__main__":
    unittest.main()
