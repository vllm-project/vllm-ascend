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
            patch.object(
                _shared_segment_stub,
                "create_shared_segment",
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

        self.assertFalse(create_segment.call_args.kwargs["mmap"])
        self.assertFalse(create_segment.call_args.kwargs["host_register"])
        self.assertEqual(create_segment.call_args.kwargs["device_id"], 7)
        self.assertIn("comm_group", create_segment.call_args.kwargs)
        self.assertNotIn("tp_group", create_segment.call_args.kwargs)
        block = create_segment.call_args.kwargs["blocks"]["pool"]
        self.assertEqual(block["shape"], (1 << 30,))
        raw.narrow.assert_called_once_with(0, alignment - 1, requested_size)
        self.assertIs(region.tensor, aligned)
        self.assertEqual(region.tensor.numel(), requested_size)
        self.assertEqual(region.segment_offset, alignment - 1)

    def test_vmm_reservation_size_covers_padding_and_heap_alignment(self):
        # Include the reported failure and both sides of the 1 GiB boundary
        # after adding the 15 bytes needed for a 16-byte-aligned tensor view.
        cases = (
            (0x4FFA00000, 0x500000000),
            ((1 << 30) - 15, 1 << 30),
            ((1 << 30) - 14, 2 << 30),
            (1 << 30, 2 << 30),
        )
        for requested_size, expected_allocation in cases:
            with self.subTest(size=requested_size):
                raw = MagicMock()
                raw.reshape.return_value = raw
                raw.data_ptr.return_value = 0x1001
                aligned = MagicMock()
                aligned.device = SimpleNamespace(type="npu")
                raw.narrow.return_value = aligned
                segment = MagicMock()
                segment.tensors.return_value = [raw]
                with patch.object(
                    _shared_segment_stub,
                    "create_shared_segment",
                    return_value=segment,
                ) as create_segment:
                    region = host_pool_module.allocate_mooncake_host_region(
                        size_bytes=requested_size,
                        alignment=16,
                        topology=HostPoolTopology(tp_rank=0, tp_size=1),
                    )

                block = create_segment.call_args.kwargs["blocks"]["pool"]
                self.assertEqual(block["shape"], (expected_allocation,))
                raw.narrow.assert_called_once_with(0, 15, requested_size)
                self.assertIs(region.tensor, aligned)
                self.assertEqual(region.segment_offset, 15)

    def test_selects_vmm_without_host_registration(self):
        with patch.object(
            _shared_segment_stub,
            "shared_segment_supported",
            return_value=True,
        ) as supported:
            self.assertEqual(host_pool_module._select_shared_segment_mode(), (False, False))

        supported.assert_called_once_with(mmap=False, host_register=False)

    def test_no_supported_mode_raises(self):
        with (
            patch.object(
                _shared_segment_stub,
                "shared_segment_supported",
                return_value=False,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "requires Ascend VMM support",
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


def test_layout_uses_segment_offsets_and_rejects_overlaps():
    pools, descriptions = [], []
    for rank in range(2):
        pool = MooncakeHostPool(
            HostMemoryRegion(torch.empty(256, dtype=torch.int8), segment_offset=8),
            HostPoolTopology(tp_rank=rank, tp_size=2),
        )
        k, v = [t.view(torch.bfloat16).reshape(4, 2, 1, 2) for t in pool.allocate_tensors([32, 32], 16)]
        views = [("layer", "k", k), ("layer", "v", v)]
        descriptions.append(pool.describe_local_views(views, 4))
        pools.append(pool)
        assert k.reshape(4, 2, 2).contiguous().data_ptr() == k.data_ptr()
    assert pools[0].data_ptr != pools[1].data_ptr
    assert descriptions[0] == descriptions[1]
    pools[1].region.segment_offset = 0
    assert pools[1].describe_local_views(views, 4) != descriptions[0]
    with unittest.TestCase().assertRaisesRegex(ValueError, "overlapping"):
        pools[1].describe_local_views([("layer", "k", k), ("layer", "v", k)], 4)
    with unittest.TestCase().assertRaisesRegex(ValueError, "outside"):
        pools[0].describe_local_views(views, 4)
