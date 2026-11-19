# SPDX-License-Identifier: Apache-2.0
"""CPU address-book regressions; no transfer engine or device is required."""

import gc
import random
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.transfer_targets import MooncakeKVRegistry


class _TensorStub:
    """Tensor metadata with controllable, potentially overlapping addresses."""

    def __init__(self, ptr: int, size: int) -> None:
        self.ptr = ptr
        self.size = size

    def data_ptr(self) -> int:
        return self.ptr

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 1


class TestDSAPDMooncakeRegistryRanges(TestCase):
    def setUp(self):
        self.registry = MooncakeKVRegistry()

    def test_empty_registries(self):
        self.assertIsNone(self.registry.get_register_location(1000, 1))
        self.assertIsNone(self.registry.resolve_offloaded_tensor(1000, 1))

    def test_cpu_buffer_boundaries_and_partial_overlap(self):
        self.registry.register_cpu_buffer(1000, 100)
        for ptr, size, expected in [
            (1000, 100, "cpu"),
            (1020, 20, "cpu"),
            (1099, 1, "cpu"),
            (1100, 0, "cpu"),
            (999, 1, None),
            (999, 2, None),
            (1050, 51, None),
            (1100, 1, None),
        ]:
            with self.subTest(ptr=ptr, size=size):
                self.assertEqual(self.registry.get_register_location(ptr, size), expected)

    def test_cpu_buffers_can_overlap_and_register_out_of_address_order(self):
        self.registry.register_cpu_buffer(200, 100)
        self.registry.register_cpu_buffer(100, 1000)
        self.registry.register_cpu_buffer(800, 10)
        self.assertEqual(self.registry.get_register_location(250, 600), "cpu")
        self.assertIsNone(self.registry.get_register_location(250, 851))

    def test_adjacent_cpu_buffers_do_not_form_one_registration(self):
        self.registry.register_cpu_buffer(1000, 100)
        self.registry.register_cpu_buffer(1100, 100)
        self.assertIsNone(self.registry.get_register_location(1050, 100))
        self.assertEqual(self.registry.get_register_location(1100, 100), "cpu")

    def test_registration_after_a_lookup_is_visible(self):
        self.assertIsNone(self.registry.get_register_location(4000, 16))
        self.assertIsNone(self.registry.resolve_offloaded_tensor(4000, 16))
        tensor = _TensorStub(4000, 32)
        self.registry.register_cpu_buffer(4000, 32)
        self.registry.register_offloaded_tensor(tensor)
        self.assertEqual(self.registry.get_register_location(4000, 16), "cpu")
        self.assertIs(self.registry.resolve_offloaded_tensor(4000, 16)[0], tensor)

    def test_invalid_and_duplicate_registrations(self):
        for ptr, size in [(0, 16), (-1, 16), (100, 0), (100, -1)]:
            self.registry.register_cpu_buffer(ptr, size)
            self.registry.register_offloaded_tensor(_TensorStub(ptr, size))
        self.assertIsNone(self.registry.get_register_location(100, 1))
        self.assertIsNone(self.registry.resolve_offloaded_tensor(100, 1))
        first = _TensorStub(1000, 100)
        duplicate = _TensorStub(1000, 100)
        for _ in range(2):
            self.registry.register_cpu_buffer(1000, 100)
            self.registry.register_offloaded_tensor(first)
        self.registry.register_offloaded_tensor(duplicate)
        self.assertIs(self.registry.resolve_offloaded_tensor(1020, 10)[0], first)

    def test_offloaded_tensor_boundaries_and_offsets(self):
        tensor = _TensorStub(1000, 100)
        self.registry.register_offloaded_tensor(tensor)
        for ptr, size, expected_offset in [
            (1000, 100, 0),
            (1020, 20, 20),
            (1099, 1, 99),
            (1100, 0, 100),
        ]:
            with self.subTest(ptr=ptr, size=size):
                resolved = self.registry.resolve_offloaded_tensor(ptr, size)
                self.assertIs(resolved[0], tensor)
                self.assertEqual(resolved[1], expected_offset)
        for ptr, size in [(999, 1), (999, 2), (1050, 51), (1100, 1)]:
            with self.subTest(ptr=ptr, size=size):
                self.assertIsNone(self.registry.resolve_offloaded_tensor(ptr, size))

    def test_first_registered_live_tensor_wins_over_address_order(self):
        tensors = [_TensorStub(200, 600), _TensorStub(100, 1000), _TensorStub(300, 400)]
        for tensor in tensors:
            self.registry.register_offloaded_tensor(tensor)
        resolved = self.registry.resolve_offloaded_tensor(350, 50)
        self.assertIs(resolved[0], tensors[0])
        self.assertEqual(resolved[1], 150)
        self.assertIs(self.registry.resolve_offloaded_tensor(150, 700)[0], tensors[1])

    def test_zero_length_at_shared_boundary_preserves_registration_order(self):
        right = _TensorStub(1100, 100)
        left = _TensorStub(1000, 100)
        self.registry.register_offloaded_tensor(right)
        self.registry.register_offloaded_tensor(left)
        self.assertIs(self.registry.resolve_offloaded_tensor(1100, 0)[0], right)

    def test_dead_candidate_falls_back_to_another_live_tensor(self):
        first = _TensorStub(200, 600)
        second = _TensorStub(100, 1000)
        self.registry.register_offloaded_tensor(first)
        self.registry.register_offloaded_tensor(second)
        self.assertIs(self.registry.resolve_offloaded_tensor(350, 50)[0], first)
        first_ref = weakref.ref(first)
        del first
        gc.collect()
        self.assertIsNone(first_ref())
        resolved = self.registry.resolve_offloaded_tensor(350, 50)
        self.assertIs(resolved[0], second)
        self.assertEqual(resolved[1], 250)

    def test_lookup_metadata_does_not_keep_tensors_alive(self):
        tensor = torch.empty(128, dtype=torch.int8, device="cpu")
        ptr = tensor.data_ptr()
        self.registry.register_offloaded_tensor(tensor)
        self.assertIs(self.registry.resolve_offloaded_tensor(ptr, 128)[0], tensor)
        tensor_ref = weakref.ref(tensor)
        del tensor
        gc.collect()
        self.assertIsNone(tensor_ref())
        self.assertIsNone(self.registry.resolve_offloaded_tensor(ptr, 128))

    def test_dead_before_first_lookup_and_reused_address(self):
        tensor = _TensorStub(4000, 128)
        self.registry.register_offloaded_tensor(tensor)
        del tensor
        gc.collect()
        self.assertIsNone(self.registry.resolve_offloaded_tensor(4000, 128))
        replacement = _TensorStub(4000, 128)
        self.registry.register_offloaded_tensor(replacement)
        self.assertIs(self.registry.resolve_offloaded_tensor(4000, 128)[0], replacement)

    def test_reset_clears_both_lookups_and_accepts_new_registrations(self):
        old = _TensorStub(1000, 128)
        self.registry.register_cpu_buffer(1000, 128)
        self.registry.register_offloaded_tensor(old)
        self.assertEqual(self.registry.get_register_location(1000, 128), "cpu")
        self.assertIsNotNone(self.registry.resolve_offloaded_tensor(1000, 128))
        self.registry.clear()
        self.assertIsNone(self.registry.get_register_location(1000, 128))
        self.assertIsNone(self.registry.resolve_offloaded_tensor(1000, 128))
        replacement = _TensorStub(1000, 128)
        self.registry.register_cpu_buffer(1000, 128)
        self.registry.register_offloaded_tensor(replacement)
        self.assertEqual(self.registry.get_register_location(1000, 128), "cpu")
        self.assertIs(self.registry.resolve_offloaded_tensor(1000, 128)[0], replacement)

    def test_byte_view_cache_remains_compatible_with_tensor_lifetime_and_reset(self):
        tensor = torch.arange(32, dtype=torch.int32, device="cpu")
        ptr = tensor.data_ptr()
        self.registry.register_offloaded_tensor(tensor)
        byte_view, offset = self.registry.resolve_offloaded_byte_tensor(ptr + 4, 8)
        self.assertEqual(offset, 4)
        self.assertTrue(torch.equal(byte_view[offset : offset + 8], tensor.view(torch.int8)[4:12]))
        tensor_ref = weakref.ref(tensor)
        del tensor, byte_view
        gc.collect()
        self.assertIsNone(tensor_ref())
        self.assertIsNone(self.registry.resolve_offloaded_byte_tensor(ptr, 4))
        self.registry.clear()
        self.assertIsNone(self.registry.resolve_offloaded_byte_tensor(ptr, 4))

    def test_large_integer_addresses_remain_exact(self):
        ptr = (1 << 64) + 123
        tensor = _TensorStub(ptr, 100)
        self.registry.register_cpu_buffer(ptr, 100)
        self.registry.register_offloaded_tensor(tensor)
        self.assertEqual(self.registry.get_register_location(ptr + 2, 98), "cpu")
        self.assertIs(self.registry.resolve_offloaded_tensor(ptr + 2, 98)[0], tensor)
        self.assertIsNone(self.registry.get_register_location(ptr + 2, 99))

    def test_randomized_queries_match_linear_range_semantics(self):
        rng = random.Random(19212)
        tensors = [_TensorStub(ptr, rng.randrange(1, 400)) for ptr in rng.sample(range(1, 3000), 80)]
        ranges = [(tensor.ptr, tensor.size) for tensor in tensors]
        tensor_refs = [weakref.ref(tensor) for tensor in tensors]
        for tensor in tensors:
            self.registry.register_cpu_buffer(tensor.ptr, tensor.size)
            self.registry.register_offloaded_tensor(tensor)
        del tensor
        # Exercise deaths after lookup metadata has already been initialized.
        self.registry.resolve_offloaded_tensor(100, 1)
        retained_tensors: list[_TensorStub | None] = list(tensors)
        del tensors
        for position in range(0, len(retained_tensors), 3):
            retained_tensors[position] = None
        gc.collect()
        for _ in range(2000):
            ptr = rng.randrange(-10, 3500)
            size = rng.randrange(-2, 500)
            expected_cpu = any(start <= ptr and ptr + size <= start + length for start, length in ranges)
            self.assertEqual(self.registry.get_register_location(ptr, size), "cpu" if expected_cpu else None)
            expected = None
            for (start, length), tensor_ref in zip(ranges, tensor_refs):
                resolved_tensor = tensor_ref()
                if resolved_tensor is not None and start <= ptr and ptr + size <= start + length:
                    expected = (resolved_tensor, ptr - start)
                    break
            actual = self.registry.resolve_offloaded_tensor(ptr, size)
            if expected is None:
                self.assertIsNone(actual)
            else:
                self.assertIs(actual[0], expected[0])
                self.assertEqual(actual[1], expected[1])

    def test_concurrent_reads_and_registration_reset(self):
        tensor = _TensorStub(1000, 128)
        barrier = threading.Barrier(3)

        def update_registry():
            barrier.wait(timeout=5)
            for _ in range(200):
                self.registry.clear()
                self.registry.register_cpu_buffer(1000, 128)
                self.registry.register_offloaded_tensor(tensor)

        def read_registry():
            barrier.wait(timeout=5)
            for _ in range(1000):
                self.assertIn(self.registry.get_register_location(1004, 16), (None, "cpu"))
                resolved = self.registry.resolve_offloaded_tensor(1004, 16)
                if resolved is not None:
                    self.assertIs(resolved[0], tensor)
                    self.assertEqual(resolved[1], 4)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(update_registry), executor.submit(read_registry), executor.submit(read_registry)]
            for future in futures:
                future.result(timeout=10)
