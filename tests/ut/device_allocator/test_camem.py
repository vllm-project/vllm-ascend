#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.device_allocator.cumem import CuMemAllocator

from tests.ut.base import PytestBase
from vllm_ascend.device_allocator.camem import (
    AllocationData,
    CaMemAllocator,
    create_and_map,
    get_pluggable_allocator,
    unmap_and_release,
)


def dummy_malloc(args):
    pass


def dummy_free(ptr):
    return (0, 0, 0, 0)


class TestCaMem(PytestBase):
    def test_lifecycle_is_inherited_from_vllm(self):
        for method_name in (
            "sleep",
            "wake_up",
            "use_allocation_tag",
            "use_memory_pool",
            "get_current_usage",
            "cudagraph_pool",
        ):
            assert getattr(CaMemAllocator, method_name) is getattr(CuMemAllocator, method_name)

    @pytest.mark.parametrize(
        "handle",
        [
            (1, 2, 3),
            ("device", 99),
            (None,),
        ],
    )
    def test_create_and_map_calls_python_create_and_map(self, handle):
        with patch("vllm_ascend.device_allocator.camem.python_create_and_map") as mock_create:
            create_and_map(handle)
            mock_create.assert_called_once_with(*handle)

    @pytest.mark.parametrize(
        "handle",
        [
            (42, "bar"),
            ("foo",),
        ],
    )
    def test_unmap_and_release_calls_python_unmap_and_release(self, handle):
        with patch("vllm_ascend.device_allocator.camem.python_unmap_and_release") as mock_release:
            unmap_and_release(handle)
            mock_release.assert_called_once_with(*handle)

    @patch("vllm_ascend.device_allocator.camem.init_module")
    @patch("vllm_ascend.device_allocator.camem.torch.npu.memory.NPUPluggableAllocator")
    def test_get_pluggable_allocator(self, mock_allocator_class, mock_init_module):
        mock_allocator_instance = MagicMock()
        mock_allocator_class.return_value = mock_allocator_instance

        def side_effect_malloc_and_free(malloc_fn, free_fn):
            malloc_fn((1, 2, 3))
            free_fn(123)

        mock_init_module.side_effect = side_effect_malloc_and_free

        allocator = get_pluggable_allocator(dummy_malloc, dummy_free)
        mock_init_module.assert_called_once_with(dummy_malloc, dummy_free)
        assert allocator == mock_allocator_instance

    def test_singleton_behavior(self):
        instance1 = CaMemAllocator.get_instance()
        instance2 = CaMemAllocator.get_instance()
        assert instance1 is instance2

    def test_python_malloc_and_free_callback(self):
        allocator = CaMemAllocator.get_instance()

        # mock allocation_handle
        handle = (1, 100, 1234, 0)
        allocator.current_tag = "test_tag"

        allocator.python_malloc_callback(handle)
        # check pointer_to_data store data
        ptr = handle[2]
        assert ptr in allocator.pointer_to_data
        data = allocator.pointer_to_data[ptr]
        assert data.handle == handle
        assert data.tag == "test_tag"

        # check free callback with cpu_backup_tensor
        data.cpu_backup_tensor = torch.zeros(1)
        with patch("vllm_ascend.device_allocator.camem.torch.npu.synchronize"):
            result_handle = allocator.python_free_callback(ptr)
        assert result_handle == handle
        assert ptr not in allocator.pointer_to_data
        assert data.cpu_backup_tensor is None

    @patch("vllm_ascend.device_allocator.camem.unmap_and_release")
    @patch("vllm_ascend.device_allocator.camem.memcpy")
    def test_sleep_offload_and_discard(self, mock_memcpy, mock_unmap):
        allocator = CaMemAllocator.get_instance()

        # prepare allocation， one tag match，one not match
        handle1 = (1, 10, 1000, 0)
        data1 = AllocationData(handle1, "tag1")
        handle2 = (2, 20, 2000, 0)
        data2 = AllocationData(handle2, "tag2")
        allocator.pointer_to_data = {
            1000: data1,
            2000: data2,
        }

        # Mock torch.empty to force pin_memory=False
        original_torch_empty = torch.empty

        def mock_torch_empty(*args, **kwargs):
            # If pin_memory was explicitly set to True, change it to False
            if "pin_memory" in kwargs and kwargs["pin_memory"] is True:
                kwargs["pin_memory"] = False
            return original_torch_empty(*args, **kwargs)

        with patch("vllm_ascend.device_allocator.camem.torch.empty", side_effect=mock_torch_empty):
            allocator.sleep(offload_tags="tag1")

        # only offload tag1, other tag2 call unmap_and_release
        assert data1.cpu_backup_tensor is not None
        assert data2.cpu_backup_tensor is None
        mock_unmap.assert_any_call(handle1)
        mock_unmap.assert_any_call(handle2)
        assert mock_unmap.call_count == 2
        assert mock_memcpy.called

    @patch("vllm_ascend.device_allocator.camem.create_and_map")
    @patch("vllm_ascend.device_allocator.camem.memcpy")
    @patch("vllm_ascend.device_allocator.camem.torch.npu.empty_cache")
    def test_wake_up_loads_and_clears_cpu_backup(self, mock_empty_cache, mock_memcpy, mock_create_and_map):
        allocator = CaMemAllocator.get_instance()

        handle = (1, 10, 1000, 0)
        tensor = torch.zeros(5, dtype=torch.uint8)
        data = AllocationData(handle, "tag1", cpu_backup_tensor=tensor, is_asleep=True)
        allocator.pointer_to_data = {1000: data}

        allocator.wake_up(tags=["tag1"])

        mock_create_and_map.assert_called_once_with(handle)
        mock_empty_cache.assert_called_once_with()
        assert data.cpu_backup_tensor is None
        assert mock_memcpy.called

    @patch("vllm_ascend.device_allocator.camem.torch.npu.empty_cache")
    @patch("vllm_ascend.device_allocator.camem.torch.npu.synchronize")
    def test_inherited_sleep_and_wake_respect_tags_and_persistent_allocations(self, mock_synchronize, mock_empty_cache):
        allocator = CaMemAllocator()
        weights_handle = (0, 16, 1000, 0)
        cache_handle = (0, 32, 2000, 0)
        persistent_handle = (0, 64, 3000, 0)
        weights = AllocationData(weights_handle, "weights")
        cache = AllocationData(cache_handle, "kv_cache")
        persistent = AllocationData(persistent_handle, CaMemAllocator.sleep_persistent_tag)
        allocator.pointer_to_data = {
            1000: weights,
            2000: cache,
            3000: persistent,
        }
        backup = torch.empty(16, dtype=torch.uint8)

        with (
            patch.object(allocator, "_copy_to_cpu", return_value=backup) as copy_to_cpu,
            patch.object(allocator, "_unmap_and_release") as unmap,
        ):
            allocator.sleep(offload_tags="weights")

        mock_synchronize.assert_called_once_with()
        mock_empty_cache.assert_called_once_with()
        copy_to_cpu.assert_called_once_with(1000, 16)
        unmap.assert_any_call(weights_handle)
        unmap.assert_any_call(cache_handle)
        assert unmap.call_count == 2
        assert weights.is_asleep and cache.is_asleep
        assert weights.cpu_backup_tensor is backup
        assert cache.cpu_backup_tensor is None
        assert not persistent.is_asleep

        with (
            patch.object(allocator, "_create_and_map") as create_and_map,
            patch.object(allocator, "_copy_to_device") as copy_to_device,
        ):
            allocator.wake_up(tags=["weights"])

        create_and_map.assert_called_once_with(weights_handle)
        copy_to_device.assert_called_once_with(1000, backup)
        assert not weights.is_asleep
        assert weights.cpu_backup_tensor is None
        assert cache.is_asleep
        assert not persistent.is_asleep

    @patch("vllm_ascend.device_allocator.camem.torch.npu.synchronize")
    def test_inherited_discard_marks_only_selected_allocations_as_asleep(self, mock_synchronize):
        allocator = CaMemAllocator()
        selected = AllocationData((0, 16, 1000, 0), "kv_cache")
        other = AllocationData((0, 32, 2000, 0), "weights")
        allocator.pointer_to_data = {1000: selected, 2000: other}

        with patch.object(allocator, "_unmap_and_release") as unmap:
            allocator.discard(("kv_cache",))

        mock_synchronize.assert_called_once_with(0)
        unmap.assert_called_once_with(selected.handle)
        assert selected.is_asleep
        assert selected.cpu_backup_tensor is None
        assert not other.is_asleep

    @patch("vllm_ascend.device_allocator.camem.torch.npu.empty_cache")
    @patch("vllm_ascend.device_allocator.camem.torch.npu.synchronize")
    def test_repeated_sleep_keeps_existing_offload_policy(self, mock_synchronize, mock_empty_cache, caplog):
        allocator = CaMemAllocator()
        backup = torch.empty(16, dtype=torch.uint8)
        data = AllocationData((0, 16, 1000, 0), "weights", cpu_backup_tensor=backup, is_asleep=True)
        allocator.pointer_to_data = {1000: data}

        with patch.object(allocator, "_unmap_and_release") as unmap:
            allocator.sleep(offload_tags=())

        mock_synchronize.assert_called_once_with()
        mock_empty_cache.assert_called_once_with()
        unmap.assert_not_called()
        assert data.cpu_backup_tensor is backup
        assert data.is_asleep
        assert "existing policy was kept" in caplog.text

    @patch("vllm_ascend.device_allocator.camem.memcpy")
    def test_npu_copy_hooks_use_acl_memcpy_directions(self, mock_memcpy):
        allocator = CaMemAllocator()
        cpu_tensor = MagicMock()
        cpu_tensor.data_ptr.return_value = 5000
        cpu_tensor.numel.return_value = 8
        cpu_tensor.element_size.return_value = 1

        with patch(
            "vllm_ascend.device_allocator.camem.torch.empty",
            return_value=cpu_tensor,
        ) as empty:
            result = allocator._copy_to_cpu(1000, 8)

        empty.assert_called_once_with(8, dtype=torch.uint8, device="cpu", pin_memory=True)
        assert result is cpu_tensor
        mock_memcpy.assert_called_once_with(5000, 5016, 1000, 8, 2)

        mock_memcpy.reset_mock()
        allocator._copy_to_device(1000, cpu_tensor)
        mock_memcpy.assert_called_once_with(1000, 1016, 5000, 8, 1)

    def test_npu_mapping_hooks_delegate_to_cann_extension(self):
        allocator = CaMemAllocator()
        handle = (0, 16, 1000, 0)

        with (
            patch("vllm_ascend.device_allocator.camem.python_create_and_map") as create,
            patch("vllm_ascend.device_allocator.camem.python_unmap_and_release") as unmap,
        ):
            allocator._create_and_map(handle)
            allocator._unmap_and_release(handle)

        create.assert_called_once_with(*handle)
        unmap.assert_called_once_with(*handle)

    def test_npu_cudagraph_pool_uses_npu_allocator_and_mempool(self):
        allocator = CaMemAllocator()
        pluggable_allocator = MagicMock()
        mem_pool = MagicMock()

        with (
            patch(
                "vllm_ascend.device_allocator.camem.get_pluggable_allocator",
                return_value=pluggable_allocator,
            ) as get_allocator,
            patch(
                "vllm_ascend.device_allocator.camem.torch.npu.memory.MemPool",
                return_value=mem_pool,
            ) as create_mem_pool,
        ):
            result = allocator._create_cudagraph_pool()

        get_allocator.assert_called_once_with(allocator.python_malloc_callback, allocator.python_free_callback)
        create_mem_pool.assert_called_once_with(pluggable_allocator._allocator)
        assert result == (mem_pool, pluggable_allocator)

    def test_use_memory_pool_context_manager(self):
        allocator = CaMemAllocator.get_instance()
        old_tag = allocator.current_tag

        # mock use_memory_pool_with_allocator
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = (MagicMock(snapshot=lambda: []), MagicMock())
        mock_ctx.__exit__.return_value = None

        with patch("vllm_ascend.device_allocator.camem.use_memory_pool_with_allocator", return_value=mock_ctx):
            with allocator.use_memory_pool(tag="my_tag"):
                assert allocator.current_tag == "my_tag"
            # restore old tag after context manager exits
            assert allocator.current_tag == old_tag
            assert allocator.allocator_and_pools["my_tag"] == [mock_ctx.__enter__.return_value]

    def test_use_memory_pool_restores_tag_when_body_raises(self):
        allocator = CaMemAllocator()
        old_tag = allocator.current_tag
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = (MagicMock(snapshot=lambda: []), MagicMock())
        mock_ctx.__exit__.return_value = None

        with (
            patch(
                "vllm_ascend.device_allocator.camem.use_memory_pool_with_allocator",
                return_value=mock_ctx,
            ),
            pytest.raises(RuntimeError, match="allocation failed"),
            allocator.use_memory_pool(tag="weights"),
        ):
            raise RuntimeError("allocation failed")

        assert allocator.current_tag == old_tag

    def test_get_current_usage(self):
        allocator = CaMemAllocator.get_instance()

        allocator.pointer_to_data = {
            1: AllocationData((0, 100, 1, 0), "tag"),
            2: AllocationData((0, 200, 2, 0), "tag"),
        }

        usage = allocator.get_current_usage()
        assert usage == 300
