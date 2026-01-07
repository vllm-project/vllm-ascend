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

from tests.ut.base import PytestBase
from vllm_ascend.device_allocator.camem import (AllocationData, CaMemAllocator,
                                                create_and_map,
                                                find_loaded_library,
                                                get_pluggable_allocator,
                                                unmap_and_release)


def dummy_malloc(args):
    pass


def dummy_free(ptr):
    return (0, 0, 0, 0)


class TestCaMem(PytestBase):

    def test_find_loaded_library_success_and_not_found(self):
        path = find_loaded_library("libc")
        assert path is not None, "Expected to find libc library"
        assert path.endswith(".so.6") or ".so" in path
        assert "libc" in path

        path = find_loaded_library("non_existent_library")
        assert path is None, "Expected to not find non-existent library"

    @pytest.mark.parametrize("handle", [
        (1, 2, 3),
        ("device", 99),
        (None, ),
    ])
    def test_create_and_map_calls_python_create_and_map(self, handle):
        with patch("vllm_ascend.device_allocator.camem.python_create_and_map"
                   ) as mock_create:
            create_and_map(handle)
            mock_create.assert_called_once_with(*handle)

    @pytest.mark.parametrize("handle", [
        (42, "bar"),
        ("foo", ),
    ])
    def test_unmap_and_release_calls_python_unmap_and_release(self, handle):
        with patch(
                "vllm_ascend.device_allocator.camem.python_unmap_and_release"
        ) as mock_release:
            unmap_and_release(handle)
            mock_release.assert_called_once_with(*handle)

    @patch("vllm_ascend.device_allocator.camem.init_module")
    @patch(
        "vllm_ascend.device_allocator.camem.torch.npu.memory.NPUPluggableAllocator"
    )
    def test_get_pluggable_allocator(self, mock_allocator_class,
                                     mock_init_module):
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
            if 'pin_memory' in kwargs and kwargs['pin_memory'] is True:
                kwargs['pin_memory'] = False
            return original_torch_empty(*args, **kwargs)

        with patch("vllm_ascend.device_allocator.camem.torch.empty",
                   side_effect=mock_torch_empty):
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
    def test_wake_up_loads_and_clears_cpu_backup(self, mock_memcpy,
                                                 mock_create_and_map):
        allocator = CaMemAllocator.get_instance()

        handle = (1, 10, 1000, 0)
        tensor = torch.zeros(5, dtype=torch.uint8)
        data = AllocationData(handle, "tag1", cpu_backup_tensor=tensor)
        allocator.pointer_to_data = {1000: data}

        allocator.wake_up(tags=["tag1"])

        mock_create_and_map.assert_called_once_with(handle)
        assert data.cpu_backup_tensor is None
        assert mock_memcpy.called

    def test_use_memory_pool_context_manager(self):
        allocator = CaMemAllocator.get_instance()
        old_tag = allocator.current_tag

        # mock use_memory_pool_with_allocator
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = "data"
        mock_ctx.__exit__.return_value = None

        with patch(
                "vllm_ascend.device_allocator.camem.use_memory_pool_with_allocator",
                return_value=mock_ctx):
            with allocator.use_memory_pool(tag="my_tag"):
                assert allocator.current_tag == "my_tag"
            # restore old tag after context manager exits
            assert allocator.current_tag == old_tag

    def test_get_current_usage(self):
        allocator = CaMemAllocator.get_instance()

        allocator.pointer_to_data = {
            1: AllocationData((0, 100, 1, 0), "tag"),
            2: AllocationData((0, 200, 2, 0), "tag"),
        }

        usage = allocator.get_current_usage()
        assert usage == 300
