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

import gc
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import PytestBase
from vllm_ascend.device_allocator import camem as camem_module
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
    @pytest.fixture(autouse=True)
    def _camem_available(self):
        # get_instance() asserts camem_available, but CPU UTs can run without
        # the vllm_ascend_C extension.
        with patch("vllm_ascend.device_allocator.camem.camem_available", True):
            yield

    @pytest.fixture
    def _npu_camem_allocator(self, monkeypatch):
        if torch.npu.is_available() is not True or camem_module.python_create_and_map is None:
            pytest.skip("requires an Ascend NPU and the CaMem extension")

        torch.npu.set_device(0)
        allocator = CaMemAllocator.get_instance()
        monkeypatch.setattr(allocator, "pointer_to_data", {})
        monkeypatch.setattr(allocator, "allocator_and_pools", {})
        yield allocator

        if any(data.is_asleep for data in allocator.pointer_to_data.values()):
            allocator.wake_up()
        gc.collect()
        torch.npu.synchronize()
        allocator.release_pools()

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

    def test_release_pools_clears_all_tags(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        pools = {
            "weights": (MagicMock(name="weights_pool"), MagicMock(name="weights_allocator")),
            "kv_cache": (MagicMock(name="kv_pool"), MagicMock(name="kv_allocator")),
        }
        monkeypatch.setattr(allocator, "allocator_and_pools", pools)

        allocator.release_pools()
        assert allocator.allocator_and_pools == {}

        # Cleanup is safe to call repeatedly, including during shutdown.
        allocator.release_pools()

    def test_shutdown_singleton_releases_pools(self, monkeypatch):
        allocator = MagicMock()
        monkeypatch.setattr(CaMemAllocator, "instance", allocator)

        CaMemAllocator._shutdown_singleton()

        allocator.release_pools.assert_called_once_with()

    def test_shutdown_unmaps_live_allocations_and_drops_backups(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        awake = MagicMock(is_asleep=False, handle=(0, 4096, 1, 2), cpu_backup_tensor=None)
        asleep = MagicMock(is_asleep=True, handle=(0, 8192, 3, 4), cpu_backup_tensor=MagicMock())
        monkeypatch.setattr(allocator, "pointer_to_data", {1: awake, 3: asleep})
        unmap = MagicMock()
        monkeypatch.setattr(allocator, "_unmap_or_fail", unmap)
        release = MagicMock()
        monkeypatch.setattr(allocator, "release_pools", release)

        allocator.shutdown()
        allocator.shutdown()

        unmap.assert_called_once_with(awake.handle)
        assert awake.is_asleep and asleep.is_asleep
        assert awake.cpu_backup_tensor is None and asleep.cpu_backup_tensor is None
        assert release.call_count == 2

    def test_shutdown_reclaims_physical_memory_with_live_tensor(self, _npu_camem_allocator):
        allocator = _npu_camem_allocator
        with allocator.use_memory_pool("weights"):
            tensor = torch.ones(64 << 20, dtype=torch.uint8, device="npu")
        torch.npu.synchronize()
        pointer = tensor.data_ptr()
        assert allocator.pointer_to_data
        before = torch.npu.mem_get_info()[0]
        allocator.shutdown()
        assert torch.npu.mem_get_info()[0] - before >= tensor.nbytes
        assert tensor.data_ptr() == pointer  # metadata only; never read unmapped storage
        assert all(data.is_asleep for data in allocator.pointer_to_data.values())
        allocator.shutdown()
        del tensor
        gc.collect()
        torch.npu.empty_cache()
        assert not allocator.pointer_to_data

    def test_discard_tags(self, _npu_camem_allocator):
        allocator = _npu_camem_allocator

        def mapped_bytes():
            return sum(data.handle[1] for data in allocator.pointer_to_data.values() if not data.is_asleep)

        with allocator.use_memory_pool("weights"):
            weights = torch.ones(1024, 1024, device="npu")

        with allocator.use_memory_pool("kv_cache"):
            kv_cache = torch.ones(512, 512, device="npu")

        mapped_before = mapped_bytes()
        allocator.discard("kv_cache")
        mapped_after_discard = mapped_bytes()
        assert mapped_before - mapped_after_discard >= kv_cache.nbytes
        assert mapped_after_discard >= weights.nbytes
        assert torch.allclose(weights, torch.ones_like(weights))

        allocator.wake_up()
        assert kv_cache.shape == (512, 512)

        allocator.sleep(offload_tags="weights")
        allocator.wake_up()
        assert torch.allclose(weights, torch.ones_like(weights))

    def test_sleep_with_only_weights_asleep(self, _npu_camem_allocator):
        allocator = _npu_camem_allocator
        with allocator.use_memory_pool("weights"):
            weights = torch.ones(1024, 1024, device="npu")
        with allocator.use_memory_pool("kv_cache"):
            kv_cache = torch.ones(512, 512, device="npu")

        allocator.sleep(offload_tags="weights")
        assert all(data.is_asleep for data in allocator.pointer_to_data.values())
        weights_backup = next(
            data.cpu_backup_tensor for data in allocator.pointer_to_data.values() if data.tag == "weights"
        )
        assert weights_backup is not None

        allocator.wake_up(tags=["kv_cache"])
        assert all(data.is_asleep for data in allocator.pointer_to_data.values() if data.tag == "weights")
        assert all(not data.is_asleep for data in allocator.pointer_to_data.values() if data.tag == "kv_cache")

        allocator.sleep(offload_tags="weights")
        assert all(data.is_asleep for data in allocator.pointer_to_data.values())
        assert (
            next(data.cpu_backup_tensor for data in allocator.pointer_to_data.values() if data.tag == "weights")
            is weights_backup
        )
        allocator.wake_up(tags=["weights"])
        assert all(not data.is_asleep for data in allocator.pointer_to_data.values() if data.tag == "weights")
        assert torch.allclose(weights, torch.ones_like(weights))
        assert all(data.is_asleep for data in allocator.pointer_to_data.values() if data.tag == "kv_cache")

        allocator.wake_up(tags=["kv_cache"])
        assert all(not data.is_asleep for data in allocator.pointer_to_data.values())
        assert kv_cache.shape == (512, 512)

    @pytest.mark.parametrize("tag", ["workspace", None], ids=["workspace", "default"])
    def test_selective_wake_restores_requested_tags(self, _npu_camem_allocator, tag):
        allocator = _npu_camem_allocator
        internal_tag = tag or allocator.default_tag
        with allocator.use_memory_pool(tag):
            internal = torch.empty(16 << 20, dtype=torch.uint8, device="npu")
        with allocator.use_memory_pool("kv_cache"):
            kv_cache = torch.empty(32 << 20, dtype=torch.uint8, device="npu")

        allocator.sleep(offload_tags=())
        assert all(data.is_asleep for data in allocator.pointer_to_data.values())

        allocator.wake_up(tags=[internal_tag])
        assert all(not data.is_asleep for data in allocator.pointer_to_data.values() if data.tag == internal_tag)
        assert all(data.is_asleep for data in allocator.pointer_to_data.values() if data.tag == "kv_cache")
        mapped_internal_bytes = sum(
            data.handle[1]
            for data in allocator.pointer_to_data.values()
            if not data.is_asleep and data.tag == internal_tag
        )
        assert mapped_internal_bytes >= internal.nbytes

        allocator.wake_up(tags=["kv_cache"])
        assert all(not data.is_asleep for data in allocator.pointer_to_data.values())
        assert kv_cache.numel() == 32 << 20

    @patch("vllm_ascend.device_allocator.camem.torch.cuda.synchronize")
    @patch("vllm_ascend.device_allocator.camem.torch.npu.synchronize")
    def test_python_malloc_and_free_callback(self, mock_npu_sync, mock_cuda_sync):
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
        mock_npu_sync.assert_called_once_with(handle[0])
        mock_cuda_sync.assert_not_called()
        assert ptr not in allocator.pointer_to_data
        assert allocator._pending_free[ptr] is data
        assert data.cpu_backup_tensor is not None
        allocator.python_free_complete_callback(ptr)
        assert ptr not in allocator.pointer_to_data
        assert ptr not in allocator._pending_free
        assert data.cpu_backup_tensor is None

    @patch("vllm_ascend.device_allocator.camem.unmap_and_release")
    @patch("vllm_ascend.device_allocator.camem.memcpy", return_value=0)
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
    @patch("vllm_ascend.device_allocator.camem.memcpy", return_value=0)
    def test_wake_up_loads_and_clears_cpu_backup(self, mock_memcpy, mock_create_and_map):
        allocator = CaMemAllocator.get_instance()

        handle = (1, 10, 1000, 0)
        tensor = torch.zeros(5, dtype=torch.uint8)
        data = AllocationData(handle, "tag1", cpu_backup_tensor=tensor, is_asleep=True)
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

        with patch("vllm_ascend.device_allocator.camem.use_memory_pool_with_allocator", return_value=mock_ctx):
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

    def test_wake_uses_npu_cache_api(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        monkeypatch.setattr(allocator, "pointer_to_data", {})
        with (
            patch.object(torch.npu, "empty_cache") as npu_empty_cache,
            patch.object(torch.accelerator, "empty_cache", side_effect=RuntimeError("not a DeviceAllocator")),
        ):
            allocator.wake_up()
        npu_empty_cache.assert_called_once_with()

    def test_aclgraph_pool_reuses_pool_and_restores_tag(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        monkeypatch.setattr(allocator, "allocator_and_pools", {})
        monkeypatch.setattr(allocator, "current_tag", "weights")
        pool = MagicMock()
        pool.id = (0, 7)
        with (
            patch.object(camem_module, "get_pluggable_allocator") as create_allocator,
            patch.object(torch.npu.memory, "MemPool", return_value=pool) as create_pool,
            patch.object(torch.npu.memory, "use_mem_pool") as enter_pool,
        ):
            with allocator.aclgraph_pool() as first:
                assert first == pool.id
                assert allocator.current_tag == camem_module.ACL_GRAPH_TAG
            assert allocator.current_tag == "weights"
            with pytest.raises(RuntimeError, match="capture failed"), allocator.aclgraph_pool() as second:
                assert second == first
                raise RuntimeError("capture failed")
            assert allocator.current_tag == "weights"
            assert create_allocator.call_count == create_pool.call_count == 1
            enter_pool.assert_not_called()

    def test_failed_unmap_does_not_mark_allocation_asleep(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        monkeypatch.setattr(allocator, "_unmap_failed", False)
        data = AllocationData((0, 16, 1000, 0), "scratch")
        monkeypatch.setattr(allocator, "pointer_to_data", {1000: data})
        with (
            patch.object(camem_module, "unmap_and_release", side_effect=RuntimeError("unmap failed")),
            pytest.raises(RuntimeError, match="unmap failed"),
        ):
            allocator.sleep(offload_tags=())
        assert not data.is_asleep
        with pytest.raises(RuntimeError, match="mapping state is unknown"):
            allocator.wake_up()

    def test_failed_restore_keeps_backup_for_retry(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        backup = torch.zeros(16, dtype=torch.uint8)
        data = AllocationData((0, 16, 1000, 0), "weights", cpu_backup_tensor=backup, is_asleep=True)
        monkeypatch.setattr(allocator, "pointer_to_data", {1000: data})
        with (
            patch.object(camem_module, "create_and_map"),
            patch.object(camem_module, "unmap_and_release") as unmap,
            patch.object(camem_module, "memcpy", return_value=507000),
        ):
            with pytest.raises(RuntimeError, match="host-to-device restore failed"):
                allocator.wake_up(tags=["weights"])
            unmap.assert_called_once_with(data.handle)
        assert data.is_asleep
        assert data.cpu_backup_tensor is backup

    def test_failed_backup_does_not_unmap(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        data = AllocationData((0, 16, 1000, 0), "weights")
        monkeypatch.setattr(allocator, "pointer_to_data", {1000: data})
        with (
            patch.object(camem_module.torch, "empty", return_value=torch.zeros(16, dtype=torch.uint8)),
            patch.object(camem_module, "unmap_and_release") as unmap,
            patch.object(camem_module, "memcpy", return_value=507000),
        ):
            with pytest.raises(RuntimeError, match="device-to-host backup failed"):
                allocator.sleep(offload_tags=("weights",))
            unmap.assert_not_called()
        assert not data.is_asleep
        assert data.cpu_backup_tensor is None

    def test_free_sync_failure_keeps_handle_and_backup(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        backup = torch.zeros(16, dtype=torch.uint8)
        data = AllocationData((0, 16, 1000, 42), "weights", cpu_backup_tensor=backup)
        monkeypatch.setattr(allocator, "pointer_to_data", {1000: data})
        monkeypatch.setattr(allocator, "_pending_free", {})
        with (
            patch.object(torch.npu, "synchronize", side_effect=RuntimeError("sync failed")),
            pytest.raises(RuntimeError, match="sync failed"),
        ):
            allocator.python_free_callback(1000)
        assert allocator.pointer_to_data[1000] is data
        assert not allocator._pending_free
        assert data.cpu_backup_tensor is backup

    def test_native_free_error_rejects_sleep_and_wake_before_device_work(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        monkeypatch.setattr(allocator, "_unmap_failed", False)
        error = MagicMock(side_effect=RuntimeError("CaMem free callback failed; restart the worker"))
        monkeypatch.setattr(camem_module, "python_check_free_error", error)
        with patch.object(torch.npu, "synchronize") as sync:
            for transition in (allocator.sleep, allocator.wake_up):
                with pytest.raises(RuntimeError, match="restart the worker"):
                    transition()
            sync.assert_not_called()

    @patch("vllm_ascend.device_allocator.camem.init_module")
    @patch("vllm_ascend.device_allocator.camem.torch.npu.memory.NPUPluggableAllocator")
    def test_free_completion_callback_is_registered(self, mock_allocator, mock_init):
        complete = MagicMock()
        get_pluggable_allocator(dummy_malloc, dummy_free, complete)
        mock_init.assert_called_once_with(dummy_malloc, dummy_free, complete)

    def test_graph_pool_release_cannot_hide_pending_free_records(self, monkeypatch):
        allocator = CaMemAllocator.get_instance()
        data = AllocationData((0, 16, 1000, 42), camem_module.ACL_GRAPH_TAG)
        monkeypatch.setattr(allocator, "pointer_to_data", {})
        monkeypatch.setattr(allocator, "_pending_free", {1000: data})
        monkeypatch.setattr(allocator, "allocator_and_pools", {})
        monkeypatch.setattr(allocator, "_unmap_failed", False)
        monkeypatch.setattr(camem_module, "python_check_free_error", None)
        with pytest.raises(RuntimeError, match="still have live owners"):
            allocator.release_aclgraph_pool()
