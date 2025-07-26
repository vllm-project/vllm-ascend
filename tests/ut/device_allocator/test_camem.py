import pytest
from unittest.mock import MagicMock, patch
from vllm_ascend.device_allocator.camem import *

def test_find_loaded_library_success_and_not_found():
    path = find_loaded_library("libc")
    assert path is not None, "Expected to find libc library"
    assert path.endswith(".so.6") or ".so" in path
    assert "libc" in path
    
    path = find_loaded_library("non_existent_library")
    assert path is None, "Expected to not find non-existent library"


@pytest.mark.parametrize("handle", [
    (1, 2, 3),
    ("device", 99),
    (None,),
])
def test_create_and_map_calls_python_create_and_map(handle):
    with patch("vllm_ascend.device_allocator.camem.python_create_and_map") as mock_create:
        create_and_map(handle)
        mock_create.assert_called_once_with(*handle)


@pytest.mark.parametrize("handle", [
    (42, "bar"),
    ("foo",),
])
def test_unmap_and_release_calls_python_unmap_and_release(handle):
    with patch("vllm_ascend.device_allocator.camem.python_unmap_and_release") as mock_release:
        unmap_and_release(handle)
        mock_release.assert_called_once_with(*handle)
        

def dummy_malloc(args):
    pass
def dummy_free(ptr):
    return (0, 0, 0, 0)
@patch("vllm_ascend.device_allocator.camem.init_module")
@patch("vllm_ascend.device_allocator.camem.torch.npu.memory.NPUPluggableAllocator")
def test_get_pluggable_allocator(mock_allocator_class,mock_init_module):
    mock_allocator_instance = MagicMock()
    mock_allocator_class.return_value = mock_allocator_instance
    allocator = get_pluggable_allocator(dummy_malloc, dummy_free)
    mock_init_module.assert_called_once_with(dummy_malloc, dummy_free)
    assert allocator == mock_allocator_instance
    
    

def test_singleton_behavior():
    instance1 = CaMemAllocator.get_instance()
    instance2 = CaMemAllocator.get_instance()
    assert instance1 is instance2


def test_python_malloc_and_free_callback():
    allocator = CaMemAllocator.get_instance()

    # 模拟 allocation_handle
    handle = (1, 100, 1234, 0)
    allocator.current_tag = "test_tag"

    allocator.python_malloc_callback(handle)
    # 检查 pointer_to_data 存储了数据
    ptr = handle[2]
    assert ptr in allocator.pointer_to_data
    data = allocator.pointer_to_data[ptr]
    assert data.handle == handle
    assert data.tag == "test_tag"

    # 测试 free callback，带有 cpu_backup_tensor
    data.cpu_backup_tensor = torch.zeros(1)
    result_handle = allocator.python_free_callback(ptr)
    assert result_handle == handle
    assert ptr not in allocator.pointer_to_data
    assert data.cpu_backup_tensor is None


@patch("vllm_ascend.device_allocator.camem.unmap_and_release")
@patch("vllm_ascend.device_allocator.camem.memcpy")
def test_sleep_offload_and_discard(mock_memcpy, mock_unmap):
    allocator = CaMemAllocator.get_instance()

    # 准备两个 allocation， 一个 tag 匹配，一个不匹配
    handle1 = (1, 10, 1000, 0)
    data1 = AllocationData(handle1, "tag1")
    handle2 = (2, 20, 2000, 0)
    data2 = AllocationData(handle2, "tag2")
    allocator.pointer_to_data = {
        1000: data1,
        2000: data2,
    }

    # 模拟 is_pin_memory_available 返回 True
    with patch("vllm_ascend.device_allocator.camem.is_pin_memory_available", return_value=True):
        allocator.sleep(offload_tags="tag1")

    # 只 offload tag1, 其他 tag2 直接调用 unmap_and_release
    assert data1.cpu_backup_tensor is not None
    assert data2.cpu_backup_tensor is None
    mock_unmap.assert_any_call(handle1)
    mock_unmap.assert_any_call(handle2)
    assert mock_unmap.call_count == 2
    assert mock_memcpy.called


@patch("vllm_ascend.device_allocator.camem.create_and_map")
@patch("vllm_ascend.device_allocator.camem.memcpy")
def test_wake_up_loads_and_clears_cpu_backup(mock_memcpy, mock_create_and_map):
    allocator = CaMemAllocator.get_instance()

    handle = (1, 10, 1000, 0)
    tensor = torch.zeros(5, dtype=torch.uint8)
    data = AllocationData(handle, "tag1", cpu_backup_tensor=tensor)
    allocator.pointer_to_data = {1000: data}

    allocator.wake_up(tags=["tag1"])

    mock_create_and_map.assert_called_once_with(handle)
    assert data.cpu_backup_tensor is None
    assert mock_memcpy.called


def test_use_memory_pool_context_manager():
    allocator = CaMemAllocator.get_instance()
    old_tag = allocator.current_tag

    # mock use_memory_pool_with_allocator 返回可上下文管理对象
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = "data"
    mock_ctx.__exit__.return_value = None

    with patch("vllm_ascend.device_allocator.camem.use_memory_pool_with_allocator", return_value=mock_ctx):
        with allocator.use_memory_pool(tag="my_tag"):
            assert allocator.current_tag == "my_tag"
        # 退出上下文后恢复旧 tag
        assert allocator.current_tag == old_tag


def test_get_current_usage():
    allocator = CaMemAllocator.get_instance()

    allocator.pointer_to_data = {
        1: AllocationData((0, 100, 1, 0), "tag"),
        2: AllocationData((0, 200, 2, 0), "tag"),
    }

    usage = allocator.get_current_usage()
    assert usage == 300