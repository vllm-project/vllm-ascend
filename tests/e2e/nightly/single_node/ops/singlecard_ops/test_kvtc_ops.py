import pytest
import torch
import torch_npu  # noqa: F401
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module")
def ops():
    if get_ascend_device_type() != AscendDeviceType.A2:
        pytest.skip("KVTC native operators are built for Ascend 910B")
    assert enable_custom_op()
    return torch.ops._C_ascend


@pytest.mark.parametrize(("block_size", "num_layers"), [(16, 2), (128, 16)])
@pytest.mark.parametrize("with_rope", [False, True], ids=["plain", "rope"])
def test_kvtc_prepare_and_restore(ops, block_size, num_layers, with_rope):
    num_blocks, num_heads = 5, 2
    head_size = 32 if with_rope else 24
    shape = (num_blocks, block_size, num_heads, head_size)
    generator = torch.Generator().manual_seed(42)
    caches = [torch.randint(-32, 32, shape, generator=generator).to(torch.bfloat16) for _ in range(num_layers)]
    device_caches = [cache.npu() for cache in caches]
    source_ids = torch.tensor([4, 0, 3, 2])
    destination_ids = torch.tensor([2, 1, 4, 0])
    destination = [torch.zeros_like(cache) for cache in device_caches]

    expected = torch.stack([cache[source_ids] for cache in caches], dim=2)
    flat_shape = (len(source_ids) * block_size, num_layers * num_heads * head_size)
    if with_rope:
        starts = torch.tensor([0, 5, 9, 15])
        phase = torch.arange(int(starts.max()) + block_size) % 4
        cos = torch.tensor([1.0, 0.0, -1.0, 0.0])[phase]
        sin = torch.tensor([0.0, 1.0, 0.0, -1.0])[phase]
        table = torch.stack((cos, sin), dim=-1).repeat_interleave(head_size // 2, dim=-1)
        inverse = torch.stack((cos, -sin), dim=-1).repeat_interleave(head_size // 2, dim=-1)
        positions = starts[:, None] + torch.arange(block_size)
        c, s = cos[positions][..., None, None, None], -sin[positions][..., None, None, None]
        first, second = expected.float().chunk(2, dim=-1)
        expected = torch.cat((first * c - second * s, second * c + first * s), dim=-1).to(torch.bfloat16)
        prepared = ops.kvtc_prepare_rope(device_caches, source_ids.npu(), starts.npu(), inverse.npu())
        status = ops.kvtc_restore_rope(prepared, destination_ids.npu(), starts.npu(), table.npu(), destination)
    else:
        prepared = ops.kvtc_prepare(device_caches, source_ids.npu())
        status = ops.kvtc_restore(prepared, destination_ids.npu(), destination)

    torch.testing.assert_close(prepared.cpu(), expected.reshape(flat_shape), rtol=0, atol=0)
    assert status.cpu().tolist() == [0]
    for source, restored in zip(caches, destination):
        expected_cache = torch.zeros_like(source)
        expected_cache[destination_ids] = source[source_ids]
        torch.testing.assert_close(restored.cpu(), expected_cache, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["meta", "npu"])
@pytest.mark.parametrize(
    ("num_heads", "head_size", "with_rope", "message"),
    [
        (1, 24, False, "32-byte aligned"),
        (2, 24, True, "divisible by 16"),
        (1, 144, True, "head size must be <=128"),
        (0, 32, False, "cache dimensions must be positive"),
    ],
)
def test_kvtc_rejects_invalid_cache_shape(ops, device, num_heads, head_size, with_rope, message):
    cache = torch.empty(3, 16, num_heads, head_size, device=device, dtype=torch.bfloat16)
    block_ids = torch.zeros(1, device=device, dtype=torch.int64)
    reconstructed = torch.empty(16, num_heads * head_size, device=device, dtype=torch.bfloat16)

    if with_rope:
        starts = torch.zeros_like(block_ids)
        table = torch.empty(16, head_size, device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError, match=message):
            ops.kvtc_prepare_rope([cache], block_ids, starts, table)
        with pytest.raises(RuntimeError, match=message):
            ops.kvtc_restore_rope(reconstructed, block_ids, starts, table, [cache])
    else:
        with pytest.raises(RuntimeError, match=message):
            ops.kvtc_prepare([cache], block_ids)
        with pytest.raises(RuntimeError, match=message):
            ops.kvtc_restore(reconstructed, block_ids, [cache])


@pytest.mark.parametrize("device", ["meta", "npu"])
def test_kvtc_rejects_empty_block_ids(ops, device):
    cache = torch.empty(3, 16, 2, 32, device=device, dtype=torch.bfloat16)
    block_ids = torch.empty(0, device=device, dtype=torch.int64)
    reconstructed = torch.empty(0, 64, device=device, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="block_ids must not be empty"):
        ops.kvtc_prepare([cache], block_ids)
    with pytest.raises(RuntimeError, match="block_ids must not be empty"):
        ops.kvtc_restore(reconstructed, block_ids, [cache])


def test_kvtc_fake_tensor_shapes(ops):
    with FakeTensorMode():
        caches = [torch.empty(3, 16, 2, 24, dtype=torch.bfloat16) for _ in range(2)]
        block_ids = torch.empty(2, dtype=torch.int64)
        prepared = ops.kvtc_prepare(caches, block_ids)
        status = ops.kvtc_restore(prepared, block_ids, caches)

    assert prepared.shape == (32, 96) and prepared.dtype == torch.bfloat16
    assert status.shape == (1,) and status.dtype == torch.int32
