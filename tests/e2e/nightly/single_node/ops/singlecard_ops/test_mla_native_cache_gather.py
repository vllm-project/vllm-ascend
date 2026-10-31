# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native paged MLA reads preserve strided pages without copying the cache pool."""

from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor, DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

BLOCK_SIZE = 128
KEY_DIM = 512
ROPE_DIM = 64
PREFIX_ELEMENTS = 256
PAGE_PADDING_ELEMENTS = 256


def _cache_views(
    backing: torch.Tensor, blocks: int, layout: str = "component_major"
) -> tuple[torch.Tensor, torch.Tensor]:
    page_stride = BLOCK_SIZE * (KEY_DIM + ROPE_DIM) + PAGE_PADDING_ELEMENTS
    if layout == "token_fused":
        key_token_stride = rope_token_stride = KEY_DIM + ROPE_DIM
        rope_offset = KEY_DIM
    else:
        key_token_stride, rope_token_stride = KEY_DIM, ROPE_DIM
        rope_offset = BLOCK_SIZE * KEY_DIM
    key = torch.as_strided(
        backing,
        (blocks, BLOCK_SIZE, 1, KEY_DIM),
        (page_stride, key_token_stride, key_token_stride, 1),
        storage_offset=PREFIX_ELEMENTS,
    )
    rope = torch.as_strided(
        backing,
        (blocks, BLOCK_SIZE, 1, ROPE_DIM),
        (page_stride, rope_token_stride, rope_token_stride, 1),
        storage_offset=PREFIX_ELEMENTS + rope_offset,
    )
    return key, rope


def _reference_rows(cache: torch.Tensor, table: torch.Tensor, lengths: list[int], starts: list[int]) -> torch.Tensor:
    # Resolve one token at a time on the CPU, independently of the device
    # gather's block traversal and the old repeat_interleave/index chain.
    rows = [
        cache[int(table[request, position // BLOCK_SIZE]), position % BLOCK_SIZE]
        for request, (length, start) in enumerate(zip(lengths, starts))
        for position in range(start, start + length)
    ]
    return torch.stack(rows) if rows else cache.new_empty((0, 1, cache.shape[-1]))


@pytest.mark.parametrize(
    "adaptor,layout",
    [
        (BaseDeviceAdaptor, "component_major"),
        (A5DeviceAdaptor, "component_major"),
        (A5DeviceAdaptor, "token_fused"),
    ],
    ids=["base-component-major", "a5-component-major", "a5-token-fused"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("no_rope", [False, True], ids=["rope", "nope-duplicate"])
@pytest.mark.parametrize(
    "lengths,starts",
    [
        ([129], [128]),
        ([135], [256]),
        ([255], [128]),
        ([129, 0, 263, 127], [128, 128 * 9999, 0, 256]),
        ([0, 0], [0, 128 * 9999]),
    ],
    ids=["tail-1", "tail-7", "tail-127", "mixed-empty", "all-empty"],
)
def test_native_mla_gather_preserves_strided_pages(adaptor, layout, dtype, no_rope, lengths, starts):
    if layout == "token_fused" and not get_current_hardware_profile().supports(HardwareCapability.MLA_FLASH):
        pytest.skip("Token-fused MLA cache gathering requires Ascend A5 inner-stride support")
    blocks = 12
    page_stride = BLOCK_SIZE * (KEY_DIM + ROPE_DIM) + PAGE_PADDING_ELEMENTS
    # Poison the prefix, page gaps and suffix. Populate on the CPU so cache
    # setup does not itself exercise a noncontiguous NPU write implementation.
    original = torch.full((PREFIX_ELEMENTS + blocks * page_stride + PREFIX_ELEMENTS,), 37, dtype=dtype)
    key_cpu, rope_cpu = _cache_views(original, blocks, layout)
    key_cpu.copy_((torch.arange(key_cpu.numel(), dtype=torch.int32) % 127 - 63).reshape(key_cpu.shape))
    rope_cpu.copy_((torch.arange(rope_cpu.numel(), dtype=torch.int32) % 97 - 48).reshape(rope_cpu.shape))
    backing = original.to("npu")
    key_cache, rope_cache = _cache_views(backing, blocks, layout)
    assert not key_cache.is_contiguous() and not rope_cache.is_contiguous()
    assert key_cache.storage_offset() > 0 and rope_cache.storage_offset() > 0
    assert key_cache.stride(0) > BLOCK_SIZE * KEY_DIM
    assert rope_cache.stride(0) > BLOCK_SIZE * ROPE_DIM
    if layout == "token_fused":
        assert key_cache.stride(1) == rope_cache.stride(1) == KEY_DIM + ROPE_DIM

    # The zero-length request deliberately has unusable entries and a large
    # aligned start: no cache page may be read for that row.
    table_cpu = torch.tensor(
        [[9, 2, 7, 1, 10, 3], [-1, -1, -1, -1, -1, -1], [8, 0, 11, 4, 2, 5], [6, 10, 3, 8, 1, 4]],
        dtype=torch.int32,
    )[: len(lengths)]
    # Slice a wider tensor to exercise the wrapper's existing seq_lens
    # contiguous conversion without changing the per-request length ABI.
    lengths_npu = torch.tensor([[length, 0] for length in lengths], dtype=torch.int32, device="npu")[:, 0]
    starts_npu = torch.tensor(starts, dtype=torch.int32, device="npu")
    value_cache = key_cache if no_rope else rope_cache
    expected_key = _reference_rows(key_cpu, table_cpu, lengths, starts)
    expected_value = expected_key if no_rope else _reference_rows(rope_cpu, table_cpu, lengths, starts)
    key = torch.full(expected_key.shape, -77, dtype=dtype, device="npu")
    value = torch.full(expected_value.shape, -77, dtype=dtype, device="npu")

    # MLA-NoPE passes the latent cache twice, rather than an empty RoPE cache:
    # the native API treats any empty operand as a no-op for both outputs.
    with patch(
        "vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache",
        wraps=torch_npu.npu_gather_pa_kv_cache,
    ) as native_gather:
        adaptor.kv_cache_load(
            key_cache, value_cache, table_cpu.to("npu"), lengths_npu, starts_npu, key=key, value=value
        )
    native_gather.assert_called_once()
    assert key.is_contiguous() and value.is_contiguous()
    torch.testing.assert_close(key.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value.cpu(), expected_value, rtol=0, atol=0)
    # Includes untouched pages and all poison regions of the shared allocation.
    torch.testing.assert_close(backing.cpu(), original, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [257, 130944], ids=["short-prefix", "profile-prefix"])
def test_native_mla_gather_does_not_materialize_the_cache_pool(tokens):
    blocks = 2048
    page_stride = BLOCK_SIZE * (KEY_DIM + ROPE_DIM) + PAGE_PADDING_ELEMENTS
    backing = torch.empty(PREFIX_ELEMENTS + blocks * page_stride, dtype=torch.bfloat16, device="npu")
    key_cache, rope_cache = _cache_views(backing, blocks)
    key_cache.fill_(3)
    rope_cache.fill_(7)
    table_columns = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    table = torch.arange(blocks - 1, blocks - table_columns - 1, -1, dtype=torch.int32, device="npu").unsqueeze(0)
    lengths = torch.tensor([tokens], dtype=torch.int32, device="npu")
    starts = torch.tensor([BLOCK_SIZE], dtype=torch.int32, device="npu")
    key = torch.empty((tokens, 1, KEY_DIM), dtype=torch.bfloat16, device="npu")
    rope = torch.empty((tokens, 1, ROPE_DIM), dtype=torch.bfloat16, device="npu")

    # Initialize the operator and its workspace before tracking transient live
    # memory. Outputs are already allocated: < 0.3 MiB for the short case and
    # about 144 MiB for the profiling-sized prefix. A full NoPE pool copy alone
    # is 256 MiB. The 128 MiB allowance for additional live memory leaves room
    # for allocator rounding/workspace while rejecting an allocator-tracked
    # temporary as large as the full pool. This does not detect copies inside
    # preallocated or externally managed workspace; a kernel trace must check
    # for those separately.
    DeviceOperator.kv_cache_load(key_cache, rope_cache, table, lengths, starts, key=key, value=rope)
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    with patch(
        "vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache",
        wraps=torch_npu.npu_gather_pa_kv_cache,
    ) as native_gather:
        DeviceOperator.kv_cache_load(key_cache, rope_cache, table, lengths, starts, key=key, value=rope)
        torch.npu.synchronize()
    peak_bytes = torch.npu.max_memory_allocated() - baseline
    native_gather.assert_called_once()
    limit_bytes = key_cache.numel() * key_cache.element_size() // 2
    assert peak_bytes < limit_bytes, f"Native MLA gather allocated {peak_bytes} bytes (limit: {limit_bytes})"
    torch.testing.assert_close(key.cpu(), torch.full(key.shape, 3, dtype=key.dtype), rtol=0, atol=0)
    torch.testing.assert_close(rope.cpu(), torch.full(rope.shape, 7, dtype=rope.dtype), rtol=0, atol=0)
