# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-context Qwen FA3 with device lengths, ragged queries and graph replay."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config.compilation import CUDAGraphMode

from tests.e2e.pull_request.one_card.test_flash_attn_precision import golden
from vllm_ascend.attention.flash_attention_v3 import (
    AscendFlashAttentionImpl,
    AscendQwenFlashAttentionMetadataBuilder,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("capacity", [4096, 32768])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("prefill", [False, True])
@torch.inference_mode()
def test_qwen_full_attention_keeps_exact_device_lengths(dtype, capacity, num_kv_heads, prefill):
    pytest.importorskip("flash_attn_npu_3")
    if not get_current_hardware_profile().supports(HardwareCapability.FLASH_ATTN_NPU_3):
        pytest.skip("Device-length ecosystem FA3 requires A2/A3")
    torch.npu.set_device(0)
    torch.manual_seed(42)
    page, num_heads, head_size, max_reqs = 128, 16, 256, 3
    blocks = capacity // page
    query_lens = [128, 17, 8] if prefill else [8, 8, 8]
    token_capacity = sum(query_lens)
    query = torch.randn(token_capacity, num_heads, head_size, dtype=dtype, device="npu")
    # Interleaved K/V allocations match the target's paged cache strides.
    page_stride = 2 * page * num_kv_heads * head_size + 128
    storage = torch.randn(max_reqs * blocks, page_stride, dtype=dtype, device="npu")
    key = storage.as_strided(
        (max_reqs * blocks, page, num_kv_heads, head_size),
        (page_stride, num_kv_heads * head_size, head_size, 1),
    )
    value = storage.as_strided(key.shape, key.stride(), storage_offset=page * num_kv_heads * head_size)
    tables = torch.randperm(max_reqs * blocks, device="npu").to(torch.int32).view(max_reqs, blocks)
    original_tables = tables.clone()
    lengths = torch.tensor([128, capacity - 7, capacity], dtype=torch.int32, device="npu")
    offsets_cpu = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
    offsets = offsets_cpu.to("npu")
    common = SimpleNamespace(
        num_reqs=max_reqs,
        num_actual_tokens=token_capacity,
        num_input_tokens=token_capacity,
        query_start_loc_cpu=offsets_cpu,
        query_start_loc=offsets,
        seq_lens=lengths,
        block_table_tensor=tables,
        slot_mapping=torch.full((token_capacity,), -1, dtype=torch.int32, device="npu"),
        max_query_len=max(query_lens),
        causal=True,
    )
    impl = AscendFlashAttentionImpl(num_heads, head_size, head_size**-0.5, num_kv_heads=num_kv_heads)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(enforce_eager=True, num_speculative_tokens=7),
        scheduler_config=SimpleNamespace(max_num_seqs=max_reqs),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL if prefill else CUDAGraphMode.FULL_DECODE_ONLY,
            static_forward_context={"target.attn": SimpleNamespace(impl=impl)},
        ),
    )
    spec = SimpleNamespace(block_size=page, dtype=dtype)
    builder = AscendQwenFlashAttentionMetadataBuilder(spec, ["target.attn"], config, torch.device("npu"))
    layer = SimpleNamespace(layer_name="target.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    capture_metadata = builder.build(0, common)
    output = torch.empty_like(query)
    impl.forward(layer, query, None, None, (key, value), capture_metadata, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        impl.forward(layer, query, None, None, (key, value), capture_metadata, output)
    q_cpu, k_cpu, v_cpu = query.cpu(), key.cpu(), value.cpu()
    # Rejection changes exact KV lengths and query widths. Reorder and remove
    # requests, then grow back to the captured bucket and the full KV boundary.
    cases = [
        ([64, 8] if prefill else [7, 8], [capacity - 9, 127], [1, 0]),
        ([16] if prefill else [3], [capacity - 5], [2]),
        ([8, 3, 8], [capacity - 8, 127, capacity], [2, 0, 1]),
        (query_lens, [128, capacity - 7, capacity], [0, 1, 2]),
    ]
    for queries, exact_lengths, order in cases:
        active = len(order)
        lengths.zero_()
        lengths[:active].copy_(torch.tensor(exact_lengths, dtype=torch.int32, device="npu"))
        cpu_bounds = torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], dtype=torch.int32)
        offsets_cpu.fill_(sum(queries))
        offsets_cpu[: active + 1].copy_(cpu_bounds)
        offsets.copy_(offsets_cpu)
        tables[:active].copy_(original_tables[order])
        common.max_query_len = max(queries)
        metadata = builder.build(0, common)
        assert metadata.block_tables.data_ptr() == capture_metadata.block_tables.data_ptr()
        for kernel, buffer in metadata.scheduler_metadata.items():
            assert buffer.data_ptr() == capture_metadata.scheduler_metadata[kernel].data_ptr()
        graph.replay()
        eager = torch.empty_like(query)
        impl.forward(layer, query, None, None, (key, value), metadata, eager)
        torch.npu.synchronize()
        references = []
        q_offset = 0
        cpu_tables = tables.cpu()
        for req, (q_len, kv_len) in enumerate(zip(queries, exact_lengths)):
            references.append(
                golden(
                    q_cpu[q_offset : q_offset + q_len],
                    k_cpu,
                    v_cpu,
                    cpu_tables[req : req + 1],
                    [kv_len],
                    None,
                    q_len,
                    True,
                )
            )
            q_offset += q_len
        reference = torch.cat(references)
        # BF16 prefill softmax accumulation has the same rounding error in
        # stock global FA3; compare against FP32 with a BF16-sized tolerance.
        atol = 4e-3 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(eager[:q_offset].cpu(), reference, rtol=2e-2, atol=atol)
        torch.testing.assert_close(output[:q_offset].cpu(), reference, rtol=2e-2, atol=atol)
