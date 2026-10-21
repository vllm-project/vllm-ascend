# SPDX-License-Identifier: Apache-2.0
"""NPU numerical coverage of the P BF16 -> D TQ4 export boundary.

This is a component test; a full model P/D run is still required for TTFT.
"""

import pytest
import torch
import torch_npu

from vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill import pack_bf16_kv
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import TurboQuantLatent


@pytest.mark.parametrize("rows", [1, 257, 4096])
@pytest.mark.parametrize("fused", [False, True])
@torch.inference_mode()
def test_bf16_export_matches_existing_tq_slot_abi(rows, fused):
    init_device_properties_triton()
    torch.manual_seed(211)
    raw = torch.randn(rows, 512, dtype=torch.bfloat16, device="npu")
    gamma = torch.randn(512, dtype=torch.bfloat16, device="npu")
    nope, _ = torch_npu.npu_rms_norm(raw, gamma)
    rope = torch.randn(rows, 64, dtype=torch.bfloat16, device="npu")
    # P retains normalized latent and post-RoPE values in separate page views.
    backing = torch.empty(rows, 576, dtype=torch.bfloat16, device="npu")
    backing[:, :512].copy_(nope)
    backing[:, 512:].copy_(rope)
    before = backing.cpu()
    quantizer = TurboQuantLatent(fused_finalize=fused)
    actual = pack_bf16_kv(backing[:, :512], backing[:, 512:], quantizer)

    # Existing online SFA storage contract: nibbles / predivided RoPE / FP16 scale.
    online = quantizer.compress(nope).reshape(rows, 258)
    scale = online[:, 256:].contiguous().view(torch.float16)
    online_rope = (rope.float() / (scale.float() + 1e-20)).bfloat16()
    expected = torch.cat((online[:, :256], online_rope.view(torch.uint8), online[:, 256:]), -1)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)
    assert actual.shape == (rows, 386)
    assert torch.isfinite(scale).all()


@torch.inference_mode()
def test_converted_cache_is_consumable_by_tq_attention():
    init_device_properties_triton()
    torch.manual_seed(212)
    rows = 2048
    nope = torch.randn(rows, 512, dtype=torch.bfloat16, device="npu")
    rope = torch.randn(rows, 64, dtype=torch.bfloat16, device="npu")
    store = TurboQuantLatent(fused_finalize=True)
    packed = pack_bf16_kv(nope, rope, store)
    cache = packed.view(rows // 256, 256, 1, 386).view(torch.int8)
    query = torch.randn(1, 64, 576, dtype=torch.bfloat16, device="npu")
    # Query's latent coordinates must match the exported signed-Hadamard K.
    query[:, :, :512].copy_(store.forward(query[:, :, :512]))
    indices = torch.arange(rows, device="npu", dtype=torch.int32).view(1, 1, rows)
    common = dict(
        block_table=torch.arange(rows // 256, dtype=torch.int32, device="npu").view(1, -1),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device="npu"),
        actual_seq_lengths_kv=torch.tensor([rows], dtype=torch.int32, device="npu"),
        scale_value=576**-0.5,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        key_quant_mode=3,
        value_quant_mode=3,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    actual = torch_npu.npu_kv_quant_sparse_flash_attention(query, cache, cache, indices, **common)
    actual = actual[0] if isinstance(actual, tuple) else actual
    assert torch.isfinite(actual).all()
    # Independently decode the byte ABI and compare attention in its rotated
    # coordinates, with a valid, unique sparse index for every key.
    codes = torch.stack((packed[:, :256] & 15, packed[:, :256] >> 4), -1).flatten(1).long()
    scale = packed[:, 384:].contiguous().view(torch.float16).float()
    decoded_nope = (store.centroids.bfloat16().float()[codes] * scale).bfloat16()
    decoded_rope = (packed[:, 256:384].contiguous().view(torch.bfloat16).float() * scale).bfloat16()
    selected = indices[0, 0].long()
    keys = torch.cat((decoded_nope, decoded_rope), -1)[selected].float()
    expected = (query[0].float() @ keys.T * 576**-0.5).softmax(-1) @ decoded_nope[selected].float()
    error = (actual.float().reshape_as(expected) - expected).norm() / expected.norm()
    assert error.item() < 0.03


@torch.inference_mode()
def test_colocated_decode_update_skips_graph_padding_and_preserves_bf16():
    from vllm_ascend.quantization.methods.kv_cache.turboquant.final_prefill import (
        BF16_SLOT_BYTES,
        TQ_SLOT_BYTES,
        ColocatedFinalPrefill,
        colocated_cache_views,
    )

    init_device_properties_triton()
    raw = torch.zeros(2 * 128 * (BF16_SLOT_BYTES + TQ_SLOT_BYTES), dtype=torch.int8, device="npu")
    cache = colocated_cache_views(raw, 2, 128)
    cache[0].normal_()
    cache[1].normal_()
    before = [t.cpu() for t in cache]
    state = ColocatedFinalPrefill(TurboQuantLatent(fused_finalize=True))
    state.bind(cache)
    slots = torch.tensor([3, 129, -1, -1], device="npu", dtype=torch.int64)
    state.update_decode_tokens(slots)
    expected = pack_bf16_kv(cache[0].flatten(0, 2)[[3, 129]], cache[1].flatten(0, 2)[[3, 129]], state.quantizer)
    actual = cache[2].cpu().view(-1, TQ_SLOT_BYTES)
    torch.testing.assert_close(actual[[3, 129]], expected.cpu().view(torch.int8), rtol=0, atol=0)
    actual[[3, 129]] = 0
    assert not actual.count_nonzero()
    for original, current in zip(before[:2], cache[:2]):
        torch.testing.assert_close(original, current.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 17, 257, 4096])
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@torch.inference_mode()
def test_direct_store_matches_existing_abi_and_preserves_source(rows, dtype):
    init_device_properties_triton()
    torch.manual_seed(171)
    nope = torch.randn(rows, 512, dtype=torch.bfloat16, device="npu")
    rope = torch.randn(rows, 64, dtype=torch.bfloat16, device="npu")
    before = [x.cpu() for x in (nope, rope)]
    quantizer = TurboQuantLatent(fused_finalize=True)
    expected = pack_bf16_kv(nope, rope, quantizer).view(dtype)
    target = torch.full((rows * 2 + 1, 386), 37, dtype=dtype, device="npu")
    slots = torch.arange(rows, device="npu", dtype=torch.int64) * 2
    for _ in range(4):
        quantizer.compress_kv_into(nope, rope, slots, target)
        torch.testing.assert_close(target[slots].cpu(), expected.cpu(), rtol=0, atol=0)
    assert torch.all(target[1::2] == 37)
    assert torch.all(target[-1] == 37)
    for a, b in zip(before, (nope, rope)):
        torch.testing.assert_close(a, b.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_incremental_side_stream_matches_serial_cache_across_steps():
    from vllm_ascend.quantization.methods.kv_cache.turboquant.final_prefill import (
        BF16_SLOT_BYTES,
        TQ_SLOT_BYTES,
        ColocatedFinalPrefill,
        PrefillQuantizationPipeline,
        colocated_cache_views,
    )
    init_device_properties_triton()
    device = torch.device("npu:0")
    pipe = PrefillQuantizationPipeline(device)
    states = []
    for _ in range(3):
        cache = colocated_cache_views(
            torch.zeros(4 * 128 * (BF16_SLOT_BYTES + TQ_SLOT_BYTES), device=device, dtype=torch.int8), 4, 128)
        state = ColocatedFinalPrefill(TurboQuantLatent(fused_finalize=True))
        state.quantizer._initialize(device)
        state.bind(cache)
        states.append(state)
    expected_steps = [[] for _ in states]
    for start, end in ((0, 129), (129, 257), (257, 511)):
        slots = torch.arange(start, end, device=device)
        bf16_before = []
        for state in states:
            state.cache[0].view(-1, 512)[start:end].normal_()
            state.cache[1].view(-1, 64)[start:end].normal_()
            bf16_before.append(tuple(t.clone() for t in state.cache[:2]))
            state.prefill_slots = slots
            pipe.submit(state)
            # Exercise unrelated main-stream vector/cube work while converting.
            matrix = torch.randn(256, 256, device=device)
            matrix = matrix @ matrix
        pipe.join()
        assert not pipe.pending
        # The next step's main stream can immediately consume completed slots.
        for index, state in enumerate(states):
            nope, rope, target = state.cache
            for before, after in zip(bf16_before[index], (nope, rope)):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            # Keep the serial reference's GEMM row count identical. Changing
            # the batch size can change FP32 reduction rounding at a bin edge.
            expected_steps[index].append(
                pack_bf16_kv(nope.view(-1, 512)[start:end],
                             rope.view(-1, 64)[start:end], state.quantizer))
            expected = torch.cat(expected_steps[index])
            torch.testing.assert_close(target.view(-1, 386)[:end].cpu(),
                                       expected.cpu().view(torch.int8),
                                       rtol=0,
                                       atol=0)
            assert not target.view(-1, 386)[end:].count_nonzero()


@torch.inference_mode()
def test_tq_attention_clears_unwritten_query_shards():
    from types import SimpleNamespace
    from unittest.mock import patch
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    query = torch.ones(4, 2, 576, device="npu", dtype=torch.bfloat16)
    impl = SimpleNamespace(scale=256**-0.5, qk_rope_head_dim=64)
    for count in (0, 1, 4):
        dirty = torch.full((4, 2, 512), float("nan"), device="npu", dtype=torch.bfloat16)
        dirty[:count].fill_(3)
        lengths = torch.tensor([count], dtype=torch.int32, device="npu")
        with patch("torch_npu.npu_kv_quant_sparse_flash_attention", return_value=dirty):
            output, _, _ = AscendSFAImpl._turboquant_sfa(
                impl,
                query,
                None,
                None,
                None,
                lengths,
                lengths,
                3,
            )
        assert torch.isfinite(output).all()
        assert torch.all(output[:count] == 3)
        assert not output[count:].count_nonzero()
