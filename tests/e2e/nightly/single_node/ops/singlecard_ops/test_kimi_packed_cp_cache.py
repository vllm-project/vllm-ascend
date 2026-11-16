# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real operators must read and write the planner's hole-spanning views."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch_npu
from fla_npu.ops.ascendc import causal_conv1d_update
from vllm.v1.kv_cache_interface import MambaSpec

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_kv_quant_sparse_flash_attention_vllm import (
    _make_inputs,
    _run_custom_op,
)
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.kimi_k3.packed_sfa_cache import map_packed_sfa_metadata, packed_sfa_cache_views
from vllm_ascend.ops.kda import run_recurrent_kda
from vllm_ascend.ops.triton.pcp_kv_cache import copy_pcp_kv_cache
from vllm_ascend.utils import enable_custom_op


@dataclass
class CacheAddresses:
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor


def map_addresses(spec, row_bytes, table, slots):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=slots.numel()),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
        speculative_config=None,
        model_config=SimpleNamespace(max_model_len=4096),
    )
    return map_packed_sfa_metadata(SimpleNamespace(vllm_config=config), CacheAddresses(table, slots), spec, row_bytes)


@torch.inference_mode()
def test_packed_main_store_pcp_read_and_sparse_attention():
    torch_npu.npu.set_device(0)
    assert enable_custom_op()
    inputs = _make_inputs()
    baseline = _run_custom_op(inputs)[0]
    source = inputs["key"].reshape(-1, 656)
    spec = AscendMLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=656,
        dtype=torch.int8,
        cache_sparse_sfa_c8=True,
        is_block_outermost=True,
    )
    row_bytes = 4 * 128 * 656
    rows = source.shape[0] // 256 + 1
    backing = torch.full((rows * row_bytes,), -7, dtype=torch.int8, device="npu")
    (cache,) = packed_sfa_cache_views(backing, spec, rows, row_bytes)
    logical_table = torch.arange(2, rows * 2, dtype=torch.int32, device="npu").view(1, -1)
    logical_slots = torch.arange(256, rows * 256, dtype=torch.int32, device="npu")
    mapped = map_addresses(spec, row_bytes, logical_table, logical_slots)
    impl = SimpleNamespace(
        pcp_enabled=False,
        enable_sparse_sfa_c8=True,
        enable_sparse_sfa_turboquant=False,
        sfa_qsfa_packed_kv_head_dim=656,
    )
    nope, rope, scales = source.split([512, 128, 16], dim=-1)
    AscendSFAImpl._store_packed_kv_cache(impl, rope, nope, scales, (cache,), mapped.slot_mapping, SimpleNamespace())
    torch.testing.assert_close(copy_pcp_kv_cache((cache,), mapped.slot_mapping).cpu(), source.cpu(), rtol=0, atol=0)
    # Read the same history via the actual sparse-attention operator.
    inputs.update(key=cache, value=cache, block_table=mapped.block_table_tensor)
    actual = _run_custom_op(inputs)[0]
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), baseline.cpu(), rtol=0, atol=0)
    assert bool((backing[:row_bytes] == -7).all())  # null row was not written


@torch.inference_mode()
def test_packed_indexer_store_and_topk():
    torch_npu.npu.set_device(0)
    assert enable_custom_op()
    torch.manual_seed(21849)
    tokens, heads = 768, 64
    key = torch.randint(-32, 32, (tokens, 1, 128), dtype=torch.int8, device="npu")
    scales = torch.rand(tokens, 1, dtype=torch.float16, device="npu") + 0.5
    query = torch.randint(-32, 32, (2, heads, 128), dtype=torch.int8, device="npu")
    weights = torch.rand(2, heads, dtype=torch.float16, device="npu")
    query_scales = torch.ones(2, heads, dtype=torch.float16, device="npu")
    spec = AscendSFAIndexerCacheSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
        is_block_outermost=True,
    )
    row_bytes = 4 * 128 * 130
    backing = torch.full((4 * row_bytes,), -7, dtype=torch.int8, device="npu")
    cache, scale_cache = packed_sfa_cache_views(backing, spec, 4, row_bytes)
    ids = torch.tensor([3, 1, 2], dtype=torch.int32, device="npu")
    table = (ids[:, None] * 2 + torch.arange(2, dtype=torch.int32, device="npu")).view(1, -1)
    slots = (ids[:, None] * 256 + torch.arange(256, dtype=torch.int32, device="npu")).flatten()
    mapped = map_addresses(spec, row_bytes, table, slots)
    scatter = torch.stack((mapped.slot_mapping // 128, mapped.slot_mapping % 128), dim=-1)
    backend = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=(cache, scale_cache)),
        enable_sparse_li_quant=True,
        _write_cache_tensor=AscendSFAIndexerBackend._write_cache_tensor,
    )
    AscendSFAIndexerBackend.write_cache(backend, key, scales, mapped.slot_mapping, scatter)
    pages, offsets = scatter.unbind(-1)
    torch.testing.assert_close(cache[pages.long(), offsets.long()].cpu(), key.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(scale_cache[pages.long(), offsets.long(), 0].cpu(), scales.cpu(), rtol=0, atol=0)

    def select(k, s, block_table):
        output = torch_npu.npu_quant_lightning_indexer(
            query,
            k,
            weights,
            query_dequant_scale=query_scales,
            key_dequant_scale=s.squeeze(2),
            query_quant_mode=0,
            key_quant_mode=0,
            actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32, device="npu"),
            actual_seq_lengths_key=torch.tensor([tokens], dtype=torch.int32, device="npu"),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
        return output[0] if isinstance(output, (tuple, list)) else output

    baseline = select(
        key.view(6, 128, 1, 128),
        scales.view(6, 128, 1, 1),
        torch.arange(6, dtype=torch.int32, device="npu").view(1, -1),
    )
    actual = select(cache, scale_cache, mapped.block_table_tensor)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), baseline.cpu(), rtol=0, atol=0)
    assert bool((backing[:row_bytes] == -7).all())


@torch.inference_mode()
def test_packed_kda_conv_and_recurrent_use_row_strides():
    torch_npu.npu.set_device(0)
    torch.manual_seed(21849)
    heads, dim, rows = 4, 128, 4
    channels = 3 * heads * dim
    spec = MambaSpec(block_size=256, shapes=((3, channels), (heads, dim, dim)), dtypes=(torch.bfloat16, torch.float32))
    row_bytes = 2 * spec.page_size_bytes
    backing = torch.zeros(rows * row_bytes, dtype=torch.int8, device="npu")
    conv, state = packed_sfa_cache_views(backing, spec, rows, row_bytes)
    conv.copy_(torch.randn_like(conv))
    state.copy_(torch.randn_like(state) * 0.01)
    dense_conv, dense_state = conv.clone(), state.clone()
    indices = torch.tensor([3, 1], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
    x = torch.randn(2, channels, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(4, channels, dtype=x.dtype, device="npu")

    def convolve(cache):
        return causal_conv1d_update(
            x,
            cache,
            weight,
            activation="silu",
            conv_state_indices=indices,
            query_start_loc=starts,
            max_query_len=1,
            null_block_id=0,
            out=torch.empty_like(x),
        )

    expected_conv = convolve(dense_conv)
    actual_conv = convolve(conv)
    torch.testing.assert_close(actual_conv.cpu(), expected_conv.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(conv.cpu(), dense_conv.cpu(), rtol=0, atol=0)
    q, k, v, gate = (torch.randn(1, 2, heads, dim, dtype=x.dtype, device="npu") for _ in range(4))
    beta = torch.rand(1, 2, heads, device="npu")
    a_log = torch.zeros(heads, device="npu")
    bias = torch.zeros(heads * dim, device="npu")

    def recurrent(cache):
        return run_recurrent_kda(q, k, v, gate, beta, cache, starts, indices, a_log, bias, lower_bound=-5.0)

    expected = recurrent(dense_state)
    actual = recurrent(state)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(state.cpu(), dense_state.cpu(), rtol=0, atol=0)
