# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re
import torch
import torch_npu  # noqa: F401

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.glm5next.cache_views import view_compressed_indexer_cache
from vllm_ascend.ops.triton.glm5_next_cann_keypool import write_keypool_caches

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3")


@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_int8_key_scale_write_preserves_padding_and_invalid_slots(use_graph):
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        model_version="glm5_next",
        tokens_per_state=4,
        indexes_kv_by_block_stride=True,
        page_size_padded=8192,
    )
    raw = torch.full((2 * spec.page_size_bytes,), 77, dtype=torch.uint8, device="npu")
    expected_raw = raw.clone()
    backend = SimpleNamespace(get_kv_cache_shape=lambda n, b, h, d: (n, b, h, d))
    cache, scales = view_compressed_indexer_cache("test", spec, raw, backend, 128)
    expected_cache, expected_scales = view_compressed_indexer_cache("expected", spec, expected_raw, backend, 128)
    pooled = torch.arange(6 * 128, device="npu").reshape(6, 128).remainder(127).to(torch.int8)
    pooled[1].zero_()
    pooled_scales = torch.tensor([1, 0.125, 0.25, 0.5, 2, 4], dtype=torch.float16, device="npu")
    slots = [-1, 0, 31, 32, 63, 64]
    metadata = SimpleNamespace(pooled_slot_mapping=torch.tensor(slots, device="npu"))
    inputs = SimpleNamespace(
        write_indices=torch.empty((0, 1), dtype=torch.int64, device="npu"),
        state_indices=torch.empty((0, 1), dtype=torch.int64, device="npu"),
        page_rows=4,
    )
    tail = torch.zeros((2, 4, 256), device="npu")
    state = torch.zeros((1, 4, 256), device="npu")

    def run():
        write_keypool_caches(
            tail,
            cache,
            None,
            None,
            pooled,
            metadata,
            SimpleNamespace(cann_inputs=inputs),
            state,
            scale_cache=scales,
            pooled_scales=pooled_scales,
        )

    run()
    if use_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        graph.replay()
    torch.npu.synchronize()
    for row, slot in enumerate(slots):
        if 0 <= slot < 64:
            expected_cache[slot // 32, slot % 32, 0].copy_(pooled[row])
            expected_scales[slot // 32, slot % 32, 0, 0].copy_(pooled_scales[row])
    # Whole-page comparison includes the interleaved scales and untouched padding.
    torch.testing.assert_close(raw.cpu(), expected_raw.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_int8_operator_reads_managed_pages(use_graph):
    ops_transformer = pytest.importorskip("cann_ops_transformer")
    # The CI-bundled ops package follows its versioned CANN install directory.
    ops_transformer_file = getattr(ops_transformer, "__file__", None)
    if ops_transformer_file is None:
        pytest.skip("Cannot determine ops-transformer installation version")
    version_match = re.search(r"cann-(\d+)\.(\d+)\.(\d+)", str(Path(ops_transformer_file).resolve()))
    if version_match is None:
        pytest.skip("Cannot determine ops-transformer installation version")
    if tuple(map(int, version_match.groups())) < (9, 3, 0):
        pytest.skip("INT8 PoolKeyIndexer requires ops-transformer 9.3.0 or later")
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        model_version="glm5_next",
        tokens_per_state=4,
        indexes_kv_by_block_stride=True,
        page_size_padded=8192,
    )
    raw = torch.zeros(20 * spec.page_size_bytes, dtype=torch.uint8, device="npu")
    backend = SimpleNamespace(get_kv_cache_shape=lambda n, b, h, d: (n, b, h, d))
    keys, scales = view_compressed_indexer_cache("operator", spec, raw, backend, 128)
    torch.manual_seed(1024)
    keys.copy_(torch.randint(-127, 128, keys.shape, device="npu", dtype=torch.int8))
    scales.copy_(torch.rand(scales.shape, device="npu") * 0.01 + 0.001)
    scales = scales.squeeze(-1)
    query = torch.randint(-127, 128, (2, 32, 128), device="npu", dtype=torch.int8)
    weights = torch.rand(2, 32, device="npu", dtype=torch.float16)
    kwargs = dict(
        actual_seq_q=torch.tensor([1, 2], device="npu", dtype=torch.int32),
        actual_seq_k=torch.tensor([0, 640], device="npu", dtype=torch.int32),
        block_table=torch.arange(20, device="npu", dtype=torch.int32).repeat(2, 1),
        q_descale=torch.full((2, 32), 0.01, device="npu", dtype=torch.float16),
        layout_q="TND",
        layout_k="PA_BBND",
        topk=2048,
        pool_size=4,
        mask_mode=3,
        quant_mode=2,
    )
    tail = torch.tensor([3, 0], device="npu", dtype=torch.int32)

    def run(k, s):
        return torch.ops.cann_ops_transformer.pool_key_indexer(query, k, weights, tail, k_descale=s, **kwargs)[0]

    expected = run(keys.contiguous(), scales.contiguous())
    actual = run(keys, scales)
    if use_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run(keys, scales)
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
