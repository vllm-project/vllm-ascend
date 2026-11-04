# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check HiSparse destinations against the existing SFA preprocessing ops."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu


def _setup():
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse integration currently targets Ascend 950")
    torch.npu.set_device(0)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_preprocess_writes_hisparse_mapping_once(cache_dtype, capture):
    _setup()
    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
    from vllm_ascend.attention.sfa_v1 import custom_kv_rmsnorm_rope

    rows, block = 5, 128
    c8 = cache_dtype == torch.float8_e4m3fn
    width = 656 if c8 else 576
    torch.manual_seed(71)
    values = torch.randn(rows, 2624, dtype=torch.bfloat16, device="npu")[:, 2048:]
    gamma = torch.ones(512, dtype=torch.bfloat16, device="npu")
    cos = torch.ones(rows, 1, 1, 64, dtype=torch.bfloat16, device="npu")
    sin = torch.zeros_like(cos)
    raw = torch.full((3, 2, block, width * cache_dtype.itemsize), 90, dtype=torch.uint8, device="npu")
    destination = raw[:, 1].view(cache_dtype)
    logical = [-1, 129, 0, 257, 128]
    slots = torch.tensor(logical, dtype=torch.int64, device="npu")
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device="npu"), hisparse_for_capture=True
    )
    scratch = (
        torch.empty(1, block, 1, 512, dtype=values.dtype, device="npu"),
        torch.empty(1, block, 1, 64, dtype=values.dtype, device="npu"),
        torch.arange(block, dtype=torch.int64, device="npu"),
    )
    impl = SimpleNamespace(
        enable_sparse_sfa_c8=c8,
        c8_cache_dtype=torch.float8_e4m3fn,
        sfa_qsfa_tile_size=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_kv_heads=1,
        _kv_staging_capacity=block,
        _kv_workspace=lambda dtype: scratch,
        kv_a_layernorm=SimpleNamespace(weight=gamma, variance_epsilon=1e-5),
        hisparse_cache=SimpleNamespace(write_target=lambda count, slot_count: (destination, slots, count)),
    )
    # The ordinary source cache has no Resident/Hot addressing. It must never
    # receive the parent's second C8 store after exec_kv has used write_target.
    source_cache = (torch.empty(0, device="npu"),)

    def forward():
        rope, latent, *scales = AscendHiSparseImpl.exec_kv(impl, values, cos, sin, source_cache, slots, metadata)
        scales = scales[0] if scales else None
        AscendHiSparseImpl._store_parallel_kv(impl, rope, latent, scales, source_cache, slots, metadata)

    graph = None
    try:
        forward()
        torch.npu.synchronize()
        if capture:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                forward()
        for count in (4, 2):
            values.neg_()
            metadata.query_start_loc[-1:].fill_(count)
            raw.fill_(90)
            if graph is None:
                forward()
            else:
                graph.replay()
            torch.npu.synchronize()
            if c8:
                rope, latent, scales = custom_kv_rmsnorm_rope(
                    values.reshape(rows, 1, 1, 576), gamma, cos, sin, 512, 64, tile_size=128
                )
                components = (latent, rope, scales)
            else:
                # Reuse native SFA as the oracle; test only the new destination,
                # strides, row mapping and padding, not RMSNorm arithmetic.
                reference_k, reference_r = torch.empty_like(scratch[0]), torch.empty_like(scratch[1])
                torch_npu.npu_kv_rmsnorm_rope_cache(
                    values.view(rows, 1, 1, 576),
                    gamma,
                    cos,
                    sin,
                    scratch[2][:rows],
                    reference_r,
                    reference_k,
                    epsilon=1e-5,
                    cache_mode="PA",
                )
                components = (reference_k.view(-1, 512)[:rows], reference_r.view(-1, 64)[:rows])
            payload = torch.cat(
                [part.contiguous().view(torch.uint8).reshape(rows, -1).cpu() for part in components], -1
            )
            expected = torch.full(raw.shape, 90, dtype=torch.uint8)
            for row, slot in enumerate(logical):
                if row < count and slot >= 0:
                    expected[slot // block, 1, slot % block] = payload[row]
            torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
