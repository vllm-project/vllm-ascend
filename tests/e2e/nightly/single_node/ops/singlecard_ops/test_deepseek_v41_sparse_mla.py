# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Aurora's slot-backed BF16 cache with the prior block-outermost views."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from tests.deepseek_v41_cache_utils import allocate_cache_views, make_cache_config
from vllm_ascend.attention.dsa_v41 import DeepseekV41EagerAttentionImpl
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

WIDTH = 512
HEADS = 32
BLOCK = 128
OLD_BLOCK_STRIDE = 393216


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("query_lens", [(1,), (3,), (1, 3)])
def test_slot_backed_attention_matches_block_outermost(ratio, query_lens):
    torch.manual_seed(41)
    config = make_cache_config(13)
    specs = {n: s for g in config.kv_cache_groups for n, s in g.kv_cache_spec.kv_cache_specs.items()}
    _, candidate = allocate_cache_views(config, "npu")
    # Original token lengths include odd compression tails and page boundaries.
    lengths = [257, 383][: len(query_lens)]
    ori_ids = [[3, 1, 5], [2, 4, 6]][: len(lengths)]
    cmp_ids = [[9, 7, 11], [8, 10, 12]][: len(lengths)]
    swa_name = "model.layers.3.self_attn.swa_cache"  # Padded fourth slot.
    cmp_name = f"model.layers.{2 if ratio == 2 else 20}.self_attn.long_kv_cache"
    names = [swa_name] + ([cmp_name] if ratio else [])
    baseline = {}
    for name in names:
        spec = specs[name]
        raw = torch.zeros(config.num_blocks * OLD_BLOCK_STRIDE, dtype=torch.uint8, device="npu")
        baseline[name] = raw.view(spec.dtype).as_strided(
            (config.num_blocks, spec.storage_block_size, 1, WIDTH),
            (OLD_BLOCK_STRIDE // spec.dtype.itemsize, WIDTH, WIDTH, 1),
        )
        ids = ori_ids if name == swa_name else cmp_ids
        cache_lengths = lengths if name == swa_name else [n // ratio for n in lengths]
        for physical_ids, count in zip(ids, cache_lengths):
            data = torch.randn(count, 1, WIDTH, dtype=torch.bfloat16, device="npu")
            for logical, physical in enumerate(physical_ids):
                start = logical * spec.storage_block_size
                end = min(start + spec.storage_block_size, count)
                if end > start:
                    baseline[name][physical, : end - start].copy_(data[start:end])
                    candidate[name][physical, : end - start].copy_(data[start:end])

    query = torch.randn(sum(query_lens), HEADS, WIDTH, dtype=torch.bfloat16, device="npu")
    cu = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device="npu")
    cmp_lens = seq_lens // ratio if ratio else None
    metadata = SimpleNamespace(
        swa=SimpleNamespace(
            num_reqs=len(lengths),
            query_start_loc=cu,
            seq_lens=seq_lens,
            block_table=torch.tensor(ori_ids, dtype=torch.int32, device="npu"),
            max_query_len=max(query_lens),
            max_seq_len=max(lengths),
        ),
        attention=SimpleNamespace(
            block_table=torch.tensor(cmp_ids, dtype=torch.int32, device="npu"),
            cache_seq_lens=cmp_lens,
            max_cache_seq_len=max(lengths) // ratio,
        )
        if ratio
        else None,
    )
    indices = None
    if ratio:
        indices = torch.full((sum(query_lens), 512), -1, dtype=torch.int32, device="npu")
        row = 0
        for length, qlen in zip(lengths, query_lens):
            for position in range(length - qlen, length):
                visible = (position + 1) // ratio
                indices[row, :visible] = torch.arange(visible, dtype=torch.int32, device="npu")
                row += 1
    impl = DeepseekV41EagerAttentionImpl.__new__(DeepseekV41EagerAttentionImpl)
    impl.role = SimpleNamespace(compress_ratio=ratio)
    impl.topology = SimpleNamespace(index_topk=512)
    outputs = []
    for caches in (baseline, candidate):
        attn = SimpleNamespace(
            head_dim=WIDTH,
            window_size=128,
            n_local_heads=HEADS,
            shared_state=SimpleNamespace(smla_metadata={}),
            dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=[caches[swa_name]])),
            attn_sink=torch.zeros(HEADS, dtype=torch.float32, device="npu"),
            softmax_scale=WIDTH**-0.5,
        )
        outputs.append(
            impl._native_attention(
                attn,
                query,
                metadata,
                source_cache=caches[cmp_name] if ratio else None,
                compressed_indices=indices,
            ).cpu()
        )
    assert torch.isfinite(outputs[0]).all() and torch.isfinite(outputs[1]).all()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
