# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch with torchrun (TP2/4/8) and pytest --noconftest."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("vllm.v1.attention.ops.shardq")

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.parallel_state import destroy_distributed_environment  # noqa: E402
from vllm.v1.attention.ops.shardq import ShardQMetadata, shard_q, unshard_out, zigzag_split  # noqa: E402

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config  # noqa: E402
from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata  # noqa: E402
from vllm_ascend.attention.sfa_shardq import build_sfa_shardq_metadata, select_shardq_topk  # noqa: E402
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata  # noqa: E402
from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402
from vllm_ascend.utils import enable_custom_op  # noqa: E402

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(WORLD not in (2, 4, 8), reason="requires isolated TP2/4/8")


@pytest.fixture(scope="module")
def rank():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.npu.set_device(local_rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=WORLD))
    with set_current_vllm_config(config):
        init_ascend_config(config)
        try:
            init_distributed_environment(
                world_size=WORLD, rank=rank, local_rank=local_rank, distributed_init_method="env://", backend="hccl"
            )
            initialize_model_parallel(tensor_model_parallel_size=WORLD, backend="hccl")
            yield rank
            torch.npu.synchronize()
        finally:
            # Failed initialization and failed kernel assertions also own groups.
            destroy_model_parallel()
            destroy_distributed_environment()
            clear_ascend_config()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_upstream_exchange_restores_uneven_tokens_and_heads(rank, dtype):
    token_ids, counts = zigzag_split(np.array([7, 12]), WORLD)
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    metadata = ShardQMetadata(
        num_decode_tokens=1,
        num_tokens_per_rank=counts.tolist(),
        token_ids=torch.tensor(token_ids, device=device),
        rank_offsets=torch.tensor(np.r_[0, np.cumsum(counts)], device=device),
    )
    full_query = torch.arange(20 * WORLD * 2 * 6).view(20, WORLD * 2, 6).to(dtype).to(device)
    local_query = full_query[:, 2 * rank : 2 * rank + 2].contiguous()
    decode, prefill = shard_q(local_query.split((4, 2), dim=-1), metadata)
    start = int(counts[:rank].sum())
    rows = torch.tensor(token_ids[start : start + counts[rank]] + 1, device=device)
    torch.testing.assert_close(prefill, full_query[rows], rtol=0, atol=0)
    result = unshard_out(prefill[..., :4].contiguous(), metadata, decode[..., :4].contiguous())
    torch.testing.assert_close(result, local_query[..., :4], rtol=0, atol=0)


@pytest.mark.parametrize("lens", [(8,), (1, 33, 19)])
def test_lightning_indexer_and_sfa_match_head_sharded_tp(rank, lens):
    assert enable_custom_op(), "SFA requires the source revision's native operators"
    torch.manual_seed(42)
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    block_size, heads, latent, rope, index_heads, index_dim = 128, 128, 512, 64, 64, 128
    tokens = sum(lens)
    starts = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    seq_lens = torch.tensor([2112 + length for length in lens], dtype=torch.int32)
    blocks_per_request = (int(seq_lens.max()) + block_size - 1) // block_size
    block_table = torch.arange(len(lens) * blocks_per_request, dtype=torch.int32).view(len(lens), -1).to(device)
    metadata = AscendSFAMetadata(
        num_actual_tokens=tokens,
        num_input_tokens=tokens,
        slot_mapping=torch.arange(tokens, device=device),
        seq_lens=seq_lens.to(device),
        seq_lens_cpu=seq_lens,
        cum_query_lens=starts[1:].to(device),
        query_start_loc=starts.to(device),
        block_table=block_table,
        positions=torch.cat([torch.arange(2112, 2112 + length) for length in lens]).to(device),
        sin=None,
        cos=None,
    )
    common = SimpleNamespace(
        num_actual_tokens=tokens,
        num_reqs=len(lens),
        max_query_len=max(lens),
        query_start_loc_cpu=starts,
        query_start_loc=metadata.query_start_loc,
        seq_lens_cpu_upper_bound=seq_lens,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(sparse_mla_shard_q_min_tokens=1),
        speculative_config=None,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
    )
    shardq = build_sfa_shardq_metadata(common, metadata, config)
    assert shardq is not None
    num_blocks = block_table.numel()
    cache = tuple(
        torch.randn(num_blocks, block_size, 1, dim, dtype=torch.bfloat16).to(device) for dim in (latent, rope)
    )
    index_cache = (torch.randn(num_blocks, block_size, 1, index_dim, dtype=torch.bfloat16).to(device),)
    index_query = torch.randn(tokens, index_heads, index_dim, dtype=torch.bfloat16).to(device)
    weights = torch.randn(tokens, index_heads, dtype=torch.bfloat16).to(device)
    indexer = SimpleNamespace(
        n_head=index_heads,
        head_dim=index_dim,
        enable_sparse_li_quant=False,
        enable_sparse_li_c8=False,
        enable_sparse_li_c4=False,
        use_torch_npu_lightning_indexer=False,
        k_cache=SimpleNamespace(kv_cache=index_cache),
    )
    index_metadata = AscendSFAIndexerMetadata(
        num_actual_tokens=tokens,
        slot_mapping=metadata.slot_mapping,
        seq_lens=metadata.seq_lens,
        cum_query_lens=metadata.cum_query_lens,
        block_table=block_table,
        sin=None,
        cos=None,
        block_size=block_size,
    )
    full_topk = DeviceOperator.indexer_select_post_process(
        index_query,
        None,
        None,
        weights,
        index_cache,
        0,
        1,
        index_metadata,
        metadata.cum_query_lens,
        metadata.seq_lens,
        False,
        False,
        False,
    )
    # Repeated block-table rows must preserve the unsplit request's causal
    # Top-K, including when the batch retains a head-sharded decode prefix.
    rows = shardq.query_indices
    local_topk = select_shardq_topk(index_query[rows], weights[rows], indexer, index_metadata, None, shardq=shardq)
    torch.testing.assert_close(local_topk.sort(-1).values, full_topk[rows].sort(-1).values, rtol=0, atol=0)

    full_query = torch.randn(tokens, heads, latent + rope, dtype=torch.bfloat16)
    query = full_query[:, rank * (heads // WORLD) : (rank + 1) * (heads // WORLD)].contiguous().to(device)
    q_nope, q_pe = query.split((latent, rope), dim=-1)
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.use_shardq = True
    impl.kv_lora_rank = latent
    impl.qk_rope_head_dim = rope
    impl.enable_sparse_sfa_turboquant = False
    impl.scale = (latent + rope) ** -0.5
    expected = impl._execute_sparse_flash_attention_process(
        q_nope, q_pe, cache, full_topk, metadata, metadata.cum_query_lens, metadata.seq_lens
    )
    metadata.shardq = shardq
    actual = impl._execute_sparse_flash_attention_process(
        q_nope, q_pe, cache, local_topk, metadata, metadata.cum_query_lens, metadata.seq_lens
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
