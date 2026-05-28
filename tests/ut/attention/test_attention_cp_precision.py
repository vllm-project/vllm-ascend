from functools import partial
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import set_random_seed

from tests.ut.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
    patch_distributed_groups,
)
from tests.ut.conftest import npu_test
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.attention.context_parallel.attention_cp import (
    AscendAttentionCPImpl,
)
from vllm_ascend.attention.context_parallel.common_cp import (
    AscendMetadataForDecode,
    AscendMetadataForPrefill,
    AscendPCPMetadata,
)

BATCH_SPECS = {
    "single_prefill": BatchSpec(seq_lens=[128], query_lens=[128]),
    "small_prefill": BatchSpec(seq_lens=[32, 48], query_lens=[32, 48]),
    "medium_prefill": BatchSpec(seq_lens=[256, 512], query_lens=[256, 512]),
    "large_prefill": BatchSpec(seq_lens=[1024, 2048], query_lens=[1024, 2048]),
    "single_decode": BatchSpec(seq_lens=[32], query_lens=[1]),
    "small_decode": BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "medium_decode": BatchSpec(seq_lens=[128, 256, 512, 1024], query_lens=[1, 1, 1, 1]),
    "mixed_small": BatchSpec(seq_lens=[32, 40, 5, 5], query_lens=[1, 1, 5, 5]),
    "mixed_medium": BatchSpec(seq_lens=[256, 512, 7, 7], query_lens=[1, 1, 7, 7]),
    "mixed_large": BatchSpec(seq_lens=[1024, 2048, 16, 16], query_lens=[1, 1, 16, 16]),
    "mtp_1_plus_3_small": BatchSpec(seq_lens=[128, 256, 512, 1024], query_lens=[4, 4, 4, 4]),
    "mtp_1_plus_3_medium": BatchSpec(seq_lens=[1024, 2048, 3072, 4096], query_lens=[4, 4, 4, 4]),
    "mtp_1_plus_3_tiny": BatchSpec(seq_lens=[64, 128], query_lens=[4, 4]),
}

MODELS = [
    "Qwen/Qwen3-8B",
]


class MockAttentionLayer:
    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self.layer_name = "model.layers.0"


def compute_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_spec: BatchSpec,
    scale: float,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute reference attention output using PyTorch SDPA.

    Iterates over each sequence in the batch and computes causal attention
    matching the FIA sparse_mode=3 behavior with splitfuse layout.
    Used for pure prefill tests where seq_lens == query_lens.
    """
    enable_gqa = num_q_heads != num_kv_heads
    all_sdpa_outputs = []
    q_offset = 0
    kv_offset = 0

    for i in range(batch_spec.batch_size):
        q_len = batch_spec.query_lens[i]
        kv_len = batch_spec.seq_lens[i]

        q_i = q[q_offset : q_offset + q_len]
        k_i = k[kv_offset : kv_offset + kv_len]
        v_i = v[kv_offset : kv_offset + kv_len]

        q_sdpa = q_i.unsqueeze(0).transpose(1, 2)
        k_sdpa = k_i.unsqueeze(0).transpose(1, 2)
        v_sdpa = v_i.unsqueeze(0).transpose(1, 2)

        context_len = kv_len - q_len
        if context_len > 0:
            attn_mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device)
            causal_mask = torch.tril(torch.ones(q_len, q_len, device=q.device))
            attn_mask[:, context_len:] = causal_mask
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                is_causal=False,
                enable_gqa=enable_gqa,
                scale=scale,
            )
        else:
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                is_causal=True,
                enable_gqa=enable_gqa,
                scale=scale,
            )

        all_sdpa_outputs.append(sdpa_out.transpose(1, 2).squeeze(0))
        q_offset += q_len
        kv_offset += kv_len

    return torch.cat(all_sdpa_outputs, dim=0)


def compute_mixed_sdpa_reference(
    full_q: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    batch_spec: BatchSpec,
    scale: float,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute per-sequence SDPA reference for mixed decode+prefill.

    Each sequence gets its own Q/K/V with causal masking
    (context tokens are visible to new tokens).
    """
    enable_gqa = num_q_heads != num_kv_heads
    all_outputs = []
    q_offset = 0
    kv_offset = 0

    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q_i = full_q[q_offset : q_offset + q_len]
        k_i = full_k[kv_offset : kv_offset + s_len]
        v_i = full_v[kv_offset : kv_offset + s_len]

        q_sdpa = q_i.unsqueeze(0).transpose(1, 2)
        k_sdpa = k_i.unsqueeze(0).transpose(1, 2)
        v_sdpa = v_i.unsqueeze(0).transpose(1, 2)

        if context_len > 0:
            attn_mask = torch.ones(q_len, s_len, dtype=torch.bool, device=full_q.device)
            causal_mask = torch.tril(torch.ones(q_len, q_len, device=full_q.device))
            attn_mask[:, context_len:] = causal_mask
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                is_causal=False,
                enable_gqa=enable_gqa,
                scale=scale,
            )
        else:
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                is_causal=True,
                enable_gqa=enable_gqa,
                scale=scale,
            )

        all_outputs.append(sdpa_out.transpose(1, 2).squeeze(0))
        q_offset += q_len
        kv_offset += s_len

    return torch.cat(all_outputs, dim=0)


def _make_kv_cache_for_decode(
    batch_spec: BatchSpec,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-populate KV cache with ALL tokens for decode-only tests.

    Places context + new tokens sequentially in cache blocks
    so that FIA paged attention can read them via block_table.
    """
    num_blocks = sum((s + block_size - 1) // block_size for s in batch_spec.seq_lens)
    num_blocks = max(num_blocks, 64)
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)

    kv_offset = 0
    block_start = 0
    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        k_full = key[kv_offset : kv_offset + s_len]
        v_full = value[kv_offset : kv_offset + s_len]

        context_k = k_full[:context_len].contiguous()
        context_v = v_full[:context_len].contiguous()
        new_k = k_full[context_len:].contiguous()
        new_v = v_full[context_len:].contiguous()

        num_blocks_for_seq = (s_len + block_size - 1) // block_size
        block_table[i, :num_blocks_for_seq] = torch.arange(
            block_start,
            block_start + num_blocks_for_seq,
            dtype=torch.int32,
            device=device,
        )

        for t_idx in range(context_len):
            blk = block_start + t_idx // block_size
            pos = t_idx % block_size
            k_cache[blk, pos] = context_k[t_idx]
            v_cache[blk, pos] = context_v[t_idx]

        for t_idx in range(q_len):
            blk = block_start + (context_len + t_idx) // block_size
            pos = (context_len + t_idx) % block_size
            k_cache[blk, pos] = new_k[t_idx]
            v_cache[blk, pos] = new_v[t_idx]

        kv_offset += s_len
        block_start += num_blocks_for_seq

    return k_cache, v_cache


def _make_kv_cache_for_mixed(
    batch_spec: BatchSpec,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-populate KV cache for mixed decode+prefill tests.

    Only decode sequences have context tokens placed in the cache.
    Prefill sequences (seq_lens == query_lens) have no context;
    their KV goes through the direct FIA prefill path.
    """
    num_blocks = sum((s + block_size - 1) // block_size for s in batch_spec.seq_lens)
    num_blocks = max(num_blocks, 64)
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)

    kv_offset = 0
    block_start = 0
    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len
        is_decode = q_len == 1

        k_full = key[kv_offset : kv_offset + s_len]
        v_full = value[kv_offset : kv_offset + s_len]

        if is_decode and context_len > 0:
            num_blocks_for_seq = (s_len + block_size - 1) // block_size
            block_table[i, :num_blocks_for_seq] = torch.arange(
                block_start,
                block_start + num_blocks_for_seq,
                dtype=torch.int32,
                device=device,
            )

            for t_idx in range(context_len):
                blk = block_start + t_idx // block_size
                pos = t_idx % block_size
                k_cache[blk, pos] = k_full[t_idx]
                v_cache[blk, pos] = v_full[t_idx]

            for t_idx in range(q_len):
                blk = block_start + (context_len + t_idx) // block_size
                pos = (context_len + t_idx) % block_size
                k_cache[blk, pos] = k_full[context_len + t_idx]
                v_cache[blk, pos] = v_full[context_len + t_idx]

            block_start += num_blocks_for_seq

        kv_offset += s_len

    return k_cache, v_cache


def build_cp_attn_metadata(
    batch_spec: BatchSpec,
    vllm_config,
    device: torch.device,
    pcp_size: int = 1,
    pcp_rank: int = 0,
    kv_cache_prepopulated: bool = False,
    decode_threshold: int = 1,
) -> AscendMetadata:
    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)

    num_reqs = common_attn_metadata.num_reqs
    num_actual_tokens = common_attn_metadata.num_actual_tokens

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

    num_decodes = sum(1 for ql in batch_spec.query_lens if ql <= decode_threshold)
    num_prefills = batch_spec.batch_size - num_decodes
    num_decode_tokens = sum(ql for ql in batch_spec.query_lens if ql <= decode_threshold)
    num_prefill_tokens = num_actual_tokens - num_decode_tokens

    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping
    query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    num_decodes_flatten = query_lens[:num_decodes].sum().item()

    seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]

    num_actual_tokens_pcp_padded = num_actual_tokens * pcp_size

    query_start_loc = query_start_loc_cpu.to(device, non_blocking=True)

    attn_mask_builder = AttentionMaskBuilder(device)
    attn_mask = attn_mask_builder.get_attention_mask(common_attn_metadata.causal, vllm_config.model_config)

    prefill_metadata = None
    if num_prefills > 0:
        prefill_query_lens = query_lens[num_decodes:]
        attn_mask_seqlens = torch.cumsum(prefill_query_lens, dim=0).tolist()
        head_attn_nomask_seqlens = attn_mask_seqlens if pcp_rank > 0 else []
        tail_attn_nomask_seqlens = attn_mask_seqlens

        total_prefill_tokens = num_prefill_tokens
        prefill_tokens_offset = num_decode_tokens

        if pcp_size > 1:
            chunk_size = total_prefill_tokens // pcp_size
            rank_start = prefill_tokens_offset + pcp_rank * chunk_size
            rank_end = rank_start + chunk_size

            q_head_idx = torch.arange(rank_start, rank_end, device=device, dtype=torch.long)
            q_tail_idx = torch.tensor([], device=device, dtype=torch.long)

            kv_total = total_prefill_tokens * pcp_size
            kv_with_q_head_mask_idx = torch.arange(
                prefill_tokens_offset,
                prefill_tokens_offset + kv_total,
                device=device,
                dtype=torch.long,
            )
            kv_with_q_head_nomask_idx = (
                torch.arange(prefill_tokens_offset, prefill_tokens_offset + kv_total, device=device, dtype=torch.long)
                if pcp_rank > 0
                else torch.tensor([], device=device, dtype=torch.long)
            )
            q_full_idx = torch.arange(chunk_size, device=device, dtype=torch.long)
            kv_with_q_tail_nomask_idx = torch.tensor([], device=device, dtype=torch.long)
            kv_with_q_tail_mask_idx = torch.tensor([], device=device, dtype=torch.long)
            pcp_allgather_restore_idx = list(range(total_prefill_tokens * pcp_size))
        else:
            q_head_idx = torch.arange(
                prefill_tokens_offset,
                prefill_tokens_offset + total_prefill_tokens,
                device=device,
                dtype=torch.long,
            )
            q_tail_idx = torch.tensor([], device=device, dtype=torch.long)
            kv_with_q_head_mask_idx = torch.arange(
                prefill_tokens_offset,
                prefill_tokens_offset + total_prefill_tokens,
                device=device,
                dtype=torch.long,
            )
            kv_with_q_head_nomask_idx = torch.tensor([], device=device, dtype=torch.long)
            kv_with_q_tail_nomask_idx = torch.tensor([], device=device, dtype=torch.long)
            kv_with_q_tail_mask_idx = torch.tensor([], device=device, dtype=torch.long)
            q_full_idx = torch.arange(total_prefill_tokens, device=device, dtype=torch.long)
            pcp_allgather_restore_idx = None

        pcp_metadata = AscendPCPMetadata(
            q_head_idx=q_head_idx,
            q_tail_idx=q_tail_idx,
            kv_with_q_head_nomask_idx=kv_with_q_head_nomask_idx,
            kv_with_q_head_mask_idx=kv_with_q_head_mask_idx,
            kv_with_q_tail_nomask_idx=kv_with_q_tail_nomask_idx,
            kv_with_q_tail_mask_idx=kv_with_q_tail_mask_idx,
            attn_mask_seqlens=attn_mask_seqlens,
            head_attn_nomask_seqlens=head_attn_nomask_seqlens,
            tail_attn_nomask_seqlens=tail_attn_nomask_seqlens,
            q_full_idx=q_full_idx,
            pcp_use_hybrid_attn=False,
            pcp_allgather_restore_idx=pcp_allgather_restore_idx,
        )

        prefill_cumsum_q = torch.cumsum(query_lens[num_decodes:], dim=0).to(device)

        prefill_metadata = AscendMetadataForPrefill(
            pcp_metadata=pcp_metadata,
            pcp_exit_fa_scatter_idx=None,
            chunked_context=None,
            block_tables=block_table[num_decodes_flatten:, ...],
            actual_seq_lengths_q=prefill_cumsum_q,
        )

    decode_metadata = None
    if num_decodes > 0:
        decode_query_lens = query_lens[:num_decodes].tolist()
        decode_seq_lens = seq_lens_cpu[:num_decodes].tolist()

        if kv_cache_prepopulated:
            num_computed_tokens_arr = np.zeros((num_decodes_flatten, pcp_size, 1), dtype=np.int32)
            flat_idx = 0
            for i in range(num_decodes):
                s_len = int(decode_seq_lens[i])
                q_len = decode_query_lens[i]
                context_len = s_len - q_len
                for t in range(q_len):
                    num_computed_tokens_arr[flat_idx, pcp_rank, 0] = context_len + t + 1
                    flat_idx += 1
        else:
            num_computed_tokens_arr = np.zeros((num_decodes_flatten, pcp_size, 1), dtype=np.int32)

        # Tile block_table for MTP: each decode request may have multiple tokens
        if num_decodes_flatten > num_decodes:
            tiled_rows = []
            for i in range(num_decodes):
                q_len = decode_query_lens[i]
                row = block_table[i : i + 1]
                tiled_rows.append(row.repeat(q_len, 1))
            decode_block_tables = torch.cat(tiled_rows, dim=0)
        else:
            decode_block_tables = block_table[:num_decodes_flatten]

        decode_metadata = AscendMetadataForDecode(
            num_computed_tokens_of_pcp_dcp=num_computed_tokens_arr,
            block_tables=decode_block_tables,
        )

    actual_seq_lengths_q = (
        torch.arange(num_decodes_flatten, device=device) + 1
        if num_decodes_flatten > 0
        else torch.tensor([], device=device)
    ).tolist() + torch.cumsum(query_lens[num_decodes:], dim=0).tolist()

    attn_metadata = AscendMetadata(
        num_actual_tokens=num_actual_tokens,
        num_decode_tokens=num_decode_tokens,
        num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
        num_decodes_flatten=num_decodes_flatten,
        block_tables=block_table,
        query_start_loc=query_start_loc,
        seq_lens=common_attn_metadata.seq_lens[:num_reqs],
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_list=seq_lens_cpu.tolist(),
        max_query_len=common_attn_metadata.max_query_len,
        actual_seq_lengths_q=actual_seq_lengths_q,
        slot_mapping=slot_mapping,
        attn_mask=attn_mask,
        attn_state=common_attn_metadata.attn_state,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        prefill=prefill_metadata,
        decode_meta=decode_metadata,
    )
    return attn_metadata


def run_cp_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    attn_metadata: AscendMetadata,
    impl: AscendAttentionCPImpl,
    device: torch.device,
    vllm_config,
) -> torch.Tensor:
    """Run CP attention forward pass with proper setup.

    Mocks reshape_and_cache (no-op) since KV cache is pre-populated
    for decode tests and not needed for prefill tests.
    """
    mock_layer_entry = MagicMock()
    for layer_name in ["placeholder"]:
        vllm_config.compilation_config.static_forward_context[layer_name] = mock_layer_entry

    num_tokens = query.shape[0]
    num_q_heads = query.shape[1]
    head_size = query.shape[2]
    output = torch.empty(num_tokens, num_q_heads, head_size, dtype=query.dtype, device=device)

    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        from vllm.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        forward_ctx.num_tokens = num_tokens
        forward_ctx.is_draft_model = False
        forward_ctx.is_draft_model_prefill = False
        forward_ctx.capturing = False
        forward_ctx.flash_comm_v1_enabled = False
        forward_ctx.flashcomm_v2_enabled = False

        mock_layer = MockAttentionLayer(device)

        import vllm_ascend.device.device_op as device_op_module

        original_reshape_and_cache = device_op_module.DeviceOperator.reshape_and_cache
        original_kv_cache_load = device_op_module.DeviceOperator.kv_cache_load

        def _mock_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
            return

        def _mock_kv_cache_load(key_cache, value_cache, block_tables, seq_lens_kv, starts, key, value):
            return

        device_op_module.DeviceOperator.reshape_and_cache = staticmethod(_mock_reshape_and_cache)
        device_op_module.DeviceOperator.kv_cache_load = staticmethod(_mock_kv_cache_load)

        try:
            output = impl.forward(mock_layer, query, key, value, kv_cache, attn_metadata, output=output)
        finally:
            device_op_module.DeviceOperator.reshape_and_cache = original_reshape_and_cache
            device_op_module.DeviceOperator.kv_cache_load = original_kv_cache_load

    return output


@pytest.fixture(autouse=True)
def default_mock_config():
    mock_config = MagicMock()
    mock_config.compilation_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    mock_config.compilation_config.static_forward_context = {}
    mock_config.parallel_config = MagicMock()
    mock_config.parallel_config.prefill_context_parallel_size = 1
    mock_config.parallel_config.decode_context_parallel_size = 1
    mock_config.parallel_config.tensor_parallel_size = 1
    mock_config.model_config = MagicMock()
    mock_config.model_config.dtype = torch.float16
    mock_config.speculative_config = None
    mock_config.cache_config = MagicMock()
    mock_config.cache_config.block_size = 128
    mock_config.kv_transfer_config = None

    with set_current_vllm_config(mock_config):
        yield mock_config


@pytest.fixture(autouse=True)
def mock_graph_params():
    with patch("vllm_ascend.compilation.acl_graph.get_graph_params") as mock_get_graph:
        graph_params = MagicMock()
        graph_params.workspaces = {}
        graph_params.handles = {}
        graph_params.attn_params = {}
        graph_params.events = {}
        mock_get_graph.return_value = graph_params
        with patch("vllm_ascend.compilation.acl_graph.get_draft_graph_params", return_value=graph_params):
            yield


def _create_cp_impl(vllm_config, device, num_q_heads, num_kv_heads, head_size, scale):
    return AscendAttentionCPImpl(
        num_heads=num_q_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
    )


def _assert_close(output, reference, rtol, atol, backend_name):
    assert output.shape == reference.shape, (
        f"[{backend_name}] shape {output.shape} != reference shape {reference.shape}"
    )
    assert output.dtype == reference.dtype, (
        f"[{backend_name}] dtype {output.dtype} != reference dtype {reference.dtype}"
    )
    assert torch.isfinite(output).all(), f"[{backend_name}] produced non-finite values"

    def error_msg(msg: str, name: str):
        return f"[{name}] output differs from SDPA baseline. {msg}"

    torch.testing.assert_close(
        output,
        reference,
        rtol=rtol,
        atol=atol,
        msg=partial(error_msg, name=backend_name),
    )


# ---------------------------------------------------------------------------
# Pure prefill (seq_lens == query_lens, no context)
# ---------------------------------------------------------------------------
@npu_test(num_npus=1, npu_type="a2")
def _test_cp_prefill_precision_no_cp(
    batch_spec: BatchSpec,
    model: str,
    *,
    block_size: int = 128,
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    set_random_seed(42)

    vllm_config = create_vllm_config(
        model_name=model,
        tensor_parallel_size=1,
        max_model_len=max(batch_spec.seq_lens) + block_size,
        block_size=block_size,
        num_gpu_blocks=8192,
    )
    device = torch.device("npu")

    num_q_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    dtype = torch.bfloat16
    scale = 1.0 / (head_size**0.5)

    num_tokens = batch_spec.compute_num_tokens()
    total_kv = sum(batch_spec.seq_lens)

    query_vllm = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    key_vllm = torch.randn(total_kv, num_kv_heads, head_size, dtype=dtype, device=device)
    value_vllm = torch.randn(total_kv, num_kv_heads, head_size, dtype=dtype, device=device)

    sdpa_output = compute_sdpa_reference(
        query_vllm,
        key_vllm,
        value_vllm,
        batch_spec,
        scale,
        num_q_heads,
        num_kv_heads,
    )

    attn_metadata = build_cp_attn_metadata(batch_spec, vllm_config, device, pcp_size=1, pcp_rank=0)

    num_blocks = sum((s + block_size - 1) // block_size for s in batch_spec.seq_lens)
    num_blocks = max(num_blocks, 64)
    kv_cache = (
        torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device),
        torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device),
    )

    impl = _create_cp_impl(vllm_config, device, num_q_heads, num_kv_heads, head_size, scale)
    output = run_cp_attention(query_vllm, key_vllm, value_vllm, kv_cache, attn_metadata, impl, device, vllm_config)

    _assert_close(output, sdpa_output, rtol, atol, "CP_Prefill")


# ---------------------------------------------------------------------------
# Pure decode (query_lens == 1, context in KV cache)
# ---------------------------------------------------------------------------
@npu_test(num_npus=1, npu_type="a2")
def _test_cp_decode_precision_no_cp(
    batch_spec: BatchSpec,
    model: str,
    *,
    block_size: int = 128,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    decode_threshold: int = 1,
):
    set_random_seed(42)

    vllm_config = create_vllm_config(
        model_name=model,
        tensor_parallel_size=1,
        max_model_len=max(batch_spec.seq_lens) + block_size,
        block_size=block_size,
        num_gpu_blocks=8192,
    )
    device = torch.device("npu")

    num_q_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    dtype = torch.bfloat16
    scale = 1.0 / (head_size**0.5)

    total_kv = sum(batch_spec.seq_lens)

    query_full = torch.randn(total_kv, num_q_heads, head_size, dtype=dtype, device=device)
    key_full = torch.randn(total_kv, num_kv_heads, head_size, dtype=dtype, device=device)
    value_full = torch.randn(total_kv, num_kv_heads, head_size, dtype=dtype, device=device)

    sdpa_output = compute_mixed_sdpa_reference(
        query_full,
        key_full,
        value_full,
        batch_spec,
        scale,
        num_q_heads,
        num_kv_heads,
    )

    # Backend inputs: only the new (decode) tokens
    all_q, all_k, all_v = [], [], []
    q_offset = 0
    kv_offset = 0
    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q_i = query_full[q_offset : q_offset + q_len].contiguous()
        k_full_i = key_full[kv_offset : kv_offset + s_len]
        v_full_i = value_full[kv_offset : kv_offset + s_len]

        all_q.append(q_i)
        all_k.append(k_full_i[context_len:])
        all_v.append(v_full_i[context_len:])

        q_offset += q_len
        kv_offset += s_len

    query_vllm = torch.cat(all_q, dim=0)
    key_vllm = torch.cat(all_k, dim=0)
    value_vllm = torch.cat(all_v, dim=0)

    attn_metadata = build_cp_attn_metadata(
        batch_spec,
        vllm_config,
        device,
        pcp_size=1,
        pcp_rank=0,
        kv_cache_prepopulated=True,
        decode_threshold=decode_threshold,
    )

    k_cache, v_cache = _make_kv_cache_for_decode(
        batch_spec,
        num_kv_heads,
        head_size,
        block_size,
        dtype,
        device,
        key_full,
        value_full,
        attn_metadata.block_tables,
    )

    # Re-tile decode block tables after _make_kv_cache_for_decode updates
    # block_table in-place. Without this, the tiled decode_block_tables
    # (created during build_cp_attn_metadata) still contains the original
    # sequential block indices, causing FIA to read from empty cache blocks.
    if attn_metadata.num_decodes_flatten > attn_metadata.num_decodes:
        tiled_rows = []
        for i in range(attn_metadata.num_decodes):
            q_len = batch_spec.query_lens[i]
            row = attn_metadata.block_tables[i : i + 1]
            tiled_rows.append(row.repeat(q_len, 1))
        attn_metadata.decode_meta.block_tables = torch.cat(tiled_rows, dim=0)

    kv_cache = (k_cache, v_cache)

    impl = _create_cp_impl(vllm_config, device, num_q_heads, num_kv_heads, head_size, scale)
    output = run_cp_attention(query_vllm, key_vllm, value_vllm, kv_cache, attn_metadata, impl, device, vllm_config)

    _assert_close(output, sdpa_output, rtol, atol, "CP_Decode")


# ---------------------------------------------------------------------------
# Mixed decode + prefill (prefill sequences have seq_lens == query_lens)
#
# The CP attention non-chunked prefill path calls FIA directly with only the
# new tokens as KV. Context tokens in the KV cache are NOT loaded for prefill
# without chunked context. Therefore prefill sequences in mixed mode must
# have seq_lens == query_lens (no context).
# ---------------------------------------------------------------------------
@npu_test(num_npus=1, npu_type="a2")
def _test_cp_mixed_precision_no_cp(
    batch_spec: BatchSpec,
    model: str,
    *,
    block_size: int = 128,
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    set_random_seed(42)

    vllm_config = create_vllm_config(
        model_name=model,
        tensor_parallel_size=1,
        max_model_len=max(batch_spec.seq_lens) + block_size,
        block_size=block_size,
        num_gpu_blocks=8192,
    )
    device = torch.device("npu")

    num_q_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    dtype = torch.bfloat16
    scale = 1.0 / (head_size**0.5)

    total_kv = sum(batch_spec.seq_lens)

    query_full = torch.randn(total_kv, num_q_heads, head_size, dtype=dtype, device=device)
    key_full = torch.randn(total_kv, num_kv_heads, head_size, dtype=dtype, device=device)
    value_full = torch.randn(total_kv, num_kv_heads, head_size, dtype=dtype, device=device)

    sdpa_output = compute_mixed_sdpa_reference(
        query_full,
        key_full,
        value_full,
        batch_spec,
        scale,
        num_q_heads,
        num_kv_heads,
    )

    # Backend inputs: only the new tokens
    all_q, all_k, all_v = [], [], []
    q_offset = 0
    kv_offset = 0
    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q_i = query_full[q_offset : q_offset + q_len].contiguous()
        k_full_i = key_full[kv_offset : kv_offset + s_len]
        v_full_i = value_full[kv_offset : kv_offset + s_len]

        all_q.append(q_i)
        all_k.append(k_full_i[context_len:])
        all_v.append(v_full_i[context_len:])

        q_offset += q_len
        kv_offset += s_len

    query_vllm = torch.cat(all_q, dim=0)
    key_vllm = torch.cat(all_k, dim=0)
    value_vllm = torch.cat(all_v, dim=0)

    attn_metadata = build_cp_attn_metadata(
        batch_spec,
        vllm_config,
        device,
        pcp_size=1,
        pcp_rank=0,
        kv_cache_prepopulated=True,
    )

    k_cache, v_cache = _make_kv_cache_for_mixed(
        batch_spec,
        num_kv_heads,
        head_size,
        block_size,
        dtype,
        device,
        key_full,
        value_full,
        attn_metadata.block_tables,
    )
    kv_cache = (k_cache, v_cache)

    impl = _create_cp_impl(vllm_config, device, num_q_heads, num_kv_heads, head_size, scale)
    output = run_cp_attention(query_vllm, key_vllm, value_vllm, kv_cache, attn_metadata, impl, device, vllm_config)

    _assert_close(output, sdpa_output, rtol, atol, "CP_Mixed")


@npu_test(num_npus=1, npu_type="a2")
class TestCPAttentionPrecision:
    """Precision tests for AscendAttentionCPImpl.

    Validates that CP attention produces results matching
    PyTorch SDPA within 1e-2 tolerance.

    Test scenarios:
    - Pure prefill (seq_lens == query_lens), PCP=1, DCP=1
    - Pure decode, PCP=1, DCP=1 (context in KV cache)
    - Mixed decode+prefill (prefill has seq_lens == query_lens), PCP=1, DCP=1
    - MTP (Multi-Token Prediction) decode, PCP=1, DCP=1
    """

    @pytest.mark.parametrize(
        "batch_spec_name",
        [
            "single_prefill",
            "small_prefill",
            "medium_prefill",
            "large_prefill",
        ],
    )
    @pytest.mark.parametrize("model", MODELS)
    @patch_distributed_groups(dcp_size=1, pcp_size=1)
    @npu_test(num_npus=1, npu_type="a2")
    def test_cp_prefill_precision(
        self,
        mock_all2all,
        mock_dcp,
        mock_pcp,
        batch_spec_name,
        model,
    ):
        batch_spec = BATCH_SPECS[batch_spec_name]
        _test_cp_prefill_precision_no_cp(batch_spec, model)

    @pytest.mark.parametrize(
        "batch_spec_name",
        [
            "single_decode",
            "small_decode",
            "medium_decode",
        ],
    )
    @pytest.mark.parametrize("model", MODELS)
    @patch_distributed_groups(dcp_size=1, pcp_size=1)
    @npu_test(num_npus=1, npu_type="a2")
    def test_cp_decode_precision(
        self,
        mock_all2all,
        mock_dcp,
        mock_pcp,
        batch_spec_name,
        model,
    ):
        batch_spec = BATCH_SPECS[batch_spec_name]
        _test_cp_decode_precision_no_cp(batch_spec, model)

    @pytest.mark.parametrize(
        "batch_spec_name",
        [
            "mixed_small",
            "mixed_medium",
            "mixed_large",
        ],
    )
    @pytest.mark.parametrize("model", MODELS)
    @patch_distributed_groups(dcp_size=1, pcp_size=1)
    @npu_test(num_npus=1, npu_type="a2")
    def test_cp_mixed_precision(
        self,
        mock_all2all,
        mock_dcp,
        mock_pcp,
        batch_spec_name,
        model,
    ):
        batch_spec = BATCH_SPECS[batch_spec_name]
        _test_cp_mixed_precision_no_cp(batch_spec, model)

    @pytest.mark.parametrize(
        "batch_spec_name",
        [
            "mtp_1_plus_3_tiny",
            "mtp_1_plus_3_small",
            "mtp_1_plus_3_medium",
        ],
    )
    @pytest.mark.parametrize("model", MODELS)
    @patch_distributed_groups(dcp_size=1, pcp_size=1)
    @npu_test(num_npus=1, npu_type="a2")
    def test_cp_mtp_decode_precision(
        self,
        mock_all2all,
        mock_dcp,
        mock_pcp,
        batch_spec_name,
        model,
    ):
        """MTP decode: each request produces 1 target + 3 speculative tokens.

        All tokens (context + new) are pre-populated in the KV cache.
        FIA paged attention with actual_seq_lengths_kv enforces causal
        masking per token within each request.
        """
        batch_spec = BATCH_SPECS[batch_spec_name]
        _test_cp_decode_precision_no_cp(batch_spec, model, decode_threshold=4)
