import math
from functools import partial
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from tests.ut.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
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


BATCH_SPECS = {
    "small_decode": BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill": BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small": BatchSpec(seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5, 5]),
    "medium_decode": BatchSpec(
        seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
        query_lens=[1, 1, 1, 1, 1, 1, 1, 1],
    ),
    "medium_prefill": BatchSpec(seq_lens=[256, 512, 1024, 2048], query_lens=[16, 16, 16, 16]),
    "mixed_medium": BatchSpec(seq_lens=[512, 1024, 2048, 512, 1024, 2048], query_lens=[1, 1, 1, 7, 7, 7]),
    "large_decode": BatchSpec(seq_lens=[2048] * 32, query_lens=[1] * 32),
    "large_prefill": BatchSpec(seq_lens=[4096] * 8, query_lens=[32] * 8),
    "mixed_large": BatchSpec(seq_lens=[1024, 2048, 4096, 1024, 2048, 4096], query_lens=[1, 1, 1, 32, 32, 32]),
    "single_decode": BatchSpec(seq_lens=[1024], query_lens=[1]),
    "single_prefill": BatchSpec(seq_lens=[1024], query_lens=[64]),
    # encoder-only
    "small_encoder_prefill": BatchSpec(seq_lens=[32, 64, 128, 256], query_lens=[32, 64, 128, 256]),
    "medium_encoder_prefill": BatchSpec(seq_lens=[256, 512, 1024, 2048], query_lens=[256, 512, 1024, 2048]),
    "mtp_1_plus_3": BatchSpec(seq_lens=[256, 512, 1024, 1536], query_lens=[4, 4, 4, 4]),
}


class MockLinear:
    def __init__(self, out_features=128, in_features=128, device=None, dtype=torch.bfloat16):
        self.weight = torch.randn(out_features, in_features, dtype=dtype, device=device) / math.sqrt(in_features)
        self.quant_method = UnquantizedLinearMethod()

    def __call__(self, x, **kwargs):
        if x.size(-1) != self.weight.size(-1):
            self.weight = torch.randn(
                self.weight.size(0), x.size(-1), dtype=self.weight.dtype, device=x.device
            ) / math.sqrt(x.size(-1))

        return (x @ self.weight.T, None)


class MockLayerNorm:
    def __init__(self, normalized_shape, device=None, dtype=torch.bfloat16):
        self.weight = torch.ones(normalized_shape, dtype=dtype, device=device)
        self.variance_epsilon = 1e-6

    def __call__(self, x):
        return x


class MockRotary:
    def __init__(self):
        pass


def create_mla_kv_cache(
    k_nope_contexts: list[torch.Tensor],
    k_pe_contexts: list[torch.Tensor],
    block_size: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    num_blocks: int,
    common_attn_metadata: AscendCommonAttentionMetadata,
):
    """Create MLA KV cache with two separate tensors.

    MLA KV cache layout:
    - k_cache (nope): (num_blocks, block_size, num_kv_heads, kv_lora_rank)
    - v_cache (rope): (num_blocks, block_size, num_kv_heads, qk_rope_head_dim)
    """
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, kv_lora_rank, dtype=dtype, device=device)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, qk_rope_head_dim, dtype=dtype, device=device)

    seq_lens = common_attn_metadata.seq_lens.cpu()
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    batch_size = len(k_nope_contexts)
    block_table = common_attn_metadata.block_table_tensor

    start_block_idx = 0
    for i in range(batch_size):
        k_nope_ctx = k_nope_contexts[i]
        k_pe_ctx = k_pe_contexts[i]
        context_len = int(seq_lens[i]) - int(query_lens[i])

        num_blocks_for_seq = (int(seq_lens[i]) + block_size - 1) // block_size
        block_table[i, :num_blocks_for_seq] = torch.arange(
            start_block_idx, start_block_idx + num_blocks_for_seq, dtype=torch.int32
        )

        k_cache_flat = k_cache[start_block_idx : start_block_idx + num_blocks_for_seq].view(
            -1, num_kv_heads, kv_lora_rank
        )
        v_cache_flat = v_cache[start_block_idx : start_block_idx + num_blocks_for_seq].view(
            -1, num_kv_heads, qk_rope_head_dim
        )
        k_cache_flat[:context_len] = k_nope_ctx[:context_len]
        v_cache_flat[:context_len] = k_pe_ctx[:context_len]

        start_block_idx += num_blocks_for_seq

    slot_mapping = common_attn_metadata.slot_mapping
    for i in range(batch_size):
        context_len_i = int(seq_lens[i]) - int(query_lens[i])
        token_offsets = torch.arange(int(query_lens[i])) + context_len_i
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = int(query_start_loc_cpu[i])
        end = int(query_start_loc_cpu[i + 1])
        slot_mapping[start:end] = block_table[i, block_indices] * block_size + token_inter_block_offsets.to(device).to(
            torch.int32
        )

    return (k_cache, v_cache)


def run_mla_attention_backend(
    kv_cache_spec: MLAAttentionSpec,
    vllm_config,
    device: torch.device,
    common_attn_metadata: AscendCommonAttentionMetadata,
    hidden_states: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.bfloat16,
    attn_type: AttentionType = AttentionType.DECODER,
):
    from vllm_ascend.ascend_config import init_ascend_config

    init_ascend_config(vllm_config)

    from vllm_ascend.ops import rotary_embedding

    hf_config = vllm_config.model_config.hf_text_config
    qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", 64)

    rotary_embedding._cos_cache = torch.ones(8192, qk_rope_head_dim, dtype=dtype, device=device)
    rotary_embedding._sin_cache = torch.zeros(8192, qk_rope_head_dim, dtype=dtype, device=device)
    rotary_embedding._cos_mla = torch.ones(8192, 1, 1, qk_rope_head_dim, dtype=dtype, device=device)
    rotary_embedding._sin_mla = torch.zeros(8192, 1, 1, qk_rope_head_dim, dtype=dtype, device=device)

    from vllm.distributed.parallel_state import GroupCoordinator

    mock_tp_group = MagicMock(spec=GroupCoordinator)
    mock_tp_group.world_size = 1
    mock_tp_group.rank = 0

    mock_weight_prefetch = MagicMock()
    mock_weight_prefetch.maybe_prefetch_mla_or_sla_weight_in_current_stream = MagicMock()

    import vllm_ascend.utils as utils_module

    original_weight_prefetch = utils_module._WEIGHT_PREFETCH_METHOD
    utils_module._WEIGHT_PREFETCH_METHOD = mock_weight_prefetch

    try:
        with patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_tp_group):
            num_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
            num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
            head_size = vllm_config.model_config.get_head_size()

            kv_lora_rank = getattr(hf_config, "kv_lora_rank", 512)
            q_lora_rank = getattr(hf_config, "q_lora_rank", 1536)
            qk_nope_head_dim = getattr(hf_config, "qk_nope_head_dim", 128)
            qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", 64)
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            v_head_dim = getattr(hf_config, "v_head_dim", 128)

            backend = get_attn_backend(head_size, dtype, None, use_mla=True, use_sparse=False, use_mm_prefix=False)
            impl_cls = backend.get_impl_cls()
            builder_cls = backend.get_builder_cls()

            builder = builder_cls(
                kv_cache_spec,
                ["placeholder"],
                vllm_config,
                device,
            )
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )

            hidden_size = num_heads * head_size
            q_proj_out_dim = num_heads * qk_head_dim
            q_b_proj_out_dim = num_heads * qk_head_dim
            kv_b_proj_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
            kv_b_proj_in_dim = kv_lora_rank
            o_proj_out_dim = head_size * num_heads
            o_proj_in_dim = num_heads * v_head_dim

            impl = impl_cls(
                num_heads=num_heads,
                head_size=head_size,
                scale=1.0 / (head_size**0.5),
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                attn_type=attn_type.value if hasattr(attn_type, "value") else attn_type,
                kv_cache_dtype="auto",
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
                q_proj=MockLinear(out_features=q_proj_out_dim, in_features=hidden_size, device=device, dtype=dtype),
                q_b_proj=MockLinear(out_features=q_b_proj_out_dim, in_features=q_lora_rank, device=device, dtype=dtype),
                kv_b_proj=MockLinear(
                    out_features=kv_b_proj_out_dim, in_features=kv_b_proj_in_dim, device=device, dtype=dtype
                ),
                o_proj=MockLinear(out_features=o_proj_out_dim, in_features=o_proj_in_dim, device=device, dtype=dtype),
                kv_a_layernorm=MockLayerNorm(normalized_shape=kv_lora_rank, device=device, dtype=dtype),
                q_a_layernorm=MockLayerNorm(normalized_shape=q_lora_rank, device=device, dtype=dtype),
                rotary_emb=MockRotary(),
                fused_qkv_a_proj=None,
                kv_a_proj_with_mqa=MockLinear(
                    out_features=num_kv_heads * (kv_lora_rank + qk_rope_head_dim),
                    in_features=hidden_size,
                    device=device,
                    dtype=dtype,
                ),
            )

            impl.fa_quant_layer = False
            impl.enable_mlapo = False
            impl.process_weights_after_loading(dtype)

            output = torch.empty_like(hidden_states)
            output = impl.forward("layer_0", hidden_states, kv_cache, attn_metadata, output=output)
    finally:
        utils_module._WEIGHT_PREFETCH_METHOD = original_weight_prefetch

    return output, impl


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def npu_interleave_rope_simple(
    x: torch.Tensor, cos: torch.Tensor | None = None, sin: torch.Tensor | None = None
) -> torch.Tensor:
    """Simulate Ascend npu_interleave_rope with default cos=1, sin=0.

    Interleave the last dimension: [x0, x1, x2, x3, ...] -> [x0, x2, ..., x1, x3, ...]
    With cos=1, sin=0 (rope disabled), the function just returns the interleaved result.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.cat([even, odd], dim=-1).contiguous()


def prefill_sdpa(
    q_nope: torch.Tensor,
    q_pe_raw: torch.Tensor,
    k_pe_c: torch.Tensor,
    k_nope_c: torch.Tensor,
    v: torch.Tensor,
    impl,
    scale: float,
    is_causal: bool = True,
    context_len: int = 0,
) -> torch.Tensor:
    q_pe = npu_interleave_rope_simple(q_pe_raw)
    q_full = torch.cat([q_nope, q_pe], dim=-1)
    k_full = torch.cat([k_nope_c, k_pe_c], dim=-1)

    q_sdpa = q_full.unsqueeze(0).transpose(1, 2)
    k_sdpa = k_full.unsqueeze(0).transpose(1, 2)
    v_sdpa = v.unsqueeze(0).transpose(1, 2)

    if context_len > 0 and is_causal:
        q_len = q_full.shape[0]
        kv_len = k_full.shape[0]
        mask = torch.tril(
            torch.ones(q_len, kv_len, device=q_full.device, dtype=torch.bool),
            diagonal=context_len,
        )
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=mask,
            enable_gqa=False,
            scale=scale,
        )
    else:
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            is_causal=is_causal,
            enable_gqa=False,
            scale=scale,
        )
    return attn_out.transpose(1, 2).squeeze(0)


def decode_sdpa(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    k_nope: torch.Tensor,
    impl,
    scale: float,
) -> torch.Tensor:
    """Compute SDPA for decode path.

    Decode path: q is W_UK projected (ql_nope), k/v are raw latent.
    """
    q_full = torch.cat([ql_nope, q_pe], dim=-1)
    k_full = torch.cat([k_nope, k_pe], dim=-1)

    q_sdpa = q_full.unsqueeze(0).transpose(1, 2)
    k_sdpa = k_full.unsqueeze(0).transpose(1, 2)
    v_sdpa = k_nope.unsqueeze(0).transpose(1, 2)

    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        is_causal=False,
        enable_gqa=(impl.num_heads != impl.num_kv_heads),
        scale=scale,
    )
    return attn_out.transpose(1, 2).squeeze(0)


def compute_mla_sdpa_reference(
    hidden_states: torch.Tensor,
    k_nope_contexts: list[torch.Tensor],
    k_pe_contexts: list[torch.Tensor],
    impl,
    batch_spec: BatchSpec,
    scale: float,
) -> torch.Tensor:
    """Compute MLA reference using SDPA as golden baseline.

    Handles three modes:
    - Decode only: q via W_UK, k/v raw latent, _v_up_proj
    - Prefill only: q raw, k/v via kv_b_proj, no _v_up_proj
    - Mixed: decode first, then prefill, combined before o_proj
    """
    rms_eps = impl.kv_a_layernorm.variance_epsilon
    rms_w = impl.kv_a_layernorm.weight

    # --- Identify decode vs prefill tokens ---
    num_decode_tokens = 0
    for ql in batch_spec.query_lens:
        if ql == 1:
            num_decode_tokens += 1

    # --- KV projection and normalization for all tokens ---
    kv_no_split = impl.kv_a_proj_with_mqa(hidden_states)[0]
    k_nope_all = kv_no_split[:, : impl.kv_lora_rank]
    k_pe_all = kv_no_split[:, impl.kv_lora_rank :]
    k_nope_normed = rms_norm(k_nope_all, rms_w, rms_eps)

    # --- Q projection for all tokens ---
    q_b = impl.q_proj(hidden_states)[0]
    q_nope_all, q_pe_all = q_b.view(-1, impl.num_heads, impl.qk_head_dim).split(
        [impl.qk_nope_head_dim, impl.qk_rope_head_dim], dim=-1
    )

    # Decode: q_nope is W_UK projected; Prefill: q_nope stays raw
    q_nope_decode = q_nope_all[:num_decode_tokens]
    q_nope_decode_t = q_nope_decode.transpose(0, 1).float()
    ql_nope_decode = torch.bmm(q_nope_decode_t, impl.W_UK_T.float()).to(q_nope_all.dtype)
    ql_nope_decode = ql_nope_decode.transpose(0, 1)

    q_pe_decode = npu_interleave_rope_simple(q_pe_all[:num_decode_tokens])

    q_nope_prefill = q_nope_all[num_decode_tokens:]
    q_pe_prefill_raw = q_pe_all[num_decode_tokens:]

    # --- Process each sequence ---
    decode_outputs = []
    prefill_outputs = []

    token_offset = 0
    for i in range(len(batch_spec.seq_lens)):
        s_len_i = batch_spec.seq_lens[i]
        q_len_i = batch_spec.query_lens[i]
        context_len_i = s_len_i - q_len_i
        is_decode = q_len_i == 1

        if is_decode:
            ql_nope_i = ql_nope_decode[token_offset : token_offset + q_len_i]
            q_pe_i = q_pe_decode[token_offset : token_offset + q_len_i]

            k_nope_dec = k_nope_normed[token_offset : token_offset + q_len_i].view(
                q_len_i, impl.num_kv_heads, impl.kv_lora_rank
            )
            k_pe_dec = k_pe_all[token_offset : token_offset + q_len_i].view(
                q_len_i, impl.num_kv_heads, impl.qk_rope_head_dim
            )
            k_pe_dec = npu_interleave_rope_simple(k_pe_dec)

            k_nope_full_i = torch.cat([k_nope_contexts[i], k_nope_dec], dim=0)
            k_pe_full_i = torch.cat([k_pe_contexts[i], k_pe_dec], dim=0)

            sdpa_out = decode_sdpa(
                ql_nope_i,
                q_pe_i,
                k_pe_full_i,
                k_nope_full_i,
                impl,
                scale,
            )
            # _v_up_proj
            sdpa_out = sdpa_out.transpose(0, 1).contiguous()
            sdpa_out = torch.bmm(sdpa_out.float(), impl.W_UV.float()).to(sdpa_out.dtype)
            sdpa_out = sdpa_out.permute(1, 0, 2)
            sdpa_out = sdpa_out.reshape(-1, impl.num_heads * impl.v_head_dim)
            decode_outputs.append(sdpa_out)

        else:
            q_nope_i = q_nope_prefill[token_offset - num_decode_tokens : token_offset - num_decode_tokens + q_len_i]
            q_pe_raw_i = q_pe_prefill_raw[token_offset - num_decode_tokens : token_offset - num_decode_tokens + q_len_i]

            k_nope_new = k_nope_normed[token_offset : token_offset + q_len_i].view(
                q_len_i, impl.num_kv_heads, impl.kv_lora_rank
            )
            k_pe_raw_new = k_pe_all[token_offset : token_offset + q_len_i].view(
                q_len_i, impl.num_kv_heads, impl.qk_rope_head_dim
            )

            # kv_b_proj on RMS-normed new tokens: k_nope_proj + v_proj
            k_nope_new_flat = k_nope_new.view(-1, impl.kv_lora_rank)
            kv_b_new = impl.kv_b_proj(k_nope_new_flat)[0].view(
                q_len_i, impl.num_heads, impl.qk_nope_head_dim + impl.v_head_dim
            )
            k_nope_proj_new, v_proj_new = kv_b_new.split([impl.qk_nope_head_dim, impl.v_head_dim], dim=-1)

            # kv_b_proj on raw context (cache stores raw data)
            k_nope_ctx_raw = (
                k_nope_contexts[i]
                .view(context_len_i, impl.num_kv_heads, impl.kv_lora_rank)
                .reshape(-1, impl.kv_lora_rank)
            )
            kv_b_ctx = impl.kv_b_proj(k_nope_ctx_raw)[0].view(
                context_len_i, impl.num_heads, impl.qk_nope_head_dim + impl.v_head_dim
            )
            k_nope_proj_ctx, v_proj_ctx = kv_b_ctx.split([impl.qk_nope_head_dim, impl.v_head_dim], dim=-1)

            k_nope_proj = torch.cat([k_nope_proj_ctx, k_nope_proj_new], dim=0)
            v_proj = torch.cat([v_proj_ctx, v_proj_new], dim=0)

            # k_pe: context raw (as stored in cache), new interleaved
            k_pe_raw_ctx = k_pe_contexts[i].view(context_len_i, impl.num_kv_heads, impl.qk_rope_head_dim)
            k_pe_new_interleaved = npu_interleave_rope_simple(k_pe_raw_new)
            k_pe_cat = torch.cat([k_pe_raw_ctx, k_pe_new_interleaved], dim=0)
            k_pe_expanded = k_pe_cat.expand(*k_nope_proj.shape[:-1], -1)

            sdpa_out = prefill_sdpa(
                q_nope_i,
                q_pe_raw_i,
                k_pe_expanded,
                k_nope_proj,
                v_proj,
                impl,
                scale,
                is_causal=True,
                context_len=context_len_i,
            )
            sdpa_out = sdpa_out.reshape(q_len_i, impl.num_heads * impl.v_head_dim)
            prefill_outputs.append(sdpa_out)

        token_offset += q_len_i

    # --- Combine outputs in token order ---
    all_outputs = decode_outputs + prefill_outputs
    attn_output = torch.cat(all_outputs, dim=0)

    # --- Output projection ---
    sdpa_output = impl.o_proj(attn_output)[0]
    return sdpa_output


def _test_mla_attention_correctness(
    batch_spec: BatchSpec,
    model: str,
    *,
    attn_type: AttentionType = AttentionType.DECODER,
    block_size: int = 128,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    tensor_parallel_size: int = 1,
):
    set_random_seed(42)

    vllm_config = create_vllm_config(
        model_name=model,
        tensor_parallel_size=1,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=8192,
    )
    device = torch.device("npu")

    hf_config = vllm_config.model_config.hf_text_config
    kv_lora_rank = getattr(hf_config, "kv_lora_rank", 512)
    qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", 64)

    batch_size = batch_spec.batch_size
    num_q_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    dtype = torch.bfloat16
    scale = 1.0 / (head_size**0.5)

    kv_cache_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=kv_lora_rank,
        dtype=dtype,
    )

    k_nope_contexts, k_pe_contexts = [], []
    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        k_nope_full = torch.randn(s_len, num_kv_heads, kv_lora_rank, dtype=dtype, device=device)
        k_pe_full = torch.randn(s_len, num_kv_heads, qk_rope_head_dim, dtype=dtype, device=device)

        k_nope_contexts.append(k_nope_full[:context_len])
        k_pe_contexts.append(k_pe_full[:context_len])

    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)
    if attn_type == AttentionType.ENCODER_ONLY:
        common_attn_metadata.causal = False

    num_blocks = sum((s + block_size - 1) // block_size for s in batch_spec.seq_lens)
    num_blocks = max(16, num_blocks)

    kv_cache = create_mla_kv_cache(
        k_nope_contexts=k_nope_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dtype=dtype,
        device=device,
        num_blocks=num_blocks,
        common_attn_metadata=common_attn_metadata,
    )

    num_tokens = common_attn_metadata.num_actual_tokens
    hidden_size = num_q_heads * head_size
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        from vllm.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        forward_ctx.num_tokens = num_tokens
        forward_ctx.is_draft_model = False
        forward_ctx.is_draft_model_prefill = False
        forward_ctx.capturing = False
        forward_ctx.flash_comm_v1_enabled = False
        forward_ctx.flashcomm_v2_enabled = False

        backend_output, impl = run_mla_attention_backend(
            kv_cache_spec,
            vllm_config,
            device,
            common_attn_metadata,
            hidden_states,
            kv_cache,
            dtype,
            attn_type=attn_type,
        )

    sdpa_output = compute_mla_sdpa_reference(
        hidden_states,
        k_nope_contexts,
        k_pe_contexts,
        impl,
        batch_spec,
        scale,
    )

    # Compare (same pattern as test_gqa.py)
    name = "MLA"
    assert backend_output.shape == sdpa_output.shape, (
        f"[{name}] shape {backend_output.shape} != SDPA shape {sdpa_output.shape}"
    )
    assert backend_output.dtype == sdpa_output.dtype, (
        f"[{name}] dtype {backend_output.dtype} != SDPA dtype {sdpa_output.dtype}"
    )

    assert torch.isfinite(backend_output).all(), f"[{name}] produced non-finite values"

    def error_msg(msg: str, backend_name: str):
        return f"[{backend_name}] output differs from SDPA baseline. {msg}"

    torch.testing.assert_close(
        backend_output,
        sdpa_output,
        rtol=rtol,
        atol=atol,
        msg=partial(error_msg, backend_name="MLA"),
    )


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "small_decode",
        "small_prefill",
        "mixed_small",
        "medium_decode",
        "medium_prefill",
        "mixed_medium",
        "large_decode",
        "large_prefill",
        "single_decode",
        "single_prefill",
        "mtp_1_plus_3",
    ],
)
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-V2"])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_mla_backend_correctness(default_vllm_config, batch_spec_name: str, model: str, tensor_parallel_size: int):
    batch_spec = BATCH_SPECS[batch_spec_name]

    _test_mla_attention_correctness(
        batch_spec,
        model,
        tensor_parallel_size=tensor_parallel_size,
    )
