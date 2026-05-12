#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""NPU precision tests for ``vllm_ascend.attention.context_parallel.sfa_cp``.

The suite follows PR #7216 style: one ``BATCH_SPECS`` table, one correctness
helper, and one parametrized test over workload, topology, dtype, TP size, and
model. In-scope branches are compared with a fp32 dense MQA reference on NPU.
PCP>1 prefill/mixed needs real per-rank scheduler metadata and is skipped here.
"""

import math
import sys
from typing import Callable
from unittest.mock import MagicMock

import pytest
import torch

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm.config import VllmConfig  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from tests.ut.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm_ascend.attention.context_parallel.sfa_cp import (  # noqa: E402
    AscendSFACPImpl,
    AscendSFACPMetadataBuilder,
)

SPARSE_COUNT = 2048
_BLOCK_SIZE = 128
_TEST_NUM_HEADS = 8
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2

_MAX_SIG_REL_ERR = 1e-2      # max |out-ref| / peak |ref|
_MAX_MEAN_SIG_ERR = 5e-3     # mean |out-ref| / mean |ref|
_MAX_REL_ERR = 1e-2          # max per-element rel err where |ref| >= floor
_SIG_FLOOR_FRAC = 5e-1       # floor = this fraction of peak |ref|


def _validate_spec(spec: BatchSpec) -> None:
    """Require ``seq_len <= SPARSE_COUNT`` so sparse matches dense reference."""
    for s, q in zip(spec.seq_lens, spec.query_lens):
        assert q <= s, f"query_len ({q}) must not exceed seq_len ({s})"
        assert s <= SPARSE_COUNT, (
            f"seq_len ({s}) must be <= SPARSE_COUNT ({SPARSE_COUNT}) so the "
            "sparse attention degenerates into dense attention for the "
            "reference comparison."
        )


_VLLM_CONFIG_CACHE: dict = {}


def _get_vllm_config(
    model: str,
    dtype: torch.dtype,
    *,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
) -> VllmConfig:
    """Cached ``VllmConfig`` for DSA/SFA-CP (fp8 quant stripped; heads capped).

    Built via ``create_vllm_config`` (PR #7216 helper). The DSA-flavored
    attributes (``kv_lora_rank``, ``qk_rope_head_dim``,
    ``num_attention_heads``, ...) are pushed into both ``hf_config`` and
    ``hf_text_config`` so the SFA-CP impl reads consistent dims regardless
    of which view it reaches for. ``quantization_config`` is stripped
    because Ascend's ``ModelConfig`` validator rejects fp8 on NPU.

    ``tensor_parallel_size > 1`` is **simulated** PR #7216 style: keep the
    real TP=1 on the underlying ``VllmConfig`` (no distributed init
    required) and instead divide the test-friendly head count by TP so the
    kernel runs with the post-TP-shard head budget. This exercises the
    kernel under different head counts without needing multiple NPUs.
    """
    key = (model, dtype, tensor_parallel_size)
    cfg = _VLLM_CONFIG_CACHE.get(key)
    if cfg is not None:
        return cfg
    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float16"
    sim_num_heads = max(1, _TEST_NUM_HEADS // tensor_parallel_size)
    cfg = create_vllm_config(
        model_name=model,
        tensor_parallel_size=1,  # always TP=1; head split is simulated
        max_model_len=max_model_len,
        dtype=dtype_str,
        block_size=_BLOCK_SIZE,
        num_gpu_blocks=4096,
        max_num_seqs=64,
        max_num_batched_tokens=max(8192, max_model_len * 2),
        enable_chunked_prefill=True,
        hf_overrides={"quantization_config": None},
        hf_config_override={
            "num_attention_heads": sim_num_heads,
            "num_key_value_heads": 1,
        },
    )
    _VLLM_CONFIG_CACHE[key] = cfg
    return cfg


# Spec name prefixes drive the SFA-CP branch under test.
BATCH_SPECS: dict[str, BatchSpec] = {
    "decode_single": BatchSpec(
        seq_lens=[1024], query_lens=[1], name="decode_single"),
    "decode_small_batch": BatchSpec(
        seq_lens=[512, 1024, 1536, 2048], query_lens=[1, 1, 1, 1],
        name="decode_small_batch"),
    "decode_large_batch": BatchSpec(
        seq_lens=[2048] * 8, query_lens=[1] * 8, name="decode_large_batch"),
    "mtp_1_plus_1": BatchSpec(
        seq_lens=[512, 1024, 1536], query_lens=[2, 2, 2], name="mtp_1_plus_1"),
    "mtp_1_plus_3": BatchSpec(
        seq_lens=[1024, 1536, 2048, 2048], query_lens=[4, 4, 4, 4],
        name="mtp_1_plus_3"),
    "mtp_1_plus_7": BatchSpec(
        seq_lens=[1024, 1536, 2048], query_lens=[8, 8, 8], name="mtp_1_plus_7"),
    "prefill_single": BatchSpec(
        seq_lens=[256], query_lens=[256], name="prefill_single"),
    "prefill_small_batch": BatchSpec(
        seq_lens=[256, 512, 384], query_lens=[256, 512, 384],
        name="prefill_small_batch"),
    "prefill_with_context": BatchSpec(
        seq_lens=[512, 1024], query_lens=[128, 256], name="prefill_with_context"),
    "mixed_small": BatchSpec(
        seq_lens=[512, 1024, 256, 512], query_lens=[1, 1, 64, 128],
        name="mixed_small"),
    "mixed_medium": BatchSpec(
        seq_lens=[1024, 1536, 2048, 256, 512], query_lens=[1, 1, 1, 64, 128],
        name="mixed_medium"),
}


def _infer_mode(spec: BatchSpec) -> str:
    """Return one of: ``decode``, ``mtp``, ``prefill``, ``mixed``."""
    name = spec.name
    for prefix in ("decode_", "mtp_", "prefill_", "mixed_"):
        if name.startswith(prefix):
            return prefix.rstrip("_")
    raise ValueError(
        f"BatchSpec name {name!r} does not start with a known mode prefix "
        "('decode_', 'mtp_', 'prefill_', 'mixed_')"
    )


def _build_topk_indices(
    seq_lens: list[int],
    query_lens: list[int],
    sparse_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Build causal topk indices with INVALID_IDX (-1) padding."""
    num_tokens = sum(query_lens)
    topk = torch.full((num_tokens, 1, sparse_count), -1, dtype=torch.int32, device=device)

    cum_q = 0
    for b, s_len in enumerate(seq_lens):
        q_len = query_lens[b]
        ctx_len = s_len - q_len
        for j in range(q_len):
            valid_end = ctx_len + j + 1
            topk[cum_q + j, 0, :valid_end] = torch.arange(valid_end, dtype=torch.int32, device=device)
        cum_q += q_len

    return topk


def _reference_sparse_attention(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope_full_per_req: list[torch.Tensor],
    k_rope_full_per_req: list[torch.Tensor],
    seq_lens: list[int],
    query_lens: list[int],
    scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Pure-PyTorch dense MQA baseline in fp32."""
    outputs: list[torch.Tensor] = []

    cum_q = 0
    for b, s_len in enumerate(seq_lens):
        q_len = query_lens[b]
        ctx_len = s_len - q_len

        K_nope = k_nope_full_per_req[b][:s_len].float()
        K_rope = k_rope_full_per_req[b][:s_len].float()
        K = torch.cat([K_nope, K_rope], dim=-1)
        V = K_nope

        for j in range(q_len):
            t = cum_q + j
            valid_end = ctx_len + j + 1

            q_n = ql_nope[t].float()
            q_p = q_pe[t].float()
            Q = torch.cat([q_n, q_p], dim=-1)

            K_b = K[:valid_end]
            V_b = V[:valid_end]

            scores = (Q @ K_b.transpose(0, 1)) * scale
            attn = torch.softmax(scores, dim=-1)
            out = attn @ V_b
            outputs.append(out.to(out_dtype))

        cum_q += q_len

    return torch.stack(outputs, dim=0)


def _build_cp_paged_kv_cache(
    seq_lens: list[int],
    k_nope_contexts: list[torch.Tensor],
    k_rope_contexts: list[torch.Tensor],
    block_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    cp_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, int]:
    """Allocate a paged KV cache that matches the *decode* CP gather/block_table layout.

    Each request is padded so its block count is exactly ``cp_size * L``
    (i.e. divisible by the total CP world size). The local rank is treated
    as "rank 0" and owns the first ``L`` of each request's logical blocks
    (the actual values in ``local_block_table`` only matter via shape because
    ``gather_kv_cross_cp`` is mocked).

    The ``gathered_*`` tensors mirror what the impl would see after both
    DCP and PCP all-gathers; layout matches the indexing produced by
    ``gather_block_table``:
        ``gathered[i*N + r*L + b]`` holds K data for logical block
        ``b*cp_size + i`` of request ``r`` (where ``N == batch_size * L``).
    """
    batch_size = len(seq_lens)

    raw_blocks_per_req = [(s + block_size - 1) // block_size for s in seq_lens]
    max_raw = max(raw_blocks_per_req)
    total_blocks_per_req = ((max_raw + cp_size - 1) // cp_size) * cp_size
    L = total_blocks_per_req // cp_size

    total_blocks = batch_size * total_blocks_per_req + 1  # +1 reserves block 0

    full_k_nope_cache = torch.zeros(
        total_blocks, block_size, 1, kv_lora_rank, dtype=dtype, device=device
    )
    full_k_rope_cache = torch.zeros(
        total_blocks, block_size, 1, qk_rope_head_dim, dtype=dtype, device=device
    )
    full_block_table = torch.zeros(
        batch_size, total_blocks_per_req, dtype=torch.int32, device=device
    )

    next_block = 1
    for r in range(batch_size):
        s_len = seq_lens[r]
        for p in range(total_blocks_per_req):
            full_block_table[r, p] = next_block
            tok_start = p * block_size
            tok_end = min(tok_start + block_size, s_len)
            length = max(tok_end - tok_start, 0)
            if length > 0:
                full_k_nope_cache[next_block, :length, 0, :] = (
                    k_nope_contexts[r][tok_start:tok_end]
                )
                full_k_rope_cache[next_block, :length, 0, :] = (
                    k_rope_contexts[r][tok_start:tok_end]
                )
            next_block += 1

    local_block_table = full_block_table[:, :L].contiguous()

    N = batch_size * L
    gathered_k_nope = torch.zeros(
        cp_size * N, block_size, 1, kv_lora_rank, dtype=dtype, device=device
    )
    gathered_k_rope = torch.zeros(
        cp_size * N, block_size, 1, qk_rope_head_dim, dtype=dtype, device=device
    )
    for i in range(cp_size):
        for r in range(batch_size):
            for b in range(L):
                p_logical = b * cp_size + i
                phys_block = int(full_block_table[r, p_logical].item())
                dst = i * N + r * L + b
                gathered_k_nope[dst] = full_k_nope_cache[phys_block]
                gathered_k_rope[dst] = full_k_rope_cache[phys_block]

    return (
        full_k_nope_cache,
        full_k_rope_cache,
        full_block_table,
        local_block_table,
        gathered_k_nope,
        gathered_k_rope,
        L,
    )


def _build_cp_prefill_compact_metadata(
    prefill_full_block_table: torch.Tensor,
    full_k_nope_cache: torch.Tensor,
    full_k_rope_cache: torch.Tensor,
    cp_size: int,
    block_size: int,
    seq_lens: list[int],
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the prefill compact metadata + matching gathered KV view.

    Reproduces the production
    ``AscendSFACPMetadataBuilder.build_prefill_compact_block_metadata``
    formula: ``block_table_cp[r, b*cp_size + i] = new_block_table[r, b] + i*M``
    where ``M = num_unique_local_blocks``. Then materialises the
    ``gathered_compact`` tensor that ``gather_kv_cross_cp_compact`` would
    produce so that the kernel, when walking ``block_table_cp[r, :]``,
    sees the contiguous K context of request ``r`` block-by-block.

    Inputs
    ------
    prefill_full_block_table:
        Shape ``(num_prefill_reqs, L * cp_size)``. The "full" (logical)
        block table for prefill requests as built by
        ``_build_cp_paged_kv_cache``. Entry ``[r, p]`` holds the physical
        block id that owns logical position ``p`` of request ``r``.

    The "local" (rank-0) prefill block table -- which the production code
    actually receives in ``attn_metadata.block_table[num_decodes:]`` -- is
    derived as ``prefill_full_block_table[:, :L]``. This matches the
    convention used by ``_build_cp_paged_kv_cache`` for the decode path.
    """
    num_prefill_reqs, total_blocks_per_req = prefill_full_block_table.shape
    assert total_blocks_per_req % cp_size == 0, (
        f"prefill_full_block_table has {total_blocks_per_req} columns, "
        f"which is not divisible by cp_size={cp_size}"
    )
    L = total_blocks_per_req // cp_size

    prefill_local_block_table = prefill_full_block_table[:, :L].contiguous()

    block_arange = torch.arange(cp_size, dtype=prefill_local_block_table.dtype, device=device)
    valid_block_ids, new_block_table_flat = prefill_local_block_table.flatten().unique(return_inverse=True)
    num_blocks = valid_block_ids.shape[0]
    block_table_cp = (
        new_block_table_flat.unsqueeze(-1).to(prefill_local_block_table)
        + (block_arange * num_blocks).view(1, 1, -1).to(prefill_local_block_table)
    ).reshape(prefill_local_block_table.shape[0], -1)

    new_block_table_2d = new_block_table_flat.view(prefill_local_block_table.shape)

    M = int(num_blocks)
    gathered_compact_nope = torch.zeros(
        cp_size * M, block_size, 1, kv_lora_rank, dtype=dtype, device=device
    )
    gathered_compact_rope = torch.zeros(
        cp_size * M, block_size, 1, qk_rope_head_dim, dtype=dtype, device=device
    )

    for r in range(num_prefill_reqs):
        s_len = seq_lens[r]
        for b_local in range(L):
            for i in range(cp_size):
                p_logical = b_local * cp_size + i
                tok_start = p_logical * block_size
                tok_end = min(tok_start + block_size, s_len)
                length = max(tok_end - tok_start, 0)
                if length <= 0:
                    continue
                src_block = int(prefill_full_block_table[r, p_logical].item())
                dst = int(new_block_table_2d[r, b_local].item()) + i * M
                gathered_compact_nope[dst, :length, 0, :] = (
                    full_k_nope_cache[src_block, :length, 0, :]
                )
                gathered_compact_rope[dst, :length, 0, :] = (
                    full_k_rope_cache[src_block, :length, 0, :]
                )

    return valid_block_ids, block_table_cp, gathered_compact_nope, gathered_compact_rope


def _make_fake_self(
    *,
    scale: float,
    pcp_size: int,
    dcp_size: int,
    gather_kv_cross_cp_fn: Callable | None = None,
    gather_kv_cross_cp_compact_fn: Callable | None = None,
) -> MagicMock:
    """Construct a ``MagicMock`` matching the slice of ``AscendSFACPImpl``
    that ``_execute_sparse_flash_attention_process`` reads.

    Bound methods that don't depend on init-time state
    (``gather_block_table``, ``_execute_sparse_flash_attention``) are
    delegated to the real (unbound) implementations so the production
    kernel call stays under test. The collective gathers
    (``gather_kv_cross_cp`` / ``gather_kv_cross_cp_compact``) are mocked
    via ``side_effect`` because they require an initialised PCP / DCP
    ``ProcessGroupHCCL``, which can't be created on a single rank.
    """
    fake_self = MagicMock()
    fake_self.scale = scale
    fake_self.pcp_size = pcp_size
    fake_self.dcp_size = dcp_size

    if gather_kv_cross_cp_fn is not None:
        fake_self.gather_kv_cross_cp = MagicMock(side_effect=gather_kv_cross_cp_fn)
    if gather_kv_cross_cp_compact_fn is not None:
        fake_self.gather_kv_cross_cp_compact = MagicMock(side_effect=gather_kv_cross_cp_compact_fn)

    fake_self.gather_block_table = lambda block_num, block_tables, block_arange: (
        AscendSFACPImpl.gather_block_table(fake_self, block_num, block_tables, block_arange)
    )
    fake_self._execute_sparse_flash_attention = lambda *args, **kwargs: (
        AscendSFACPImpl._execute_sparse_flash_attention(fake_self, *args, **kwargs)
    )
    fake_self._align_to_graph_bucket_tokens = lambda x, m: x
    return fake_self


def _run_sfa_cp_kernel(
    *,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    full_k_nope_cache: torch.Tensor,
    full_k_rope_cache: torch.Tensor,
    local_block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    scale: float,
    pcp_size: int,
    dcp_size: int,
    num_decodes: int,
    num_decode_tokens: int,
    num_prefills: int,
    device: torch.device,
    gathered_k_nope: torch.Tensor | None = None,
    gathered_k_rope: torch.Tensor | None = None,
    valid_block_ids: torch.Tensor | None = None,
    block_table_cp: torch.Tensor | None = None,
    gathered_compact_nope: torch.Tensor | None = None,
    gathered_compact_rope: torch.Tensor | None = None,
    prefill_q_cum_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the SFA-CP kernel with mocked CP collectives."""
    cp_size = pcp_size * dcp_size

    gather_kv_cross_cp_fn = None
    gather_kv_cross_cp_compact_fn = None

    if num_decode_tokens > 0:
        assert gathered_k_nope is not None and gathered_k_rope is not None, (
            "decode branch requires gathered_k_nope/gathered_k_rope"
        )
        gathered_lookup = {
            id(full_k_nope_cache): gathered_k_nope,
            id(full_k_rope_cache): gathered_k_rope,
        }

        def gather_kv_cross_cp_fn(kv: torch.Tensor, block_tables: torch.Tensor):
            gathered = gathered_lookup.get(id(kv))
            assert gathered is not None, (
                "gather_kv_cross_cp called with an unexpected kv tensor"
            )
            return gathered, block_tables.numel()

    if num_prefills > 0:
        assert (
            valid_block_ids is not None
            and block_table_cp is not None
            and gathered_compact_nope is not None
            and gathered_compact_rope is not None
            and prefill_q_cum_seqlens is not None
        ), (
            "prefill branch requires valid_block_ids / block_table_cp / "
            "gathered_compact_{nope,rope} / prefill_q_cum_seqlens"
        )
        gathered_compact_lookup = {
            id(full_k_nope_cache): gathered_compact_nope,
            id(full_k_rope_cache): gathered_compact_rope,
        }

        def gather_kv_cross_cp_compact_fn(kv: torch.Tensor, vbid: torch.Tensor):
            gathered = gathered_compact_lookup.get(id(kv))
            assert gathered is not None, (
                "gather_kv_cross_cp_compact called with an unexpected kv tensor"
            )
            return gathered

    fake_self = _make_fake_self(
        scale=scale,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        gather_kv_cross_cp_fn=gather_kv_cross_cp_fn,
        gather_kv_cross_cp_compact_fn=gather_kv_cross_cp_compact_fn,
    )

    fake_attn_metadata = MagicMock()
    fake_attn_metadata.block_table = local_block_table
    fake_attn_metadata.num_decodes = num_decodes
    fake_attn_metadata.num_decode_tokens = num_decode_tokens
    fake_attn_metadata.num_prefills = num_prefills

    fake_sfa_cp_metadata = MagicMock()
    fake_sfa_cp_metadata.block_arange = torch.arange(cp_size, dtype=torch.int32, device=device)
    if num_prefills > 0:
        fake_sfa_cp_metadata.valid_block_ids = valid_block_ids
        fake_sfa_cp_metadata.block_table_cp = block_table_cp
        fake_sfa_cp_metadata.prefill_q_cum_seqlens = prefill_q_cum_seqlens
    fake_attn_metadata.sfa_cp_metadata = fake_sfa_cp_metadata

    cum_lens_arg = cum_query_lens if num_decode_tokens > 0 else prefill_q_cum_seqlens

    return AscendSFACPImpl._execute_sparse_flash_attention_process(
        fake_self,
        ql_nope,
        q_pe,
        (full_k_nope_cache, full_k_rope_cache),
        topk_indices,
        fake_attn_metadata,
        cum_lens_arg,
        seq_lens_tensor,
    )


def _record_and_assert(
    backend_output: torch.Tensor,
    reference_output: torch.Tensor,
    tag: str,
    *,
    dtype: torch.dtype,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> tuple[float, float]:
    """Assert numerical closeness and record signal-relative metrics."""
    assert backend_output.shape == reference_output.shape, (
        f"[{tag}] backend shape {tuple(backend_output.shape)} != "
        f"reference shape {tuple(reference_output.shape)}"
    )
    assert backend_output.dtype == reference_output.dtype, (
        f"[{tag}] backend dtype {backend_output.dtype} != "
        f"reference dtype {reference_output.dtype}"
    )
    assert torch.isfinite(backend_output).all(), (
        f"[{tag}] sparse flash attention produced non-finite values"
    )

    torch.testing.assert_close(
        backend_output,
        reference_output,
        rtol=rtol,
        atol=atol,
        msg=lambda m: f"[SFA-CP:{tag}] kernel output diverges from baseline. {m}",
    )

    ref_f32 = reference_output.float()
    out_f32 = backend_output.float()
    diff = (out_f32 - ref_f32).abs()
    ref_abs = ref_f32.abs()
    peak = float(ref_abs.max())
    mean_ref_abs = float(ref_abs.mean())
    sig_floor = peak * _SIG_FLOOR_FRAC

    max_abs_err = float(diff.max())
    mean_abs_err = float(diff.mean())
    max_sig_rel_err = max_abs_err / peak if peak > 0 else 0.0
    mean_sig_rel_err = mean_abs_err / mean_ref_abs if mean_ref_abs > 0 else 0.0

    significant_mask = ref_abs >= sig_floor
    if significant_mask.any():
        per_elem_rel = diff[significant_mask] / ref_abs[significant_mask]
        max_rel_err_sig = float(per_elem_rel.max())
    else:
        max_rel_err_sig = 0.0

    assert max_sig_rel_err < _MAX_SIG_REL_ERR, (
        f"[SFA-CP:{tag}] dtype={dtype} signal-relative max error "
        f"{max_sig_rel_err * 100:.4f}% exceeds 1% budget "
        f"(peak={peak:.4e}, max_abs_err={max_abs_err:.4e})"
    )
    assert mean_sig_rel_err < _MAX_MEAN_SIG_ERR, (
        f"[SFA-CP:{tag}] dtype={dtype} signal-relative mean error "
        f"{mean_sig_rel_err * 100:.4f}% exceeds 0.5% drift budget "
        f"(mean_ref_abs={mean_ref_abs:.4e}, mean_abs_err={mean_abs_err:.4e})"
    )
    assert max_rel_err_sig < _MAX_REL_ERR, (
        f"[SFA-CP:{tag}] dtype={dtype} per-element relative error on "
        f">={int(_SIG_FLOOR_FRAC * 100)}%-of-peak elements "
        f"{max_rel_err_sig * 100:.4f}% exceeds 1% budget "
        f"(peak={peak:.4e}, max_abs_err={max_abs_err:.4e})"
    )

    return max_abs_err, max_rel_err_sig


def _make_synthetic_kv_contexts(
    seq_lens: list[int],
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-request K context generated independently so values across
    requests are uncorrelated."""
    k_nope = [
        torch.randn(s, kv_lora_rank, dtype=dtype, device=device) * 0.1
        for s in seq_lens
    ]
    k_rope = [
        torch.randn(s, qk_rope_head_dim, dtype=dtype, device=device) * 0.1
        for s in seq_lens
    ]
    return k_nope, k_rope


def _test_sfa_cp_correctness(
    batch_spec: BatchSpec,
    model: str,
    *,
    pcp_size: int,
    dcp_size: int,
    dtype: torch.dtype = torch.bfloat16,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    tensor_parallel_size: int = 1,
) -> None:
    """Test ``AscendSFACPImpl`` against a fp32 dense MQA reference."""
    mode = _infer_mode(batch_spec)
    assert not (pcp_size > 1 and mode in ("prefill", "mixed")), (
        f"PCP>1 {mode} is out of scope; the parametrize whitelist "
        "should not have generated this combination"
    )

    torch.manual_seed(2026)
    _validate_spec(batch_spec)

    vllm_config = _get_vllm_config(
        model, dtype, tensor_parallel_size=tensor_parallel_size,
    )
    device = torch.device("npu")

    seq_lens = list(batch_spec.seq_lens)
    query_lens = list(batch_spec.query_lens)
    batch_size = batch_spec.batch_size
    num_tokens = batch_spec.compute_num_tokens()
    cp_size = pcp_size * dcp_size

    cache_config = vllm_config.cache_config
    hf_text = vllm_config.model_config.hf_text_config
    block_size = cache_config.block_size
    kv_lora_rank = hf_text.kv_lora_rank
    qk_rope_head_dim = hf_text.qk_rope_head_dim
    num_heads = hf_text.num_attention_heads

    head_dim = kv_lora_rank + qk_rope_head_dim
    scale = 1.0 / math.sqrt(head_dim)

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=block_size, device=device,
    )

    k_nope_contexts, k_rope_contexts = _make_synthetic_kv_contexts(
        seq_lens, kv_lora_rank, qk_rope_head_dim, dtype, device,
    )
    (full_k_nope_cache, full_k_rope_cache, full_block_table, local_block_table,
     gathered_k_nope, gathered_k_rope, _L) = _build_cp_paged_kv_cache(
        seq_lens=seq_lens,
        k_nope_contexts=k_nope_contexts,
        k_rope_contexts=k_rope_contexts,
        block_size=block_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        cp_size=cp_size,
        dtype=dtype,
        device=device,
    )

    ql_nope = torch.randn(
        num_tokens, num_heads, kv_lora_rank, dtype=dtype, device=device,
    ) * 0.1
    q_pe = torch.randn(
        num_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=device,
    ) * 0.1
    topk_indices = _build_topk_indices(seq_lens, query_lens, SPARSE_COUNT, device)

    cum_query_lens = common_attn_metadata.query_start_loc[1:].to(torch.int32)
    seq_lens_tensor = common_attn_metadata.seq_lens.to(torch.int32)

    decode_gathered_k_nope: torch.Tensor | None = None
    decode_gathered_k_rope: torch.Tensor | None = None
    prefill_full_block_table: torch.Tensor | None = None

    if mode in ("decode", "mtp"):
        num_decodes = batch_size
        num_decode_tokens = num_tokens
        num_prefills = 0
        decode_gathered_k_nope = gathered_k_nope
        decode_gathered_k_rope = gathered_k_rope
        n_decode_reqs = batch_size
    elif mode == "prefill":
        num_decodes = 0
        num_decode_tokens = 0
        num_prefills = batch_size
        prefill_full_block_table = full_block_table
        n_decode_reqs = 0
    else:  # mixed (chunked-prefill)
        n_decode_reqs = sum(1 for q in query_lens if q == 1)
        assert all(q == 1 for q in query_lens[:n_decode_reqs]), (
            f"mixed spec {batch_spec.name}: decode requests must come first"
        )
        assert all(q > 1 for q in query_lens[n_decode_reqs:]), (
            f"mixed spec {batch_spec.name}: prefill requests must come after decode"
        )
        num_decodes = n_decode_reqs
        num_decode_tokens = n_decode_reqs
        num_prefills = batch_size - n_decode_reqs
        prefill_full_block_table = full_block_table[n_decode_reqs:]

        L_decode = local_block_table[:n_decode_reqs].shape[1]
        N_decode = n_decode_reqs * L_decode
        decode_gathered_k_nope = torch.zeros(
            cp_size * N_decode, block_size, 1, kv_lora_rank,
            dtype=dtype, device=device,
        )
        decode_gathered_k_rope = torch.zeros(
            cp_size * N_decode, block_size, 1, qk_rope_head_dim,
            dtype=dtype, device=device,
        )
        for i in range(cp_size):
            for r in range(n_decode_reqs):
                for b in range(L_decode):
                    p_logical = b * cp_size + i
                    phys_block = int(full_block_table[r, p_logical].item())
                    dst = i * N_decode + r * L_decode + b
                    decode_gathered_k_nope[dst] = full_k_nope_cache[phys_block]
                    decode_gathered_k_rope[dst] = full_k_rope_cache[phys_block]

    valid_block_ids = None
    block_table_cp = None
    gathered_compact_nope = None
    gathered_compact_rope = None
    prefill_q_cum_seqlens = None
    if num_prefills > 0:
        valid_block_ids, block_table_cp, gathered_compact_nope, gathered_compact_rope = (
            _build_cp_prefill_compact_metadata(
                prefill_full_block_table=prefill_full_block_table,
                full_k_nope_cache=full_k_nope_cache,
                full_k_rope_cache=full_k_rope_cache,
                cp_size=cp_size,
                block_size=block_size,
                seq_lens=seq_lens[n_decode_reqs:],
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                dtype=dtype,
                device=device,
            )
        )
        if n_decode_reqs > 0:
            prefill_q_cum_seqlens = (
                cum_query_lens[n_decode_reqs:]
                - cum_query_lens[n_decode_reqs - 1]
            )
        else:
            prefill_q_cum_seqlens = cum_query_lens

    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        backend_output = _run_sfa_cp_kernel(
            ql_nope=ql_nope,
            q_pe=q_pe,
            full_k_nope_cache=full_k_nope_cache,
            full_k_rope_cache=full_k_rope_cache,
            local_block_table=local_block_table,
            topk_indices=topk_indices,
            cum_query_lens=cum_query_lens,
            seq_lens_tensor=seq_lens_tensor,
            scale=scale,
            pcp_size=pcp_size,
            dcp_size=dcp_size,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            device=device,
            gathered_k_nope=decode_gathered_k_nope,
            gathered_k_rope=decode_gathered_k_rope,
            valid_block_ids=valid_block_ids,
            block_table_cp=block_table_cp,
            gathered_compact_nope=gathered_compact_nope,
            gathered_compact_rope=gathered_compact_rope,
            prefill_q_cum_seqlens=prefill_q_cum_seqlens,
        )
    reference_output = _reference_sparse_attention(
        ql_nope=ql_nope,
        q_pe=q_pe,
        k_nope_full_per_req=k_nope_contexts,
        k_rope_full_per_req=k_rope_contexts,
        seq_lens=seq_lens,
        query_lens=query_lens,
        scale=scale,
        out_dtype=dtype,
    )

    dt = "bf16" if dtype == torch.bfloat16 else "fp16"
    tag = (
        f"{mode}|{batch_spec.name}|pcp={pcp_size}|dcp={dcp_size}"
        f"|tp={tensor_parallel_size}|{dt}"
    )
    _record_and_assert(
        backend_output, reference_output, tag,
        dtype=dtype, atol=atol, rtol=rtol,
    )


# Whitelist of ``(batch_spec_name, pcp_size, dcp_size)`` combinations that are
# in scope for the single-rank precision suite. ``PCP > 1`` prefill / mixed
# requires per-rank PCP scheduler metadata (``q_head_idx``, ``q_tail_idx``,
# ``q_full_idx``, ...) that only a multi-rank job can produce faithfully -- so
# those combos are intentionally absent from the matrix (covered structurally
# by ``tests/ut/attention/test_sfa_cp.py``).
_TOPOLOGIES_ALL = [(2, 2), (2, 1), (1, 2), (1, 4)]
_TOPOLOGIES_PCP1 = [(1, 2), (1, 4)]
_TEST_CASES: list[tuple[str, int, int]] = [
    (name, pcp, dcp)
    for name in BATCH_SPECS
    for (pcp, dcp) in (
        _TOPOLOGIES_ALL if _infer_mode(BATCH_SPECS[name]) in ("decode", "mtp")
        else _TOPOLOGIES_PCP1
    )
]


@pytest.mark.parametrize(
    "batch_spec_name,pcp_size,dcp_size",
    _TEST_CASES,
    ids=[f"{n}-pcp{p}-dcp{d}" for (n, p, d) in _TEST_CASES],
)
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-V3.2-Exp"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_sfa_cp_correctness(
    batch_spec_name: str,
    model: str,
    pcp_size: int,
    dcp_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> None:
    """Test SFA-CP correctness across workload, topology, dtype, and TP size."""
    _test_sfa_cp_correctness(
        BATCH_SPECS[batch_spec_name],
        model,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
    )


def test_sfa_cp_build_prefill_compact_block_metadata_layout() -> None:
    """Validate kernel-facing prefill block-table remapping."""
    device = torch.device("npu")
    cp_size = 4
    block_arange = torch.arange(cp_size, dtype=torch.int32, device=device)

    block_table = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 1, 2],
        ],
        dtype=torch.int32,
        device=device,
    )

    fake_self = MagicMock()
    fake_self.block_arange_buffer = block_arange
    valid_block_ids, block_table_cp = AscendSFACPMetadataBuilder.build_prefill_compact_block_metadata(
        fake_self, block_table, num_decodes=2
    )

    assert valid_block_ids.numel() == 6, (
        f"expected 6 unique prefill blocks, got {valid_block_ids.numel()}"
    )
    assert block_table_cp.shape == (2, 4 * cp_size), (
        f"block_table_cp has wrong shape {tuple(block_table_cp.shape)}"
    )

    pos_of_1 = (valid_block_ids == 1).nonzero(as_tuple=False).item()
    assert int(block_table_cp[0, 0].item()) == pos_of_1
    pos_of_2 = (valid_block_ids == 2).nonzero(as_tuple=False).item()
    assert int(block_table_cp[0, cp_size].item()) == pos_of_2
