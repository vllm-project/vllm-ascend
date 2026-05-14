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
"""NPU precision tests: ``AscendMlaCPImpl`` vs fp32 reference (``mla_cp.py``)."""

import math
import sys
from contextlib import ExitStack
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm.config import VllmConfig  # noqa: E402

from tests.ut.attention.utils import BatchSpec, create_vllm_config  # noqa: E402
from tests.ut.conftest import npu_test  # noqa: E402
from vllm_ascend.attention.attention_v1 import AscendAttentionState  # noqa: E402
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaCPImpl  # noqa: E402
from vllm_ascend.attention.mla_v1 import AscendMLAImpl  # noqa: E402
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

_BLOCK_SIZE = 128
_TEST_NUM_HEADS = 16
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128

DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2
# fp16 carries an extra ULP of cumulative rounding through MLA's W_UV
# down-projection that SFA does not have, so single edge-cases at
# near-zero reference values can land 1 ULP outside the bf16 envelope.
# Widening here to 2e-2 still represents <0.5% peak-relative error
# (peaks are ~4 in mixed_prefill) -- well inside the "<1%" requirement
# enforced by the signal-relative assertion in ``_record_and_assert``.
FP16_RTOL = 2e-2
FP16_ATOL = 2e-2

_MAX_SIG_REL_ERR = 1e-2      # max |out-ref| / peak |ref|
_MAX_MEAN_SIG_ERR = 5e-3     # mean |out-ref| / mean |ref|
_MAX_REL_ERR = 1e-2          # max per-element rel err where |ref| >= floor
_SIG_FLOOR_FRAC = 5e-1       # floor = this fraction of peak |ref|


def _validate_spec(spec: BatchSpec) -> None:
    """Require ``query_len <= seq_len`` per request."""
    for s, q in zip(spec.seq_lens, spec.query_lens):
        assert q <= s, f"query_len ({q}) must not exceed seq_len ({s})"


_VLLM_CONFIG_CACHE: dict = {}


def _get_vllm_config(
    model: str,
    dtype: torch.dtype,
    *,
    max_model_len: int = 8192,
    tensor_parallel_size: int = 1,
) -> VllmConfig:
    """Cached ``VllmConfig`` for MLA-CP (fp8 quant stripped; heads capped).

    Mirrors the ``test_sfa_cp_precision.py`` strategy: ``tensor_parallel_size > 1``
    is **simulated** by keeping real TP=1 on the underlying ``VllmConfig``
    (no distributed init required) and instead dividing the test-friendly
    head count by TP so the kernel runs with the post-TP-shard head budget.
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
        max_num_seqs=256,
        max_num_batched_tokens=max(8192, max_model_len * 2),
        enable_chunked_prefill=True,
        hf_overrides={"quantization_config": None},
        hf_config_override={
            "num_attention_heads": sim_num_heads,
            "num_key_value_heads": 1,
            "kv_lora_rank": _KV_LORA_RANK,
            "qk_nope_head_dim": _QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
            "v_head_dim": _V_HEAD_DIM,
        },
    )
    _VLLM_CONFIG_CACHE[key] = cfg
    return cfg


# Spec name prefixes drive the MLA-CP branch under test.
BATCH_SPECS: dict[str, BatchSpec] = {
    "decode_single": BatchSpec(
        seq_lens=[1024], query_lens=[1], name="decode_single"),
    "decode_small_batch": BatchSpec(
        seq_lens=[64, 128, 256, 512], query_lens=[1, 1, 1, 1],
        name="decode_small_batch"),
    "decode_large_batch": BatchSpec(
        seq_lens=[2048] * 16, query_lens=[1] * 16, name="decode_large_batch"),
    "mtp_1_plus_1": BatchSpec(
        seq_lens=[256, 512, 1024], query_lens=[2, 2, 2], name="mtp_1_plus_1"),
    "mtp_1_plus_3": BatchSpec(
        seq_lens=[256, 512, 1024, 1536], query_lens=[4, 4, 4, 4],
        name="mtp_1_plus_3"),
    "mtp_1_plus_7": BatchSpec(
        seq_lens=[512, 1024, 2048], query_lens=[8, 8, 8], name="mtp_1_plus_7"),
    "prefill_single": BatchSpec(
        seq_lens=[256], query_lens=[256], name="prefill_single"),
    "prefill_small_batch": BatchSpec(
        seq_lens=[128, 256, 384], query_lens=[128, 256, 384],
        name="prefill_small_batch"),
    "prefill_medium_batch": BatchSpec(
        seq_lens=[512, 1024], query_lens=[512, 1024],
        name="prefill_medium_batch"),
    "mixed_small": BatchSpec(
        seq_lens=[64, 128, 256, 512], query_lens=[1, 1, 64, 128],
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


def _make_synthetic_kv_contexts(
    seq_lens: list[int],
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-request K context (latent space) generated independently so values
    across requests are uncorrelated."""
    k_nope = [
        torch.randn(s, kv_lora_rank, dtype=dtype, device=device)
        for s in seq_lens
    ]
    k_pe = [
        torch.randn(s, qk_rope_head_dim, dtype=dtype, device=device)
        for s in seq_lens
    ]
    return k_nope, k_pe


def _build_paged_kv_cache(
    seq_lens: list[int],
    k_nope_contexts: list[torch.Tensor],
    k_pe_contexts: list[torch.Tensor],
    block_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paged ``k_nope`` / ``k_pe`` caches for the decode / MTP path."""
    batch_size = len(seq_lens)
    blocks_per_seq = [(s + block_size - 1) // block_size for s in seq_lens]
    total_blocks = sum(blocks_per_seq) + 1
    max_blocks_per_seq = max(blocks_per_seq)

    k_nope_cache = torch.zeros(
        total_blocks, block_size, 1, kv_lora_rank, dtype=dtype, device=device,
    )
    k_pe_cache = torch.zeros(
        total_blocks, block_size, 1, qk_rope_head_dim, dtype=dtype, device=device,
    )
    block_table = torch.zeros(
        batch_size, max_blocks_per_seq, dtype=torch.int32, device=device,
    )

    next_block_id = 1
    for b, s_len in enumerate(seq_lens):
        n_blocks = blocks_per_seq[b]
        for i in range(n_blocks):
            block_id = next_block_id
            block_table[b, i] = block_id
            tok_start = i * block_size
            tok_end = min(tok_start + block_size, s_len)
            length = tok_end - tok_start
            k_nope_cache[block_id, :length, 0, :] = k_nope_contexts[b][tok_start:tok_end]
            k_pe_cache[block_id, :length, 0, :] = k_pe_contexts[b][tok_start:tok_end]
            next_block_id += 1

    return k_nope_cache, k_pe_cache, block_table


def _make_w_uv(
    num_heads: int,
    kv_lora_rank: int,
    v_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(
        num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device=device,
    ) * (1.0 / math.sqrt(kv_lora_rank))


def _decode_reference(
    q_nope_latent: torch.Tensor,
    q_pe_latent: torch.Tensor,
    k_nope_contexts: list[torch.Tensor],
    k_pe_contexts: list[torch.Tensor],
    seq_lens: list[int],
    query_lens: list[int],
    W_UV: torch.Tensor,
    scale: float,
    causal: bool,
    num_heads: int,
    v_head_dim: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Pure-PyTorch dense MLA decode baseline in fp32 (latent-space attention
    followed by ``W_UV`` projection back into ``v_head_dim``)."""
    outputs: list[torch.Tensor] = []
    cum_q = 0
    for b, (s_len, q_len) in enumerate(zip(seq_lens, query_lens)):
        ctx_len = s_len - q_len

        K_nope = k_nope_contexts[b].float()
        K_pe = k_pe_contexts[b].float()
        V_lat = K_nope
        K_full = torch.cat([K_nope, K_pe], dim=-1)

        for j in range(q_len):
            t = cum_q + j
            valid_end = ctx_len + j + 1 if causal else s_len

            q_n = q_nope_latent[t].float()
            q_p = q_pe_latent[t].float()
            Q = torch.cat([q_n, q_p], dim=-1)

            K_b = K_full[:valid_end]
            V_b = V_lat[:valid_end]

            scores = (Q @ K_b.transpose(0, 1)) * scale
            attn = torch.softmax(scores, dim=-1)
            outputs.append(attn @ V_b)

        cum_q += q_len

    O_lat = torch.stack(outputs, dim=0)
    O_proj = torch.bmm(O_lat.transpose(0, 1), W_UV.float())
    O_final = O_proj.transpose(0, 1).contiguous()
    return O_final.reshape(O_final.shape[0], num_heads * v_head_dim).to(out_dtype)


def _prefill_reference(
    q_nope_full: list[torch.Tensor],
    q_pe_full: list[torch.Tensor],
    k_nope_full_per_req: list[torch.Tensor],
    k_pe_full_per_req: list[torch.Tensor],
    v_full_per_req: list[torch.Tensor],
    scale: float,
    num_heads: int,
    v_head_dim: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Pure-PyTorch dense MLA prefill baseline in fp32 (causal)."""
    outputs: list[torch.Tensor] = []
    for q_nope, q_pe, k_nope, k_pe, v in zip(
        q_nope_full, q_pe_full, k_nope_full_per_req, k_pe_full_per_req, v_full_per_req,
    ):
        q_len = q_nope.shape[0]
        Q = torch.cat([q_nope, q_pe], dim=-1).float()
        K = torch.cat([k_nope, k_pe], dim=-1).float()
        V = v.float()

        scores = torch.matmul(
            Q.transpose(0, 1), K.transpose(0, 1).transpose(-1, -2),
        ) * scale
        causal_mask = torch.triu(
            torch.ones(q_len, q_len, dtype=torch.bool, device=Q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, V.transpose(0, 1)).transpose(0, 1).contiguous()
        outputs.append(o.reshape(q_len, num_heads * v_head_dim).to(out_dtype))

    return torch.cat(outputs, dim=0)


def _make_fake_cp_group(world_size: int) -> MagicMock:
    """Single-rank fake (PCP or DCP) collective group.

    ``all_gather(t, dim)`` is mocked as ``cat([t] * world_size, dim=dim)``
    so downstream view / permute / npu_attention_update shapes line up.
    For ``world_size == 1`` it is the identity.
    """
    grp = MagicMock()
    grp.world_size = world_size
    grp.rank_in_group = 0
    grp.device_group = MagicMock() if world_size > 1 else None
    if world_size > 1:
        grp.all_gather = MagicMock(
            side_effect=lambda t, dim: torch.cat([t] * world_size, dim=dim),
        )
    else:
        grp.all_gather = MagicMock(side_effect=lambda t, dim: t)
    return grp


def _patch_distributed_groups_cp(pcp_size: int, dcp_size: int) -> list:
    """Patch CP distributed groups + collectives for single-rank simulation.

    Mirrors the ``test_sfa_cp_precision.py`` strategy: ``pcp_size`` /
    ``dcp_size`` can be > 1 even though only one NPU is available, by
    providing fake groups + side-effect mocks for the collectives that
    ``_process_attn_out_lse`` / ``_npu_attention_update`` exercise on the
    decode path. ``torch.distributed.all_to_all_single`` is mocked as
    ``output.copy_(input)``: in a real multi-rank run, all-to-all permutes
    head-group ownership across DCP ranks; with identical per-rank data
    the no-op preserves the data the kernel-update pipeline expects.
    """
    fake_pcp = _make_fake_cp_group(pcp_size)
    fake_dcp = _make_fake_cp_group(dcp_size)

    def _fake_all_to_all_single(output, input_, *args, **kwargs):
        output.copy_(input_)
        return None

    common_cp = "vllm_ascend.attention.context_parallel.common_cp"
    return [
        patch(f"{common_cp}.get_pcp_group", return_value=fake_pcp),
        patch(f"{common_cp}.get_dcp_group", return_value=fake_dcp),
        patch(f"{common_cp}.get_decode_context_model_parallel_world_size",
              return_value=dcp_size),
        patch("torch.distributed.all_to_all_single",
              side_effect=_fake_all_to_all_single),
    ]


def _patch_extra_ctx(module_path: str):
    """Patch ``_EXTRA_CTX`` so the kernel sees a benign forward context."""
    fake_ctx = MagicMock()
    fake_ctx.is_draft_model = False
    fake_ctx.is_draft_model_prefill = False
    fake_ctx.capturing = False
    return patch(f"{module_path}._EXTRA_CTX", fake_ctx)


def _populate_impl_attrs(
    impl: Any,
    *,
    scale: float,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    W_UV: torch.Tensor,
    vllm_config: VllmConfig,
    speculative_config,
    pcp_size: int,
    dcp_size: int,
) -> None:
    """Populate ``AscendMlaCPImpl`` attributes that the kernel reads."""
    impl.vllm_config = vllm_config
    impl.scale = scale
    impl.num_heads = num_heads
    impl.num_kv_heads = 1
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_nope_head_dim = qk_nope_head_dim
    impl.qk_rope_head_dim = qk_rope_head_dim
    impl.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    impl.v_head_dim = v_head_dim
    impl.fa_quant_layer = False
    impl.enable_kv_nz = False
    impl.W_UV = W_UV
    impl.layer_name = "test_layer"
    impl.speculative_config = speculative_config
    impl.pcp_size = pcp_size
    impl.dcp_size = dcp_size
    impl.pcp_rank = 0
    impl.dcp_rank = 0
    impl.pcp_group = None if pcp_size == 1 else MagicMock()
    impl.dcp_group = None if dcp_size == 1 else MagicMock()


def _make_fake_self(*, dtype: torch.dtype, **kwargs) -> MagicMock:
    """``MagicMock`` self for the decode path (delegates ``_v_up_proj`` and
    ``_compute_prefill_context`` to the real unbound implementations)."""
    fake_self = MagicMock()
    _populate_impl_attrs(fake_self, **kwargs)
    fake_self.dtype = dtype
    fake_self._v_up_proj = lambda x: AscendMlaCPImpl._v_up_proj(fake_self, x)
    fake_self._compute_prefill_context = lambda *a, **kw: (
        AscendMLAImpl._compute_prefill_context(fake_self, *a, **kw)
    )
    return fake_self


def _make_real_impl(**kwargs) -> AscendMlaCPImpl:
    """Real ``AscendMlaCPImpl`` instance (used by the prefill path which
    relies on inherited ``AscendMLAImpl._compute_prefill_context``)."""
    impl = object.__new__(AscendMlaCPImpl)
    _populate_impl_attrs(impl, **kwargs)
    return impl


def _build_prefill_attn_mask(device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(device)


def _make_decode_metadata(
    *,
    seq_lens: list[int],
    query_lens: list[int],
    block_table: torch.Tensor,
    attn_state,
    attn_mask: torch.Tensor | None,
    cp_seq_len: list[int],
) -> MagicMock:
    num_decode_tokens = sum(query_lens)
    decode_meta = MagicMock()
    decode_meta.block_table = block_table
    decode_meta.seq_lens_list = list(seq_lens)
    decode_meta.actual_seq_lengths_q = list(range(1, num_decode_tokens + 1))
    decode_meta.attn_mask = attn_mask
    decode_meta.cp_seq_len = cp_seq_len

    attn_metadata = MagicMock()
    attn_metadata.attn_state = attn_state
    attn_metadata.decode = decode_meta
    return attn_metadata


def _make_prefill_metadata(
    *, query_lens: list[int], attn_mask: torch.Tensor,
) -> MagicMock:
    actual_seq_lengths_q = [sum(query_lens[: i + 1]) for i in range(len(query_lens))]
    prefill_meta = MagicMock()
    prefill_meta.actual_seq_lengths_q = actual_seq_lengths_q
    prefill_meta.attn_mask = attn_mask
    prefill_meta.chunked_context = None
    prefill_meta.pcp_metadata = None

    attn_metadata = MagicMock()
    attn_metadata.prefill = prefill_meta
    return attn_metadata


def _run_mla_cp_decode_kernel(
    *,
    q_nope_latent: torch.Tensor,
    q_pe_latent: torch.Tensor,
    k_nope_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    query_lens: list[int],
    attn_state,
    causal: bool,
    speculative_config,
    block_size: int,
    pcp_size: int,
    dcp_size: int,
    dtype: torch.dtype,
    impl_kwargs: dict,
    device: torch.device,
) -> torch.Tensor:
    """Drive ``AscendMlaCPImpl._forward_decode`` with mocked CP collectives.

    Single-rank simulation of PCP / DCP > 1:
      - We provide the FULL local KV (rank 0 "owns everything"),
        ``cp_seq_len`` = full per-request seq_lens, and the full block
        table. The kernel produces the *full* attention for every head it
        sees (no real per-rank chunking).
      - When ``dcp_size > 1`` the production preprocess (``reorg_decode_q``)
        all-gathers ``q_nope`` / ``q_pe`` along the head dim. We emulate
        that by ``repeat(1, dcp_size, 1)`` so the kernel runs with
        ``num_heads * dcp_size`` heads (matching ``_forward_decode``'s own
        head-count branch).
      - The downstream ``_process_attn_out_lse`` + ``_npu_attention_update``
        reduction over ``PCP * DCP`` contributions is therefore over
        identical full-attention outputs (one per virtual rank), which
        reduces back to the same full attention.
    """
    attn_mask = _build_prefill_attn_mask(device) if causal else None

    if attn_state == AscendAttentionState.SpecDecoding:
        cp_seq_len: list[int] = []
        for s_len, q_len in zip(seq_lens, query_lens):
            for j in range(q_len):
                cp_seq_len.append(s_len - q_len + j + 1)
        per_token_rows = []
        for b, q_len in enumerate(query_lens):
            for _ in range(q_len):
                per_token_rows.append(block_table[b])
        block_table = torch.stack(per_token_rows, dim=0).contiguous()
    else:
        cp_seq_len = list(seq_lens)

    attn_metadata = _make_decode_metadata(
        seq_lens=seq_lens,
        query_lens=query_lens,
        block_table=block_table,
        attn_state=attn_state,
        attn_mask=attn_mask,
        cp_seq_len=cp_seq_len,
    )

    q_nope = q_nope_latent
    q_pe = q_pe_latent
    if dcp_size > 1:
        q_nope = q_nope.repeat(1, dcp_size, 1)
        q_pe = q_pe.repeat(1, dcp_size, 1)

    fake_self = _make_fake_self(
        dtype=dtype,
        speculative_config=speculative_config,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        **impl_kwargs,
    )

    with ExitStack() as stack:
        for p in _patch_distributed_groups_cp(pcp_size, dcp_size):
            stack.enter_context(p)
        stack.enter_context(_patch_extra_ctx(
            "vllm_ascend.attention.context_parallel.mla_cp"))
        return AscendMlaCPImpl._forward_decode(
            fake_self,
            q_nope,
            q_pe,
            k_nope_cache,
            k_pe_cache,
            block_size,
            attn_metadata,
        )


def _run_mla_cp_prefill_kernel(
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    value: torch.Tensor,
    k_nope_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    query_lens: list[int],
    pcp_size: int,
    dcp_size: int,
    impl_kwargs: dict,
    device: torch.device,
) -> torch.Tensor:
    """Drive ``AscendMlaCPImpl._forward_prefill`` with mocked CP collectives.

    Single-rank simulation only covers ``pcp_size == 1`` (DCP > 1 OK);
    PCP > 1 prefill requires per-rank PCP scheduler metadata that only a
    real multi-rank job can produce faithfully (consistent with the SFA-CP
    precision matrix, see the topology whitelist below).
    """
    assert pcp_size == 1, "PCP > 1 prefill is out of scope"

    attn_mask = _build_prefill_attn_mask(device)
    attn_metadata = _make_prefill_metadata(
        query_lens=query_lens, attn_mask=attn_mask,
    )

    impl = _make_real_impl(
        speculative_config=None,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        **impl_kwargs,
    )

    with ExitStack() as stack:
        for p in _patch_distributed_groups_cp(pcp_size, dcp_size):
            stack.enter_context(p)
        stack.enter_context(_patch_extra_ctx("vllm_ascend.attention.mla_v1"))
        stack.enter_context(_patch_extra_ctx(
            "vllm_ascend.attention.context_parallel.mla_cp"))
        return AscendMlaCPImpl._forward_prefill(
            impl,
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            value,
            (k_nope_cache, k_pe_cache),
            attn_metadata,
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
    """Assert numerical closeness and record signal-relative metrics.

    The MLA-CP decode kernel returns fp32 (online-softmax LSE reduction is
    accumulated in fp32 and never down-cast), so we cast the backend output
    to the reference dtype before comparing -- this differs from SFA-CP,
    which preserves input dtype end-to-end.
    """
    assert backend_output.shape == reference_output.shape, (
        f"[{tag}] backend shape {tuple(backend_output.shape)} != "
        f"reference shape {tuple(reference_output.shape)}"
    )
    assert torch.isfinite(backend_output).all(), (
        f"[{tag}] MLA-CP attention produced non-finite values"
    )
    if backend_output.dtype != reference_output.dtype:
        backend_output = backend_output.to(reference_output.dtype)

    torch.testing.assert_close(
        backend_output,
        reference_output,
        rtol=rtol,
        atol=atol,
        msg=lambda m: f"[MLA-CP:{tag}] kernel output diverges from baseline. {m}",
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

    print(
        f"[MLA-CP-precision] {tag} "
        f"peak={peak:.4e} max_abs_err={max_abs_err:.4e} "
        f"max_sig_rel_err={max_sig_rel_err * 100:.4f}% "
        f"mean_sig_rel_err={mean_sig_rel_err * 100:.4f}% "
        f"max_rel_err_sig(>={int(_SIG_FLOOR_FRAC * 100)}%peak)="
        f"{max_rel_err_sig * 100:.4f}%"
    )

    assert max_sig_rel_err < _MAX_SIG_REL_ERR, (
        f"[MLA-CP:{tag}] dtype={dtype} signal-relative max error "
        f"{max_sig_rel_err * 100:.4f}% exceeds 1% budget "
        f"(peak={peak:.4e}, max_abs_err={max_abs_err:.4e})"
    )
    assert mean_sig_rel_err < _MAX_MEAN_SIG_ERR, (
        f"[MLA-CP:{tag}] dtype={dtype} signal-relative mean error "
        f"{mean_sig_rel_err * 100:.4f}% exceeds 0.5% drift budget "
        f"(mean_ref_abs={mean_ref_abs:.4e}, mean_abs_err={mean_abs_err:.4e})"
    )
    assert max_rel_err_sig < _MAX_REL_ERR, (
        f"[MLA-CP:{tag}] dtype={dtype} per-element relative error on "
        f">={int(_SIG_FLOOR_FRAC * 100)}%-of-peak elements "
        f"{max_rel_err_sig * 100:.4f}% exceeds 1% budget "
        f"(peak={peak:.4e}, max_abs_err={max_abs_err:.4e})"
    )

    return max_abs_err, max_rel_err_sig


def _test_mla_cp_correctness(
    batch_spec: BatchSpec,
    model: str,
    *,
    pcp_size: int,
    dcp_size: int,
    dtype: torch.dtype = torch.bfloat16,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    tensor_parallel_size: int = 1,
) -> None:
    """Test ``AscendMlaCPImpl`` against a fp32 dense MLA reference."""
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
    num_tokens = batch_spec.compute_num_tokens()

    cache_config = vllm_config.cache_config
    hf_text = vllm_config.model_config.hf_text_config
    block_size = cache_config.block_size
    kv_lora_rank = hf_text.kv_lora_rank
    qk_nope_head_dim = hf_text.qk_nope_head_dim
    qk_rope_head_dim = hf_text.qk_rope_head_dim
    v_head_dim = hf_text.v_head_dim
    num_heads = hf_text.num_attention_heads

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    scale = 1.0 / math.sqrt(qk_head_dim)

    k_nope_contexts, k_pe_contexts = _make_synthetic_kv_contexts(
        seq_lens, kv_lora_rank, qk_rope_head_dim, dtype, device,
    )
    k_nope_cache, k_pe_cache, block_table = _build_paged_kv_cache(
        seq_lens, k_nope_contexts, k_pe_contexts,
        block_size, kv_lora_rank, qk_rope_head_dim, dtype, device,
    )
    q_nope_latent = torch.randn(
        num_tokens, num_heads, kv_lora_rank, dtype=dtype, device=device,
    )
    q_pe_latent = torch.randn(
        num_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=device,
    )
    W_UV = _make_w_uv(num_heads, kv_lora_rank, v_head_dim, dtype, device)

    q_nope_full = [
        torch.randn(q, num_heads, qk_nope_head_dim, dtype=dtype, device=device)
        for q in query_lens
    ]
    q_pe_full = [
        torch.randn(q, num_heads, qk_rope_head_dim, dtype=dtype, device=device)
        for q in query_lens
    ]
    k_nope_full = [
        torch.randn(q, num_heads, qk_nope_head_dim, dtype=dtype, device=device)
        for q in query_lens
    ]
    k_pe_full = [
        torch.randn(q, num_heads, qk_rope_head_dim, dtype=dtype, device=device)
        for q in query_lens
    ]
    v_full = [
        torch.randn(q, num_heads, v_head_dim, dtype=dtype, device=device)
        for q in query_lens
    ]

    impl_kwargs = dict(
        scale=scale,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        W_UV=W_UV,
        vllm_config=vllm_config,
    )

    dt = "bf16" if dtype == torch.bfloat16 else "fp16"
    tag_base = (
        f"{mode}|{batch_spec.name}|pcp={pcp_size}|dcp={dcp_size}"
        f"|tp={tensor_parallel_size}|{dt}"
    )

    if mode in ("decode", "mtp"):
        causal = mode == "mtp"
        if mode == "mtp":
            spec_window = query_lens[0]
            speculative_config = MagicMock()
            speculative_config.num_speculative_tokens = spec_window - 1
            attn_state = AscendAttentionState.SpecDecoding
        else:
            speculative_config = None
            attn_state = AscendAttentionState.DecodeOnly

        backend_output = _run_mla_cp_decode_kernel(
            q_nope_latent=q_nope_latent,
            q_pe_latent=q_pe_latent,
            k_nope_cache=k_nope_cache,
            k_pe_cache=k_pe_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            query_lens=query_lens,
            attn_state=attn_state,
            causal=causal,
            speculative_config=speculative_config,
            block_size=block_size,
            pcp_size=pcp_size,
            dcp_size=dcp_size,
            dtype=dtype,
            impl_kwargs=impl_kwargs,
            device=device,
        )
        reference_output = _decode_reference(
            q_nope_latent=q_nope_latent,
            q_pe_latent=q_pe_latent,
            k_nope_contexts=k_nope_contexts,
            k_pe_contexts=k_pe_contexts,
            seq_lens=seq_lens,
            query_lens=query_lens,
            W_UV=W_UV,
            scale=scale,
            causal=causal,
            num_heads=num_heads,
            v_head_dim=v_head_dim,
            out_dtype=dtype,
        )
        _record_and_assert(
            backend_output, reference_output, tag_base,
            dtype=dtype, atol=atol, rtol=rtol,
        )
        return

    if mode == "prefill":
        backend_output = _run_mla_cp_prefill_kernel(
            q_nope=torch.cat(q_nope_full, dim=0),
            q_pe=torch.cat(q_pe_full, dim=0),
            k_nope=torch.cat(k_nope_full, dim=0),
            k_pe=torch.cat(k_pe_full, dim=0),
            value=torch.cat(v_full, dim=0),
            k_nope_cache=k_nope_cache,
            k_pe_cache=k_pe_cache,
            query_lens=query_lens,
            pcp_size=pcp_size,
            dcp_size=dcp_size,
            impl_kwargs=impl_kwargs,
            device=device,
        )
        reference_output = _prefill_reference(
            q_nope_full=q_nope_full,
            q_pe_full=q_pe_full,
            k_nope_full_per_req=k_nope_full,
            k_pe_full_per_req=k_pe_full,
            v_full_per_req=v_full,
            scale=scale,
            num_heads=num_heads,
            v_head_dim=v_head_dim,
            out_dtype=dtype,
        )
        _record_and_assert(
            backend_output, reference_output, tag_base,
            dtype=dtype, atol=atol, rtol=rtol,
        )
        return

    n_decode_reqs = sum(1 for q in query_lens if q == 1)
    assert all(q == 1 for q in query_lens[:n_decode_reqs]), (
        f"mixed spec {batch_spec.name}: decode requests must come first"
    )
    assert all(q > 1 for q in query_lens[n_decode_reqs:]), (
        f"mixed spec {batch_spec.name}: prefill requests must come after decode"
    )

    decode_backend = _run_mla_cp_decode_kernel(
        q_nope_latent=q_nope_latent[:n_decode_reqs],
        q_pe_latent=q_pe_latent[:n_decode_reqs],
        k_nope_cache=k_nope_cache,
        k_pe_cache=k_pe_cache,
        block_table=block_table[:n_decode_reqs],
        seq_lens=seq_lens[:n_decode_reqs],
        query_lens=query_lens[:n_decode_reqs],
        attn_state=AscendAttentionState.ChunkedPrefill,
        causal=False,
        speculative_config=None,
        block_size=block_size,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        dtype=dtype,
        impl_kwargs=impl_kwargs,
        device=device,
    )
    decode_reference = _decode_reference(
        q_nope_latent=q_nope_latent[:n_decode_reqs],
        q_pe_latent=q_pe_latent[:n_decode_reqs],
        k_nope_contexts=k_nope_contexts[:n_decode_reqs],
        k_pe_contexts=k_pe_contexts[:n_decode_reqs],
        seq_lens=seq_lens[:n_decode_reqs],
        query_lens=query_lens[:n_decode_reqs],
        W_UV=W_UV,
        scale=scale,
        causal=False,
        num_heads=num_heads,
        v_head_dim=v_head_dim,
        out_dtype=dtype,
    )
    _record_and_assert(
        decode_backend, decode_reference, f"{tag_base}|mixed_decode",
        dtype=dtype, atol=atol, rtol=rtol,
    )

    prefill_backend = _run_mla_cp_prefill_kernel(
        q_nope=torch.cat(q_nope_full[n_decode_reqs:], dim=0),
        q_pe=torch.cat(q_pe_full[n_decode_reqs:], dim=0),
        k_nope=torch.cat(k_nope_full[n_decode_reqs:], dim=0),
        k_pe=torch.cat(k_pe_full[n_decode_reqs:], dim=0),
        value=torch.cat(v_full[n_decode_reqs:], dim=0),
        k_nope_cache=k_nope_cache,
        k_pe_cache=k_pe_cache,
        query_lens=query_lens[n_decode_reqs:],
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        impl_kwargs=impl_kwargs,
        device=device,
    )
    prefill_reference = _prefill_reference(
        q_nope_full=q_nope_full[n_decode_reqs:],
        q_pe_full=q_pe_full[n_decode_reqs:],
        k_nope_full_per_req=k_nope_full[n_decode_reqs:],
        k_pe_full_per_req=k_pe_full[n_decode_reqs:],
        v_full_per_req=v_full[n_decode_reqs:],
        scale=scale,
        num_heads=num_heads,
        v_head_dim=v_head_dim,
        out_dtype=dtype,
    )
    _record_and_assert(
        prefill_backend, prefill_reference, f"{tag_base}|mixed_prefill",
        dtype=dtype, atol=atol, rtol=rtol,
    )


# Whitelist of ``(batch_spec_name, pcp_size, dcp_size)`` combinations that are
# in scope for the single-rank precision suite. ``PCP > 1`` prefill / mixed
# requires per-rank PCP scheduler metadata (``q_head_idx``, ``q_tail_idx``,
# ``q_full_idx``, ...) that only a multi-rank job can produce faithfully -- so
# those combos are intentionally absent from the matrix (covered structurally
# by ``tests/ut/attention/test_mla_cp.py``). The non-CP ``(1, 1)`` baseline is
# covered by ``tests/ut/attention/test_mla_v1_precision.py``.
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


@npu_test()
@pytest.mark.parametrize(
    "batch_spec_name,pcp_size,dcp_size",
    _TEST_CASES,
    ids=[f"{n}-pcp{p}-dcp{d}" for (n, p, d) in _TEST_CASES],
)
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-V3.2-Exp"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_mla_cp_correctness(
    batch_spec_name: str,
    model: str,
    pcp_size: int,
    dcp_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> None:
    """Test MLA-CP correctness across workload, topology, dtype, and TP size."""
    atol = FP16_ATOL if dtype == torch.float16 else DEFAULT_ATOL
    rtol = FP16_RTOL if dtype == torch.float16 else DEFAULT_RTOL
    _test_mla_cp_correctness(
        BATCH_SPECS[batch_spec_name],
        model,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        dtype=dtype,
        atol=atol,
        rtol=rtol,
        tensor_parallel_size=tensor_parallel_size,
    )
