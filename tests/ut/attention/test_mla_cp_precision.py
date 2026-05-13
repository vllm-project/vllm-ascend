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

# ruff: noqa: I001 - import order: load ``_torch_npu_inductor_shim`` before ``vllm`` / ``vllm_ascend``.
import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

import tests.ut.attention._torch_npu_inductor_shim as _torch_npu_inductor_shim  # noqa: F401
from vllm.config import VllmConfig

from tests.ut.attention.utils import BatchSpec, create_vllm_config
from tests.ut.conftest import npu_test
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaCPImpl
from vllm_ascend.attention.mla_v1 import AscendMLAImpl

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 1

BLOCK_SIZE = 128

_MLA_UT_MAX_MODEL_LEN = 8192
_MLA_VLLM_CONFIG_CACHE: dict[tuple[str, str, int], VllmConfig] = {}

QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)

DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2

_MAX_SIG_REL_ERR = 1e-2
_MAX_MEAN_SIG_ERR = 5e-3
_MAX_REL_ERR = 1e-2
_SIG_FLOOR_FRAC = 5e-1


def _get_mla_precision_vllm_config(
    model: str,
    dtype: torch.dtype,
    *,
    tensor_parallel_size: int = 1,
) -> VllmConfig:
    """Cached config; MLA dims via ``hf_config_override``; DeepSeek strips ``quantization_config``."""
    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float16"
    cache_key = (model, dtype_str, tensor_parallel_size)
    cached = _MLA_VLLM_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    hf_overrides: dict | None = None
    if "deepseek" in model.lower():
        hf_overrides = {"quantization_config": None}
    cfg = create_vllm_config(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=_MLA_UT_MAX_MODEL_LEN,
        dtype=dtype_str,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=4096,
        max_num_seqs=256,
        max_num_batched_tokens=max(8192, _MLA_UT_MAX_MODEL_LEN * 2),
        enable_chunked_prefill=True,
        hf_overrides=hf_overrides or {},
        hf_config_override={
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "kv_lora_rank": KV_LORA_RANK,
            "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "v_head_dim": V_HEAD_DIM,
        },
    )
    _MLA_VLLM_CONFIG_CACHE[cache_key] = cfg
    return cfg


PURE_DECODE_SPECS: dict[str, BatchSpec] = {
    "pure_decode_single": BatchSpec(
        seq_lens=[1024],
        query_lens=[1],
        name="pure_decode_single",
    ),
    "pure_decode_small_batch": BatchSpec(
        seq_lens=[64, 128, 256, 512],
        query_lens=[1, 1, 1, 1],
        name="pure_decode_small_batch",
    ),
    "pure_decode_large_batch": BatchSpec(
        seq_lens=[2048] * 16,
        query_lens=[1] * 16,
        name="pure_decode_large_batch",
    ),
}

PURE_PREFILL_SPECS: dict[str, BatchSpec] = {
    "pure_prefill_single": BatchSpec(
        seq_lens=[256],
        query_lens=[256],
        name="pure_prefill_single",
    ),
    "pure_prefill_small_batch": BatchSpec(
        seq_lens=[128, 256, 384],
        query_lens=[128, 256, 384],
        name="pure_prefill_small_batch",
    ),
    "pure_prefill_medium_batch": BatchSpec(
        seq_lens=[512, 1024],
        query_lens=[512, 1024],
        name="pure_prefill_medium_batch",
    ),
}

MIXED_SPECS: dict[str, BatchSpec] = {
    "mixed_small": BatchSpec(
        seq_lens=[64, 128, 256, 512],
        query_lens=[1, 1, 64, 128],
        name="mixed_small",
    ),
    "mixed_medium": BatchSpec(
        seq_lens=[1024, 1536, 2048, 256, 512],
        query_lens=[1, 1, 1, 64, 128],
        name="mixed_medium",
    ),
}

MTP_SPECS: dict[str, BatchSpec] = {
    "mtp_1_plus_1": BatchSpec(
        seq_lens=[256, 512, 1024],
        query_lens=[2, 2, 2],
        name="mtp_1_plus_1",
    ),
    "mtp_1_plus_3": BatchSpec(
        seq_lens=[256, 512, 1024, 1536],
        query_lens=[4, 4, 4, 4],
        name="mtp_1_plus_3",
    ),
    "mtp_1_plus_7": BatchSpec(
        seq_lens=[512, 1024, 2048],
        query_lens=[8, 8, 8],
        name="mtp_1_plus_7",
    ),
}

_SPECS_BY_SCENARIO: dict[str, dict[str, BatchSpec]] = {
    "pure_decode": PURE_DECODE_SPECS,
    "pure_prefill": PURE_PREFILL_SPECS,
    "mixed": MIXED_SPECS,
    "mtp": MTP_SPECS,
}


def _mla_cp_precision_cases() -> list[tuple[str, str, torch.dtype]]:
    cases: list[tuple[str, str, torch.dtype]] = []
    for name in PURE_DECODE_SPECS:
        for dtype in (torch.bfloat16, torch.float16):
            cases.append(("pure_decode", name, dtype))
    for name in PURE_PREFILL_SPECS:
        cases.append(("pure_prefill", name, torch.bfloat16))
    for name in MIXED_SPECS:
        cases.append(("mixed", name, torch.bfloat16))
    for name in MTP_SPECS:
        cases.append(("mtp", name, torch.bfloat16))
    return cases


MLA_CP_PRECISION_CASES = _mla_cp_precision_cases()


def _mla_cp_case_id(case: tuple[str, str, torch.dtype]) -> str:
    scenario, name, dtype = case
    dt = "bf16" if dtype == torch.bfloat16 else "fp16"
    return f"{scenario}|{name}|{dt}"


@dataclass
class _MLAFixture:
    batch_spec: BatchSpec
    model: str
    vllm_config: VllmConfig

    k_nope_cache: torch.Tensor
    k_pe_cache: torch.Tensor
    block_table: torch.Tensor

    k_nope_contexts: list[torch.Tensor]
    k_pe_contexts: list[torch.Tensor]

    q_nope_latent: torch.Tensor
    q_pe_latent: torch.Tensor

    q_nope_full: list[torch.Tensor] = field(default_factory=list)
    q_pe_full: list[torch.Tensor] = field(default_factory=list)
    k_nope_full_per_req: list[torch.Tensor] = field(default_factory=list)
    k_pe_full_per_req: list[torch.Tensor] = field(default_factory=list)
    v_full_per_req: list[torch.Tensor] = field(default_factory=list)

    W_UV: torch.Tensor = None


def _build_paged_kv_cache(
    seq_lens: list[int],
    k_nope_contexts: list[torch.Tensor],
    k_pe_contexts: list[torch.Tensor],
    block_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(seq_lens)
    blocks_per_seq = [(s + block_size - 1) // block_size for s in seq_lens]
    total_blocks = sum(blocks_per_seq) + 1
    max_blocks_per_seq = max(blocks_per_seq)

    k_nope_cache = torch.zeros(
        total_blocks,
        block_size,
        num_kv_heads,
        kv_lora_rank,
        dtype=dtype,
        device=device,
    )
    k_pe_cache = torch.zeros(
        total_blocks,
        block_size,
        num_kv_heads,
        qk_rope_head_dim,
        dtype=dtype,
        device=device,
    )
    block_table = torch.zeros(
        batch_size,
        max_blocks_per_seq,
        dtype=torch.int32,
        device=device,
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
    num_heads: int, kv_lora_rank: int, v_head_dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.randn(num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device=device) * (
        1.0 / math.sqrt(kv_lora_rank)
    )


def _build_fixture(
    batch_spec: BatchSpec,
    dtype: torch.dtype,
    device: torch.device,
    model: str,
    *,
    tensor_parallel_size: int = 1,
) -> _MLAFixture:
    vllm_config = _get_mla_precision_vllm_config(
        model,
        dtype,
        tensor_parallel_size=tensor_parallel_size,
    )
    num_tokens = batch_spec.compute_num_tokens()
    assert num_tokens == sum(batch_spec.query_lens)

    seq_lens = list(batch_spec.seq_lens)

    k_nope_contexts = [torch.randn(s, KV_LORA_RANK, dtype=dtype, device=device) for s in seq_lens]
    k_pe_contexts = [torch.randn(s, QK_ROPE_HEAD_DIM, dtype=dtype, device=device) for s in seq_lens]

    k_nope_cache, k_pe_cache, block_table = _build_paged_kv_cache(
        seq_lens=seq_lens,
        k_nope_contexts=k_nope_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=BLOCK_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        dtype=dtype,
        device=device,
    )

    q_nope_latent = torch.randn(num_tokens, NUM_HEADS, KV_LORA_RANK, dtype=dtype, device=device)
    q_pe_latent = torch.randn(num_tokens, NUM_HEADS, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)

    q_nope_full: list[torch.Tensor] = []
    q_pe_full: list[torch.Tensor] = []
    k_nope_full_per_req: list[torch.Tensor] = []
    k_pe_full_per_req: list[torch.Tensor] = []
    v_full_per_req: list[torch.Tensor] = []
    for q_len in batch_spec.query_lens:
        q_nope_full.append(torch.randn(q_len, NUM_HEADS, QK_NOPE_HEAD_DIM, dtype=dtype, device=device))
        q_pe_full.append(torch.randn(q_len, NUM_HEADS, QK_ROPE_HEAD_DIM, dtype=dtype, device=device))
        k_nope_full_per_req.append(torch.randn(q_len, NUM_HEADS, QK_NOPE_HEAD_DIM, dtype=dtype, device=device))
        k_pe_full_per_req.append(torch.randn(q_len, NUM_HEADS, QK_ROPE_HEAD_DIM, dtype=dtype, device=device))
        v_full_per_req.append(torch.randn(q_len, NUM_HEADS, V_HEAD_DIM, dtype=dtype, device=device))

    W_UV = _make_w_uv(NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM, dtype, device)

    return _MLAFixture(
        batch_spec=batch_spec,
        model=model,
        vllm_config=vllm_config,
        k_nope_cache=k_nope_cache,
        k_pe_cache=k_pe_cache,
        block_table=block_table,
        k_nope_contexts=k_nope_contexts,
        k_pe_contexts=k_pe_contexts,
        q_nope_latent=q_nope_latent,
        q_pe_latent=q_pe_latent,
        q_nope_full=q_nope_full,
        q_pe_full=q_pe_full,
        k_nope_full_per_req=k_nope_full_per_req,
        k_pe_full_per_req=k_pe_full_per_req,
        v_full_per_req=v_full_per_req,
        W_UV=W_UV,
    )


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
    out_dtype: torch.dtype,
) -> torch.Tensor:
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

            q_nope = q_nope_latent[t].float()
            q_pe = q_pe_latent[t].float()
            Q = torch.cat([q_nope, q_pe], dim=-1)

            K_b = K_full[:valid_end]
            V_b = V_lat[:valid_end]

            scores = (Q @ K_b.transpose(0, 1)) * scale
            attn = torch.softmax(scores, dim=-1)
            o_lat = attn @ V_b
            outputs.append(o_lat)

        cum_q += q_len

    O_lat = torch.stack(outputs, dim=0)
    O_proj = torch.bmm(O_lat.transpose(0, 1), W_UV.float())
    O_final = O_proj.transpose(0, 1).contiguous()
    O_final = O_final.reshape(O_final.shape[0], NUM_HEADS * V_HEAD_DIM)
    return O_final.to(out_dtype)


def _prefill_reference(
    q_nope_full: list[torch.Tensor],
    q_pe_full: list[torch.Tensor],
    k_nope_full_per_req: list[torch.Tensor],
    k_pe_full_per_req: list[torch.Tensor],
    v_full_per_req: list[torch.Tensor],
    scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for q_nope, q_pe, k_nope, k_pe, v in zip(
        q_nope_full, q_pe_full, k_nope_full_per_req, k_pe_full_per_req, v_full_per_req
    ):
        q_len = q_nope.shape[0]
        Q = torch.cat([q_nope, q_pe], dim=-1).float()
        K = torch.cat([k_nope, k_pe], dim=-1).float()
        V = v.float()

        scores = torch.matmul(Q.transpose(0, 1), K.transpose(0, 1).transpose(-1, -2)) * scale

        causal_mask = torch.triu(
            torch.ones(q_len, q_len, dtype=torch.bool, device=Q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)

        o = torch.matmul(attn, V.transpose(0, 1))
        o = o.transpose(0, 1).contiguous()
        outputs.append(o.reshape(q_len, NUM_HEADS * V_HEAD_DIM).to(out_dtype))

    return torch.cat(outputs, dim=0)


def _init_mla_cp_impl_common(
    impl: Any,
    *,
    W_UV: torch.Tensor,
    speculative_config,
    vllm_config: VllmConfig,
    pcp_size: int = 1,
    dcp_size: int = 1,
) -> None:
    impl.vllm_config = vllm_config
    impl.scale = SCALE
    impl.num_heads = NUM_HEADS
    impl.num_kv_heads = NUM_KV_HEADS
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_nope_head_dim = QK_NOPE_HEAD_DIM
    impl.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    impl.qk_head_dim = QK_HEAD_DIM
    impl.v_head_dim = V_HEAD_DIM
    impl.fa_quant_layer = False
    impl.enable_kv_nz = False
    impl.speculative_config = speculative_config
    impl.W_UV = W_UV
    impl.layer_name = "test_layer"
    impl.pcp_size = pcp_size
    impl.dcp_size = dcp_size
    impl.pcp_rank = 0
    impl.dcp_rank = 0
    impl.pcp_group = None if pcp_size == 1 else MagicMock()
    impl.dcp_group = None if dcp_size == 1 else MagicMock()


def _make_fake_self(
    *,
    W_UV: torch.Tensor,
    speculative_config,
    dtype: torch.dtype,
    vllm_config: VllmConfig,
    pcp_size: int = 1,
    dcp_size: int = 1,
) -> MagicMock:
    fake_self = MagicMock()
    _init_mla_cp_impl_common(
        fake_self,
        W_UV=W_UV,
        speculative_config=speculative_config,
        vllm_config=vllm_config,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
    )
    fake_self.dtype = dtype
    fake_self._v_up_proj = lambda x: AscendMlaCPImpl._v_up_proj(fake_self, x)
    fake_self._compute_prefill_context = lambda *args, **kwargs: (
        AscendMLAImpl._compute_prefill_context(fake_self, *args, **kwargs)
    )
    return fake_self


def _make_real_cp_impl(
    *,
    W_UV: torch.Tensor,
    speculative_config,
    vllm_config: VllmConfig,
    pcp_size: int = 1,
    dcp_size: int = 1,
) -> AscendMlaCPImpl:
    impl = object.__new__(AscendMlaCPImpl)
    _init_mla_cp_impl_common(
        impl,
        W_UV=W_UV,
        speculative_config=speculative_config,
        vllm_config=vllm_config,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
    )
    return impl


def _make_decode_metadata(
    *,
    seq_lens: list[int],
    query_lens: list[int],
    block_table: torch.Tensor,
    attn_state,
    attn_mask: torch.Tensor | None,
    cp_seq_len: list[int] | None = None,
) -> MagicMock:
    num_decode_tokens = sum(query_lens)
    if cp_seq_len is None:
        cp_seq_len = list(seq_lens)

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
    *,
    query_lens: list[int],
    attn_mask: torch.Tensor,
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


def _build_prefill_attn_mask(device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(device)


def _patch_extra_ctx(module_path: str):
    fake_ctx = MagicMock()
    fake_ctx.is_draft_model = False
    fake_ctx.is_draft_model_prefill = False
    fake_ctx.capturing = False
    return patch(f"{module_path}._EXTRA_CTX", fake_ctx)


def _patch_distributed_groups_size_1():
    fake_pcp = MagicMock()
    fake_pcp.world_size = 1
    fake_pcp.rank_in_group = 0
    fake_pcp.device_group = None

    fake_dcp = MagicMock()
    fake_dcp.world_size = 1
    fake_dcp.rank_in_group = 0
    fake_dcp.device_group = None

    return [
        patch(
            "vllm_ascend.attention.context_parallel.common_cp.get_pcp_group",
            return_value=fake_pcp,
        ),
        patch(
            "vllm_ascend.attention.context_parallel.common_cp.get_dcp_group",
            return_value=fake_dcp,
        ),
        patch(
            "vllm_ascend.attention.context_parallel.common_cp.get_decode_context_model_parallel_world_size",
            return_value=1,
        ),
    ]


def _run_forward_decode(
    fixture: _MLAFixture,
    *,
    speculative_config,
    attn_state,
    causal: bool,
    device: torch.device,
    dtype: torch.dtype,
    decode_token_slice: slice | None = None,
) -> torch.Tensor:
    batch_spec = fixture.batch_spec
    if decode_token_slice is None:
        seq_lens = list(batch_spec.seq_lens)
        query_lens = list(batch_spec.query_lens)
        q_nope = fixture.q_nope_latent
        q_pe = fixture.q_pe_latent
        block_table = fixture.block_table
    else:
        n_decode_reqs = decode_token_slice.stop
        seq_lens = list(batch_spec.seq_lens[:n_decode_reqs])
        query_lens = list(batch_spec.query_lens[:n_decode_reqs])
        n_tokens = sum(query_lens)
        q_nope = fixture.q_nope_latent[:n_tokens]
        q_pe = fixture.q_pe_latent[:n_tokens]
        block_table = fixture.block_table[:n_decode_reqs]

    if causal:
        attn_mask = _build_prefill_attn_mask(device)
    else:
        attn_mask = None

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

    fake_self = _make_fake_self(
        W_UV=fixture.W_UV,
        speculative_config=speculative_config,
        dtype=dtype,
        vllm_config=fixture.vllm_config,
        pcp_size=1,
        dcp_size=1,
    )

    with ExitStack() as stack:
        for p in _patch_distributed_groups_size_1():
            stack.enter_context(p)
        stack.enter_context(_patch_extra_ctx("vllm_ascend.attention.context_parallel.mla_cp"))
        return AscendMlaCPImpl._forward_decode(
            fake_self,
            q_nope,
            q_pe,
            fixture.k_nope_cache,
            fixture.k_pe_cache,
            BLOCK_SIZE,
            attn_metadata,
        )


def _run_forward_prefill(
    fixture: _MLAFixture,
    *,
    device: torch.device,
    dtype: torch.dtype,
    prefill_req_start: int = 0,
) -> torch.Tensor:
    batch_spec = fixture.batch_spec
    q_nope = torch.cat(fixture.q_nope_full[prefill_req_start:], dim=0)
    q_pe = torch.cat(fixture.q_pe_full[prefill_req_start:], dim=0)
    k_nope = torch.cat(fixture.k_nope_full_per_req[prefill_req_start:], dim=0)
    k_pe = torch.cat(fixture.k_pe_full_per_req[prefill_req_start:], dim=0)
    value = torch.cat(fixture.v_full_per_req[prefill_req_start:], dim=0)
    query_lens = list(batch_spec.query_lens[prefill_req_start:])

    attn_mask = _build_prefill_attn_mask(device)
    attn_metadata = _make_prefill_metadata(
        query_lens=query_lens,
        attn_mask=attn_mask,
    )

    impl = _make_real_cp_impl(
        W_UV=fixture.W_UV,
        speculative_config=None,
        vllm_config=fixture.vllm_config,
        pcp_size=1,
        dcp_size=1,
    )

    dummy_kv = (fixture.k_nope_cache, fixture.k_pe_cache)

    with _patch_extra_ctx("vllm_ascend.attention.mla_v1"):
        return AscendMlaCPImpl._forward_prefill(
            impl,
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            value,
            dummy_kv,
            attn_metadata,
        )


def _record_and_assert(
    backend_output: torch.Tensor,
    reference_output: torch.Tensor,
    tag: str,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_sig_rel_err: float = _MAX_SIG_REL_ERR,
    max_mean_sig_err: float = _MAX_MEAN_SIG_ERR,
    max_rel_err_sig: float = _MAX_REL_ERR,
) -> tuple[float, float, float, float]:
    assert backend_output.shape == reference_output.shape, (
        f"[{tag}] backend shape {tuple(backend_output.shape)} != reference shape {tuple(reference_output.shape)}"
    )
    assert torch.isfinite(backend_output).all(), f"[{tag}] kernel output contains non-finite values"
    if backend_output.dtype != reference_output.dtype:
        backend_output = backend_output.to(reference_output.dtype)

    ref_f32 = reference_output.float()
    out_f32 = backend_output.float()
    diff = (out_f32 - ref_f32).abs()
    ref_abs = ref_f32.abs()

    peak = float(ref_abs.max().item())
    mean_ref_abs = float(ref_abs.mean().item())
    max_abs_err = float(diff.max().item())
    mean_abs_err = float(diff.mean().item())
    sig_floor = peak * _SIG_FLOOR_FRAC

    max_sig_rel = max_abs_err / peak if peak > 0 else 0.0
    mean_sig_rel = mean_abs_err / mean_ref_abs if mean_ref_abs > 0 else 0.0

    significant_mask = ref_abs >= sig_floor
    if significant_mask.any():
        per_elem_rel = diff[significant_mask] / ref_abs[significant_mask]
        max_rel_sig = float(per_elem_rel.max().item())
    else:
        max_rel_sig = 0.0

    print(
        f"[MLA-CP-precision] {tag}: "
        f"peak={peak:.4e} max_abs_err={max_abs_err:.4e} "
        f"max_sig_rel={max_sig_rel * 100:.4f}% "
        f"mean_sig_rel={mean_sig_rel * 100:.4f}% "
        f"max_rel_sig(>={int(_SIG_FLOOR_FRAC * 100)}%peak)="
        f"{max_rel_sig * 100:.4f}%"
    )

    torch.testing.assert_close(
        backend_output,
        reference_output,
        rtol=rtol,
        atol=atol,
        msg=lambda m: f"[MLA-CP:{tag}] kernel output diverges from baseline. {m}",
    )

    assert max_sig_rel < max_sig_rel_err, (
        f"[MLA-CP:{tag}] signal-relative max error "
        f"{max_sig_rel * 100:.4f}% exceeds {max_sig_rel_err * 100:.2f}% "
        f"budget (peak={peak:.4e}, max_abs_err={max_abs_err:.4e})"
    )
    assert mean_sig_rel < max_mean_sig_err, (
        f"[MLA-CP:{tag}] signal-relative mean error "
        f"{mean_sig_rel * 100:.4f}% exceeds {max_mean_sig_err * 100:.2f}% "
        f"drift budget (mean_ref_abs={mean_ref_abs:.4e}, "
        f"mean_abs_err={mean_abs_err:.4e})"
    )
    assert max_rel_sig < max_rel_err_sig, (
        f"[MLA-CP:{tag}] per-element relative error on "
        f">={int(_SIG_FLOOR_FRAC * 100)}%-of-peak elements "
        f"{max_rel_sig * 100:.4f}% exceeds {max_rel_err_sig * 100:.2f}% "
        f"budget (peak={peak:.4e}, max_abs_err={max_abs_err:.4e})"
    )
    return peak, max_abs_err, max_sig_rel, mean_sig_rel


def _hf_model_tail(model: str) -> str:
    return model.rsplit("/", maxsplit=1)[-1]


def _run_mla_cp_precision_case(
    scenario: str,
    batch_spec: BatchSpec,
    dtype: torch.dtype,
    model: str,
    *,
    tensor_parallel_size: int,
) -> None:
    device = torch.device("npu")
    fixture = _build_fixture(
        batch_spec,
        dtype,
        device,
        model,
        tensor_parallel_size=tensor_parallel_size,
    )
    tail = _hf_model_tail(model)
    tp_tag = f"|tp={tensor_parallel_size}"

    if scenario == "pure_decode":
        backend_output = _run_forward_decode(
            fixture,
            speculative_config=None,
            attn_state=AscendAttentionState.DecodeOnly,
            causal=False,
            device=device,
            dtype=dtype,
        )
        reference_output = _decode_reference(
            q_nope_latent=fixture.q_nope_latent,
            q_pe_latent=fixture.q_pe_latent,
            k_nope_contexts=fixture.k_nope_contexts,
            k_pe_contexts=fixture.k_pe_contexts,
            seq_lens=batch_spec.seq_lens,
            query_lens=batch_spec.query_lens,
            W_UV=fixture.W_UV,
            scale=SCALE,
            causal=False,
            out_dtype=dtype,
        )
        _record_and_assert(
            backend_output,
            reference_output,
            tag=f"decode|{batch_spec.name}|{tail}|{dtype}{tp_tag}",
        )
        return

    if scenario == "pure_prefill":
        backend_output = _run_forward_prefill(
            fixture,
            device=device,
            dtype=dtype,
            prefill_req_start=0,
        )
        reference_output = _prefill_reference(
            q_nope_full=fixture.q_nope_full,
            q_pe_full=fixture.q_pe_full,
            k_nope_full_per_req=fixture.k_nope_full_per_req,
            k_pe_full_per_req=fixture.k_pe_full_per_req,
            v_full_per_req=fixture.v_full_per_req,
            scale=SCALE,
            out_dtype=dtype,
        )
        _record_and_assert(
            backend_output,
            reference_output,
            tag=f"prefill|{batch_spec.name}|{tail}|{dtype}{tp_tag}",
        )
        return

    if scenario == "mixed":
        n_decode_reqs = sum(1 for q in batch_spec.query_lens if q == 1)
        assert all(q == 1 for q in batch_spec.query_lens[:n_decode_reqs]), batch_spec
        assert all(q > 1 for q in batch_spec.query_lens[n_decode_reqs:]), batch_spec

        n_decode_tokens = n_decode_reqs
        decode_backend = _run_forward_decode(
            fixture,
            speculative_config=None,
            attn_state=AscendAttentionState.ChunkedPrefill,
            causal=False,
            device=device,
            dtype=dtype,
            decode_token_slice=slice(0, n_decode_tokens),
        )
        decode_reference = _decode_reference(
            q_nope_latent=fixture.q_nope_latent[:n_decode_tokens],
            q_pe_latent=fixture.q_pe_latent[:n_decode_tokens],
            k_nope_contexts=fixture.k_nope_contexts[:n_decode_reqs],
            k_pe_contexts=fixture.k_pe_contexts[:n_decode_reqs],
            seq_lens=batch_spec.seq_lens[:n_decode_reqs],
            query_lens=batch_spec.query_lens[:n_decode_reqs],
            W_UV=fixture.W_UV,
            scale=SCALE,
            causal=False,
            out_dtype=dtype,
        )
        _record_and_assert(
            decode_backend,
            decode_reference,
            tag=f"mixed_decode|{batch_spec.name}|{tail}|{dtype}{tp_tag}",
        )

        prefill_backend = _run_forward_prefill(
            fixture,
            device=device,
            dtype=dtype,
            prefill_req_start=n_decode_reqs,
        )
        prefill_reference = _prefill_reference(
            q_nope_full=fixture.q_nope_full[n_decode_reqs:],
            q_pe_full=fixture.q_pe_full[n_decode_reqs:],
            k_nope_full_per_req=fixture.k_nope_full_per_req[n_decode_reqs:],
            k_pe_full_per_req=fixture.k_pe_full_per_req[n_decode_reqs:],
            v_full_per_req=fixture.v_full_per_req[n_decode_reqs:],
            scale=SCALE,
            out_dtype=dtype,
        )
        _record_and_assert(
            prefill_backend,
            prefill_reference,
            tag=f"mixed_prefill|{batch_spec.name}|{tail}|{dtype}{tp_tag}",
        )
        return

    if scenario == "mtp":
        assert len(set(batch_spec.query_lens)) == 1, batch_spec
        spec_window = batch_spec.query_lens[0]
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = spec_window - 1

        backend_output = _run_forward_decode(
            fixture,
            speculative_config=speculative_config,
            attn_state=AscendAttentionState.SpecDecoding,
            causal=True,
            device=device,
            dtype=dtype,
        )
        reference_output = _decode_reference(
            q_nope_latent=fixture.q_nope_latent,
            q_pe_latent=fixture.q_pe_latent,
            k_nope_contexts=fixture.k_nope_contexts,
            k_pe_contexts=fixture.k_pe_contexts,
            seq_lens=batch_spec.seq_lens,
            query_lens=batch_spec.query_lens,
            W_UV=fixture.W_UV,
            scale=SCALE,
            causal=True,
            out_dtype=dtype,
        )
        _record_and_assert(
            backend_output,
            reference_output,
            tag=f"mtp|{batch_spec.name}|spec={spec_window}|{tail}|{dtype}{tp_tag}",
        )
        return

    raise AssertionError(f"unknown scenario: {scenario!r}")


@npu_test()
@pytest.mark.parametrize("case", MLA_CP_PRECISION_CASES, ids=_mla_cp_case_id)
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-V3.2-Exp"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_mla_cp_attention_precision(
    case: tuple[str, str, torch.dtype],
    model: str,
    tensor_parallel_size: int,
) -> None:
    scenario, batch_spec_name, dtype = case
    torch.manual_seed(2026)
    batch_spec = _SPECS_BY_SCENARIO[scenario][batch_spec_name]
    _run_mla_cp_precision_case(
        scenario,
        batch_spec,
        dtype,
        model,
        tensor_parallel_size=tensor_parallel_size,
    )
