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
"""NPU precision tests for AscendSFA sparse flash attention.

Same scaffolding as PR #7216 (``BatchSpec``, ``create_vllm_config``,
``create_common_attn_metadata``): parametrized over ``BATCH_SPECS``,
``tensor_parallel_size`` in ``[1, 2, 4]``, dtypes, and model; kernel vs
fp32 dense MQA reference.
"""

import math
import os
import sys
from unittest.mock import MagicMock

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

import pytest
import torch

# Metrics log: survives vLLM stdout/stderr redirection; path via env.
_METRICS_LOG_PATH = os.environ.get(
    "SFA_V1_PRECISION_METRICS_LOG",
    "/tmp/sfa_v1_precision_metrics.log",
)
with open(_METRICS_LOG_PATH, "w", encoding="utf-8"):
    pass


def _emit_metric(line: str) -> None:
    """Append ``line`` to the metrics log and echo it to stderr."""
    with open(_METRICS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass


if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm.forward_context import set_forward_context  # noqa: E402

from tests.ut.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from tests.ut.conftest import npu_test  # noqa: E402
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl  # noqa: E402

SPARSE_COUNT = 2048  # indexer_select_post_process (sfa_v1)

DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2

# Signal-relative checks (per-element |err|/|ref| near zero ref is unstable).
_MAX_SIG_REL_ERR = 1e-2  # max |out-ref| / peak |ref|
_MAX_MEAN_SIG_ERR = 5e-3  # mean |out-ref| / mean |ref|
_MAX_REL_ERR = 1e-2  # max per-element rel err where |ref| >= floor
_SIG_FLOOR_FRAC = 5e-1  # floor = this fraction of peak |ref|

_BLOCK_SIZE = 128
_TEST_NUM_HEADS = 8

BATCH_SPECS: dict[str, BatchSpec] = {
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
    "pure_prefill_with_context": BatchSpec(
        seq_lens=[512, 1024],
        query_lens=[128, 256],
        name="pure_prefill_with_context",
    ),
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


def _validate_spec(spec: BatchSpec) -> None:
    """Require seq_len <= SPARSE_COUNT so sparse matches dense reference."""
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
):
    """Cached ``VllmConfig`` for DSA/SFA (fp8 quant stripped; heads capped for UT)."""
    key = (model, dtype, tensor_parallel_size)
    cfg = _VLLM_CONFIG_CACHE.get(key)
    if cfg is not None:
        return cfg
    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float16"
    cfg = create_vllm_config(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype=dtype_str,
        block_size=_BLOCK_SIZE,
        num_gpu_blocks=4096,
        max_num_seqs=64,
        max_num_batched_tokens=max(8192, max_model_len * 2),
        enable_chunked_prefill=True,
        hf_overrides={"quantization_config": None},
        hf_config_override={
            "num_attention_heads": _TEST_NUM_HEADS,
            "num_key_value_heads": 1,
        },
    )
    _VLLM_CONFIG_CACHE[key] = cfg
    return cfg


def _build_paged_kv_cache_from_metadata(
    common_attn_metadata,
    seq_lens: list[int],
    k_nope_contexts: list[torch.Tensor],
    k_rope_contexts: list[torch.Tensor],
    block_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paged k_nope / k_rope caches from metadata block layout."""
    blocks_per_seq = [(s + block_size - 1) // block_size for s in seq_lens]
    total_blocks = sum(blocks_per_seq) + 1

    k_nope_cache = torch.zeros(total_blocks, block_size, 1, kv_lora_rank, dtype=dtype, device=device)
    k_rope_cache = torch.zeros(total_blocks, block_size, 1, qk_rope_head_dim, dtype=dtype, device=device)

    block_table = common_attn_metadata.block_table_tensor
    block_table.zero_()
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
            k_rope_cache[block_id, :length, 0, :] = k_rope_contexts[b][tok_start:tok_end]
            next_block_id += 1

    return k_nope_cache, k_rope_cache, block_table


def _build_topk_indices(
    seq_lens: list[int],
    query_lens: list[int],
    sparse_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Causal top-k indices; shape ``(T, 1, sparse_count)`` int32, -1 pad."""
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
    k_nope_cache: torch.Tensor,
    k_rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    query_lens: list[int],
    scale: float,
    block_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fp32 dense MQA softmax baseline over causal prefix."""
    batch_size = len(seq_lens)
    outputs: list[torch.Tensor] = []

    cum_q = 0
    for b in range(batch_size):
        s_len = seq_lens[b]
        q_len = query_lens[b]
        ctx_len = s_len - q_len

        n_blocks = (s_len + block_size - 1) // block_size
        block_ids = block_table[b, :n_blocks].long()
        k_blocks = k_nope_cache[block_ids]
        k_rope_blocks = k_rope_cache[block_ids]

        k_full = k_blocks.reshape(n_blocks * block_size, -1)[:s_len]
        k_rope_full = k_rope_blocks.reshape(n_blocks * block_size, -1)[:s_len]

        K = torch.cat([k_full, k_rope_full], dim=-1).float()
        V = k_full.float()

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


def _run_sfa_kernel(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope_cache: torch.Tensor,
    k_rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Call kernel via MagicMock self (only ``scale`` needed)."""
    fake_self = MagicMock()
    fake_self.scale = scale

    fake_attn_metadata = MagicMock()
    fake_attn_metadata.block_table = block_table

    return AscendSFAImpl._execute_sparse_flash_attention_process(
        fake_self,
        ql_nope,
        q_pe,
        (k_nope_cache, k_rope_cache),
        topk_indices,
        fake_attn_metadata,
        cum_query_lens,
        seq_lens_tensor,
    )


def _run_precision_check(
    spec: BatchSpec,
    dtype: torch.dtype,
    vllm_config,
    *,
    tensor_parallel_size: int,
) -> None:
    torch.manual_seed(2026)
    _validate_spec(spec)

    device = torch.device("npu")
    seq_lens = list(spec.seq_lens)
    query_lens = list(spec.query_lens)
    batch_size = spec.batch_size
    num_tokens = spec.compute_num_tokens()

    cache_config = vllm_config.cache_config
    hf_text = vllm_config.model_config.hf_text_config
    block_size = cache_config.block_size
    qk_rope_head_dim = hf_text.qk_rope_head_dim
    kv_lora_rank = hf_text.kv_lora_rank
    num_heads = hf_text.num_attention_heads

    head_dim = kv_lora_rank + qk_rope_head_dim
    scale = 1.0 / math.sqrt(head_dim)

    common_attn_metadata = create_common_attn_metadata(spec, block_size=block_size, device=device)

    k_nope_contexts = [torch.randn(s, kv_lora_rank, dtype=dtype, device=device) for s in seq_lens]
    k_rope_contexts = [torch.randn(s, qk_rope_head_dim, dtype=dtype, device=device) for s in seq_lens]

    k_nope_cache, k_rope_cache, block_table = _build_paged_kv_cache_from_metadata(
        common_attn_metadata=common_attn_metadata,
        seq_lens=seq_lens,
        k_nope_contexts=k_nope_contexts,
        k_rope_contexts=k_rope_contexts,
        block_size=block_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dtype=dtype,
        device=device,
    )

    ql_nope = torch.randn(num_tokens, num_heads, kv_lora_rank, dtype=dtype, device=device)
    q_pe = torch.randn(num_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=device)

    topk_indices = _build_topk_indices(seq_lens, query_lens, SPARSE_COUNT, device)

    cum_query_lens = torch.tensor(
        [sum(query_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device=device,
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        backend_output = _run_sfa_kernel(
            ql_nope=ql_nope,
            q_pe=q_pe,
            k_nope_cache=k_nope_cache,
            k_rope_cache=k_rope_cache,
            block_table=block_table,
            topk_indices=topk_indices,
            cum_query_lens=cum_query_lens,
            seq_lens_tensor=seq_lens_tensor,
            scale=scale,
        )
    reference_output = _reference_sparse_attention(
        ql_nope=ql_nope,
        q_pe=q_pe,
        k_nope_cache=k_nope_cache,
        k_rope_cache=k_rope_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_lens=query_lens,
        scale=scale,
        block_size=block_size,
        out_dtype=dtype,
    )

    tag = f"{spec.name},tp={tensor_parallel_size}"
    assert backend_output.shape == reference_output.shape, (
        f"[{tag}] backend shape {tuple(backend_output.shape)} != reference shape {tuple(reference_output.shape)}"
    )
    assert backend_output.dtype == reference_output.dtype, (
        f"[{tag}] backend dtype {backend_output.dtype} != reference dtype {reference_output.dtype}"
    )
    assert torch.isfinite(backend_output).all(), f"[{tag}] sparse flash attention produced non-finite values"

    torch.testing.assert_close(
        backend_output,
        reference_output,
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
        msg=lambda m: f"[SFA:{tag}] kernel output diverges from baseline. {m}",
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

    _emit_metric(
        f"[SFA:{spec.name}] tp={tensor_parallel_size} dtype={dtype} "
        f"peak={peak:.4e} "
        f"max_abs_err={max_abs_err:.4e} "
        f"max_sig_rel_err={max_sig_rel_err * 100:.4f}% "
        f"mean_sig_rel_err={mean_sig_rel_err * 100:.4f}% "
        f"max_rel_err_sig(>={int(_SIG_FLOOR_FRAC * 100)}%peak)="
        f"{max_rel_err_sig * 100:.4f}%"
    )

    assert max_sig_rel_err < _MAX_SIG_REL_ERR, (
        f"[SFA:{tag}] dtype={dtype} signal-relative max error "
        f"{max_sig_rel_err * 100:.4f}% exceeds 1% budget "
        f"(peak={peak:.4e}, max_abs_err={max_abs_err:.4e})"
    )
    assert mean_sig_rel_err < _MAX_MEAN_SIG_ERR, (
        f"[SFA:{tag}] dtype={dtype} signal-relative mean error "
        f"{mean_sig_rel_err * 100:.4f}% exceeds 0.5% drift budget "
        f"(mean_ref_abs={mean_ref_abs:.4e}, mean_abs_err={mean_abs_err:.4e})"
    )
    assert max_rel_err_sig < _MAX_REL_ERR, (
        f"[SFA:{tag}] dtype={dtype} per-element relative error on "
        f">={int(_SIG_FLOOR_FRAC * 100)}%-of-peak elements "
        f"{max_rel_err_sig * 100:.4f}% exceeds 1% budget "
        f"(peak={peak:.4e}, max_abs_err={max_abs_err:.4e})"
    )


@npu_test()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-V3.2-Exp"])
@pytest.mark.parametrize("batch_spec_name", list(BATCH_SPECS.keys()))
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_sfa_sparse_flash_attention_precision(
    batch_spec_name: str,
    model: str,
    dtype: torch.dtype,
    tensor_parallel_size: int,
) -> None:
    """SFA kernel vs fp32 dense MQA reference (decode, prefill, mixed, MTP)."""
    vllm_config = _get_vllm_config(model, dtype, tensor_parallel_size=tensor_parallel_size)
    _run_precision_check(
        BATCH_SPECS[batch_spec_name],
        dtype,
        vllm_config,
        tensor_parallel_size=tensor_parallel_size,
    )
