# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import argparse
import copy
import importlib
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.config import set_current_vllm_config

from tests.e2e.pull_request.one_card.attention_utils import create_vllm_config
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import SFA_FIA_SHARED_PREFILL_TOPK_WIDTH, AscendSFAImpl, AscendSFAMetadata

# Register the native torch.ops._C_ascend kernels without asking mypy to
# statically analyze the binary extension module.
importlib.import_module("vllm_ascend.vllm_ascend_C")

pytestmark = pytest.mark.skipif(
    torch.npu.device_count() < 1,
    reason="Shared SFA FIA native checks require at least one NPU.",
)

_DEVICE = torch.device("npu")
_DTYPE = torch.bfloat16
_DENSE_PREFILL_BLOCK_SIZE = 128
_DENSE_PREFILL_KV_HEADS = 1
_DENSE_PREFILL_QUERY_HEADS = 4
_DENSE_PREFILL_LATENT_DIM = 512
_DENSE_PREFILL_ROPE_DIM = 64
_SHARED_PREFILL_RTOL = 1e-2
_SHARED_PREFILL_ATOL = 1e-2
# Keep matcher tolerance aligned with existing one-card attention precision tests.


def _build_topk_indices(
    seq_lens: tuple[int, ...],
    query_lens: tuple[int, ...],
    sparse_count: int,
) -> torch.Tensor:
    num_tokens = sum(query_lens)
    topk_indices = torch.full(
        (num_tokens, 1, sparse_count),
        -1,
        dtype=torch.int32,
        device=_DEVICE,
    )
    cursor = 0
    for s_len, q_len in zip(seq_lens, query_lens, strict=True):
        context_len = s_len - q_len
        for j in range(q_len):
            visible = min(s_len, context_len + j + 1, sparse_count)
            if visible:
                topk_indices[cursor, 0, :visible] = torch.arange(visible, device=_DEVICE, dtype=torch.int32)
            cursor += 1
    return topk_indices


def _build_prefill_case(
    query_lens: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    seed: int,
) -> tuple[AscendSFAMetadata, tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    assert len(query_lens) == len(kv_lengths)

    num_requests = len(query_lens)
    num_tokens = sum(query_lens)
    seq_lens_cpu = torch.tensor(kv_lengths, dtype=torch.int32)
    cum_query_lens_cpu = torch.tensor(query_lens, dtype=torch.int32).cumsum(0).to(torch.int32)

    seq_lens = seq_lens_cpu.to(_DEVICE)
    cum_query_lens = cum_query_lens_cpu.to(_DEVICE)

    blocks_per_seq = [(s_len + _DENSE_PREFILL_BLOCK_SIZE - 1) // _DENSE_PREFILL_BLOCK_SIZE for s_len in kv_lengths]
    max_blocks = max(blocks_per_seq)
    block_table = torch.zeros((num_requests, max_blocks), dtype=torch.int32, device=_DEVICE)

    next_block = 1
    request_block_tables: list[torch.Tensor] = []
    for request_id, num_blocks in enumerate(blocks_per_seq):
        block_ids = torch.arange(num_blocks, dtype=torch.int32, device=_DEVICE) + next_block
        block_table[request_id, :num_blocks] = block_ids
        request_block_tables.append(block_ids)
        next_block += num_blocks

    total_blocks = next_block
    k_nope_cache = torch.zeros(
        (total_blocks, _DENSE_PREFILL_BLOCK_SIZE, 1, _DENSE_PREFILL_LATENT_DIM),
        dtype=_DTYPE,
        device=_DEVICE,
    )
    k_rope_cache = torch.zeros(
        (total_blocks, _DENSE_PREFILL_BLOCK_SIZE, 1, _DENSE_PREFILL_ROPE_DIM),
        dtype=_DTYPE,
        device=_DEVICE,
    )
    for request_id, s_len in enumerate(kv_lengths):
        kv_context_nope = torch.randn(s_len, _DENSE_PREFILL_LATENT_DIM, dtype=_DTYPE, device=_DEVICE)
        kv_context_rope = torch.randn(s_len, _DENSE_PREFILL_ROPE_DIM, dtype=_DTYPE, device=_DEVICE)
        request_blocks = request_block_tables[request_id]
        for block_offset, block_id in enumerate(request_blocks):
            start = block_offset * _DENSE_PREFILL_BLOCK_SIZE
            block_len = min(_DENSE_PREFILL_BLOCK_SIZE, max(s_len - start, 0))
            if block_len <= 0:
                continue
            k_nope_cache[block_id, :block_len, 0] = kv_context_nope[start : start + block_len]
            k_rope_cache[block_id, :block_len, 0] = kv_context_rope[start : start + block_len]

    attn_mask = torch.ones(
        SFA_FIA_SHARED_PREFILL_TOPK_WIDTH,
        SFA_FIA_SHARED_PREFILL_TOPK_WIDTH,
        dtype=torch.int8,
        device=_DEVICE,
    ).triu(1)

    ql_nope = torch.randn(
        num_tokens,
        _DENSE_PREFILL_QUERY_HEADS,
        _DENSE_PREFILL_LATENT_DIM,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    q_pe = torch.randn(
        num_tokens,
        _DENSE_PREFILL_QUERY_HEADS,
        _DENSE_PREFILL_ROPE_DIM,
        dtype=_DTYPE,
        device=_DEVICE,
    )

    metadata = AscendSFAMetadata(
        num_actual_tokens=num_tokens,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32, device=_DEVICE),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        cum_query_lens=cum_query_lens,
        cum_query_lens_cpu=cum_query_lens_cpu,
        block_table=block_table,
        sin=torch.empty(0, device=_DEVICE),
        cos=torch.empty(0, device=_DEVICE),
    )
    metadata.attn_state = AscendAttentionState.PrefillCacheHit
    metadata.num_input_tokens = num_tokens
    metadata.num_actual_tokens = num_tokens
    metadata.num_decodes = 0
    metadata.num_decode_tokens = 0
    metadata.block_size = _DENSE_PREFILL_BLOCK_SIZE
    metadata.attn_mask = attn_mask
    metadata._sfa_fia_shared_prefill_plan = None

    topk_indices = _build_topk_indices(
        seq_lens=tuple(kv_lengths),
        query_lens=query_lens,
        sparse_count=SFA_FIA_SHARED_PREFILL_TOPK_WIDTH,
    )

    return metadata, (k_nope_cache, k_rope_cache), ql_nope, q_pe, topk_indices


def _create_shared_prefill_impl(seed: int) -> AscendSFAImpl:
    torch.manual_seed(seed)
    vllm_config = create_vllm_config(
        model_name="tests/ut/_fake_weight",
        dtype=_DTYPE,
        tensor_parallel_size=1,
        max_model_len=8192,
        block_size=_DENSE_PREFILL_BLOCK_SIZE,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        hf_overrides={"max_position_embeddings": 8192},
        hf_config_override={
            "num_attention_heads": _DENSE_PREFILL_QUERY_HEADS,
            "num_key_value_heads": _DENSE_PREFILL_KV_HEADS,
        },
    )
    vllm_config.model_config.get_head_size = lambda: _DENSE_PREFILL_LATENT_DIM + _DENSE_PREFILL_ROPE_DIM  # type: ignore[method-assign]
    vllm_config.speculative_config = None
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.decode_context_parallel_size = 1
    vllm_config.model_config.enforce_eager = True
    vllm_config.scheduler_config.enforce_eager = True
    vllm_config.scheduler_config.max_num_batched_tokens = 8192
    vllm_config.additional_config = {"refresh": True, "enable_sfa_fia_shared_prefill": True}

    with (
        set_current_vllm_config(vllm_config),
        patch("vllm_ascend.attention.sfa_v1.get_tensor_model_parallel_world_size", return_value=1),
    ):
        init_ascend_config(vllm_config)
        return AscendSFAImpl(
            num_heads=_DENSE_PREFILL_QUERY_HEADS,
            head_size=_DENSE_PREFILL_LATENT_DIM + _DENSE_PREFILL_ROPE_DIM,
            scale=0.125,
            num_kv_heads=_DENSE_PREFILL_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            kv_lora_rank=_DENSE_PREFILL_LATENT_DIM,
            qk_nope_head_dim=_DENSE_PREFILL_LATENT_DIM,
            qk_rope_head_dim=_DENSE_PREFILL_ROPE_DIM,
            qk_head_dim=_DENSE_PREFILL_LATENT_DIM + _DENSE_PREFILL_ROPE_DIM,
            v_head_dim=128,
            q_lora_rank=256,
            q_proj=MagicMock(),
            q_b_proj=MagicMock(),
            kv_b_proj=MagicMock(),
            o_proj=MagicMock(),
            kv_a_proj_with_mqa=None,
            kv_a_layernorm=MagicMock(),
            q_a_layernorm=MagicMock(),
            rotary_emb=MagicMock(),
            indexer=None,
            skip_topk=True,
            fused_qkv_a_proj=MagicMock(),
            topk_indices_buffer=torch.empty(SFA_FIA_SHARED_PREFILL_TOPK_WIDTH * 8, dtype=torch.int32),
            layer_name="model.layers.0",
        )


def _snapshot_prefill_inputs(
    metadata: AscendSFAMetadata,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    topk_indices: torch.Tensor,
) -> dict[str, torch.Tensor | AscendSFAMetadata | tuple[torch.Tensor, torch.Tensor]]:
    return {
        "metadata": copy.deepcopy(metadata),
        "kv_cache": (kv_cache[0].clone(), kv_cache[1].clone()),
        "topk_indices": topk_indices.clone(),
    }


def _assert_no_input_mutation(
    baseline_metadata: AscendSFAMetadata,
    baseline_snapshot: dict[str, torch.Tensor | AscendSFAMetadata | tuple[torch.Tensor, torch.Tensor]],
    baseline_kv_cache: tuple[torch.Tensor, torch.Tensor],
    baseline_topk_indices: torch.Tensor,
) -> None:
    metadata_snapshot = baseline_snapshot["metadata"]
    assert isinstance(metadata_snapshot, AscendSFAMetadata)
    assert torch.equal(baseline_topk_indices, baseline_snapshot["topk_indices"])
    assert torch.equal(baseline_metadata.block_table, metadata_snapshot.block_table)
    assert torch.equal(baseline_metadata.seq_lens_cpu, metadata_snapshot.seq_lens_cpu)
    assert torch.equal(baseline_metadata.cum_query_lens_cpu, metadata_snapshot.cum_query_lens_cpu)
    assert torch.equal(baseline_metadata.seq_lens, metadata_snapshot.seq_lens)
    assert torch.equal(baseline_metadata.cum_query_lens, metadata_snapshot.cum_query_lens)
    assert isinstance(baseline_snapshot["kv_cache"], tuple)
    assert torch.equal(baseline_kv_cache[0], baseline_snapshot["kv_cache"][0])
    assert torch.equal(baseline_kv_cache[1], baseline_snapshot["kv_cache"][1])
    assert torch.equal(baseline_topk_indices, baseline_snapshot["topk_indices"])


def _expected_tail_rows(
    query_lens: tuple[int, ...],
    kv_lengths: tuple[int, ...],
) -> int:
    eligible_rows = sum(
        min(query, max(0, SFA_FIA_SHARED_PREFILL_TOPK_WIDTH - (kv_len - query)))
        for query, kv_len in zip(query_lens, kv_lengths, strict=True)
    )
    return sum(query_lens) - eligible_rows


def _run_shared_prefill_candidate(
    impl: AscendSFAImpl,
    metadata: AscendSFAMetadata,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    topk_indices: torch.Tensor,
    expected_tail_rows: int,
) -> torch.Tensor:
    with (
        patch.object(
            torch_npu, "npu_fused_infer_attention_score", wraps=torch_npu.npu_fused_infer_attention_score
        ) as fused_mock,
        patch.object(
            AscendSFAImpl,
            "_execute_sparse_flash_attention_process",
            wraps=impl._execute_sparse_flash_attention_process,
        ) as sparse_mock,
    ):
        candidate = impl._try_sfa_fia_shared_prefill(ql_nope, q_pe, kv_cache, metadata, topk_indices)
    assert candidate is not None
    assert fused_mock.call_count == 1
    assert sparse_mock.call_count == 1
    sparse_q = sparse_mock.call_args.args[0]
    assert sparse_q.shape[0] == expected_tail_rows
    return candidate


def _run_shared_prefill_case(
    query_lens: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    impl = _create_shared_prefill_impl(seed)
    baseline_metadata, kv_cache, ql_nope, q_pe, topk_indices = _build_prefill_case(query_lens, kv_lengths, seed)
    candidate_metadata = copy.deepcopy(baseline_metadata)
    snapshot = _snapshot_prefill_inputs(candidate_metadata, kv_cache, topk_indices)

    baseline = impl._execute_sparse_flash_attention_process(
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        baseline_metadata,
        baseline_metadata.cum_query_lens,
        baseline_metadata.seq_lens,
        block_table=baseline_metadata.block_table,
    )
    candidate = _run_shared_prefill_candidate(
        impl,
        candidate_metadata,
        kv_cache,
        ql_nope,
        q_pe,
        topk_indices,
        expected_tail_rows=_expected_tail_rows(query_lens, kv_lengths),
    )
    candidate_repeat = _run_shared_prefill_candidate(
        impl,
        candidate_metadata,
        kv_cache,
        ql_nope,
        q_pe,
        topk_indices,
        expected_tail_rows=_expected_tail_rows(query_lens, kv_lengths),
    )

    assert candidate_metadata._sfa_fia_shared_prefill_plan is not None
    _assert_no_input_mutation(candidate_metadata, snapshot, kv_cache, topk_indices)
    torch.testing.assert_close(
        candidate, candidate_repeat, rtol=_SHARED_PREFILL_RTOL, atol=_SHARED_PREFILL_ATOL, check_dtype=False
    )

    return candidate, baseline


def _run_sparsity_fallback_case(
    query_lens: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    seed: int,
) -> None:
    impl = _create_shared_prefill_impl(seed)
    metadata, kv_cache, ql_nope, q_pe, topk_indices = _build_prefill_case(query_lens, kv_lengths, seed)

    candidate = impl._try_sfa_fia_shared_prefill(ql_nope, q_pe, kv_cache, metadata, topk_indices)
    assert candidate is None


def _compare_case(query_lens: tuple[int, ...], kv_lengths: tuple[int, ...], seed: int) -> None:
    candidate, baseline = _run_shared_prefill_case(query_lens, kv_lengths, seed)

    torch.testing.assert_close(
        candidate,
        baseline,
        rtol=_SHARED_PREFILL_RTOL,
        atol=_SHARED_PREFILL_ATOL,
        check_dtype=False,
    )
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(baseline).all()


def test_shared_prefill_native_single_request_q4096_prefix2048_tail2048():
    _compare_case((4096,), (4096,), 2026)


def test_shared_prefill_native_grouped_request_q1536_kv2560_dense2048_tail1024():
    _compare_case((1536, 1536), (2560, 2560), 2027)


def test_shared_prefill_native_exact_three_request_q2267_multisegment():
    query_lens = (2267, 2267, 2267)
    kv_lengths = (2267, 2267, 2267)
    impl = _create_shared_prefill_impl(2028)
    baseline_metadata, kv_cache, ql_nope, q_pe, topk_indices = _build_prefill_case(query_lens, kv_lengths, 2028)
    baseline_metadata.attn_state = AscendAttentionState.PrefillNoCache
    candidate_metadata = copy.deepcopy(baseline_metadata)
    snapshot = _snapshot_prefill_inputs(candidate_metadata, kv_cache, topk_indices)

    baseline = impl._execute_sparse_flash_attention_process(
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        baseline_metadata,
        baseline_metadata.cum_query_lens,
        baseline_metadata.seq_lens,
        block_table=baseline_metadata.block_table,
    )
    with (
        patch.object(
            torch_npu,
            "npu_fused_infer_attention_score",
            wraps=torch_npu.npu_fused_infer_attention_score,
        ) as fused_mock,
        patch.object(
            AscendSFAImpl,
            "_execute_sparse_flash_attention_process",
            wraps=impl._execute_sparse_flash_attention_process,
        ) as sparse_mock,
    ):
        candidate = impl._try_sfa_fia_shared_prefill(ql_nope, q_pe, kv_cache, candidate_metadata, topk_indices)

    assert candidate is not None
    assert fused_mock.call_count == 3
    for request, call in enumerate(fused_mock.call_args_list):
        dense = call.kwargs
        assert dense["query"].shape[0] == SFA_FIA_SHARED_PREFILL_TOPK_WIDTH
        assert dense["actual_seq_lengths"] == [SFA_FIA_SHARED_PREFILL_TOPK_WIDTH]
        assert dense["actual_seq_lengths_kv"] == [SFA_FIA_SHARED_PREFILL_TOPK_WIDTH]
        torch.testing.assert_close(
            dense["block_table"],
            candidate_metadata.block_table[request : request + 1],
        )

    sparse_mock.assert_called_once()
    tail = sparse_mock.call_args.args
    assert tail[0].shape[0] == 657
    assert tail[5].tolist() == [219, 438, 657]
    assert tail[6].tolist() == [2267, 2267, 2267]
    torch.testing.assert_close(sparse_mock.call_args.kwargs["block_table"], candidate_metadata.block_table)

    candidate_repeat = impl._try_sfa_fia_shared_prefill(ql_nope, q_pe, kv_cache, candidate_metadata, topk_indices)
    assert candidate_repeat is not None
    assert candidate_metadata._sfa_fia_shared_prefill_plan is None
    _assert_no_input_mutation(candidate_metadata, snapshot, kv_cache, topk_indices)
    torch.testing.assert_close(
        candidate, candidate_repeat, rtol=_SHARED_PREFILL_RTOL, atol=_SHARED_PREFILL_ATOL, check_dtype=False
    )
    torch.testing.assert_close(
        candidate, baseline, rtol=_SHARED_PREFILL_RTOL, atol=_SHARED_PREFILL_ATOL, check_dtype=False
    )
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(baseline).all()


def test_shared_prefill_native_fallback_when_dense_total_not_admitted():
    _run_sparsity_fallback_case((277,), (277,), 2030)


def test_shared_prefill_native_fallback_on_unsupported_block_size():
    impl = _create_shared_prefill_impl(2031)
    metadata, kv_cache, ql_nope, q_pe, topk_indices = _build_prefill_case((4096,), (4096,), 2031)
    metadata.block_size = 64
    candidate = impl._try_sfa_fia_shared_prefill(ql_nope, q_pe, kv_cache, metadata, topk_indices)
    assert candidate is None


def _run_case_latency(
    query_lens: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    seed: int,
    iters: int,
    warmup: int,
) -> tuple[list[float], list[float]]:
    baseline_metadata, kv_cache, ql_nope, q_pe, topk_indices = _build_prefill_case(query_lens, kv_lengths, seed)
    candidate_metadata = copy.deepcopy(baseline_metadata)
    impl = _create_shared_prefill_impl(seed)

    def _call_baseline() -> None:
        impl._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            baseline_metadata,
            baseline_metadata.cum_query_lens,
            baseline_metadata.seq_lens,
            block_table=baseline_metadata.block_table,
        )

    def _call_candidate() -> None:
        output = impl._try_sfa_fia_shared_prefill(ql_nope, q_pe, kv_cache, candidate_metadata, topk_indices)
        assert output is not None

    for _ in range(warmup):
        _call_baseline()
        torch.npu.synchronize()
        _call_candidate()
        torch.npu.synchronize()

    candidate_samples: list[float] = []
    baseline_samples: list[float] = []
    for i in range(iters):
        if i % 2 == 0:
            torch.npu.synchronize()
            start = time.perf_counter()
            _call_baseline()
            torch.npu.synchronize()
            baseline_samples.append((time.perf_counter() - start) * 1000)

            torch.npu.synchronize()
            start = time.perf_counter()
            _call_candidate()
            torch.npu.synchronize()
            candidate_samples.append((time.perf_counter() - start) * 1000)
        else:
            torch.npu.synchronize()
            start = time.perf_counter()
            _call_candidate()
            torch.npu.synchronize()
            candidate_samples.append((time.perf_counter() - start) * 1000)

            torch.npu.synchronize()
            start = time.perf_counter()
            _call_baseline()
            torch.npu.synchronize()
            baseline_samples.append((time.perf_counter() - start) * 1000)

    return baseline_samples, candidate_samples


def _main_benchmark() -> None:
    parser = argparse.ArgumentParser(description="Run minimal native shared-prefill SFA benchmark")
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    cases = {
        "single_q4096": ((4096,), (4096,), 2026),
        "group_q1536_kv2560": ((1536, 1536), (2560, 2560), 2027),
    }

    for name, (query_lens, kv_lengths, seed) in cases.items():
        baseline_samples_ms, candidate_samples_ms = _run_case_latency(
            query_lens,
            kv_lengths,
            seed,
            iters=args.iters,
            warmup=args.warmup,
        )
        baseline_sorted = sorted(baseline_samples_ms)
        candidate_sorted = sorted(candidate_samples_ms)
        baseline_mean = sum(baseline_samples_ms) / len(baseline_samples_ms)
        candidate_mean = sum(candidate_samples_ms) / len(candidate_samples_ms)
        baseline_median = baseline_sorted[len(baseline_sorted) // 2]
        candidate_median = candidate_sorted[len(candidate_sorted) // 2]
        print(
            f"[SFA_SHARED_PREFILL_BENCH] case={name} baseline_samples_ms={baseline_samples_ms} "
            f"candidate_samples_ms={candidate_samples_ms} "
            f"baseline_mean_ms={baseline_mean:.4f} baseline_median_ms={baseline_median:.4f} "
            f"candidate_mean_ms={candidate_mean:.4f} candidate_median_ms={candidate_median:.4f}"
        )


if __name__ == "__main__":
    _main_benchmark()
