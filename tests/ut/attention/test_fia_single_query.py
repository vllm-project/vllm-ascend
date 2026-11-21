# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for single-query TND FIA speculative-verification layout, replay, and isolation."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    FIAParamProvider,
)
from vllm_ascend.attention.fia_single_query import (
    _MAX_INDEX_CACHE_ENTRIES,
    MAX_TND_FIA_BATCH,
    is_all_single_query,
    prepare_fia_layout,
    prepare_graph_fia_layout,
    single_query_layout,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile


def _make_vllm_config(
    *,
    num_speculative_tokens: int = 2,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL_DECODE_ONLY,
    runner_type: str = "generate",
    enable_c8_quant: bool = False,
    cache_dtype: str = "auto",
    pcp_size: int = 1,
    dcp_size: int = 1,
    parallel_drafting: bool = False,
) -> SimpleNamespace:
    spec_cfg = (
        SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=parallel_drafting,
            speculator_config=SimpleNamespace(parallel_drafting=parallel_drafting),
            use_dspark=lambda: False,
        )
        if num_speculative_tokens > 0
        else None
    )
    quant_cfg = SimpleNamespace(enable_c8_quant=enable_c8_quant) if enable_c8_quant else None
    return SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=spec_cfg,
        quant_config=quant_cfg,
        cache_config=SimpleNamespace(block_size=16, cache_dtype=cache_dtype),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        ),
        model_config=SimpleNamespace(
            max_model_len=4096,
            runner_type=runner_type,
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(sliding_window=None),
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        scheduler_config=SimpleNamespace(max_num_seqs=16, enable_chunked_prefill=True),
    )


def _make_common_metadata(
    *,
    query_ends: list[int],
    seq_lens: list[int],
    block_table: torch.Tensor,
    num_reqs: int,
    num_actual_tokens: int,
    num_input_tokens: int,
    max_query_len: int,
    is_prefilling: list[bool] | None,
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill,
    causal: bool = True,
) -> AscendCommonAttentionMetadata:
    q_loc = torch.tensor([0, *query_ends], dtype=torch.int32)
    seq_t = torch.tensor(seq_lens, dtype=torch.int32)
    prefill_t = torch.tensor(is_prefilling, dtype=torch.bool) if is_prefilling is not None else None
    return AscendCommonAttentionMetadata(
        query_start_loc=q_loc,
        query_start_loc_cpu=q_loc,
        seq_lens=seq_t,
        seq_lens_cpu=seq_t,
        _seq_lens_cpu=seq_t,
        num_computed_tokens_cpu=None,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        num_input_tokens=num_input_tokens,
        max_query_len=max_query_len,
        block_table_tensor=block_table,
        slot_mapping=torch.arange(max(num_actual_tokens, 1), dtype=torch.int32),
        positions=torch.arange(max(num_input_tokens, 1), dtype=torch.int64),
        max_seq_len=4096,
        attn_state=attn_state,
        is_prefilling=prefill_t,
        causal=causal,
    )


def _make_backend_impl(
    vllm_config: SimpleNamespace,
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    enable_c8_quant: bool = False,
    kv_cache_dtype: torch.dtype = torch.bfloat16,
    is_draft_layer: bool = False,
    sliding_window: int | None = None,
) -> AscendAttentionBackendImpl:
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.vllm_config = vllm_config
    impl.num_heads = key_cache.shape[2]
    impl.num_kv_heads = key_cache.shape[2]
    impl.head_size = key_cache.shape[3]
    impl.scale = impl.head_size**-0.5
    impl.sliding_window = sliding_window
    impl._non_causal_swa_attn_mask = None
    impl.sinks = None
    impl.attn_type = "decoder"
    impl.use_bnsd_kv_cache = False
    impl.key_cache = key_cache
    impl.value_cache = value_cache
    impl.enable_c8_quant = enable_c8_quant
    impl.kv_cache_dtype = kv_cache_dtype
    impl._use_max_workspace_for_fia_graph = False
    impl._layer_name = "model.layers.3.self_attn.attn"
    impl._is_draft_layer = is_draft_layer
    impl._fia_index_cache = {}
    return impl


def _reference_paged_causal_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_ends: list[int],
    kv_lengths: list[int],
) -> torch.Tensor:
    """Exact FP64 multi-head paged causal attention reference [T, H, D]."""
    num_tokens, num_heads, head_dim = query.shape
    num_blocks, block_size = k_cache.shape[0], k_cache.shape[1]
    k_pages = k_cache.view(num_blocks, block_size, num_heads, head_dim)
    v_pages = v_cache.view(num_blocks, block_size, num_heads, head_dim)
    out = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float64)
    start = 0
    for row, (end, kv_len) in enumerate(zip(query_ends, kv_lengths, strict=True)):
        width = end - start
        pages = block_table[row]
        flat_k = k_pages[pages].reshape(-1, num_heads, head_dim)[:kv_len].to(torch.float64)
        flat_v = v_pages[pages].reshape(-1, num_heads, head_dim)[:kv_len].to(torch.float64)
        for offset in range(width):
            visible = kv_len - width + 1 + offset
            q_tok = query[start + offset].to(torch.float64)
            for h in range(num_heads):
                scores = (q_tok[h] @ flat_k[:visible, h].T) * (head_dim**-0.5)
                probs = torch.softmax(scores, dim=-1)
                out[start + offset, h] = probs @ flat_v[:visible, h]
        start = end
    return out


def test_hardware_profile_enables_single_query_spec_verify_on_a5_only():
    assert get_hardware_profile(AscendDeviceType.A5).supports(HardwareCapability.SINGLE_QUERY_SPEC_VERIFY_FIA)
    assert not get_hardware_profile(AscendDeviceType.A3).supports(HardwareCapability.SINGLE_QUERY_SPEC_VERIFY_FIA)
    assert not get_hardware_profile(AscendDeviceType.A2).supports(HardwareCapability.SINGLE_QUERY_SPEC_VERIFY_FIA)


def test_is_all_single_query_rejects_zero_width_or_multi_token_rows():
    assert is_all_single_query([1, 2, 3], 3)
    assert not is_all_single_query([0, 2], 2)
    assert not is_all_single_query([2, 3], 3)
    assert not is_all_single_query([], 0)


@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_single_query_layout_matches_causal_fp64_reference_across_widths(width: int):
    torch.manual_seed(0)
    num_heads, head_dim, block_size = 2, 32, 16
    block_table = torch.tensor([[2, 5, 1, 7], [0, 3, 6, 4]], dtype=torch.int32)
    # Row 0 has cached context (kv_len=37); row 1 is a brand-new short extend (kv_len == width).
    query_ends = [width, 2 * width]
    kv_lengths = [37, width]
    query = torch.randn(2 * width, num_heads, head_dim, dtype=torch.bfloat16)
    k_cache = torch.randn(8, block_size, num_heads, head_dim, dtype=torch.bfloat16)
    v_cache = torch.randn(8, block_size, num_heads, head_dim, dtype=torch.bfloat16)

    new_ends, new_kv_lens, rows = single_query_layout(query_ends, kv_lengths, max_expand_width=width)
    assert new_ends == list(range(1, 2 * width + 1))
    assert new_kv_lens == list(range(37 - width + 1, 38)) + list(range(1, width + 1))
    assert rows == [0] * width + [1] * width

    expanded_table = block_table[torch.tensor(rows, dtype=torch.long)]
    ref_multi = _reference_paged_causal_attention(query, k_cache, v_cache, block_table, query_ends, kv_lengths)
    ref_single = _reference_paged_causal_attention(query, k_cache, v_cache, expanded_table, new_ends, new_kv_lens)
    assert torch.equal(ref_multi, ref_single)


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_full_decode_graph_capture_and_short_extend_or_varlen_replays(cudagraph_mode: CUDAGraphMode):
    cfg = _make_vllm_config(num_speculative_tokens=2, cudagraph_mode=cudagraph_mode)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=torch.bfloat16)
    a5_profile = get_hardware_profile(AscendDeviceType.A5)
    k_cache = torch.randn(16, 16, 2, 32, dtype=torch.bfloat16)
    v_cache = torch.randn(16, 16, 2, 32, dtype=torch.bfloat16)
    impl = _make_backend_impl(cfg, key_cache=k_cache, value_cache=v_cache)

    with (
        patch("vllm_ascend.attention.attention_v1.get_current_hardware_profile", return_value=a5_profile),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(capturing=False, is_draft_model=False),
        ),
    ):
        builder = AscendAttentionMetadataBuilder(spec, ["model.layers.3.self_attn.attn"], cfg, torch.device("cpu"))
        builder.attn_mask_builder = MagicMock()
        builder.attn_mask_builder.get_attention_mask.return_value = torch.ones((16, 16), dtype=torch.bool)
        assert builder.enable_single_query_spec_verify is True

        # Capture pure decode dummy graph of 6 tokens (2 requests x 3 tokens).
        cap_table = torch.tensor([[4, 5, 0, 0], [6, 7, 0, 0]], dtype=torch.int32)
        cap_common = _make_common_metadata(
            query_ends=[3, 6],
            seq_lens=[3, 3],
            block_table=cap_table,
            num_reqs=2,
            num_actual_tokens=6,
            num_input_tokens=6,
            max_query_len=3,
            is_prefilling=[False, False],
            attn_state=AscendAttentionState.DecodeOnly,
        )
        cap_meta = builder.build_for_graph_capture(cap_common, AscendAttentionState.DecodeOnly)

        captured_tasks: list[tuple[dict, FIAParamProvider]] = []
        with (
            patch("vllm_ascend.attention.attention_v1.get_capture_resource", return_value=torch.empty(1)),
            patch(
                "vllm_ascend.attention.attention_v1.register_task",
                side_effect=lambda fn, kwargs, provider: captured_tasks.append((kwargs, provider)),
            ),
        ):
            impl.full_graph_fia(
                query=torch.randn(6, 2, 32, dtype=torch.bfloat16),
                key=k_cache,
                value=v_cache,
                attn_metadata=cap_meta,
                output=torch.empty(6, 2, 32, dtype=torch.bfloat16),
            )
        assert len(captured_tasks) == 1
        cap_kwargs, provider = captured_tasks[0]
        assert provider.single_query_fia is True
        assert cap_kwargs["atten_mask"] is None
        assert cap_kwargs["sparse_mode"] == 0
        assert cap_kwargs["actual_seq_lengths"] == [1, 2, 3, 4, 5, 6]

        replay_cases = [
            # 1) Short extend with qlen=1 and is_prefilling=True (unpadded tail to 6 graph tokens).
            (
                _make_common_metadata(
                    query_ends=[1],
                    seq_lens=[49],
                    block_table=torch.tensor([[9, 8, 7, 6]], dtype=torch.int32),
                    num_reqs=1,
                    num_actual_tokens=1,
                    num_input_tokens=6,
                    max_query_len=1,
                    is_prefilling=[True],
                    attn_state=AscendAttentionState.DecodeOnly,
                ),
                [49, 1, 1, 1, 1, 1],
                [[9, 8, 7, 6]] * 6,
            ),
            # 2) Short extend with qlen=K+1=3, zero-width padded request [3, 3], is_prefilling=True.
            (
                _make_common_metadata(
                    query_ends=[3, 3],
                    seq_lens=[120, 0],
                    block_table=torch.tensor([[3, 2, 1, 0], [0, 0, 0, 0]], dtype=torch.int32),
                    num_reqs=2,
                    num_actual_tokens=3,
                    num_input_tokens=6,
                    max_query_len=3,
                    is_prefilling=[True, False],
                    attn_state=AscendAttentionState.ChunkedPrefill,
                ),
                [118, 119, 120, 1, 1, 1],
                [[3, 2, 1, 0]] * 6,
            ),
            # 3) Mixed varlen decode-graph replay: [spec_verify qlen=3 (False), short_extend qlen=1 (True), pad->6].
            (
                _make_common_metadata(
                    query_ends=[3, 4, 6],
                    seq_lens=[200, 65, 1],
                    block_table=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]], dtype=torch.int32),
                    num_reqs=3,
                    num_actual_tokens=4,
                    num_input_tokens=6,
                    max_query_len=3,
                    is_prefilling=[False, True, False],
                    attn_state=AscendAttentionState.ChunkedPrefill,
                ),
                [198, 199, 200, 65, 1, 1],
                [[1, 2, 3, 4]] * 3 + [[5, 6, 7, 8]] + [[0, 0, 0, 0]] * 2,
            ),
            # 4) Pure qlen=1 decode replay across 2 requests padded to 6 tokens.
            (
                _make_common_metadata(
                    query_ends=[1, 2, 6],
                    seq_lens=[30, 40, 1],
                    block_table=torch.tensor([[11, 12, 0, 0], [13, 14, 0, 0], [0, 0, 0, 0]], dtype=torch.int32),
                    num_reqs=3,
                    num_actual_tokens=2,
                    num_input_tokens=6,
                    max_query_len=1,
                    is_prefilling=[False, False, False],
                    attn_state=AscendAttentionState.DecodeOnly,
                ),
                [30, 40, 1, 1, 1, 1],
                [[11, 12, 0, 0], [13, 14, 0, 0]] + [[0, 0, 0, 0]] * 4,
            ),
        ]

        for replay_common, expected_kv_lens, expected_rows in replay_cases:
            replay_meta = builder.build(0, replay_common)
            resolved = provider.resolve({"model.layers.3.self_attn.attn": replay_meta})
            assert resolved["actual_seq_lengths"] == [1, 2, 3, 4, 5, 6]
            assert resolved["actual_seq_lengths_kv"] == expected_kv_lens
            assert resolved["block_table"].tolist() == expected_rows
            assert resolved["atten_mask"] is None
            assert resolved["sparse_mode"] == 0

    with pytest.raises(AssertionError, match="FIA replay phase differs from capture"):
        provider.resolve({"model.layers.3.self_attn.attn": SimpleNamespace(fia_single_query_graph=None)})


@pytest.mark.parametrize(
    "attn_state",
    [AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit],
)
def test_mixed_spec_verify_short_extend_and_long_prefill_keeps_decode_slice_sparse_mode_0(
    attn_state: AscendAttentionState,
):
    torch.manual_seed(7)
    cfg = _make_vllm_config(num_speculative_tokens=2, cudagraph_mode=CUDAGraphMode.NONE)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=torch.bfloat16)
    a5_profile = get_hardware_profile(AscendDeviceType.A5)
    k_cache = torch.randn(16, 16, 2, 32, dtype=torch.bfloat16)
    v_cache = torch.randn(16, 16, 2, 32, dtype=torch.bfloat16)
    impl = _make_backend_impl(cfg, key_cache=k_cache, value_cache=v_cache)
    block_table = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int32)
    query = torch.randn(15, 2, 32, dtype=torch.bfloat16)

    with (
        patch("vllm_ascend.attention.attention_v1.get_current_hardware_profile", return_value=a5_profile),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(capturing=False, is_draft_model=False),
        ),
    ):
        builder = AscendAttentionMetadataBuilder(spec, ["model.layers.3.self_attn.attn"], cfg, torch.device("cpu"))
        builder.attn_mask_builder = MagicMock()
        builder.attn_mask_builder.get_attention_mask.return_value = torch.ones((16, 16), dtype=torch.bool)

        # Req 0: spec_verify width=3 (False); Req 1: short_extend width=2 (True); Req 2: prefill width=10 (True).
        common = _make_common_metadata(
            query_ends=[3, 5, 15],
            seq_lens=[39, 18, 25],
            block_table=block_table,
            num_reqs=3,
            num_actual_tokens=15,
            num_input_tokens=15,
            max_query_len=10,
            is_prefilling=[False, True, True],
            attn_state=attn_state,
        )
        meta = builder.build(0, common)
        assert meta.num_decodes == 2
        assert meta.num_decode_tokens == 5
        assert meta.num_prefills == 1

        fia_calls: list[dict] = []

        def fake_fia(**kwargs):
            fia_calls.append(kwargs)
            ref = _reference_paged_causal_attention(
                kwargs["query"],
                k_cache,
                v_cache,
                kwargs["block_table"],
                list(kwargs["actual_seq_lengths"]),
                list(kwargs["actual_seq_lengths_kv"]),
            )
            return ref.to(kwargs["query"].dtype), None

        out = torch.empty_like(query)
        with patch("vllm_ascend.attention.attention_v1.DeviceOperator.npu_fused_infer_attention_score", fake_fia):
            impl.forward_fused_infer_attention(query, k_cache, v_cache, meta, out, kv_cache=(k_cache, v_cache))

        assert len(fia_calls) == 2
        # Decode slice unfolds BOTH spec_verify and short_extend into single-query sparse_mode=0!
        assert fia_calls[0]["actual_seq_lengths"] == [1, 2, 3, 4, 5]
        assert fia_calls[0]["actual_seq_lengths_kv"] == [37, 38, 39, 17, 18]
        assert fia_calls[0]["atten_mask"] is None
        assert fia_calls[0]["sparse_mode"] == 0
        # Trailing long prefill stays unexpanded on sparse_mode=3!
        assert fia_calls[1]["actual_seq_lengths"] == [10]
        assert fia_calls[1]["actual_seq_lengths_kv"] == [25]
        assert fia_calls[1]["atten_mask"] is not None
        assert fia_calls[1]["sparse_mode"] == 3

        expected = _reference_paged_causal_attention(query, k_cache, v_cache, block_table, [3, 5, 15], [39, 18, 25]).to(
            torch.bfloat16
        )
        assert torch.equal(out, expected)

        # Also verify non-split eager batch [spec_verify width=3 (False), short_extend width=2 (True)] (num_prefills=0)
        # runs in a single FIA call with sparse_mode=0!
        decode_only_common = _make_common_metadata(
            query_ends=[3, 5],
            seq_lens=[39, 18],
            block_table=block_table[:2],
            num_reqs=2,
            num_actual_tokens=5,
            num_input_tokens=5,
            max_query_len=3,
            is_prefilling=[False, True],
            attn_state=attn_state,
        )
        decode_only_meta = builder.build(0, decode_only_common)
        assert decode_only_meta.num_decodes == 2
        assert decode_only_meta.num_prefills == 0
        fia_calls.clear()
        out_short = torch.empty_like(query[:5])
        with patch("vllm_ascend.attention.attention_v1.DeviceOperator.npu_fused_infer_attention_score", fake_fia):
            impl.forward_fused_infer_attention(
                query[:5], k_cache, v_cache, decode_only_meta, out_short, kv_cache=(k_cache, v_cache)
            )
        assert len(fia_calls) == 1
        assert fia_calls[0]["actual_seq_lengths"] == [1, 2, 3, 4, 5]
        assert fia_calls[0]["actual_seq_lengths_kv"] == [37, 38, 39, 17, 18]
        assert fia_calls[0]["atten_mask"] is None
        assert fia_calls[0]["sparse_mode"] == 0
        assert torch.equal(out_short, expected[:5])


def test_target_vs_draft_and_c8_isolation_and_cache_invalidation():
    a5_profile = get_hardware_profile(AscendDeviceType.A5)
    target_cfg = _make_vllm_config(num_speculative_tokens=2, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    target_spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=torch.bfloat16)
    eagle_draft_spec = copy.copy(target_spec)
    object.__setattr__(eagle_draft_spec, "is_draft", True)
    swa_spec = SlidingWindowSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=torch.bfloat16, sliding_window=128)
    k_cache = torch.randn(8, 16, 2, 32, dtype=torch.bfloat16)
    v_cache = torch.randn(8, 16, 2, 32, dtype=torch.bfloat16)

    with (
        patch("vllm_ascend.attention.attention_v1.get_current_hardware_profile", return_value=a5_profile),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(capturing=False, is_draft_model=False),
        ),
    ):
        target_builder = AscendAttentionMetadataBuilder(
            target_spec, ["model.layers.3.self_attn.attn"], target_cfg, torch.device("cpu")
        )
        target_builder.attn_mask_builder = MagicMock()
        target_builder.attn_mask_builder.get_attention_mask.return_value = torch.ones((16, 16), dtype=torch.bool)
        assert target_builder.enable_single_query_spec_verify is True

        # 1) Draft builders (is_draft=True, runner_type='draft', mtp.* layers), C8, SWA, PCP, DCP, parallel_drafting:
        eagle_builder = AscendAttentionMetadataBuilder(
            eagle_draft_spec, ["draft.layers.0.self_attn.attn"], target_cfg, torch.device("cpu")
        )
        standalone_draft_builder = AscendAttentionMetadataBuilder(
            target_spec,
            ["model.layers.0.self_attn.attn"],
            _make_vllm_config(runner_type="draft"),
            torch.device("cpu"),
        )
        mtp_builder = AscendAttentionMetadataBuilder(
            target_spec, ["mtp.layers.0.self_attn.attn"], target_cfg, torch.device("cpu")
        )
        c8_builder = AscendAttentionMetadataBuilder(
            target_spec,
            ["model.layers.3.self_attn.attn"],
            _make_vllm_config(enable_c8_quant=True),
            torch.device("cpu"),
        )
        swa_builder = AscendAttentionMetadataBuilder(
            swa_spec, ["model.layers.0.self_attn.attn"], target_cfg, torch.device("cpu")
        )
        pard_builder = AscendAttentionMetadataBuilder(
            target_spec,
            ["model.layers.3.self_attn.attn"],
            _make_vllm_config(parallel_drafting=True),
            torch.device("cpu"),
        )
        for disabled_builder in (
            eagle_builder,
            standalone_draft_builder,
            mtp_builder,
            c8_builder,
            swa_builder,
            pard_builder,
        ):
            assert disabled_builder.enable_single_query_spec_verify is False

        # 2) Even if standalone EAGLE prefill or MTP draft or C8 receives a target AscendMetadata object directly
        #    in V2 eager (where _EXTRA_CTX.is_draft_model is False), runtime guards keep it unexpanded!
        block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        target_meta = target_builder.build(
            0,
            _make_common_metadata(
                query_ends=[3, 6],
                seq_lens=[40, 50],
                block_table=block_table,
                num_reqs=2,
                num_actual_tokens=6,
                num_input_tokens=6,
                max_query_len=3,
                is_prefilling=[False, False],
                attn_state=AscendAttentionState.ChunkedPrefill,
            ),
        )
        assert target_meta.fia_max_expand_width == 3
        assert target_meta.fia_single_query_graph is not None

        query = torch.randn(6, 2, 32, dtype=torch.bfloat16)
        target_impl = _make_backend_impl(target_cfg, key_cache=k_cache, value_cache=v_cache, is_draft_layer=False)
        draft_impl = _make_backend_impl(target_cfg, key_cache=k_cache, value_cache=v_cache, is_draft_layer=True)
        c8_impl = _make_backend_impl(
            target_cfg, key_cache=k_cache, value_cache=v_cache, enable_c8_quant=True, kv_cache_dtype=torch.int8
        )

        seen: list[tuple[list[int], int]] = []

        def record_fia(**kwargs):
            seen.append((list(kwargs["actual_seq_lengths"]), kwargs["sparse_mode"]))
            return torch.zeros_like(kwargs["query"]), None

        with patch("vllm_ascend.attention.attention_v1.DeviceOperator.npu_fused_infer_attention_score", record_fia):
            draft_impl.forward_fused_infer_attention(
                query, k_cache, v_cache, target_meta, torch.empty_like(query), kv_cache=(k_cache, v_cache)
            )
            c8_impl.forward_fused_infer_attention(
                query, k_cache, v_cache, target_meta, torch.empty_like(query), kv_cache=(k_cache, v_cache)
            )
            target_impl.forward_fused_infer_attention(
                query, k_cache, v_cache, target_meta, torch.empty_like(query), kv_cache=(k_cache, v_cache)
            )

        assert seen == [
            ([3, 6], 3),  # draft_impl: unexpanded!
            ([3, 6], 3),  # c8_impl: unexpanded!
            ([1, 2, 3, 4, 5, 6], 0),  # target_impl: unfolded!
        ]

        # 3) Long prefill in FULL_DECODE_ONLY mode must NOT build fia_single_query_graph or unfold.
        long_prefill_meta = target_builder.build(
            0,
            _make_common_metadata(
                query_ends=[12],
                seq_lens=[12],
                block_table=torch.tensor([[1, 2]], dtype=torch.int32),
                num_reqs=1,
                num_actual_tokens=12,
                num_input_tokens=12,
                max_query_len=12,
                is_prefilling=[True],
                attn_state=AscendAttentionState.PrefillCacheHit,
            ),
        )
        assert long_prefill_meta.fia_single_query_graph is None
        q_ends, kv_lens, _ = prepare_fia_layout(
            long_prefill_meta, [12], [12], torch.tensor([[1, 2]], dtype=torch.int32)
        )
        assert (q_ends, kv_lens) == ([12], [12])

        # 4) In-place seq_lens_list mutation (e.g. draft step updates) never hits stale cached kv_lengths.
        _, kv_step1, table_step1 = prepare_fia_layout(target_meta, [3, 6], [40, 50], block_table)
        _, kv_step1_cached, table_step1_cached = prepare_fia_layout(target_meta, [3, 6], [40, 50], block_table)
        assert kv_step1 == [38, 39, 40, 48, 49, 50]
        assert table_step1_cached is table_step1
        _, kv_step2, _ = prepare_fia_layout(target_meta, [3, 6], [41, 51], block_table)
        assert kv_step2 == [39, 40, 41, 49, 50, 51]


def test_speculator_update_decode_attn_metadata_clears_single_query_fields():
    speculator_path = (
        Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/target_dependent_ar/speculator.py"
    )
    tree = ast.parse(speculator_path.read_text())
    cls_node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendTargetDependentARSpeculator"
    )
    fn_node = next(
        n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == "_update_decode_attn_metadata"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    mod = ast.fix_missing_locations(ast.Module(body=[future, fn_node], type_ignores=[]))
    scope = {"Any": object, "torch": torch}
    exec(compile(mod, str(speculator_path), "exec"), scope)
    update_fn = scope["_update_decode_attn_metadata"]

    dummy_speculator = SimpleNamespace(
        attn_architecture="GQA",
        use_dcp=False,
        _get_seq_lens_cpu=lambda n: torch.tensor([20, 30][:n], dtype=torch.int32),
        _calc_next_seq_lens_cpu=lambda seq_lens_cpu, num_reqs, num_reqs_padded, step: seq_lens_cpu + step,
    )
    meta = SimpleNamespace(
        seq_lens_cpu=torch.tensor([20, 30], dtype=torch.int32),
        seq_lens_list=[20, 30],
        actual_seq_lengths_q=[3, 6],
        fia_max_expand_width=3,
        fia_single_query=(("cached",), [1, 2, 3, 4, 5, 6], [18, 19, 20, 28, 29, 30], torch.zeros((6, 2))),
        fia_single_query_graph=([1, 2, 3, 4, 5, 6], [18, 19, 20, 28, 29, 30], torch.zeros((6, 2))),
    )
    update_fn(dummy_speculator, {"draft.layers.0.self_attn.attn": meta}, step=1, num_reqs=2)
    assert meta.seq_lens_list == [21, 31]
    assert meta.actual_seq_lengths_q == [1, 2]
    assert meta.fia_max_expand_width == 0
    assert meta.fia_single_query is None
    assert meta.fia_single_query_graph is None


@pytest.mark.parametrize(
    ("device_type", "runner_type", "num_speculative_tokens"),
    [
        (AscendDeviceType.A2, "pooling", 0),
        (AscendDeviceType.A3, "pooling", 0),
        (AscendDeviceType.A5, "pooling", 0),
        (AscendDeviceType.A5, "pooling", 2),
        (AscendDeviceType.A3, "generate", 2),
        (AscendDeviceType.A5, "generate", 0),
    ],
)
def test_one_token_causal_pooling_with_spec_verify_disabled_keeps_mask_and_sparse_mode_3(
    device_type: AscendDeviceType,
    runner_type: str,
    num_speculative_tokens: int,
):
    profile = get_hardware_profile(device_type)
    cfg = _make_vllm_config(
        num_speculative_tokens=num_speculative_tokens,
        cudagraph_mode=CUDAGraphMode.NONE,
        runner_type=runner_type,
    )
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=torch.bfloat16)
    k_cache = torch.randn(8, 16, 2, 32, dtype=torch.bfloat16)
    v_cache = torch.randn(8, 16, 2, 32, dtype=torch.bfloat16)
    impl = _make_backend_impl(cfg, key_cache=k_cache, value_cache=v_cache)
    mask = torch.ones((16, 16), dtype=torch.bool)

    with (
        patch("vllm_ascend.attention.attention_v1.get_current_hardware_profile", return_value=profile),
        patch("vllm_ascend.attention.attention_v1.envs_vllm.VLLM_BATCH_INVARIANT", False),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(capturing=False, is_draft_model=False),
        ),
    ):
        builder = AscendAttentionMetadataBuilder(spec, ["model.layers.0.self_attn.attn"], cfg, torch.device("cpu"))
        builder.attn_mask_builder = MagicMock()
        builder.attn_mask_builder.get_attention_mask.return_value = mask
        assert builder.enable_single_query_spec_verify is False

        # One-token cached causal batch (V2 pooling state builder selects PrefillCacheHit).
        common = _make_common_metadata(
            query_ends=[1],
            seq_lens=[17],
            block_table=torch.tensor([[2, 3]], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=1,
            num_input_tokens=1,
            max_query_len=1,
            is_prefilling=[True],
            attn_state=AscendAttentionState.PrefillCacheHit,
            causal=True,
        )
        meta = builder.build(0, common)
        assert meta.fia_max_expand_width == 0

        fia_calls: list[dict] = []

        def record_fia(**kwargs):
            fia_calls.append(kwargs)
            return torch.zeros_like(kwargs["query"]), None

        query = torch.randn(1, 2, 32, dtype=torch.bfloat16)
        with patch("vllm_ascend.attention.attention_v1.DeviceOperator.npu_fused_infer_attention_score", record_fia):
            impl.forward_fused_infer_attention(
                query, k_cache, v_cache, meta, torch.empty_like(query), kv_cache=(k_cache, v_cache)
            )

        assert len(fia_calls) == 1
        assert fia_calls[0]["actual_seq_lengths"] == [1]
        assert fia_calls[0]["actual_seq_lengths_kv"] == [17]
        assert fia_calls[0]["atten_mask"] is mask
        assert fia_calls[0]["sparse_mode"] == 3


def test_device_index_cache_reuses_indices_tensor_across_steps():
    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    index_cache: dict = {}
    meta = SimpleNamespace(
        num_actual_tokens=6,
        fia_max_expand_width=3,
        fia_single_query=None,
    )

    # Step 1 populates index_cache; step 2 (with incremented kv_lengths) reuses the exact indices tensor.
    _, kv1, _ = prepare_fia_layout(meta, [3, 6], [40, 50], block_table, index_cache)
    assert kv1 == [38, 39, 40, 48, 49, 50]
    assert len(index_cache) == 1
    cached_indices, cached_stream = next(iter(index_cache.values()))
    assert cached_stream is None

    _, kv2, _ = prepare_fia_layout(meta, [3, 6], [41, 51], block_table, index_cache)
    assert kv2 == [39, 40, 41, 49, 50, 51]
    assert len(index_cache) == 1
    assert next(iter(index_cache.values()))[0] is cached_indices

    graph_out = prepare_graph_fia_layout([3, 6], [42, 52], block_table, 6, 6, index_cache)
    assert graph_out is not None
    graph_ends, graph_kv, _ = graph_out
    assert graph_ends == [1, 2, 3, 4, 5, 6]
    assert graph_kv == [40, 41, 42, 50, 51, 52]
    assert next(iter(index_cache.values()))[0] is cached_indices


def test_tnd_batch_4096_boundary_eager_and_graph():
    # 1024 requests x 4 tokens = 4096 unfolded TND entries (at limit -> unfolds).
    q_ends_4096 = [4 * (i + 1) for i in range(1024)]
    kv_lens_4096 = [32 + i for i in range(1024)]
    table_1025 = torch.arange(2050, dtype=torch.int32).view(1025, 2)

    meta_4096 = SimpleNamespace(
        num_actual_tokens=4096,
        fia_max_expand_width=4,
        fia_single_query=None,
    )
    ends_ok, kv_ok, tbl_ok = prepare_fia_layout(meta_4096, q_ends_4096, kv_lens_4096, table_1025[:1024])
    assert len(ends_ok) == MAX_TND_FIA_BATCH
    assert is_all_single_query(ends_ok, 4096)
    assert tbl_ok is not None and tbl_ok.shape == (4096, 2)

    # Adding 1 more single-token request -> 4097 unfolded entries (> 4096 -> falls back to unexpanded 1025 rows).
    q_ends_4097 = [*q_ends_4096, 4097]
    kv_lens_4097 = [*kv_lens_4096, 20]
    meta_4097 = SimpleNamespace(
        num_actual_tokens=4097,
        fia_max_expand_width=4,
        fia_single_query=None,
    )
    ends_fb, kv_fb, tbl_fb = prepare_fia_layout(meta_4097, q_ends_4097, kv_lens_4097, table_1025)
    assert ends_fb == q_ends_4097
    assert kv_fb == kv_lens_4097
    assert tbl_fb is table_1025
    assert not is_all_single_query(ends_fb, 4097)

    # Graph layout: 4096 succeeds, 4097 returns None so full_graph_fia stays on masked sparse_mode=3.
    graph_ok = prepare_graph_fia_layout(q_ends_4096, kv_lens_4096, table_1025[:1024], 4096, 4096)
    assert graph_ok is not None and len(graph_ok[0]) == 4096
    assert prepare_graph_fia_layout(q_ends_4097, kv_lens_4097, table_1025, 4097, 4097) is None


def test_pcp_dcp_batch_invariant_cross_page_and_full_graph_mode_guards():
    a5_profile = get_hardware_profile(AscendDeviceType.A5)
    a3_profile = get_hardware_profile(AscendDeviceType.A3)
    a2_profile = get_hardware_profile(AscendDeviceType.A2)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=torch.bfloat16)

    with (
        patch("vllm_ascend.attention.attention_v1.get_current_hardware_profile", return_value=a5_profile),
        patch("vllm_ascend.attention.attention_v1.envs_vllm.VLLM_BATCH_INVARIANT", False),
    ):
        pcp_builder = AscendAttentionMetadataBuilder(
            spec, ["model.layers.0.self_attn.attn"], _make_vllm_config(pcp_size=2), torch.device("cpu")
        )
        dcp_builder = AscendAttentionMetadataBuilder(
            spec, ["model.layers.0.self_attn.attn"], _make_vllm_config(dcp_size=2), torch.device("cpu")
        )
        full_only_builder = AscendAttentionMetadataBuilder(
            spec,
            ["model.layers.0.self_attn.attn"],
            _make_vllm_config(cudagraph_mode=CUDAGraphMode.FULL),
            torch.device("cpu"),
        )
        assert pcp_builder.enable_single_query_spec_verify is False
        assert dcp_builder.enable_single_query_spec_verify is False
        assert full_only_builder.enable_single_query_spec_verify is False

    # A2 and A3 enable single-query spec verify when VLLM_BATCH_INVARIANT=1.
    for profile in (a2_profile, a3_profile):
        with (
            patch("vllm_ascend.attention.attention_v1.get_current_hardware_profile", return_value=profile),
            patch("vllm_ascend.attention.attention_v1.envs_vllm.VLLM_BATCH_INVARIANT", True),
        ):
            inv_builder = AscendAttentionMetadataBuilder(
                spec, ["model.layers.0.self_attn.attn"], _make_vllm_config(), torch.device("cpu")
            )
            assert inv_builder.enable_single_query_spec_verify is True

    # FULL_AND_PIECEWISE with full_graph_mode=False skips prepare_graph_fia_layout; cross-page KV lengths
    # (31, 32, 33 across block_size=16 boundary 32) and multi-request 1-token PrefillCacheHit match reference.
    k_cache = torch.randn(8, 16, 2, 32, dtype=torch.bfloat16)
    v_cache = torch.randn(8, 16, 2, 32, dtype=torch.bfloat16)
    cfg_fp = _make_vllm_config(num_speculative_tokens=2, cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
    impl = _make_backend_impl(cfg_fp, key_cache=k_cache, value_cache=v_cache)

    with (
        patch("vllm_ascend.attention.attention_v1.get_current_hardware_profile", return_value=a5_profile),
        patch("vllm_ascend.attention.attention_v1.envs_vllm.VLLM_BATCH_INVARIANT", False),
        patch(
            "vllm_ascend.ascend_forward_context.get_forward_context",
            return_value=SimpleNamespace(capturing=False, is_draft_model=False),
        ),
    ):
        builder = AscendAttentionMetadataBuilder(spec, ["model.layers.0.self_attn.attn"], cfg_fp, torch.device("cpu"))
        builder.attn_mask_builder = MagicMock()
        builder.attn_mask_builder.get_attention_mask.return_value = torch.ones((16, 16), dtype=torch.bool)

        cross_page_table = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.int32)
        common_cross = _make_common_metadata(
            query_ends=[3, 4],
            seq_lens=[33, 18],
            block_table=cross_page_table,
            num_reqs=2,
            num_actual_tokens=4,
            num_input_tokens=4,
            max_query_len=3,
            is_prefilling=[False, False],
            attn_state=AscendAttentionState.ChunkedPrefill,
        )
        meta_piecewise = builder.build(0, common_cross, full_graph_mode=False)
        assert meta_piecewise.fia_single_query_graph is None
        meta_full = builder.build(0, common_cross, full_graph_mode=True)
        assert meta_full.fia_single_query_graph is not None

        q_cross = torch.randn(4, 2, 32, dtype=torch.bfloat16)
        with patch(
            "vllm_ascend.attention.attention_v1.DeviceOperator.npu_fused_infer_attention_score",
            side_effect=lambda **kw: (
                _reference_paged_causal_attention(
                    kw["query"],
                    k_cache,
                    v_cache,
                    kw["block_table"],
                    kw["actual_seq_lengths"],
                    kw["actual_seq_lengths_kv"],
                )
                .to(torch.bfloat16)
                .view(-1),
                None,
            ),
        ) as mock_fia:
            out_cross = impl.forward_fused_infer_attention(
                q_cross, k_cache, v_cache, meta_piecewise, torch.empty_like(q_cross), kv_cache=(k_cache, v_cache)
            )
        assert mock_fia.call_args.kwargs["actual_seq_lengths"] == [1, 2, 3, 4]
        assert mock_fia.call_args.kwargs["actual_seq_lengths_kv"] == [31, 32, 33, 18]
        assert mock_fia.call_args.kwargs["sparse_mode"] == 0
        ref_cross = _reference_paged_causal_attention(q_cross, k_cache, v_cache, cross_page_table, [3, 4], [33, 18]).to(
            torch.bfloat16
        )
        assert torch.equal(out_cross, ref_cross)

        # Multi-request all-1-token PrefillCacheHit under target spec-verify enabled skips mask safely with 1 query/row.
        common_1tok = _make_common_metadata(
            query_ends=[1, 2],
            seq_lens=[17, 33],
            block_table=cross_page_table,
            num_reqs=2,
            num_actual_tokens=2,
            num_input_tokens=2,
            max_query_len=1,
            is_prefilling=[False, False],
            attn_state=AscendAttentionState.PrefillCacheHit,
        )
        meta_1tok = builder.build(0, common_1tok, full_graph_mode=False)
        q_1tok = torch.randn(2, 2, 32, dtype=torch.bfloat16)
        seen_1tok: list[dict] = []
        with patch(
            "vllm_ascend.attention.attention_v1.DeviceOperator.npu_fused_infer_attention_score",
            side_effect=lambda **kw: (seen_1tok.append(kw), (torch.zeros_like(kw["query"]), None))[1],
        ):
            impl.forward_fused_infer_attention(
                q_1tok, k_cache, v_cache, meta_1tok, torch.empty_like(q_1tok), kv_cache=(k_cache, v_cache)
            )
        assert seen_1tok[0]["actual_seq_lengths"] == [1, 2]
        assert seen_1tok[0]["actual_seq_lengths_kv"] == [17, 33]
        assert seen_1tok[0]["atten_mask"] is None
        assert seen_1tok[0]["sparse_mode"] == 0


def test_device_index_cache_stream_isolation_and_eviction_records_stream():
    block_table = torch.arange(600, dtype=torch.int32).view(300, 2)
    index_cache: dict = {}

    stream_a = SimpleNamespace(npu_stream=101)
    stream_b = SimpleNamespace(npu_stream=202)
    recorded_streams: list[Any] = []

    class _TrackedIndices:
        def __init__(self, tensor: torch.Tensor):
            self.tensor = tensor

        def record_stream(self, stream: Any) -> None:
            recorded_streams.append(stream)

    # Populate 256 distinct non-identity row patterns on stream_a, then trigger 257th insert on stream_b.
    with patch(
        "vllm_ascend.attention.fia_single_query._current_device_stream",
        return_value=(stream_a, 101),
    ):
        for i in range(_MAX_INDEX_CACHE_ENTRIES):
            num_tokens = i + 2
            q_ends = [*range(1, i + 1), i + 2]
            kv_lens = [10] * (i + 1)
            meta = SimpleNamespace(num_actual_tokens=num_tokens, fia_max_expand_width=3, fia_single_query=None)
            prepare_fia_layout(meta, q_ends, kv_lens, block_table, index_cache)
            if i == 0:
                first_key = next(iter(index_cache.keys()))
                first_indices, first_stream = index_cache[first_key]
                index_cache[first_key] = (_TrackedIndices(first_indices), first_stream)

    assert len(index_cache) == _MAX_INDEX_CACHE_ENTRIES
    # Same rows on stream_b misses stream_a's key, evicts entry 0, and calls record_stream(stream_a).
    with patch(
        "vllm_ascend.attention.fia_single_query._current_device_stream",
        return_value=(stream_b, 202),
    ):
        meta_b = SimpleNamespace(num_actual_tokens=2, fia_max_expand_width=3, fia_single_query=None)
        prepare_fia_layout(meta_b, [2], [10], block_table, index_cache)

    assert len(index_cache) == _MAX_INDEX_CACHE_ENTRIES
    assert recorded_streams == [stream_a]
