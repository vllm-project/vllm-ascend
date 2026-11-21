# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU unit tests for head_dim=512 speculative paged verify and decode."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.attention_v1 import (
    SWA_INT_MAX,
    AscendAttentionBackendImpl,
    AscendAttentionState,
)
from vllm_ascend.attention.large_head_paged_verify import (
    AscendLargeHeadPagedAttentionBackendImpl,
    AscendLargeHeadPagedBackend,
    AscendLargeHeadPagedMetadata,
    AscendLargeHeadPagedMetadataBuilder,
    LargeHeadPagedVerifyParamProvider,
    build_paged_verify_mask,
    large_head_decode_graph_params,
    large_head_verify_graph_params,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    expand_pa_per_token_rows,
    requires_contiguous_pa_kv_cache,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.device.utils import FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE, _gather_paged_kv_to_dense
from vllm_ascend.platform import NPUPlatform, _should_use_large_head_paged_backend


def _selector_config(**overrides):
    defaults = dict(
        head_size=FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=128,
        use_mla=False,
        has_sliding_window=False,
        sliding_window=None,
        use_sparse=False,
        use_pcp=False,
        use_dcp=False,
        use_kv_connector=False,
        has_sink=False,
        use_non_causal=False,
        use_per_head_quant_scales=False,
        use_adaptive_verification=False,
        use_rswa=False,
        use_hisparse=False,
        attn_type=AttentionType.DECODER,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mock_vllm_config(
    *,
    with_spec: bool = True,
    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    runner_type: str = "generate",
    max_cudagraph_capture_size: int | None = None,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type=runner_type,
            max_model_len=8192,
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(model_type="gemma4", architectures=["Gemma4ForConditionalGeneration"]),
            hf_text_config=SimpleNamespace(model_type="gemma4_text"),
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        cache_config=SimpleNamespace(block_size=8, cache_dtype="auto"),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True, max_num_seqs=8, max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            max_cudagraph_capture_size=max_cudagraph_capture_size,
        ),
        speculative_config=(
            SimpleNamespace(
                num_speculative_tokens=3,
                parallel_drafting=False,
                speculator_config=SimpleNamespace(parallel_drafting=False),
                use_dspark=lambda: False,
            )
            if with_spec
            else None
        ),
        kv_transfer_config=None,
        quant_config=None,
        use_v2_model_runner=True,
    )


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3])
def test_select_large_head_paged_backend_enabled_on_a2_a3(monkeypatch, device):
    cfg = _mock_vllm_config(with_spec=True)
    selector = _selector_config()
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.platform.get_current_hardware_profile", lambda: get_hardware_profile(device))
    assert _should_use_large_head_paged_backend(selector) is True

    monkeypatch.setattr("vllm_ascend.platform._validate_fa3_backend", lambda *_: False)
    monkeypatch.setattr("vllm_ascend.platform._validate_flash_attn_backend", lambda *_: False)
    monkeypatch.setattr("vllm_ascend.platform._validate_flash_attn_npu_3", lambda *_: False)
    assert (
        NPUPlatform._select_attn_backend_cls(selector)
        == "vllm_ascend.attention.large_head_paged_verify.AscendLargeHeadPagedBackend"
    )


@pytest.mark.parametrize(
    ("device", "with_spec", "runner_type", "selector_overrides"),
    [
        (AscendDeviceType.A5, True, "generate", {}),
        (AscendDeviceType._310P, True, "generate", {}),
        (AscendDeviceType.A2, False, "generate", {}),
        (AscendDeviceType.A2, True, "pooling", {}),
        (AscendDeviceType.A2, True, "generate", {"head_size": 256}),
        (AscendDeviceType.A2, True, "generate", {"has_sliding_window": True}),
        (AscendDeviceType.A2, True, "generate", {"attn_type": AttentionType.ENCODER_ONLY}),
        (AscendDeviceType.A2, True, "generate", {"use_mla": True}),
        (AscendDeviceType.A2, True, "generate", {"use_sparse": True}),
        (AscendDeviceType.A2, True, "generate", {"use_pcp": True}),
        (AscendDeviceType.A2, True, "generate", {"use_dcp": True}),
        (AscendDeviceType.A2, True, "generate", {"use_kv_connector": True}),
        (AscendDeviceType.A2, True, "generate", {"has_sink": True}),
        (AscendDeviceType.A2, True, "generate", {"use_non_causal": True}),
        (AscendDeviceType.A2, True, "generate", {"use_per_head_quant_scales": True}),
        (AscendDeviceType.A2, True, "generate", {"use_adaptive_verification": True}),
        (AscendDeviceType.A2, True, "generate", {"use_rswa": True}),
        (AscendDeviceType.A2, True, "generate", {"use_hisparse": True}),
    ],
)
def test_select_large_head_paged_backend_exclusions(monkeypatch, device, with_spec, runner_type, selector_overrides):
    cfg = _mock_vllm_config(with_spec=with_spec, runner_type=runner_type)
    selector = _selector_config(**selector_overrides)
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.platform.get_current_hardware_profile", lambda: get_hardware_profile(device))
    assert _should_use_large_head_paged_backend(selector) is False


def test_kv_cache_shape_and_contiguous_cache_contract(monkeypatch):
    a2_profile = get_hardware_profile(AscendDeviceType.A2)
    a5_profile = get_hardware_profile(AscendDeviceType.A5)

    with (
        patch("vllm_ascend.attention.large_head_paged_verify.get_current_hardware_profile", return_value=a2_profile),
        patch("vllm.config.get_current_vllm_config_or_none", return_value=_mock_vllm_config()),
    ):
        shape_bnsd = AscendLargeHeadPagedBackend.get_kv_cache_shape(16, 128, 4, 512)
        assert shape_bnsd == (2, 16, 4, 128, 512)

    with (
        patch("vllm_ascend.attention.large_head_paged_verify.get_current_hardware_profile", return_value=a2_profile),
        patch(
            "vllm.config.get_current_vllm_config_or_none",
            return_value=_mock_vllm_config(cudagraph_mode=CUDAGraphMode.FULL),
        ),
    ):
        shape_full = AscendLargeHeadPagedBackend.get_kv_cache_shape(16, 128, 4, 512)
        assert shape_full == (2, 16, 128, 4, 512)

    with (
        patch("vllm_ascend.attention.large_head_paged_verify.get_current_hardware_profile", return_value=a5_profile),
        patch("vllm.config.get_current_vllm_config_or_none", return_value=_mock_vllm_config()),
    ):
        shape_a5 = AscendLargeHeadPagedBackend.get_kv_cache_shape(16, 128, 4, 512)
        assert shape_a5 == (2, 16, 128, 4, 512)

    cfg = _mock_vllm_config(with_spec=True)
    spec = FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=512, dtype=torch.bfloat16)
    impl = AscendLargeHeadPagedAttentionBackendImpl.__new__(AscendLargeHeadPagedAttentionBackendImpl)
    impl.sliding_window = None
    assert requires_contiguous_pa_kv_cache(SimpleNamespace(impl=impl), cfg, spec) is True

    padded_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=512,
        dtype=torch.bfloat16,
        page_size_padded=spec.real_page_size_bytes + 64,
    )
    assert requires_contiguous_pa_kv_cache(SimpleNamespace(impl=impl), cfg, padded_spec) is False

    base_impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    base_impl.sliding_window = None
    assert requires_contiguous_pa_kv_cache(SimpleNamespace(impl=base_impl), cfg, spec) is False


def test_build_paged_verify_mask_oracle_and_pad_row_safety():
    # Batch: real row seq=6, real minimal seq=4 (k=3 -> qlen=4), padded/dummy rows seq=1 and seq=0.
    seq_lens = torch.tensor([6, 4, 1, 0], dtype=torch.int32)
    block_table = torch.zeros((4, 1), dtype=torch.int32)
    mask = build_paged_verify_mask(
        seq_lens,
        block_table,
        num_speculative_tokens=3,
        block_size=8,
        device=torch.device("cpu"),
    )
    assert mask.shape == (4, 1, 4, 8)
    assert mask.dtype == torch.bool

    expected = torch.ones((4, 1, 4, 8), dtype=torch.bool)
    # seq=6, qlen=4 -> allowed last positions: [2, 3, 4, 5]
    for j in range(4):
        expected[0, 0, j, : 2 + j + 1] = False
        expected[1, 0, j, : j + 1] = False
        # Padded/dummy rows clamp_min(0) -> position 0 stays unmasked (False) to prevent all-True NaNs.
        expected[2, 0, j, 0] = False
        expected[3, 0, j, 0] = False

    torch.testing.assert_close(mask, expected)


def test_expand_pa_per_token_rows_oracle_and_idempotency():
    metadata = AscendLargeHeadPagedMetadata(
        block_tables=torch.tensor([[10, 11], [20, 21], [99, 99]], dtype=torch.int32),
        actual_seq_lengths_q=[4, 6, 6],
        seq_lens_list=[10, 5, 0],
        seq_lens=torch.tensor([10, 5, 0], dtype=torch.int32),
    )
    assert expand_pa_per_token_rows(metadata) is True
    assert metadata.seq_lens.tolist() == [7, 8, 9, 10, 4, 5]
    torch.testing.assert_close(
        metadata.block_tables,
        torch.tensor([[10, 11], [10, 11], [10, 11], [10, 11], [20, 21], [20, 21]], dtype=torch.int32),
    )
    # Second call must be idempotent.
    assert expand_pa_per_token_rows(metadata) is True
    assert metadata.seq_lens.tolist() == [7, 8, 9, 10, 4, 5]


def test_gather_paged_kv_to_dense_bnsd_matches_natural(monkeypatch):
    num_blocks, block_size, num_kv_heads, head_size = 8, 8, 2, 16
    natural_k = torch.randn((num_blocks, block_size, num_kv_heads, head_size), dtype=torch.float32)
    natural_v = torch.randn((num_blocks, block_size, num_kv_heads, head_size), dtype=torch.float32)
    bnsd_k = natural_k.permute(0, 2, 1, 3).contiguous()
    bnsd_v = natural_v.permute(0, 2, 1, 3).contiguous()

    # Verify BNSD selects needed blocks on the contiguous pool before permuting.
    orig_index_select = torch.Tensor.index_select
    observed_contiguous: list[bool] = []

    def spy_index_select(tensor, dim, index):
        if tensor.shape == bnsd_k.shape:
            observed_contiguous.append(tensor.is_contiguous())
        return orig_index_select(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", spy_index_select)

    for block_table, seq_lens in (
        (torch.tensor([[5, 1]], dtype=torch.int32), [5]),
        (torch.tensor([[2, 0], [3, 1]], dtype=torch.int32), [16, 16]),
        (torch.tensor([[2, 0], [3, 1], [6, 7]], dtype=torch.int32), [10, 5, 16]),
    ):
        out_nat_k, out_nat_v = _gather_paged_kv_to_dense(
            natural_k, natural_v, block_table, seq_lens, num_kv_heads, head_size
        )
        out_bnsd_k, out_bnsd_v = _gather_paged_kv_to_dense(
            bnsd_k, bnsd_v, block_table, seq_lens, num_kv_heads, head_size
        )
        torch.testing.assert_close(out_nat_k, out_bnsd_k)
        torch.testing.assert_close(out_nat_v, out_bnsd_v)
        assert out_bnsd_k.is_contiguous() and out_bnsd_v.is_contiguous()

    assert observed_contiguous and all(observed_contiguous)

    # Explicit use_bnsd handles square dims where block_size == num_kv_heads.
    sq_nat_k = torch.randn((3, 4, 4, 8), dtype=torch.float32)
    sq_nat_v = torch.randn((3, 4, 4, 8), dtype=torch.float32)
    sq_bnsd_k = sq_nat_k.permute(0, 2, 1, 3).contiguous()
    sq_bnsd_v = sq_nat_v.permute(0, 2, 1, 3).contiguous()
    bt_sq = torch.tensor([[1, 2]], dtype=torch.int32)
    exp_k, exp_v = _gather_paged_kv_to_dense(sq_nat_k, sq_nat_v, bt_sq, [6], 4, 8, use_bnsd=False)
    act_k, act_v = _gather_paged_kv_to_dense(sq_bnsd_k, sq_bnsd_v, bt_sq, [6], 4, 8, use_bnsd=True)
    torch.testing.assert_close(exp_k, act_k)
    torch.testing.assert_close(exp_v, act_v)


def test_metadata_builder_capture_vs_runtime_mask(monkeypatch):
    cfg = _mock_vllm_config(with_spec=True, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    spec = FullAttentionSpec(block_size=8, num_kv_heads=2, head_size=512, dtype=torch.bfloat16)
    monkeypatch.setattr(
        "vllm_ascend.attention.attention_v1.get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    monkeypatch.setattr("vllm_ascend.attention.utils.get_current_vllm_config", lambda: cfg)
    builder = AscendLargeHeadPagedMetadataBuilder(spec, ["layer.59"], cfg, torch.device("cpu"))
    common = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([6, 8], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=2,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        max_seq_len=8,
        causal=True,
        attn_state=AscendAttentionState.ChunkedPrefill,
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int32),
    )

    cap_meta = builder.build_for_cudagraph_capture(common)
    assert cap_meta.paged_verify_mask is None
    assert cap_meta.use_pa_graph is False

    rt_meta = builder.build(0, common)
    assert rt_meta.paged_verify_mask is None
    assert rt_meta.use_pa_graph is False


def test_bnsd_verify_capture_param_provider_and_eager_routing(monkeypatch):
    cfg = _mock_vllm_config(with_spec=True, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1.get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.attention.utils.get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1.is_pcp_decode_sharding_enabled", lambda *_: False)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1.needs_layer_aware_fia_graph_replay", lambda: True)
    monkeypatch.setattr(
        "vllm_ascend.attention.large_head_paged_verify.get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    monkeypatch.setattr(
        "vllm_ascend.attention.attention_v1.get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    extra_ctx = SimpleNamespace(capturing=False, is_draft_model=False)
    monkeypatch.setattr("vllm_ascend.attention.large_head_paged_verify._EXTRA_CTX", extra_ctx)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1._EXTRA_CTX", extra_ctx)

    impl = AscendLargeHeadPagedAttentionBackendImpl(
        num_heads=8,
        head_size=512,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )
    impl._layer_name = "layer.59"
    assert impl.use_bnsd_kv_cache is True

    k_cache = torch.zeros((4, 2, 8, 512), dtype=torch.bfloat16)
    v_cache = torch.zeros((4, 2, 8, 512), dtype=torch.bfloat16)
    impl.key_cache, impl.value_cache = k_cache, v_cache

    query = torch.randn((8, 8, 512), dtype=torch.bfloat16)
    key = torch.randn((8, 2, 512), dtype=torch.bfloat16)
    value = torch.randn((8, 2, 512), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    cap_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[4, 8],
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        seq_lens_list=[6, 8],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_prefills=0,
        causal=True,
    )

    captured_ws = torch.zeros(128, dtype=torch.uint8)
    get_ws_mock = MagicMock(return_value=captured_ws)
    reg_mock = MagicMock()
    monkeypatch.setattr(
        "torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace",
        get_ws_mock,
        raising=False,
    )
    monkeypatch.setattr("vllm_ascend.attention.large_head_paged_verify.register_task", reg_mock)
    monkeypatch.setattr(
        "vllm_ascend.attention.large_head_paged_verify.get_capture_resource",
        lambda _key, factory, use_max_workspace=False: factory(),
    )

    extra_ctx.capturing = True
    try:
        impl.forward_fused_infer_attention(query, key, value, cap_meta, output)
    finally:
        extra_ctx.capturing = False

    get_ws_mock.assert_called_once()
    assert reg_mock.call_count == 1
    op_arg, kwargs_arg, provider_arg = reg_mock.call_args[0]
    assert op_arg is torch_npu.npu_fused_infer_attention_score_v2.out
    assert kwargs_arg["input_layout"] == "BSND"
    assert kwargs_arg["sparse_mode"] == 3
    assert kwargs_arg["pre_tokens"] == SWA_INT_MAX
    assert kwargs_arg["next_tokens"] == 0
    assert kwargs_arg["block_size"] == 8
    assert kwargs_arg["actual_seq_qlen"] == [4, 4]
    assert kwargs_arg["atten_mask"].shape == (2048, 2048)
    assert kwargs_arg["atten_mask"].dtype == torch.int8
    assert kwargs_arg["out"][0].data_ptr() == output.data_ptr()
    assert provider_arg == LargeHeadPagedVerifyParamProvider("layer.59")

    # Provider omits atten_mask and clamps padded/dummy KV lengths to >= qlen.
    pad_meta = AscendLargeHeadPagedMetadata(
        block_tables=torch.zeros((4, 2), dtype=torch.int32),
        actual_seq_lengths_q=[4, 8, 12, 16],
        seq_lens=torch.tensor([6, 8, 1, 0], dtype=torch.int32),
        seq_lens_list=[6, 8, 1, 0],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_prefills=0,
        causal=True,
    )
    resolved = provider_arg.resolve({"layer.59": pad_meta})
    assert "atten_mask" not in resolved
    assert resolved["actual_seq_kvlen"] == [6, 8, 4, 4]

    # Eager BSND verify path uses sparse_mode=3 + static [2048, 2048] mask without dynamic mask alloc.
    fia_v2_mock = MagicMock(return_value=(torch.ones((2, 4, 8, 512), dtype=torch.bfloat16), torch.zeros(1)))
    monkeypatch.setattr("torch_npu.npu_fused_infer_attention_score_v2", fia_v2_mock)
    rt_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[4, 8],
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        seq_lens_list=[6, 8],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_prefills=0,
        causal=True,
    )
    out = impl.forward_fused_infer_attention(query, key, value, rt_meta, output)
    assert out is output
    assert fia_v2_mock.call_count == 1
    assert fia_v2_mock.call_args.kwargs["input_layout"] == "BSND"
    assert fia_v2_mock.call_args.kwargs["sparse_mode"] == 3
    assert fia_v2_mock.call_args.kwargs["pre_tokens"] == SWA_INT_MAX
    assert fia_v2_mock.call_args.kwargs["next_tokens"] == 0
    assert fia_v2_mock.call_args.kwargs["atten_mask"].shape == (2048, 2048)
    assert fia_v2_mock.call_args.kwargs["atten_mask"].dtype == torch.int8
    torch.testing.assert_close(output, torch.ones_like(output))

    # Eager draft decode (qlen=1, _EXTRA_CTX.is_draft_model=True) also uses BSND paged op without mask.
    fia_v2_mock.reset_mock()
    fia_v2_mock.return_value = (torch.full((2, 1, 8, 512), 2.0, dtype=torch.bfloat16), torch.zeros(1))
    draft_q = torch.randn((2, 8, 512), dtype=torch.bfloat16)
    draft_out = torch.empty_like(draft_q)
    draft_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[1, 2],
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        seq_lens_list=[6, 8],
        attn_state=AscendAttentionState.DecodeOnly,
        num_prefills=0,
        causal=True,
    )
    extra_ctx.is_draft_model = True
    try:
        impl.forward_fused_infer_attention(draft_q, draft_q[:, :2], draft_q[:, :2], draft_meta, draft_out)
    finally:
        extra_ctx.is_draft_model = False
    assert fia_v2_mock.call_count == 1
    assert fia_v2_mock.call_args.kwargs["input_layout"] == "BSND"
    assert fia_v2_mock.call_args.kwargs["sparse_mode"] == 0
    assert fia_v2_mock.call_args.kwargs["actual_seq_qlen"] == [1, 1]
    assert fia_v2_mock.call_args.kwargs["atten_mask"] is None
    torch.testing.assert_close(draft_out, torch.full_like(draft_out, 2.0))

    # PrefillNoCache must never enter eager verify even with uniform qlen=4.
    fia_v2_mock.reset_mock()
    nocache_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[4, 8],
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        seq_lens_list=[4, 4],
        attn_state=AscendAttentionState.PrefillNoCache,
        num_prefills=2,
        causal=True,
    )
    assert impl._should_eager_bnsd_verify(nocache_meta, query.shape[0]) is False

    # FULL mode helpers emit PagedAttention kwargs with expanded block tables.
    pa_meta = AscendLargeHeadPagedMetadata(
        block_tables=torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]], dtype=torch.int32),
        actual_seq_lengths_q=[2, 4],
        seq_lens=torch.tensor([5, 6, 7, 8], dtype=torch.int32),
        seq_lens_list=[6, 8],
        use_pa_graph=True,
    )
    pa_verify = large_head_verify_graph_params("layer.59", pa_meta)
    assert set(pa_verify) == {"block_table", "context_lens"}
    assert pa_verify["block_table"].shape[0] == 4
    pa_decode = large_head_decode_graph_params(pa_meta, [11, 12])
    assert set(pa_decode) == {"block_table", "context_lens"}
    assert pa_decode["context_lens"].tolist() == [11, 12]


def test_eager_verify_synthetic_short_row_falls_back_to_bnsd_mask(monkeypatch):
    cfg = _mock_vllm_config(with_spec=True, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    _patch_large_head_env(monkeypatch, cfg)
    impl = _new_large_head_impl(
        torch.zeros((4, 2, 8, 512), dtype=torch.bfloat16),
        torch.zeros((4, 2, 8, 512), dtype=torch.bfloat16),
    )
    fia_v2_mock = MagicMock(return_value=(torch.full((2, 8, 4, 512), 1.25, dtype=torch.bfloat16), torch.zeros(1)))
    monkeypatch.setattr("torch_npu.npu_fused_infer_attention_score_v2", fia_v2_mock)

    query = torch.randn((8, 8, 512), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    short_meta = AscendLargeHeadPagedMetadata(
        block_tables=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        actual_seq_lengths_q=[4, 8],
        seq_lens=torch.tensor([2, 8], dtype=torch.int32),
        seq_lens_list=[2, 8],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_prefills=0,
        causal=True,
    )
    impl.forward_fused_infer_attention(query, query[:, :2], query[:, :2], short_meta, output)
    assert fia_v2_mock.call_count == 1
    assert fia_v2_mock.call_args.kwargs["input_layout"] == "BNSD"
    assert fia_v2_mock.call_args.kwargs["sparse_mode"] == 0
    assert fia_v2_mock.call_args.kwargs["atten_mask"].shape == (2, 1, 4, 16)
    assert fia_v2_mock.call_args.kwargs["atten_mask"].dtype == torch.bool
    torch.testing.assert_close(output, torch.full_like(output, 1.25))


def _patch_large_head_env(monkeypatch, cfg, *, kv_layout: str | None = None):
    extra_ctx = SimpleNamespace(capturing=False, is_draft_model=False)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1.get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.attention.utils.get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1.is_pcp_decode_sharding_enabled", lambda *_: False)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1.needs_layer_aware_fia_graph_replay", lambda: True)
    monkeypatch.setattr(
        "vllm_ascend.attention.large_head_paged_verify.get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    monkeypatch.setattr(
        "vllm_ascend.attention.attention_v1.get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    monkeypatch.setattr("vllm_ascend.attention.large_head_paged_verify._EXTRA_CTX", extra_ctx)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1._EXTRA_CTX", extra_ctx)
    if kv_layout is not None:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", kv_layout)
        monkeypatch.setattr("vllm.envs.VLLM_KV_CACHE_LAYOUT", kv_layout, raising=False)
    return extra_ctx


def _new_large_head_impl(k_cache: torch.Tensor, v_cache: torch.Tensor) -> AscendLargeHeadPagedAttentionBackendImpl:
    impl = AscendLargeHeadPagedAttentionBackendImpl(
        num_heads=8,
        head_size=512,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )
    impl._layer_name = "layer.59"
    impl.key_cache, impl.value_cache = k_cache, v_cache
    return impl


def test_full_mode_metadata_isolation_and_full_graph_pa(monkeypatch):
    cfg = _mock_vllm_config(with_spec=True, cudagraph_mode=CUDAGraphMode.FULL, max_cudagraph_capture_size=8)
    extra_ctx = _patch_large_head_env(monkeypatch, cfg, kv_layout="NHD")
    spec = FullAttentionSpec(block_size=8, num_kv_heads=2, head_size=512, dtype=torch.bfloat16)
    builder = AscendLargeHeadPagedMetadataBuilder(spec, ["layer.59"], cfg, torch.device("cpu"))

    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    common_small = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([6, 8], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=2,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        max_seq_len=8,
        causal=True,
        attn_state=AscendAttentionState.ChunkedPrefill,
        block_table_tensor=block_table,
        slot_mapping=torch.arange(8, dtype=torch.int32),
    )
    meta = builder.build(0, common_small)
    assert meta.use_pa_graph is True
    assert meta.block_tables.shape == (2, 2)
    assert meta.seq_lens.tolist() == [6, 8]
    assert meta.pa_block_tables is not None and meta.pa_block_tables.shape == (8, 2)
    assert meta.pa_context_lens is not None
    assert meta.pa_context_lens.tolist() == [3, 4, 5, 6, 5, 6, 7, 8]
    resolved = large_head_verify_graph_params("layer.59", meta)
    assert resolved["block_table"] is meta.pa_block_tables
    assert resolved["context_lens"] is meta.pa_context_lens

    # Runtime batches above max_cudagraph_capture_size skip PA graph expansion cleanly.
    common_large = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 10], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 10], dtype=torch.int32),
        seq_lens=torch.tensor([6, 10], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([6, 10], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=2,
        num_actual_tokens=10,
        num_input_tokens=10,
        max_query_len=6,
        max_seq_len=10,
        causal=True,
        attn_state=AscendAttentionState.ChunkedPrefill,
        block_table_tensor=block_table,
        slot_mapping=torch.arange(10, dtype=torch.int32),
    )
    large_meta = builder.build(0, common_large)
    assert large_meta.use_pa_graph is True
    assert large_meta.block_tables.shape == (2, 2)
    assert large_meta.pa_block_tables is None
    assert large_meta.pa_context_lens is None

    cap_meta = builder.build_for_cudagraph_capture(common_small)
    assert cap_meta.block_tables.shape == (2, 2)
    assert cap_meta.pa_block_tables is not None and cap_meta.pa_block_tables.shape == (8, 2)

    impl = _new_large_head_impl(
        torch.zeros((4, 8, 2, 512), dtype=torch.bfloat16),
        torch.zeros((4, 8, 2, 512), dtype=torch.bfloat16),
    )
    assert impl.use_bnsd_kv_cache is False

    get_ws_mock = MagicMock(return_value=torch.zeros(64, dtype=torch.uint8))
    reg_mock = MagicMock()
    monkeypatch.setattr("torch_npu._npu_paged_attention_get_workspace", get_ws_mock, raising=False)
    monkeypatch.setattr("vllm_ascend.attention.large_head_paged_verify.register_task", reg_mock)
    monkeypatch.setattr(
        "vllm_ascend.attention.large_head_paged_verify.get_capture_resource",
        lambda _key, factory, use_max_workspace=False: factory(),
    )

    q = torch.randn((8, 8, 512), dtype=torch.bfloat16)
    out = torch.empty_like(q)
    extra_ctx.capturing = True
    try:
        impl.forward_fused_infer_attention(q, q[:, :2], q[:, :2], cap_meta, out)
    finally:
        extra_ctx.capturing = False

    get_ws_mock.assert_called_once()
    reg_mock.assert_called_once()
    op_arg, kwargs_arg, provider_arg = reg_mock.call_args[0]
    assert op_arg is torch_npu._npu_paged_attention
    assert kwargs_arg["block_table"] is cap_meta.pa_block_tables
    assert kwargs_arg["context_lens"] is cap_meta.pa_context_lens
    assert provider_arg == LargeHeadPagedVerifyParamProvider("layer.59")


def test_eager_mixed_chunked_prefill_phase_split_and_nonuniform_decode(monkeypatch):
    cfg = _mock_vllm_config(with_spec=True, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    _patch_large_head_env(monkeypatch, cfg)
    impl = _new_large_head_impl(
        torch.zeros((8, 2, 8, 512), dtype=torch.bfloat16),
        torch.zeros((8, 2, 8, 512), dtype=torch.bfloat16),
    )

    fia_v2_mock = MagicMock(return_value=(torch.full((2, 4, 8, 512), 1.5, dtype=torch.bfloat16), torch.zeros(1)))
    fusion_mock = MagicMock(return_value=(torch.full((6, 8, 512), 2.5, dtype=torch.bfloat16), None))
    gather_calls: list[dict[str, object]] = []

    def spy_gather(k_cache, v_cache, block_table, seq_lens, num_kv_heads, head_size, *, use_bnsd=None):
        gather_calls.append({"block_table": block_table, "seq_lens": list(seq_lens), "use_bnsd": use_bnsd})
        return _gather_paged_kv_to_dense(
            k_cache, v_cache, block_table, seq_lens, num_kv_heads, head_size, use_bnsd=use_bnsd
        )

    monkeypatch.setattr("torch_npu.npu_fused_infer_attention_score_v2", fia_v2_mock)
    monkeypatch.setattr("torch_npu.npu_fusion_attention", fusion_mock)
    monkeypatch.setattr("vllm_ascend.attention.large_head_paged_verify._gather_paged_kv_to_dense", spy_gather)

    block_table = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    q_mixed = torch.randn((14, 8, 512), dtype=torch.bfloat16)
    k_mixed = torch.randn((14, 2, 512), dtype=torch.bfloat16)
    v_mixed = torch.randn((14, 2, 512), dtype=torch.bfloat16)

    # Case 1: 2 uniform decode/verify requests + 1 fresh prefill -> BSND sparse_mode=3, 0 dense gathers.
    fresh_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[4, 8, 14],
        seq_lens=torch.tensor([6, 8, 6], dtype=torch.int32),
        seq_lens_list=[6, 8, 6],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_decodes=2,
        num_prefills=1,
        num_decode_tokens=8,
        causal=True,
    )
    out_fresh = torch.empty_like(q_mixed)
    impl.forward_fused_infer_attention(q_mixed, k_mixed, v_mixed, fresh_meta, out_fresh)
    assert fia_v2_mock.call_count == 1
    assert fia_v2_mock.call_args.kwargs["input_layout"] == "BSND"
    assert fia_v2_mock.call_args.kwargs["sparse_mode"] == 3
    assert fia_v2_mock.call_args.kwargs["actual_seq_qlen"] == [4, 4]
    assert fia_v2_mock.call_args.kwargs["actual_seq_kvlen"] == [6, 8]
    assert fia_v2_mock.call_args.kwargs["atten_mask"].shape == (2048, 2048)
    assert fia_v2_mock.call_args.kwargs["atten_mask"].dtype == torch.int8
    assert fusion_mock.call_count == 1
    assert fusion_mock.call_args.kwargs["actual_seq_qlen"] == [6]
    assert fusion_mock.call_args.kwargs["actual_seq_kvlen"] == [6]
    assert not gather_calls
    torch.testing.assert_close(out_fresh[:8], torch.full_like(out_fresh[:8], 1.5))
    torch.testing.assert_close(out_fresh[8:14], torch.full_like(out_fresh[8:14], 2.5))

    # Case 2: cached prefill gathers only the single prefill row with use_bnsd=True.
    fia_v2_mock.reset_mock()
    fusion_mock.reset_mock()
    cached_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[4, 8, 14],
        seq_lens=torch.tensor([6, 8, 10], dtype=torch.int32),
        seq_lens_list=[6, 8, 10],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_decodes=2,
        num_prefills=1,
        num_decode_tokens=8,
        causal=True,
    )
    out_cached = torch.empty_like(q_mixed)
    impl.forward_fused_infer_attention(q_mixed, k_mixed, v_mixed, cached_meta, out_cached)
    assert fia_v2_mock.call_count == 1
    assert fia_v2_mock.call_args.kwargs["input_layout"] == "BSND"
    assert fia_v2_mock.call_args.kwargs["sparse_mode"] == 3
    assert len(gather_calls) == 1
    assert gather_calls[0]["seq_lens"] == [10]
    assert gather_calls[0]["use_bnsd"] is True
    torch.testing.assert_close(gather_calls[0]["block_table"], torch.tensor([[4, 5]], dtype=torch.int32))
    assert fusion_mock.call_args.kwargs["actual_seq_qlen"] == [6]
    assert fusion_mock.call_args.kwargs["actual_seq_kvlen"] == [10]

    # Case 3: non-uniform decodes expand per-token to single-query BSND FIA-v2 without mask or gather.
    fia_v2_mock.reset_mock()
    fusion_mock.reset_mock()
    gather_calls.clear()
    fia_v2_mock.return_value = (torch.full((5, 1, 8, 512), 3.5, dtype=torch.bfloat16), torch.zeros(1))
    q_nonunif = torch.randn((5, 8, 512), dtype=torch.bfloat16)
    nonunif_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table[:2],
        actual_seq_lengths_q=[4, 5],
        seq_lens=torch.tensor([10, 5], dtype=torch.int32),
        seq_lens_list=[10, 5],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_decodes=2,
        num_prefills=0,
        num_decode_tokens=5,
        causal=True,
    )
    out_nonunif = torch.empty_like(q_nonunif)
    impl.forward_fused_infer_attention(q_nonunif, q_nonunif[:, :2], q_nonunif[:, :2], nonunif_meta, out_nonunif)
    assert fia_v2_mock.call_count == 1
    assert fia_v2_mock.call_args.kwargs["input_layout"] == "BSND"
    assert fia_v2_mock.call_args.kwargs["sparse_mode"] == 0
    assert fia_v2_mock.call_args.kwargs["actual_seq_qlen"] == [1, 1, 1, 1, 1]
    assert fia_v2_mock.call_args.kwargs["actual_seq_kvlen"] == [7, 8, 9, 10, 5]
    assert fia_v2_mock.call_args.kwargs["atten_mask"] is None
    assert fia_v2_mock.call_args.kwargs["block_table"].shape == (5, 2)
    assert fusion_mock.call_count == 0
    assert not gather_calls
    torch.testing.assert_close(out_nonunif, torch.full_like(out_nonunif, 3.5))


def test_eager_mixed_chunked_prefill_full_mode_uses_pa_for_decode(monkeypatch):
    cfg = _mock_vllm_config(with_spec=True, cudagraph_mode=CUDAGraphMode.FULL)
    _patch_large_head_env(monkeypatch, cfg, kv_layout="NHD")
    impl = _new_large_head_impl(
        torch.zeros((8, 8, 2, 512), dtype=torch.bfloat16),
        torch.zeros((8, 8, 2, 512), dtype=torch.bfloat16),
    )
    assert impl.use_bnsd_kv_cache is False

    def fake_pa(*, out, **_kwargs):
        out.fill_(4.0)

    pa_mock = MagicMock(side_effect=fake_pa)
    fusion_mock = MagicMock(return_value=(torch.full((6, 8, 512), 5.0, dtype=torch.bfloat16), None))
    gather_spy = MagicMock(side_effect=_gather_paged_kv_to_dense)
    monkeypatch.setattr("torch_npu._npu_paged_attention", pa_mock, raising=False)
    monkeypatch.setattr("torch_npu.npu_fusion_attention", fusion_mock)
    monkeypatch.setattr("vllm_ascend.attention.large_head_paged_verify._gather_paged_kv_to_dense", gather_spy)

    q = torch.randn((14, 8, 512), dtype=torch.bfloat16)
    k = torch.randn((14, 2, 512), dtype=torch.bfloat16)
    v = torch.randn((14, 2, 512), dtype=torch.bfloat16)
    out = torch.empty_like(q)
    meta = AscendLargeHeadPagedMetadata(
        block_tables=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
        actual_seq_lengths_q=[4, 8, 14],
        seq_lens=torch.tensor([6, 8, 6], dtype=torch.int32),
        seq_lens_list=[6, 8, 6],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_decodes=2,
        num_prefills=1,
        num_decode_tokens=8,
        causal=True,
    )
    impl.forward_fused_infer_attention(q, k, v, meta, out)
    assert pa_mock.call_count == 1
    assert pa_mock.call_args.kwargs["block_table"].shape == (8, 2)
    assert pa_mock.call_args.kwargs["context_lens"].tolist() == [3, 4, 5, 6, 5, 6, 7, 8]
    assert fusion_mock.call_count == 1
    assert fusion_mock.call_args.kwargs["actual_seq_qlen"] == [6]
    assert fusion_mock.call_args.kwargs["actual_seq_kvlen"] == [6]
    assert gather_spy.call_count == 0
    torch.testing.assert_close(out[:8], torch.full_like(out[:8], 4.0))
    torch.testing.assert_close(out[8:14], torch.full_like(out[8:14], 5.0))
