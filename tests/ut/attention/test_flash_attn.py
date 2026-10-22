# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.compilation.backends import set_model_tag
from vllm.v1.attention.selector import AttentionSelectorConfig

from vllm_ascend.attention.fa_v1 import AscendFlashAttentionImpl


@pytest.fixture
def dflash2_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(runner_type="generate"),
        speculative_config=SimpleNamespace(
            method="dflash",
            target_model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")),
            draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
            draft_sample_method="greedy",
            enforce_eager=True,
        ),
    )


def make_impl(query, *, num_kv_heads=1, window=None, layout="PA_BBND", sinks=None):
    impl = AscendFlashAttentionImpl.__new__(AscendFlashAttentionImpl)
    impl.num_heads, impl.num_kv_heads = query.shape[1], num_kv_heads
    impl.scale, impl.sliding_window = query.shape[-1] ** -0.5, window
    impl.use_bnsd_kv_cache, impl.sinks = layout == "PA_BNBD", sinks
    impl.key_cache = impl.value_cache = None
    return impl


@pytest.fixture
def select_flash_attn(monkeypatch):
    from vllm_ascend import platform
    from vllm_ascend.device.hardware_profile import AttentionBackendFamily

    profile = SimpleNamespace(supports=lambda _: True, attention_backend_family=AttentionBackendFamily.STANDARD)
    monkeypatch.setattr(platform, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(platform, "_validate_fa3_backend", lambda *_: False)

    def select(config, selector):
        monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
        with set_model_tag("dflash_head"):
            backend = platform.NPUPlatform.get_attn_backend_cls(None, selector)
        return backend == "vllm_ascend.attention.fa_v1.AscendFlashAttentionBackend"

    return select, profile


def test_flash_attn_is_selected_only_for_a5_dflash2_eager_drafts(select_flash_attn, dflash2_config):
    select, profile = select_flash_attn
    config = dflash2_config
    selector = AttentionSelectorConfig(head_size=128, block_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto")
    assert select(config, selector)
    profile.supports = lambda _: False
    assert not select(config, selector)
    profile.supports = lambda _: True
    config.speculative_config = None
    config.model_config = None
    assert not select(config, selector)
    assert not select(None, selector)


@pytest.mark.parametrize("tag", ["backbone", "eagle_head", "dflash2_candidate_selector"])
def test_flash_attn_requires_live_dflash_loader_tag(monkeypatch, select_flash_attn, dflash2_config, tag):
    from vllm.compilation import backends

    from vllm_ascend import platform

    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: dflash2_config)
    monkeypatch.setattr(backends, "model_tag", tag)
    selector = AttentionSelectorConfig(head_size=128, block_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto")
    assert not platform._validate_flash_attn_backend((False, False, False), selector)


@pytest.mark.parametrize("sampling", ["greedy", "probabilistic"])
def test_flash_attn_selection_is_independent_of_draft_sampling(select_flash_attn, dflash2_config, sampling):
    select, _ = select_flash_attn
    dflash2_config.speculative_config.draft_sample_method = sampling
    selector = AttentionSelectorConfig(head_size=128, block_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto")
    assert select(dflash2_config, selector)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "eagle3"),
        ("method", "mtp"),
        ("enforce_eager", False),
        ("enforce_eager", None),
        ("architectures", ["DFlashDraftModel"]),
        ("architectures", None),
    ],
)
def test_other_drafts_and_graph_mode_retain_legacy_backend(select_flash_attn, dflash2_config, field, value):
    select, _ = select_flash_attn
    spec = dflash2_config.speculative_config
    setattr(spec.draft_model_config if field == "architectures" else spec, field, value)
    selector = AttentionSelectorConfig(head_size=128, block_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto")
    assert not select(dflash2_config, selector)


@pytest.mark.parametrize("model_type", ["qwen3", "deepseek_v32", "glm5_next"])
def test_other_target_architectures_retain_legacy_backend(select_flash_attn, dflash2_config, model_type):
    select, _ = select_flash_attn
    dflash2_config.speculative_config.target_model_config.hf_text_config.model_type = model_type
    selector = AttentionSelectorConfig(head_size=128, block_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto")
    assert not select(dflash2_config, selector)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_mla", True),
        ("use_sparse", True),
        ("use_pcp", True),
        ("use_dcp", True),
        ("use_compress", True),
        ("use_mm_prefix", True),
        ("use_rswa", True),
        ("use_batch_invariant", True),
        ("use_per_head_quant_scales", True),
        ("dtype", torch.float32),
        ("kv_cache_dtype", "fp8"),
        ("kv_cache_dtype", "int8"),
    ],
)
def test_unsupported_gqa_features_retain_legacy_backend(select_flash_attn, dflash2_config, field, value):
    select, _ = select_flash_attn
    selector = AttentionSelectorConfig(head_size=128, block_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto")
    if field == "use_compress":
        selector = SimpleNamespace(**{**selector._asdict(), "use_mla": True, field: value})
    else:
        selector = selector._replace(**{field: value})
        if field == "use_sparse":
            selector = selector._replace(use_mla=True)
    assert not select(dflash2_config, selector)


@pytest.mark.parametrize(
    "window,causal,mask_mode,win_right",
    [(None, False, 0, -1), (None, True, 3, -1), (2048, False, 4, 2047), (2048, True, 4, 0)],
)
def test_flash_attn_reuses_plans_within_each_draft_step(monkeypatch, window, causal, mask_mode, win_right):
    from vllm_ascend.attention.fa_v1 import AscendFlashAttentionMetadata

    plan = torch.empty(16, dtype=torch.int32)
    query = torch.zeros(12, 8, 128, dtype=torch.bfloat16)
    sinks = torch.zeros(8)
    impl = make_impl(query, window=window, sinks=sinks)
    output = torch.empty_like(query)
    ops = SimpleNamespace(flash_attn_metadata=Mock(return_value=plan), flash_attn=Mock(return_value=(query, None)))
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", ops)
    metadata_args = dict(
        query_start_loc_gpu=torch.tensor([0, 6, 12, -1], dtype=torch.int32),
        block_tables=torch.tensor([[1, 0], [3, 2], [0, 0]], dtype=torch.int32),
        max_query_len=6,
        causal=causal,
        flash_attn_mask=torch.ones(2, 2, dtype=torch.int8),
    )
    for step, lengths in enumerate(([256, 254, 0], [253, 251, 0]), start=1):
        metadata = AscendFlashAttentionMetadata(seq_lens_gpu=torch.tensor(lengths, dtype=torch.int32), **metadata_args)
        for _ in range(6):
            impl.forward_impl(query, None, None, None, metadata, output)
        assert ops.flash_attn_metadata.call_count == step
        assert ops.flash_attn.call_count == 6 * step
        meta_args = ops.flash_attn_metadata.call_args.kwargs
        args = ops.flash_attn.call_args.kwargs
        assert args["metadata"] is plan
        assert args["sinks"] is sinks
        assert args["attn_mask"] is (metadata.flash_attn_mask if mask_mode else None)
        torch.testing.assert_close(meta_args["seqused_kv"], torch.tensor(lengths, dtype=torch.int32).clamp(min=1))
        torch.testing.assert_close(meta_args["cu_seqlens_q"], torch.tensor([0, 6, 12, 12], dtype=torch.int32))
        assert meta_args["mask_mode"] == mask_mode
        assert meta_args["win_left"] == (2047 if window else -1)
        assert meta_args["win_right"] == win_right


@pytest.mark.parametrize("cpu_mirrors", [False, True])
def test_metadata_builder_does_not_materialize_exact_device_lengths(monkeypatch, cpu_mirrors):
    from vllm_ascend.attention import fa_v1 as module
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    builder = module.AscendFlashAttentionMetadataBuilder.__new__(module.AscendFlashAttentionMetadataBuilder)
    builder.model_config = SimpleNamespace(runner_type="draft")
    builder.attn_mask_builder = SimpleNamespace(
        get_splitfuse_attn_mask=Mock(return_value=torch.ones(2, 2)),
    )
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=12,
        query_start_loc_cpu=torch.tensor([0, 999, 1998], dtype=torch.int32) if cpu_mirrors else None,
        query_start_loc=torch.tensor([0, 6, 12], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 2, dtype=torch.int32),
        seq_lens=torch.tensor([256, 254], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([999, 999], dtype=torch.int32) if cpu_mirrors else None,
        seq_lens_cpu=torch.tensor([888, 888], dtype=torch.int32) if cpu_mirrors else None,
        max_seq_len=256,
        slot_mapping=torch.arange(12),
        causal=False,
        max_query_len=6,
        attn_mask=None,
        attn_state=AscendAttentionState.ChunkedPrefill,
    )

    def forbidden_host_materialization(*args, **kwargs):
        raise AssertionError("draft length metadata must not be materialized or staged on the host")

    with monkeypatch.context() as patches:
        for operation in ("tolist", "item", "cpu"):
            patches.setattr(torch.Tensor, operation, forbidden_host_materialization)
        result = builder.build_for_drafting(common, draft_index=0)
    assert result.seq_lens_gpu.data_ptr() == common.seq_lens.data_ptr()
    assert result.query_start_loc_gpu.data_ptr() == common.query_start_loc.data_ptr()
    assert not result.causal
    assert result.flash_attn_mask is builder.attn_mask_builder.get_splitfuse_attn_mask.return_value


def test_metadata_builder_slices_request_extent():
    from vllm_ascend.attention.fa_v1 import AscendFlashAttentionMetadataBuilder

    builder = AscendFlashAttentionMetadataBuilder.__new__(AscendFlashAttentionMetadataBuilder)
    builder.model_config = SimpleNamespace(runner_type="generate")
    builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: None)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=12,
        query_start_loc=torch.tensor([0, 6, 12, 999, 999], dtype=torch.int32),
        seq_lens=torch.tensor([100, 200, 999, 999], dtype=torch.int32),
        block_table_tensor=torch.zeros(4, 8, dtype=torch.int32),
        slot_mapping=torch.arange(24, dtype=torch.int32),
        max_query_len=6,
        causal=False,
        attn_state=None,
    )
    metadata = builder.build(0, common)
    assert metadata.block_tables.shape == (2, 8)
    assert metadata.slot_mapping.shape == (12,)
    torch.testing.assert_close(metadata.query_start_loc_gpu, torch.tensor([0, 6, 12], dtype=torch.int32))
    torch.testing.assert_close(metadata.seq_lens_gpu, torch.tensor([100, 200], dtype=torch.int32))
    assert metadata.seq_lens_gpu.data_ptr() == common.seq_lens.data_ptr()
    assert metadata.query_start_loc_gpu.data_ptr() == common.query_start_loc.data_ptr()


def test_distinct_attention_geometries_do_not_share_plans(monkeypatch):
    from vllm_ascend.attention.fa_v1 import AscendFlashAttentionMetadata

    ops = SimpleNamespace(
        flash_attn_metadata=Mock(return_value=torch.empty(16)),
        flash_attn=Mock(side_effect=lambda query, *args, **kwargs: (query, None)),
    )
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", ops)
    metadata = AscendFlashAttentionMetadata(
        query_start_loc_gpu=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens_gpu=torch.tensor([256], dtype=torch.int32),
        block_tables=torch.zeros(1, 2, dtype=torch.int32),
        max_query_len=6,
        causal=False,
        flash_attn_mask=torch.ones(2, 2),
    )
    geometries = [
        (8, 1, 128, None, "PA_BBND"),
        (16, 1, 128, None, "PA_BBND"),
        (8, 2, 128, None, "PA_BBND"),
        (8, 1, 64, None, "PA_BBND"),
        (8, 1, 128, 2048, "PA_BBND"),
        (8, 1, 128, None, "PA_BNBD"),
    ]
    for _ in range(2):
        for heads, kv_heads, dim, window, layout in geometries:
            query = torch.empty(6, heads, dim)
            impl = make_impl(query, num_kv_heads=kv_heads, window=window, layout=layout)
            impl.forward_impl(query, None, None, None, metadata, torch.empty_like(query))
    assert ops.flash_attn_metadata.call_count == len(geometries)
    assert ops.flash_attn.call_count == 2 * len(geometries)
