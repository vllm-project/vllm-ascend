# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder

import vllm_ascend.attention.sfa_v1 as sfa
import vllm_ascend.attention.sfa_v1 as sparse_mla
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily


def _builder(block_size, a5, monkeypatch, rope_dim=0):
    indexer = SimpleNamespace(
        topk_output_width=17,
        get_topk_lengths=lambda positions: torch.where(positions == 0, 1, 7),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(
            max_model_len=max(4096, block_size + 2),
            get_head_size=lambda: 512,
            hf_text_config=SimpleNamespace(num_attention_heads=4, kv_lora_rank=512),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=4),
        speculative_config=None,
        compilation_config=SimpleNamespace(
            static_forward_context={
                "layer": SimpleNamespace(qk_rope_head_dim=rope_dim, impl=SimpleNamespace(indexer=indexer))
            }
        ),
    )
    monkeypatch.setattr(
        sparse_mla,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED if a5 else None),
    )
    monkeypatch.setattr(sfa, "get_ascend_config", lambda: SimpleNamespace(c8_reshape_optim_enabled=False))
    monkeypatch.setattr(sfa, "select_common_block_size", lambda *args: 128)
    monkeypatch.setattr(
        sfa, "AttentionMaskBuilder", lambda device: SimpleNamespace(get_attention_mask=lambda *args: None)
    )
    monkeypatch.setattr(
        sfa, "get_cos_and_sin_mla", lambda *args, **kwargs: pytest.fail("NoPE must not build rotary tables")
    )

    def base_init(self, spec, names, cfg, device, metadata_cls, supports_dcp):
        self.kv_cache_spec, self.vllm_config, self.device = spec, cfg, device
        self.model_config, self.metadata_cls = cfg.model_config, metadata_cls

    with patch.object(MLACommonMetadataBuilder, "__init__", base_init):
        return sfa.AscendSFAMetadataBuilder(
            SimpleNamespace(block_size=block_size), ["layer"], config, torch.device("cpu")
        )


def _common(block_size):
    split = block_size // 128
    pages = torch.tensor([[7, 2], [5, -1]], dtype=torch.int32)
    expanded = (pages.unsqueeze(-1) * split + torch.arange(split, dtype=torch.int32)).reshape(2, -1)
    expanded[1, split:] = -1
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        num_input_tokens=4,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([block_size + 2, 1], dtype=torch.int32),
        _seq_lens_cpu=None,
        seq_lens_cpu=None,
        slot_mapping=torch.tensor([2 * block_size, 2 * block_size + 1, 5 * block_size, -1]),
        positions=torch.tensor([block_size, block_size + 1, 0, 0]),
        block_table_tensor=expanded,
        max_query_len=2,
        max_seq_len=block_size + 2,
        group_len=None,
        group_key_idx=None,
        group_key_cache_idx=None,
        attn_state=sfa.AscendAttentionState.ChunkedPrefill,
        causal=True,
    )


@pytest.mark.parametrize("block_size", [128, 384, 640, 2304, 4352])
@pytest.mark.parametrize("a5", [False, True])
def test_shared_sfa_nope_metadata_pages_lengths_and_draft_buffers(monkeypatch, block_size, a5):
    builder = _builder(block_size, a5, monkeypatch)
    seen = []

    def plan(**kwargs):
        seen.append(kwargs["ori_topk_length"].clone())
        return torch.full((1024,), len(seen), dtype=torch.int32)

    monkeypatch.setattr(sparse_mla, "sparse_flash_mla_metadata", plan)
    common = _common(block_size)
    first = builder.build(0, common)
    assert type(first) is sfa.AscendSFAMetadata
    assert first.cos is None and first.sin is None and first.seq_lens_cpu is None
    assert first.num_prefills == 1 and first.num_decode_tokens == 1
    torch.testing.assert_close(first.block_table, common.block_table_tensor)
    assert first.block_size == 128
    address = first.block_table.data_ptr()
    second = builder.build(0, common)
    assert second.block_table.data_ptr() == address
    draft = builder.build_for_drafting(common, 1)
    assert draft.block_table.data_ptr() != address
    if a5:
        torch.testing.assert_close(
            seen[0],
            torch.tensor([[7], [7], [1], [0]], dtype=torch.int32),
        )
        assert second.smla_metadata.data_ptr() == first.smla_metadata.data_ptr()
        assert draft.smla_metadata.data_ptr() != first.smla_metadata.data_ptr()
        assert draft.smla_topk_length.data_ptr() != first.smla_topk_length.data_ptr()
    else:
        assert first.smla_metadata is None and not seen
    common.block_table_tensor[0, : builder.nope_states[None].split] = 3 * builder.nope_states[
        None
    ].split + torch.arange(builder.nope_states[None].split)
    builder.build(0, common)
    page_multiplier = builder.nope_states[None].split
    assert first.block_table[0, 0] == 3 * page_multiplier
    assert draft.block_table[0, 0] == 7 * page_multiplier


def test_a3_sparse_mla_preserves_storage_addresses(monkeypatch):
    block_size = 2304
    builder = _builder(block_size, False, monkeypatch)
    metadata = builder.build(0, _common(block_size))
    cache = torch.arange(8 * block_size * 8, dtype=torch.float32).reshape(8, block_size, 1, 8)
    query = torch.ones(3, 2, 8)
    indices = torch.tensor(
        [
            [[0, 127, 128, block_size - 1, block_size]],
            [[0, 128, block_size - 1, block_size, block_size + 1]],
            [[0, -1, -1, -1, -1]],
        ]
    )
    pages = torch.tensor([[7, 2], [5, -1]])

    def op(**kwargs):
        viewed = kwargs["key"]
        assert viewed is kwargs["value"]
        kernel_block_size = 128 if block_size > 1024 else block_size
        assert viewed.shape == (8 * (block_size // kernel_block_size), kernel_block_size, 1, 8)
        assert viewed.data_ptr() == cache.data_ptr()
        for token, request in enumerate([0, 0, 1]):
            positions = indices[token, 0]
            positions = positions[positions >= 0]
            kernel_pages = kwargs["block_table"][request, positions // kernel_block_size]
            storage_pages = pages[request, positions // block_size]
            actual = viewed[kernel_pages, positions % kernel_block_size]
            expected = cache[storage_pages, positions % block_size]
            torch.testing.assert_close(actual, expected)
        return (query,)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention", op, raising=False)
    torch.testing.assert_close(sparse_mla.sparse_mla(query, cache, indices.int(), metadata, 0.5), query)


def test_a5_sparse_mla_splits_oversized_storage_pages(monkeypatch):
    block_size = 4352
    monkeypatch.setattr(
        sparse_mla,
        "sparse_flash_mla_metadata",
        lambda **kwargs: torch.zeros(sparse_mla.SMLA_METADATA_SIZE, dtype=torch.int32),
    )
    builder = _builder(block_size, True, monkeypatch)
    metadata = builder.build(0, _common(block_size))
    cache = torch.arange(8 * block_size * 8, dtype=torch.float32).reshape(8, block_size, 1, 8)
    query = torch.ones(3, 2, 8)
    indices = torch.tensor(
        [
            [[0, 127, 128, block_size - 1, block_size]],
            [[0, 128, block_size - 1, block_size, block_size + 1]],
            [[0, -1, -1, -1, -1]],
        ],
        dtype=torch.int32,
    )
    pages = torch.tensor([[7, 2], [5, -1]])

    def op(q, **kwargs):
        viewed = kwargs["ori_kv"]
        assert viewed.shape == (8 * (block_size // 128), 128, 1, 8)
        assert viewed.data_ptr() == cache.data_ptr()
        assert kwargs["ori_block_table"] is metadata.block_table
        for token, request in enumerate([0, 0, 1]):
            positions = indices[token, 0]
            positions = positions[positions >= 0]
            operator_pages = kwargs["ori_block_table"][request, positions // 128]
            storage_pages = pages[request, positions // block_size]
            actual = viewed[operator_pages, positions % 128]
            expected = cache[storage_pages, positions % block_size]
            torch.testing.assert_close(actual, expected)
        return (q,)

    monkeypatch.setattr(sparse_mla, "sparse_flash_mla", op)
    assert metadata.block_size == 128
    torch.testing.assert_close(sparse_mla.sparse_mla(query, cache, indices, metadata, 0.5), query)


def test_sparse_mla_rejects_incompatible_storage_page_size():
    cache = torch.empty(1, 384, 1, 8)
    with pytest.raises(ValueError, match="384 is not divisible by operator block size 256"):
        sparse_mla._view_cache_as_operator_pages(cache, 256)


def test_sparse_mla_rejects_noncontiguous_oversized_storage_pages():
    backing = torch.empty(2, 2304 * 8 + 1)
    cache = backing[:, : 2304 * 8].view(2, 2304, 1, 8)
    assert not cache.is_contiguous()
    with pytest.raises(ValueError, match="must support a zero-copy operator-page view"):
        sparse_mla._view_cache_as_operator_pages(cache, 128)


def test_rope_sfa_keeps_rotary_tables_and_kernel_pages(monkeypatch):
    builder = _builder(384, False, monkeypatch, rope_dim=64)
    common = _common(384)
    common.seq_lens_cpu = common.seq_lens.clone()
    cos, sin = torch.randn(4, 64), torch.randn(4, 64)
    with patch.object(sfa, "get_cos_and_sin_mla", return_value=(cos, sin)) as rotary:
        metadata = builder.build(0, common)
    rotary.assert_called_once()
    torch.testing.assert_close(rotary.call_args.args[0], common.positions)
    assert rotary.call_args.kwargs == {"use_cache": True}
    assert metadata.cos.data_ptr() == cos.data_ptr()
    assert metadata.sin.data_ptr() == sin.data_ptr()
    assert metadata.block_table.data_ptr() == common.block_table_tensor.data_ptr()
    assert not builder.nope_states and metadata.smla_metadata is None


@pytest.mark.parametrize("sfa_c8,li_c8", [(False, False), (True, False), (False, True), (True, True)])
def test_rope_sfa_preserves_cache_composition_and_device_dispatch(sfa_c8, li_c8):
    impl = object.__new__(sfa.AscendSFAImpl)
    impl.qk_rope_head_dim = 64
    impl.layer_name = "model.layers.0.self_attn"
    impl.has_indexer = True
    impl.enable_sparse_sfa_c8, impl.enable_sparse_li_c8 = sfa_c8, li_c8
    main = tuple(torch.empty(1) for _ in range(1 if sfa_c8 else 2))
    indexer = tuple(torch.empty(1) for _ in range(2 if li_c8 else 1))
    impl.indexer = SimpleNamespace(k_cache=SimpleNamespace(kv_cache=indexer), num_cache_tensors=len(indexer))
    composed = impl._compose_sfa_kv_cache(main)
    assert all(actual is expected for actual, expected in zip(composed, (*main, *indexer), strict=True))
    inputs = [object() for _ in range(6)]
    q, rope, indices, metadata, query_lens, seq_lens = inputs
    with patch.object(sfa.DeviceOperator, "execute_sparse_flash_attention_process") as dispatch:
        result = impl._execute_sparse_flash_attention_process(
            q, rope, composed, indices, metadata, query_lens, seq_lens
        )
    assert result is dispatch.return_value
    dispatch.assert_called_once_with(impl, q, rope, composed, indices, metadata, query_lens, seq_lens, block_table=None)


@pytest.mark.parametrize("a5", [False, True])
def test_nope_operator_masks_unwritten_graph_rows(monkeypatch, a5):
    query = torch.ones(3, 2, 128)
    cache = torch.zeros(2, 128, 1, 128)
    metadata = SimpleNamespace(
        smla_metadata=torch.empty(1024, dtype=torch.int32) if a5 else None,
        smla_topk_length=torch.tensor([[1], [1], [0]], dtype=torch.int32),
        smla_sinks=torch.ones(2, dtype=torch.float32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        block_size=128,
        # Row 2 is graph padding. query_start_loc's terminator was extended to
        # cover it, so only num_actual_tokens tells it apart from a real row.
        num_actual_tokens=2,
    )

    def op(*args, **kwargs):
        result = query.clone()
        result[2] = float("nan")
        return (result,)

    if a5:
        monkeypatch.setattr(sparse_mla, "sparse_flash_mla", op)
    else:
        monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention", op, raising=False)
    output = sparse_mla.sparse_mla(query, cache, torch.tensor([[[0]], [[0]], [[-1]]], dtype=torch.int32), metadata, 0.5)
    torch.testing.assert_close(output[:2], query[:2])
    assert (output[2] == 0).all()


def test_a5_smla_uses_original_cache_sorted_indices_and_stable_metadata(monkeypatch):
    buffer = torch.empty(sparse_mla.SMLA_METADATA_SIZE, dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([8, 4], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=8,
        block_table=torch.tensor([[2], [1]], dtype=torch.int32),
        smla_metadata=None,
        smla_topk_length=torch.full((2, 1), 3, dtype=torch.int32),
        block_size=8,
        smla_sinks=torch.ones(2, dtype=torch.float32),
        num_actual_tokens=2,
    )

    def build(**kwargs):
        assert isinstance(kwargs["max_seqlen_q"], int) and isinstance(kwargs["max_seqlen_ori_kv"], int)
        assert kwargs["has_ori_kv"] and not kwargs["has_cmp_kv"]
        # The sparse ori_kv scenario is mask mode 0; anything else makes the
        # kernel skip its softmax initialisation on ordinary decode steps.
        assert kwargs["ori_mask_mode"] == 0 and kwargs["cmp_mask_mode"] == 0
        assert kwargs["ori_win_left"] == -1 and kwargs["ori_win_right"] == -1
        assert kwargs["ori_topk_length"] is metadata.smla_topk_length
        return torch.full_like(buffer, metadata.max_seq_len)

    monkeypatch.setattr(sparse_mla, "sparse_flash_mla_metadata", build)
    sparse_mla.build_smla_metadata(metadata, buffer, 2, 128, 7)
    address = metadata.smla_metadata.data_ptr()
    metadata.max_seq_len = 12
    sparse_mla.build_smla_metadata(metadata, buffer, 2, 128, 7)
    assert metadata.smla_metadata.data_ptr() == address
    assert (metadata.smla_metadata == 12).all()
    query = torch.ones(2, 2, 128, dtype=torch.bfloat16)
    cache = torch.zeros(3, 8, 1, 128, dtype=torch.bfloat16)
    indices = torch.tensor([[[7, 6, -1, 0]], [[2, -1, 1, 0]]], dtype=torch.int32)

    def smla(q, **kwargs):
        assert kwargs["ori_kv"] is cache
        assert kwargs["sinks"] is metadata.smla_sinks
        assert kwargs.get("cmp_kv") is None
        assert kwargs["metadata"] is buffer
        assert kwargs["layout_kv"] == "PA_BBND" and kwargs["topk_value_mode"] == 1
        assert kwargs["ori_mask_mode"] == 0 and kwargs["cmp_mask_mode"] == 0
        assert kwargs["ori_win_left"] == -1 and kwargs["ori_win_right"] == -1
        torch.testing.assert_close(
            kwargs["ori_sparse_indices"], torch.tensor([[[0, 6, 7, -1]], [[0, 1, 2, -1]]], dtype=torch.int32)
        )
        torch.testing.assert_close(kwargs["ori_topk_length"], metadata.smla_topk_length)
        assert kwargs["cu_seqlens_q"] is metadata.query_start_loc
        return q + 1, torch.empty(0)

    monkeypatch.setattr(sparse_mla, "sparse_flash_mla", smla)
    torch.testing.assert_close(sparse_mla.sparse_mla(query, cache, indices, metadata, 0.5), query + 1)


def test_a5_smla_rebuilds_plan_for_a_trimmed_query(monkeypatch):
    """Eager and piecewise steps hand the operator only the unpadded rows.

    The plan written during metadata construction still describes the padded
    batch, so the call has to regenerate it - and take the top-k lengths from
    the indices it is actually passing rather than from the earlier prediction.
    """
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        seq_lens=torch.tensor([8, 0], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=8,
        block_table=torch.tensor([[2], [0]], dtype=torch.int32),
        smla_metadata=torch.zeros(sparse_mla.SMLA_METADATA_SIZE, dtype=torch.int32),
        # Predicted for the padded batch: four rows, three entries each.
        smla_topk_length=torch.full((4, 1), 3, dtype=torch.int32),
        smla_sinks=torch.ones(2, dtype=torch.float32),
        block_size=8,
        num_actual_tokens=1,
    )
    regenerated = torch.full((sparse_mla.SMLA_METADATA_SIZE,), 5, dtype=torch.int32)

    def plan(**kwargs):
        # Regenerated over the rows being passed, not the padded ones.
        torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor([0, 1, 1], dtype=torch.int32))
        torch.testing.assert_close(kwargs["ori_topk_length"], torch.tensor([[2]], dtype=torch.int32))
        assert kwargs["ori_mask_mode"] == 0
        return regenerated

    def smla(q, **kwargs):
        assert kwargs["metadata"] is regenerated
        assert kwargs["metadata"] is not metadata.smla_metadata
        torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor([0, 1, 1], dtype=torch.int32))
        # Two non-negative entries survive the sort; the prediction said three.
        torch.testing.assert_close(kwargs["ori_topk_length"], torch.tensor([[2]], dtype=torch.int32))
        return q + 1, torch.empty(0)

    monkeypatch.setattr(sparse_mla, "sparse_flash_mla_metadata", plan)
    monkeypatch.setattr(sparse_mla, "sparse_flash_mla", smla)
    query = torch.ones(1, 2, 8, dtype=torch.bfloat16)
    cache = torch.zeros(3, 8, 1, 8, dtype=torch.bfloat16)
    indices = torch.tensor([[[5, -1, 1, -1]]], dtype=torch.int32)
    torch.testing.assert_close(sparse_mla.sparse_mla(query, cache, indices, metadata, 0.5), query + 1)


def test_build_smla_metadata_rejects_a_plan_of_the_wrong_size(monkeypatch):
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=4,
        smla_topk_length=torch.ones(1, 1, dtype=torch.int32),
    )
    monkeypatch.setattr(sparse_mla, "sparse_flash_mla_metadata", lambda **_: torch.zeros(8, dtype=torch.int32))
    buffer = torch.empty(sparse_mla.SMLA_METADATA_SIZE, dtype=torch.int32)
    with pytest.raises(ValueError, match="int32 values"):
        sparse_mla.build_smla_metadata(metadata, buffer, 2, 128, 7)


def _sinks_owner(builder):
    """The SparseMLAMetadataState that owns the persistent sinks."""
    return builder.nope_states[None]


def _stub_metadata_op(monkeypatch):
    monkeypatch.setattr(
        sparse_mla,
        "sparse_flash_mla_metadata",
        lambda **kwargs: torch.zeros(sparse_mla.SMLA_METADATA_SIZE, dtype=torch.int32),
    )


@pytest.mark.parametrize("block_size", [128, 2304])
def test_a5_smla_sinks_are_builder_owned_and_stable(monkeypatch, block_size):
    """The sinks tensor is allocated once by the builder state and reused
    verbatim by every metadata build, so ACL Graph replay always finds it at
    the address captured on the first run."""
    _stub_metadata_op(monkeypatch)
    builder = _builder(block_size, True, monkeypatch)

    first = builder.build(0, _common(block_size))
    second = builder.build(0, _common(block_size))
    state = _sinks_owner(builder)

    assert first.smla_sinks is state.sinks
    assert second.smla_sinks is state.sinks
    assert first.smla_sinks.data_ptr() == second.smla_sinks.data_ptr()
    assert first.smla_sinks.shape == (state.num_heads,)
    assert first.smla_sinks.dtype == torch.float32
    torch.testing.assert_close(
        first.smla_sinks,
        torch.full((state.num_heads,), sparse_mla.SMLA_DEFAULT_SINK_VALUE, dtype=torch.float32),
    )


def test_a5_smla_passes_persistent_sinks_to_operator(monkeypatch):
    """sparse_mla() forwards the builder-owned tensor itself, not a copy."""
    _stub_metadata_op(monkeypatch)
    builder = _builder(128, True, monkeypatch)
    metadata = builder.build(0, _common(128))
    state = _sinks_owner(builder)
    seen = {}

    def op(q, **kwargs):
        seen["sinks"] = kwargs["sinks"]
        return (q,)

    monkeypatch.setattr(sparse_mla, "sparse_flash_mla", op)
    query = torch.ones(3, state.num_heads, 8)
    sparse_mla.sparse_mla(query, torch.zeros(8, 128, 1, 8), torch.zeros(3, 1, 4, dtype=torch.int32), metadata, 0.5)

    assert seen["sinks"] is metadata.smla_sinks
    assert seen["sinks"] is state.sinks


def test_a5_smla_sinks_outlive_the_operator_call(monkeypatch):
    """Regression for the 507011 kernel trap: a sinks tensor referenced only by
    the operator call's kwargs is freed as soon as the call returns, while the
    captured graph still holds its address."""
    _stub_metadata_op(monkeypatch)
    builder = _builder(128, True, monkeypatch)
    metadata = builder.build(0, _common(128))
    state = _sinks_owner(builder)
    monkeypatch.setattr(sparse_mla, "sparse_flash_mla", lambda q, **kwargs: (q,))
    query = torch.ones(3, state.num_heads, 8)
    sparse_mla.sparse_mla(query, torch.zeros(8, 128, 1, 8), torch.zeros(3, 1, 4, dtype=torch.int32), metadata, 0.5)

    ref = weakref.ref(metadata.smla_sinks)
    del metadata
    gc.collect()
    assert ref() is not None, "sinks must outlive the metadata that exposes it"
    assert ref() is state.sinks


def _smla_metadata_stub(sinks):
    return SimpleNamespace(
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        smla_metadata=torch.zeros(sparse_mla.SMLA_METADATA_SIZE, dtype=torch.int32),
        smla_topk_length=torch.ones(1, 1, dtype=torch.int32),
        smla_sinks=sinks,
        block_size=8,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
    )


def test_sparse_mla_rejects_missing_sinks():
    with pytest.raises(RuntimeError, match="persistent sinks"):
        sparse_mla.sparse_mla(
            torch.ones(1, 2, 8),
            torch.zeros(1, 8, 1, 8),
            torch.zeros(1, 1, 2, dtype=torch.int32),
            _smla_metadata_stub(None),
            0.5,
        )
