# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
        model_config=SimpleNamespace(
            max_model_len=4096,
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


@pytest.mark.parametrize("block_size", [128, 384, 2304])
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
    uses_storage_pages = a5 or block_size <= 1024
    expected_table = (
        torch.tensor([[7, 2], [5, -1]], dtype=torch.int32) if uses_storage_pages else common.block_table_tensor
    )
    torch.testing.assert_close(first.block_table, expected_table)
    assert first.block_size == (block_size if uses_storage_pages else 128)
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
    page_multiplier = 1 if uses_storage_pages else builder.nope_states[None].split
    assert first.block_table[0, 0] == 3 * page_multiplier
    assert draft.block_table[0, 0] == 7 * page_multiplier


@pytest.mark.parametrize("block_size,page_padding_bytes", [(384, 95232), (2304, 0)])
def test_a3_sparse_mla_preserves_storage_addresses(monkeypatch, block_size, page_padding_bytes):
    builder = _builder(block_size, False, monkeypatch)
    metadata = builder.build(0, _common(block_size))
    cache = torch.arange(8 * block_size * 8, dtype=torch.float32).reshape(8, block_size, 1, 8)
    if page_padding_bytes:
        backing = torch.empty(8, block_size * 8 + page_padding_bytes // cache.element_size())
        padded_cache = backing[:, : block_size * 8].view_as(cache)
        padded_cache.copy_(cache)
        cache = padded_cache
        assert not cache.is_contiguous()
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
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        block_size=128,
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
    )

    def build(**kwargs):
        assert isinstance(kwargs["max_seqlen_q"], int) and isinstance(kwargs["max_seqlen_ori_kv"], int)
        assert kwargs["has_ori_kv"] and not kwargs["has_cmp_kv"]
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
        assert kwargs["sinks"] is None
        assert kwargs.get("cmp_kv") is None
        assert kwargs["metadata"] is buffer
        assert kwargs["layout_kv"] == "PA_BBND" and kwargs["topk_value_mode"] == 1
        torch.testing.assert_close(
            kwargs["ori_sparse_indices"], torch.tensor([[[0, 6, 7, -1]], [[0, 1, 2, -1]]], dtype=torch.int32)
        )
        torch.testing.assert_close(kwargs["ori_topk_length"], metadata.smla_topk_length)
        return q + 1, torch.empty(0)

    monkeypatch.setattr(sparse_mla, "sparse_flash_mla", smla)
    torch.testing.assert_close(sparse_mla.sparse_mla(query, cache, indices, metadata, 0.5), query + 1)
