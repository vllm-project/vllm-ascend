# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
)

_KERNEL_BLOCK_SIZE = 128


def _make_builder(pcp_size: int = 1) -> AscendSFAIndexerMetadataBuilder:
    kv_cache_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=160,
        dtype=torch.uint8,
    )
    layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
    vllm_config = MagicMock()
    vllm_config.parallel_config.prefill_context_parallel_size = pcp_size
    with patch(
        "vllm_ascend.attention.indexer.select_common_block_size",
        return_value=_KERNEL_BLOCK_SIZE,
    ):
        return AscendSFAIndexerMetadataBuilder(
            kv_cache_spec,
            layer_names,
            vllm_config,
            torch.device("cpu"),
        )


def _make_common_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        num_input_tokens=4,
        slot_mapping=torch.tensor([1, 2, 3, 4, 5]),
        positions=torch.tensor([0, 1, 0, 1, 9]),
        query_start_loc=torch.tensor([0, 2, 4]),
        seq_lens=torch.tensor([5, 6, 7]),
        block_table_tensor=torch.arange(6).view(3, 2),
        group_len=MagicMock(name="group_len"),
        group_key_idx=MagicMock(name="group_key_idx"),
        group_key_cache_idx=MagicMock(name="group_key_cache_idx"),
    )


def test_sfa_indexer_backend_contract():
    assert AscendSFAIndexerBackend.accept_output_buffer
    assert AscendSFAIndexerBackend.get_name() == "ASCEND_SFA_INDEXER"
    assert AscendSFAIndexerBackend.get_builder_cls() is AscendSFAIndexerMetadataBuilder
    assert AscendSFAIndexerBackend.get_kv_cache_shape(8, 128, 1, 160) == (
        8,
        128,
        1,
        160,
    )
    assert AscendSFAIndexerBackend.get_supported_kernel_block_sizes() == [128]


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_builds_kernel_metadata(mock_cos_sin, mock_get_ascend_config):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    cos = torch.zeros(5, 1, 1, 8)
    sin = torch.zeros(5, 1, 1, 8)
    mock_cos_sin.return_value = (cos, sin)

    builder = _make_builder()
    assert builder.reorder_batch_threshold is None
    assert builder.get_cudagraph_support(MagicMock(), MagicMock()) is AttentionCGSupport.UNIFORM_BATCH

    common = _make_common_metadata()
    metadata = builder.build(0, common)

    assert isinstance(metadata, AscendSFAIndexerMetadata)
    assert metadata.num_actual_tokens == 4
    assert torch.equal(metadata.slot_mapping, common.slot_mapping[:4])
    assert torch.equal(metadata.seq_lens, common.seq_lens[:2])
    assert torch.equal(metadata.cum_query_lens, common.query_start_loc[1:3])
    assert torch.equal(metadata.block_table, common.block_table_tensor[:2])
    assert metadata.block_size == _KERNEL_BLOCK_SIZE
    assert metadata.group_len is common.group_len
    assert metadata.group_key_idx is common.group_key_idx
    assert metadata.group_key_cache_idx is common.group_key_cache_idx

    positions = mock_cos_sin.call_args.args[0]
    assert torch.equal(positions, common.positions[:4])
    assert mock_cos_sin.call_args.kwargs["use_cache"] is True
    assert torch.equal(metadata.cos, cos[:4])
    assert torch.equal(metadata.sin, sin[:4])


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_emits_full_slot_mapping_under_pcp(mock_cos_sin, mock_get_ascend_config):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder(pcp_size=2)
    common = _make_common_metadata()
    metadata = builder.build(0, common)

    # Under PCP the commit writes the gathered prefill region too, so the
    # builder emits the full slot mapping instead of the input-token slice.
    assert metadata.slot_mapping is common.slot_mapping


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
@patch("vllm_ascend.attention.indexer.torch.ops._C_ascend.store_kv_block_metadata", create=True)
def test_sfa_indexer_metadata_builder_primes_reshape_optim(
    mock_store_kv_block_metadata,
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = True
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder()
    common = _make_common_metadata()
    metadata = builder.build(0, common)

    mock_store_kv_block_metadata.assert_called_once_with(
        metadata.slot_mapping,
        common.group_len,
        common.group_key_idx,
        common.group_key_cache_idx,
        _KERNEL_BLOCK_SIZE,
    )


@pytest.mark.parametrize("source", ["_seq_lens_cpu", "seq_lens_cpu", None])
def test_metadata_cpu_state_bridge_without_device_copy(source):
    common = _make_common_metadata()
    common.attn_state = AscendAttentionState.PrefillCacheHit
    common._seq_lens_cpu = None
    common.seq_lens_cpu = None
    cpu_lengths = torch.tensor([5, 6, 7])
    if source is not None:
        setattr(common, source, cpu_lengths)
    with (
        patch("vllm_ascend.attention.indexer.get_ascend_config") as config,
        patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla") as rope,
    ):
        config.return_value.c8_reshape_optim_enabled = False
        rope.return_value = (torch.zeros(5, 8), torch.zeros(5, 8))
        metadata = _make_builder().build(0, common)
    assert metadata.attn_state is common.attn_state
    if source is None:
        assert metadata.seq_lens_cpu is None
    else:
        assert metadata.seq_lens_cpu.tolist() == [5, 6]
        assert metadata.seq_lens_cpu.data_ptr() == cpu_lengths.data_ptr()


class _DeviceTable:
    """Mock only device placement; slice real CPU storage for host assertions."""

    def __init__(self, table, device):
        self.table = table
        self.device = device

    def __getitem__(self, key):
        return self.table[key]


@pytest.fixture
def full_visible_indexer():
    config = SimpleNamespace(enable_sfa_full_visible_index_bypass=True, enable_sparse_sfa_c8=False)
    with (
        patch("vllm_ascend.attention.indexer.get_ascend_config", return_value=config),
        patch.object(AscendSFAIndexerBackend, "_full_visible_index_tables", {}),
    ):
        impl = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
        impl.allow_short_prefill_indexer_scoring_skip = True
        impl.enable_sparse_li_c8 = False
        impl._pcp_active = False
        impl._dsa_cp_active = False
        impl._speculative_active = False
        impl.topk_tokens = 2048
        device = torch.device("npu", 0)
        table = impl._get_or_create_full_visible_index_table(torch.device("cpu"))
        impl._full_visible_index_table = _DeviceTable(table, device)
        yield impl, config, table, device


def _full_visible_metadata(total=192, query=64):
    return SimpleNamespace(
        attn_state=AscendAttentionState.PrefillCacheHit,
        seq_lens_cpu=torch.tensor([total]),
        seq_lens=torch.tensor([total]),
        cum_query_lens=torch.tensor([query]),
        num_actual_tokens=query,
        num_decode_tokens=0,
        block_size=128,
        block_table=torch.zeros(1, 16, dtype=torch.int32),
        slot_mapping=torch.arange(total - query, total),
        actual_seq_lengths_query=torch.tensor([query]),
        actual_seq_lengths_key=torch.tensor([total]),
    )


def test_exact_rr_table_and_shared_storage(full_visible_indexer):
    impl, _, table, _ = full_visible_indexer
    assert table.shape == (2049, 2048)
    assert table.dtype == torch.int32
    assert table.untyped_storage().nbytes() == 2049 * 2048 * 4
    assert torch.all(table[0] == -1)
    assert table is impl._get_or_create_full_visible_index_table(torch.device("cpu"))
    for visible in range(1, 2049):
        # Independent scalar block/offset oracle, including partial blocks.
        expected = [
            block + offset for offset in range(128) for block in range(0, visible, 128) if block + offset < visible
        ]
        row = table[visible].tolist()
        assert row[:visible] == expected
        assert set(row[:visible]) == set(range(visible))
        assert row[visible:] == [-1] * (2048 - visible)


@pytest.mark.parametrize("total,query", [(1, 1), (64, 64), (192, 192), (2048, 2048), (2048, 64)])
def test_eligible_views_alias_table(full_visible_indexer, total, query):
    impl, _, table, device = full_visible_indexer
    metadata = _full_visible_metadata(total, query)
    if total == query:
        metadata.attn_state = AscendAttentionState.PrefillNoCache
    before = table.clone()
    rows = impl._get_full_visible_topk_indices(metadata, query, device)
    assert rows.shape == (query, 1, 2048)
    assert rows.untyped_storage().data_ptr() == table.untyped_storage().data_ptr()
    assert rows.storage_offset() == (total - query + 1) * 2048
    destination = torch.empty(query, 2048, dtype=torch.int32)
    destination.copy_(rows.squeeze(1))
    assert destination.untyped_storage().data_ptr() != table.untyped_storage().data_ptr()
    assert torch.equal(table, before)
    assert torch.equal(destination, table[total - query + 1 : total + 1])


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("config", "enable_sfa_full_visible_index_bypass", False),
        ("config", "enable_sparse_sfa_c8", True),
        ("impl", "allow_short_prefill_indexer_scoring_skip", False),
        ("impl", "enable_sparse_li_c8", True),
        ("impl", "_pcp_active", True),
        ("impl", "_dsa_cp_active", True),
        ("impl", "_speculative_active", True),
        ("impl", "topk_tokens", 1024),
        ("impl", "_full_visible_index_table", None),
        ("metadata", "attn_state", AscendAttentionState.DecodeOnly),
        ("metadata", "attn_state", AscendAttentionState.SpecDecoding),
        ("metadata", "attn_state", AscendAttentionState.ChunkedPrefill),
        ("metadata", "attn_state", None),
        ("metadata", "attn_state", AscendAttentionState.PrefillNoCache),
        ("metadata", "num_decode_tokens", 1),
        ("metadata", "block_size", 64),
        ("metadata", "seq_lens_cpu", None),
        ("metadata", "seq_lens_cpu", torch.tensor([2049])),
        ("metadata", "seq_lens_cpu", torch.tensor([63])),
        ("metadata", "seq_lens_cpu", torch.tensor([192.5])),
        ("metadata", "seq_lens_cpu", torch.tensor([192, 192])),
        ("metadata", "seq_lens_cpu", torch.empty(1, device="meta", dtype=torch.int64)),
        ("metadata", "block_table", torch.zeros(2, 16)),
        ("metadata", "block_table", torch.zeros(1, 1)),
        ("metadata", "seq_lens", torch.tensor([192, 192])),
        ("metadata", "cum_query_lens", torch.tensor([32, 64])),
        ("metadata", "num_actual_tokens", 65),
    ],
)
def test_narrow_fallbacks(full_visible_indexer, target, field, value):
    impl, config, _, device = full_visible_indexer
    metadata = _full_visible_metadata()
    setattr({"impl": impl, "config": config, "metadata": metadata}[target], field, value)
    assert impl._get_full_visible_topk_indices(metadata, 64, device) is None


def test_unsupported_device_and_empty_query(full_visible_indexer):
    impl, _, _, device = full_visible_indexer
    metadata = _full_visible_metadata()
    assert impl._get_full_visible_topk_indices(metadata, 64, torch.device("cpu")) is None
    assert impl._get_full_visible_topk_indices(metadata, 64, torch.device("npu", 1)) is None
    metadata.num_actual_tokens = 0
    assert impl._get_full_visible_topk_indices(metadata, 0, device) is None


@pytest.mark.parametrize("enabled,compute_topk", [(True, True), (False, True), (True, False)])
def test_forward_cache_write_before_bypass_or_scorer(full_visible_indexer, enabled, compute_topk):
    impl, config, table, device = full_visible_indexer
    config.enable_sfa_full_visible_index_bypass = enabled
    metadata = _full_visible_metadata()
    events = []
    keys = torch.ones(64, 128)
    impl.forward_k = MagicMock(return_value=(keys, None))
    impl.write_cache = MagicMock(side_effect=lambda *a, **kw: events.append("write"))
    original = impl._get_full_visible_topk_indices

    def eligible(*args):
        events.append("eligibility")
        return original(*args)

    impl._get_full_visible_topk_indices = MagicMock(side_effect=eligible)
    impl.head_dim = 128
    impl.n_head = 1
    impl.qk_rope_head_dim = 64
    impl.is_rope_neox_style = True
    impl.use_torch_npu_lightning_indexer = False
    impl.k_cache = SimpleNamespace(kv_cache=(MagicMock(),))
    impl.wk_weights_proj = MagicMock(return_value=(torch.zeros(64, 129), None))
    impl.wq_b = MagicMock(return_value=(torch.zeros(64, 128), None))
    hidden = SimpleNamespace(shape=(64, 128), device=device, dtype=torch.float32)
    scored = torch.full((64, 1, 2048), -2, dtype=torch.int32)

    def score(*args):
        events.append("score")
        return scored

    with (
        patch("vllm_ascend.attention.indexer.HAS_TRITON", True),
        patch("vllm_ascend.attention.indexer.rope_forward_triton_siso", side_effect=lambda x, *a, **kw: x),
        patch("vllm_ascend.attention.indexer.DeviceOperator.indexer_select_post_process", side_effect=score),
    ):
        result = impl.forward(hidden, torch.zeros(64, 128), None, None, keys, metadata, compute_topk)
    impl.write_cache.assert_called_once_with(keys, None, metadata.slot_mapping, indexer_attn_metadata=metadata)
    if not compute_topk:
        assert result is None
        assert events == ["write"]
    elif enabled:
        assert events == ["write", "eligibility"]
        assert result.untyped_storage().data_ptr() == table.untyped_storage().data_ptr()
        impl.wk_weights_proj.assert_not_called()
        impl.wq_b.assert_not_called()
    else:
        assert events == ["write", "eligibility", "score"]
        assert result is scored


def test_specialized_backend_falls_back(full_visible_indexer):
    _, _, table, device = full_visible_indexer

    class SpecializedIndexer(AscendSFAIndexerBackend):
        pass

    impl = SpecializedIndexer.__new__(SpecializedIndexer)
    impl.allow_short_prefill_indexer_scoring_skip = True
    impl._full_visible_index_table = _DeviceTable(table, device)
    assert impl._get_full_visible_topk_indices(_full_visible_metadata(), 64, device) is None
