# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real sparse builder and consumer contracts, with CPU tensors only."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention import indexer as indexer_module
from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.device import device_op
from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.ops import rotary_embedding
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import DraftWindowProviderStatus
from vllm_ascend.worker.v2.spec_decode.draft_window_sfa import (
    freeze_sfa_read_metadata,
    validate_sfa_read_consumers,
)


def common_metadata():
    query = torch.tensor([0, 2], dtype=torch.int32)
    lengths = torch.tensor([131], dtype=torch.int32)
    return AscendCommonAttentionMetadata(
        query_start_loc=query,
        query_start_loc_cpu=query,
        seq_lens=lengths,
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=131,
        block_table_tensor=torch.tensor([[11, 12]], dtype=torch.int32),
        slot_mapping=torch.tensor([1537, 1538]),
        positions=torch.tensor([257, 258]),
        num_input_tokens=2,
    )


def builder_for_cpu():
    # Isolate configuration/allocation setup, while exercising the actual
    # build and per-slot-address derived-buffer writer unchanged.
    builder = object.__new__(AscendSFAIndexerMetadataBuilder)
    builder.use_pcp = builder.use_dcp = builder.use_dsa_cp = False
    builder.kernel_block_size = 128
    builder._rope_capacity = 4
    builder._rope_buffers = {}
    return builder


def test_real_indexer_builder_reuse_does_not_overwrite_owned_rotary_reads(monkeypatch):
    cache = torch.arange(2048, dtype=torch.float32).view(512, 4)
    for name, value in {
        "_cos_cache": cache,
        "_sin_cache": -cache,
        "_cos_mla": torch.zeros(4, 1, 1, 4),
        "_sin_mla": torch.zeros(4, 1, 1, 4),
    }.items():
        monkeypatch.setattr(rotary_embedding, name, value)
    monkeypatch.setattr(indexer_module, "get_cos_and_sin_mla", rotary_embedding.get_cos_and_sin_mla)
    builder = builder_for_cpu()
    common = common_metadata()
    first = builder.build_for_drafting(common, draft_index=0)
    borrowed_cos = first.cos
    expected = first.cos.clone()
    retained = freeze_sfa_read_metadata({"a": first, "b": first}, ("a", "b"))
    second = builder.build_for_drafting(common.replace(positions=torch.tensor([17, 18])), draft_index=0)
    assert second.cos.data_ptr() == borrowed_cos.data_ptr()
    assert not torch.equal(borrowed_cos, expected)  # Real overwrite counterexample.
    assert torch.equal(first.cos, expected)
    assert len(retained) == 2
    assert first.slot_mapping.data_ptr() == common.slot_mapping.data_ptr()
    assert first.block_table.data_ptr() == common.block_table_tensor.data_ptr()
    assert first.actual_seq_lengths_key is first.seq_lens
    assert first.actual_seq_lengths_query is first.cum_query_lens
    assert torch.equal(first.cos[:, 0, 0], cache[torch.tensor([257, 258])])


def test_indexer_and_sparse_attention_forward_same_rebased_read_coordinates(monkeypatch):
    common = common_metadata()
    metadata = SimpleNamespace(block_table=common.block_table_tensor)
    indices = torch.tensor([[[0, 130]], [[1, 130]]], dtype=torch.int32)
    seen = {}

    def select(**kwargs):
        seen["indexer"] = kwargs
        return indices, None

    def attention(**kwargs):
        seen["attention"] = kwargs
        return torch.empty(2, 1, 4)

    monkeypatch.setattr(device_op.torch_npu, "npu_lightning_indexer", select, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention", attention, raising=False)
    query = torch.empty(2, 1, 4)
    cache = torch.empty(20, 128, 1, 4)
    actual_query = common.query_start_loc[1:]
    selected = BaseDeviceAdaptor.indexer_select_post_process(
        query,
        None,
        None,
        torch.ones(2, 1),
        (cache,),
        0,
        1,
        metadata,
        actual_query,
        common.seq_lens,
        False,
        False,
        True,
    )
    BaseDeviceAdaptor.execute_sparse_flash_attention_process(
        SimpleNamespace(scale=1.0, qk_rope_head_dim=4),
        query,
        query,
        (cache, cache),
        selected,
        metadata,
        actual_query,
        common.seq_lens,
    )
    assert seen["indexer"]["block_table"] is seen["attention"]["block_table"]
    assert seen["indexer"]["actual_seq_lengths_key"] is seen["attention"]["actual_seq_lengths_kv"]
    assert seen["attention"]["sparse_indices"] is indices
    # Kernel source uses index // page_size to address the supplied table:
    # relative 130 means logical 258 at S=128, physical block 12 + offset 2.
    assert int(common.block_table_tensor[0, 130 // 128]) * 128 + 130 % 128 == 1538
    assert common.positions.tolist() == [257, 258]


class TestSFAImpl:
    indexer: SimpleNamespace
    kv_sharing_target_layer_name: str | None
    runtime_has_indexer = True
    skip_topk = False
    use_index_cache = False


class TestIndexerImpl:
    pass


def dependency_fixture():
    impl = TestSFAImpl()
    impl.indexer = SimpleNamespace(impl=TestIndexerImpl(), k_cache=SimpleNamespace(prefix="draft.indexer.k_cache"))
    impl.kv_sharing_target_layer_name = "target.attn"
    routes = [
        SimpleNamespace(layers=("draft.attn",), status=DraftWindowProviderStatus.DIRECT_SFA),
        SimpleNamespace(layers=("draft.indexer.k_cache",), status=DraftWindowProviderStatus.DIRECT_SFA_INDEXER),
    ]
    return impl, routes


def test_shared_physical_cache_resolves_draft_owned_indexer_metadata(monkeypatch):
    impl, routes = dependency_fixture()
    layers = {"draft.attn": SimpleNamespace(impl=impl)}
    validate_sfa_read_consumers(layers, routes, sfa_impl_type=TestSFAImpl, indexer_impl_type=TestIndexerImpl)
    from vllm_ascend.attention import sfa_v1

    owned = object()
    monkeypatch.setattr(
        sfa_v1, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"draft.indexer.k_cache": owned})
    )
    assert AscendSFAImpl._get_indexer_attn_metadata(impl) is owned


@pytest.mark.parametrize("field", ["skip_topk", "use_index_cache"])
def test_target_indexcache_is_not_remapped_as_a_window_local_selection(field):
    impl, routes = dependency_fixture()
    setattr(impl, field, True)
    with pytest.raises(NotImplementedError, match="window-local selection producer"):
        validate_sfa_read_consumers(
            {"draft.attn": SimpleNamespace(impl=impl)},
            routes,
            sfa_impl_type=TestSFAImpl,
            indexer_impl_type=TestIndexerImpl,
        )


def test_missing_indexer_dependency_is_not_covered_by_main_sfa_provider():
    impl, routes = dependency_fixture()
    with pytest.raises(NotImplementedError, match="indexer read metadata provider missing"):
        validate_sfa_read_consumers(
            {"draft.attn": SimpleNamespace(impl=impl)},
            routes[:1],
            sfa_impl_type=TestSFAImpl,
            indexer_impl_type=TestIndexerImpl,
        )
