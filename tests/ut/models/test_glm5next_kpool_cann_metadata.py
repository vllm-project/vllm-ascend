# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.attention.indexer_kpool as backend_module
from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolMetadataBuilder, AscendIndexerKPoolTailMetadataBuilder
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import AscendMultiKVCacheGroupMTPProposer
from vllm_ascend.utils import vllm_version_is

pytestmark = pytest.mark.usefixtures("keypool_kernels")


@pytest.fixture(autouse=True)
def disable_cp(monkeypatch):
    monkeypatch.setattr("vllm_ascend.attention.indexer_kpool.enable_dsa_cp", lambda: False)


def _config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=2),
        model_config=SimpleNamespace(max_model_len=8192),
    )


def _builder(config):
    ratio = {"compress_ratio": 4} if vllm_version_is("0.28.0") else {"tokens_per_state": 4}
    spec = AscendMLAAttentionSpec(
        block_size=config.cache_config.block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        model_version="glm5_next",
        **ratio,
    )
    return AscendIndexerKPoolMetadataBuilder(spec, ["layer.indexer.k_cache"], config, torch.device("cpu"))


class _Common(SimpleNamespace):
    @property
    def _seq_lens_cpu(self):
        raise AssertionError("CANN metadata must not read host lengths")


def _common():
    return _Common(
        num_reqs=2,
        num_input_tokens=10,
        num_actual_tokens=8,
        query_start_loc=torch.tensor([0, 6, 8], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, -1], [1, -1]], dtype=torch.int32),
        slot_mapping=torch.tensor([2, -1, 4, 5, 6, 7, 135, 136, -1, -1]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7, 7, 8, 3, 7]),
    )


@pytest.mark.parametrize("unpadded", [None, "positions", "slot_mapping"])
@pytest.mark.parametrize("dsa_cp", [False, True])
def test_metadata_delegates_to_kernel_and_keeps_capture_storage(unpadded, dsa_cp, keypool_kernels, monkeypatch):
    monkeypatch.setattr(backend_module, "enable_dsa_cp", lambda: dsa_cp)
    monkeypatch.setattr(backend_module, "get_tp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=1))
    builder, common = _builder(_config()), _common()
    if unpadded:
        setattr(common, unpadded, getattr(common, unpadded)[: common.num_actual_tokens])
    first = builder.build(0, common)
    args = keypool_kernels.build_keypool_metadata.call_args.args
    assert args[0] is common
    assert args[2:] == (10, 8 if unpadded else 10, 128, 4)
    kwargs = keypool_kernels.build_keypool_metadata.call_args.kwargs
    assert kwargs["query_metadata"] is first.query_metadata
    assert kwargs["local_start"] == (5 if dsa_cp else 0)
    if dsa_cp:
        assert first.query_metadata.positions.data_ptr() == first.positions[5:].data_ptr()
        assert first.query_metadata.positions.shape == (5,)
        assert first.query_metadata.num_actual_tokens == 3
        assert first.query_metadata.cum_query_lens.dtype == torch.int64
        assert first.query_metadata.seq_lens.dtype == torch.int64
        assert first.query_metadata.pool_tail.dtype == torch.int64
    assert first.keypool_state_table.tolist() == [[1, 3, 3, 3], [2, 4, 4, 4]]
    fields = ("start_pos", "query_start_loc", "pool_tail", "pooled_slot_mapping", "keypool_state_table")
    pointers = [getattr(first, f).data_ptr() for f in fields]
    common.seq_lens.add_(1)
    common.positions.add_(1)
    second = builder.build(0, common)
    assert pointers == [getattr(second, f).data_ptr() for f in fields]
    if dsa_cp:
        for field in ("positions", "cum_query_lens", "seq_lens", "pool_tail"):
            assert getattr(first.query_metadata, field).data_ptr() == getattr(second.query_metadata, field).data_ptr()


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("dsa_cp", [False, True])
def test_tail_inputs_pass_page_stride_and_isolate_step_buffers(width, dsa_cp, keypool_kernels, monkeypatch):
    monkeypatch.setattr(backend_module, "enable_dsa_cp", lambda: dsa_cp)
    config, common = _config(), _common()
    spec = AscendIndexerKPoolTailSpec(
        block_size=7,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float32,
        sliding_window=4,
        compress_ratio=4,
        page_size_padded=81920,
    )
    builder = AscendIndexerKPoolTailMetadataBuilder(spec, ["layer.indexer.tail_cache"], config, torch.device("cpu"))
    common.max_query_len = width
    common.num_actual_tokens = common.num_input_tokens = 2 * width
    common.query_start_loc = torch.tensor([0, width, 2 * width], dtype=torch.int32)
    common.seq_lens = torch.tensor([16 + width, 27 + width], dtype=torch.int32)
    common.slot_mapping = torch.arange(2 * width)
    first = builder.build(0, common)
    args = keypool_kernels.build_tail_inputs.call_args.args
    assert args[0] is common
    assert args[1:5] == (4, 7, 80, width)
    assert first.cann_inputs is keypool_kernels.build_tail_inputs.return_value
    first_buffers = args[5]
    other = _Common(**vars(common))
    other.slot_mapping = common.slot_mapping.clone()
    isolated = builder.build_for_drafting(other, 1)
    other_buffers = keypool_kernels.build_tail_inputs.call_args.args[5]
    assert other_buffers["writes"].data_ptr() != first_buffers["writes"].data_ptr()
    common.query_start_loc[-1] = -1
    common.num_actual_tokens = width
    builder.build_for_graph_capture(common)
    assert keypool_kernels.build_tail_inputs.call_args.args[5] is first_buffers
    assert isolated.cann_inputs is keypool_kernels.build_tail_inputs.return_value


@pytest.mark.parametrize("field", ["positions", "slot_mapping"])
def test_incomplete_inputs_fail_before_kernel_launch(field, keypool_kernels):
    common = _common()
    setattr(common, field, getattr(common, field)[: common.num_actual_tokens - 1])
    with pytest.raises(ValueError, match="must cover every actual token"):
        _builder(_config()).build(0, common)
    keypool_kernels.build_keypool_metadata.assert_not_called()


@pytest.mark.parametrize("logical,primary", [(128, True), (256, False), (1152, False)])
def test_mtp_preserves_expanded_kernel_pages(logical, primary):
    config, common = _config(), _common()
    config.cache_config.block_size = logical
    config.parallel_config = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    builder = _builder(config)
    split, width = logical // 128, (8192 + logical - 1) // logical
    physical = torch.arange(1, 2 * width + 1, dtype=torch.int32).view(2, width)
    table = (physical[:, :, None] * split + torch.arange(split)).reshape(2, -1)
    common.block_table_tensor = table
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._uses_multi_group_kv_cache, proposer.kv_cache_gid, proposer.max_model_len = True, 0, 8192
    proposer.vllm_config = config
    gid = 0 if primary else 1
    proposer._multi_group_slot_mapping_buffers = {(gid, 0): common.slot_mapping.clone()}
    proposer._multi_group_block_table_buffers = {gid: torch.zeros_like(table)}
    cache = SimpleNamespace(
        is_circular_group=False,
        compute_slot_mapping=lambda *_: None,
        get_device_tensor=lambda: table,
        slot_mapping=SimpleNamespace(gpu=common.slot_mapping),
    )
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=[cache, cache]))
    group = SimpleNamespace(kv_cache_group_id=gid, get_metadata_builder=lambda: builder)
    draft = proposer._common_attn_metadata_for_draft_group(common, group, common.num_input_tokens)
    torch.testing.assert_close(builder.build(0, draft).block_table, table)
