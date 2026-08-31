from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoEWithLoRA,
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    has_lora,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
)
from vllm_ascend.lora.punica_npu import GMM_TOKEN_THRESHOLD, PunicaWrapperNPU


def test_ascend_fused_moe_lora_initializes_skipped_upstream_fields() -> None:
    parallel_config = SimpleNamespace(tp_size=8, tp_rank=3, ep_rank=0, use_ep=False)
    shared_experts = torch.nn.Module()
    base_layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            hidden_dim=4096,
            num_local_experts=256,
            num_experts=256,
            intermediate_size_per_partition=256,
            experts_per_token=8,
            moe_parallel_config=parallel_config,
            is_act_and_mul=True,
        ),
        _shared_experts=shared_experts,
    )

    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)

    assert wrapper._lora_stream is None
    assert wrapper._events is None
    assert wrapper.enable_moe_shared_loras is False
    assert wrapper._shared_experts is shared_experts
    assert wrapper.n_slices == 256 * 3


def test_moe_lora_apply_uses_adapter_enabled() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        adapter_enabled="all_enabled",
        fully_sharded=False,
        tp_rank=0,
    )
    routing = (torch.tensor([0]), torch.tensor([0]))

    moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        lora_routing=routing,
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=routing,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["adapter_enabled"] == "all_enabled"
    assert calls[1].kwargs["adapter_enabled"] == "all_enabled"
    assert calls[0].kwargs["fully_sharded"] is False
    assert calls[1].kwargs["fully_sharded"] is False
    assert calls[1].kwargs["offset"] == 0


def test_moe_lora_apply_propagates_fully_sharded_metadata() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked=(torch.empty(1, 1, 16, 8),),
        adapter_enabled="all_enabled",
        fully_sharded=True,
        tp_rank=3,
    )
    routing = (torch.tensor([0]), torch.tensor([0]))

    moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        lora_routing=routing,
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=routing,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["fully_sharded"] is True
    assert calls[1].kwargs["fully_sharded"] is True
    assert calls[1].kwargs["offset"] == 48


def test_punica_fully_sharded_moe_gathers_rank_shards() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)

    def shrink(_, __, output, ___, ____):
        output.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()
    lora_a = (torch.zeros(2, 2, 2, 3),)
    lora_b = (torch.zeros(2, 2, 5, 4),)

    with (
        patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather",
            side_effect=lambda value: torch.cat((value, value + 10), dim=-1),
        ) as all_gather,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce") as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 5),
            x=torch.zeros(2, 3),
            lora_a_stacked=lora_a,
            lora_b_stacked=lora_b,
            expert_ids=torch.tensor([0, 1]),
            adapter_enabled=torch.tensor([1, 1]),
            fully_sharded=True,
            token_lora_mapping=torch.tensor([0, 1]),
        )

    all_gather.assert_called_once()
    all_reduce.assert_not_called()
    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(
        expand_args[0],
        torch.tensor([[1.0, 2.0, 11.0, 12.0], [3.0, 4.0, 13.0, 14.0]]),
    )
    assert expand_args[1].shape == (4, 5, 4)
    assert torch.equal(expand_args[3], torch.tensor([0, 3]))


def test_punica_fully_sharded_moe_reduces_partial_rank() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)

    def shrink(_, __, output, ___, ____):
        output.copy_(torch.arange(8, dtype=torch.float32).view(2, 4))

    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()
    lora_a = (torch.zeros(2, 2, 4, 3),)
    lora_b = (torch.zeros(2, 2, 5, 4),)

    with (
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather") as all_gather,
        patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value + 10,
        ) as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 10),
            x=torch.zeros(2, 3),
            lora_a_stacked=lora_a,
            lora_b_stacked=lora_b,
            expert_ids=torch.tensor([0, 1]),
            adapter_enabled=torch.tensor([1, 1]),
            fully_sharded=True,
            offset=5,
            token_lora_mapping=torch.tensor([0, 1]),
        )

    all_gather.assert_not_called()
    all_reduce.assert_called_once()
    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(
        expand_args[0],
        torch.arange(8, dtype=torch.float32).view(2, 4) + 10,
    )
    assert expand_args[1].shape == (4, 5, 4)
    assert expand_args[4] == 5


def test_allgather_routing_preserves_multi_adapter_and_base_mapping() -> None:
    context = SimpleNamespace(
        top_k=2,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0, -1, 1])),
    )
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]])
    # Original flat rows [0..5] land at these expert-sorted positions.
    expanded_row_idx = torch.tensor([2, 0, 1, 3, 4, 5])

    expert_ids, lora_slots = _recover_moe_lora_routing_allgather(context, expanded_row_idx, topk_ids)

    assert torch.equal(expert_ids, torch.tensor([0, 0, 1, 1, 1, 1]))
    assert torch.equal(lora_slots, torch.tensor([0, -1, 0, -1, 1, 1]))


def test_all2all_routing_uses_local_experts_and_exchanged_adapters() -> None:
    context = SimpleNamespace(
        local_num_experts=3,
        exchanged_lora_indices=torch.tensor([1, -1, 0, 2]),
    )

    expert_ids, lora_slots = _recover_moe_lora_routing_all2all(
        context,
        group_list=torch.tensor([2, 0, 2]),
    )

    assert torch.equal(expert_ids, torch.tensor([0, 0, 2, 2]))
    assert torch.equal(lora_slots, torch.tensor([1, -1, 0, 2]))


def test_has_lora_follows_batch_metadata() -> None:
    assert not has_lora(None)
    assert not has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=True)))
    assert has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False)))


@pytest.mark.parametrize(
    ("index_mapping", "expected_no_lora"),
    [((0, 0), True), ((0, 1), False), ((2, 0), False)],
)
def test_decode_metadata_refreshes_no_lora(index_mapping, expected_no_lora) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    mapping = SimpleNamespace(index_mapping=index_mapping)
    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [], 2, 100)
    assert wrapper.no_lora is expected_no_lora
    assert bool(wrapper._no_lora_cpu) is expected_no_lora


@pytest.mark.parametrize(
    ("is_prefill", "index_mapping", "expected_count"),
    [
        (True, (42, 42, 42), 1),
        (True, (42, 7, 42), 2),
        (True, (42, 0, 42), 1),
        (False, (42, 42, 42), 1),
    ],
)
def test_metadata_tracks_host_known_active_moe_lora_count(is_prefill, index_mapping, expected_count) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    mapping = SimpleNamespace(is_prefill=is_prefill, index_mapping=index_mapping)

    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [7, 42, None], 3, 100)

    assert wrapper.num_active_moe_loras == expected_count


def test_dsa_sgmv_metadata_splits_mixed_batch_once_per_step() -> None:
    class CopyCountingBuffer:
        def __init__(self, tensor):
            self.tensor = tensor
            self.copy_count = 0

        def copy_(self, source, *, non_blocking=False):
            self.copy_count += 1
            self.tensor.copy_(source, non_blocking=non_blocking)
            return self

        def __getitem__(self, index):
            return self.tensor[index]

    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.device = torch.device("cpu")
    wrapper._token_lora_indices = torch.tensor([0, 0, 1, 1, 1, -1])
    with patch("vllm_ascend.lora.punica_npu.PIN_MEMORY", False):
        wrapper._init_dsa_sgmv_metadata_buffers(max_batches=4)
    wrapper._host_sgmv_metadata = wrapper._encode_sgmv_metadata((0, 0, 1, 1, 1, -1))
    wrapper._dsa_sgmv_metadata_buffer = CopyCountingBuffer(wrapper._dsa_sgmv_metadata_buffer)

    wrapper.prepare_dsa_sgmv_metadata(
        num_decode_tokens=2,
        num_actual_tokens=6,
    )

    decode_metadata = wrapper.get_dsa_sgmv_metadata(wrapper._token_lora_indices[:2])
    prefill_indices = wrapper._token_lora_indices[2:6]
    prefill_metadata = wrapper.get_dsa_sgmv_metadata(prefill_indices)

    assert decode_metadata.token_offset == 0
    assert decode_metadata.token_nums == 2
    assert decode_metadata.seq_start_locs[: decode_metadata.batches].tolist() == [0]
    assert decode_metadata.seq_lengths[: decode_metadata.batches].tolist() == [2]
    assert decode_metadata.lora_indices[: decode_metadata.batches].tolist() == [0]

    assert prefill_metadata.token_offset == 2
    assert prefill_metadata.token_nums == 4
    assert prefill_metadata.seq_start_locs[: prefill_metadata.batches].tolist() == [0, 3]
    assert prefill_metadata.seq_lengths[: prefill_metadata.batches].tolist() == [3, 1]
    assert prefill_metadata.lora_indices[: prefill_metadata.batches].tolist() == [1, -1]
    assert torch.count_nonzero(prefill_metadata.seq_lengths[prefill_metadata.batches :]) == 0
    assert wrapper.get_dsa_sgmv_metadata(prefill_indices) is prefill_metadata
    assert wrapper._dsa_sgmv_metadata_buffer.copy_count == 1


def test_prefill_and_dsa_reuse_one_host_rle_across_split_group() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.device = torch.device("cpu")
    wrapper._token_lora_indices = torch.tensor([0, 0, 0, 0, 1, -1])
    with patch("vllm_ascend.lora.punica_npu.PIN_MEMORY", False):
        wrapper._init_prefill_sgmv_metadata_buffers(max_batches=4)
        wrapper._init_dsa_sgmv_metadata_buffers(max_batches=4)

    def fake_base_update_metadata(self, mapping, *_args, **_kwargs):
        self._update_prefill_metadata(torch.empty(0, dtype=torch.long))
        self.is_prefill = mapping.is_prefill

    mapping = SimpleNamespace(
        is_prefill=True,
        index_mapping=(7, 7, 7, 7, 42, 0),
    )
    with (
        patch.object(
            PunicaWrapperNPU,
            "_encode_sgmv_metadata",
            wraps=PunicaWrapperNPU._encode_sgmv_metadata,
        ) as encode_sgmv_metadata,
        patch.object(
            PunicaWrapperBase,
            "update_metadata",
            new=fake_base_update_metadata,
        ),
    ):
        wrapper.update_metadata(mapping, [7, 42, None], 3, 100)
        wrapper.prepare_dsa_sgmv_metadata(
            num_decode_tokens=2,
            num_actual_tokens=6,
        )

    assert encode_sgmv_metadata.call_count == 1
    seq_start_locs, seq_lengths, lora_indices, batches, max_length, token_nums = wrapper.prefill_metadata
    assert seq_start_locs.tolist() == [0, 4, 5]
    assert seq_lengths.tolist() == [4, 1, 1]
    assert lora_indices.tolist() == [0, 1, -1]
    assert (batches, max_length, token_nums) == (3, 4, 6)

    decode_metadata = wrapper.get_dsa_sgmv_metadata(wrapper._token_lora_indices[:2])
    prefill_metadata = wrapper.get_dsa_sgmv_metadata(wrapper._token_lora_indices[2:])
    assert decode_metadata.seq_lengths[: decode_metadata.batches].tolist() == [2]
    assert decode_metadata.lora_indices[: decode_metadata.batches].tolist() == [0]
    assert prefill_metadata.seq_start_locs[: prefill_metadata.batches].tolist() == [0, 2, 3]
    assert prefill_metadata.seq_lengths[: prefill_metadata.batches].tolist() == [2, 1, 1]
    assert prefill_metadata.lora_indices[: prefill_metadata.batches].tolist() == [0, 1, -1]


def test_dsa_sgmv_metadata_marks_base_only_segment() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.device = torch.device("cpu")
    wrapper._token_lora_indices = torch.tensor([-1, -1, 0, 0])
    with patch("vllm_ascend.lora.punica_npu.PIN_MEMORY", False):
        wrapper._init_dsa_sgmv_metadata_buffers(max_batches=2)
    wrapper._host_sgmv_metadata = wrapper._encode_sgmv_metadata((-1, -1, 0, 0))

    wrapper.prepare_dsa_sgmv_metadata(
        num_decode_tokens=2,
        num_actual_tokens=4,
    )

    decode_metadata = wrapper.get_dsa_sgmv_metadata(wrapper._token_lora_indices[:2])
    prefill_metadata = wrapper.get_dsa_sgmv_metadata(wrapper._token_lora_indices[2:])
    assert decode_metadata.no_lora
    assert not prefill_metadata.no_lora
    assert bool(decode_metadata.no_lora_dispatch)
    assert not bool(prefill_metadata.no_lora_dispatch)


def test_dsa_shrink_dispatch_uses_segment_local_flags_and_mapping_views() -> None:
    num_decode_tokens = 2
    num_prefill_tokens = GMM_TOKEN_THRESHOLD + 1
    num_actual_tokens = num_decode_tokens + num_prefill_tokens
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.device = torch.device("cpu")
    wrapper._token_lora_indices = torch.zeros(num_actual_tokens, dtype=torch.long)
    with patch("vllm_ascend.lora.punica_npu.PIN_MEMORY", False):
        wrapper._init_dsa_sgmv_metadata_buffers(max_batches=2)
    wrapper._host_sgmv_metadata = wrapper._encode_sgmv_metadata((0,) * num_actual_tokens)

    wrapper.prepare_dsa_sgmv_metadata(
        num_decode_tokens=num_decode_tokens,
        num_actual_tokens=num_actual_tokens,
    )

    decode_metadata = wrapper.get_dsa_sgmv_metadata(wrapper._token_lora_indices[:num_decode_tokens])
    prefill_metadata = wrapper.get_dsa_sgmv_metadata(wrapper._token_lora_indices[num_decode_tokens:])
    assert not bool(decode_metadata.use_gmm_shrink)
    assert bool(prefill_metadata.use_gmm_shrink)
    assert not bool(decode_metadata.use_gmm_expand)
    assert bool(prefill_metadata.use_gmm_expand)
    assert decode_metadata.token_lora_indices.data_ptr() == wrapper._token_lora_indices.data_ptr()
    assert prefill_metadata.token_lora_indices.untyped_storage().data_ptr() == (
        wrapper._token_lora_indices.untyped_storage().data_ptr()
    )
    assert prefill_metadata.token_lora_indices.storage_offset() == num_decode_tokens


@pytest.mark.parametrize(
    ("token_count", "expected_use_gmm"),
    [
        (GMM_TOKEN_THRESHOLD, False),
        (GMM_TOKEN_THRESHOLD + 1, True),
    ],
)
def test_group_gemm_dispatch_uses_token_threshold(token_count, expected_use_gmm) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    mapping = SimpleNamespace(index_mapping=(7,) * token_count)

    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [7], 1, 100)

    assert bool(wrapper._use_gmm_shrink_cpu) is expected_use_gmm
    assert bool(wrapper._use_gmm_expand_cpu) is expected_use_gmm
    assert not bool(wrapper._no_lora_cpu)


def test_group_gemm_dispatch_forwards_dense_lora_metadata() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._seq_start_locs = torch.tensor([0, 2])
    wrapper._seq_lengths = torch.tensor([2, 1])
    wrapper._lora_indices_per_batch = torch.tensor([0, 1])
    wrapper._token_lora_indices = torch.tensor([0, 0, 1])
    wrapper.batch_size = 2
    wrapper.max_length = 2
    wrapper.token_nums = 3
    wrapper._use_gmm_shrink_cpu = torch.tensor(True)
    wrapper._use_gmm_expand_cpu = torch.tensor(True)
    wrapper._no_lora_cpu = torch.tensor(False)

    x = torch.ones(3, 4)
    shrink_output = (torch.zeros(3, 2),)
    lora_a_stacked = (torch.zeros(2, 1, 2, 4),)
    expand_output = torch.zeros(3, 5)
    lora_b_stacked = (torch.zeros(2, 1, 5, 2),)

    with (
        patch("vllm_ascend.lora.punica_npu._dispatch_lora_shrink") as dispatch_shrink,
        patch("vllm_ascend.lora.punica_npu._dispatch_lora_expand") as dispatch_expand,
    ):
        wrapper.add_shrink(shrink_output, x, lora_a_stacked, 0.5)
        wrapper.add_expand(expand_output, shrink_output, lora_b_stacked, (5,))

    shrink_args = dispatch_shrink.call_args.args
    assert torch.equal(shrink_args[0][0], shrink_output[0])
    assert shrink_args[2][0] is lora_a_stacked[0]
    assert torch.equal(shrink_args[3], torch.tensor([0, 1]))
    assert torch.equal(shrink_args[4], torch.tensor([2, 1]))
    assert torch.equal(shrink_args[5], torch.tensor([0, 0, 1]))
    assert shrink_args[6] == 0.5
    assert shrink_args[7] is wrapper._use_gmm_shrink_cpu
    assert shrink_args[8] is wrapper._no_lora_cpu

    expand_args = dispatch_expand.call_args.args
    assert torch.equal(expand_args[0], expand_output)
    assert torch.equal(expand_args[1][0], shrink_output[0])
    assert expand_args[2][0] is lora_b_stacked[0]
    assert torch.equal(expand_args[3], torch.tensor([0, 1]))
    assert torch.equal(expand_args[4], torch.tensor([2, 1]))
    assert torch.equal(expand_args[5], torch.tensor([0, 0, 1]))
    assert expand_args[6] == [5]
    assert expand_args[7:9] == (0, True)
    assert expand_args[9] is wrapper._use_gmm_expand_cpu
    assert expand_args[10] is wrapper._no_lora_cpu


def test_dsa_metadata_routes_shrink_and_expand_to_native_bgmv() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.sgmv_shrink = Mock()
    wrapper.sgmv_expand_slice = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()
    lora_indices = torch.tensor([0])
    seq_lengths = torch.tensor([2])
    token_lora_indices = torch.tensor([0, 0])
    metadata = SimpleNamespace(
        no_lora=False,
        lora_indices=lora_indices,
        seq_lengths=seq_lengths,
        token_lora_indices=token_lora_indices,
        op_args=(
            torch.tensor([0]),
            seq_lengths,
            lora_indices,
            1,
            2,
            2,
        ),
    )
    x = torch.ones(2, 4)
    shrink_output = (torch.zeros(2, 2).transpose(0, 1),)
    assert not shrink_output[0].is_contiguous()
    lora_a_stacked = (torch.zeros(1, 1, 2, 4),)
    expand_output = torch.zeros(2, 5)
    lora_b_stacked = (torch.zeros(1, 1, 5, 2),)

    with (
        patch("vllm_ascend.lora.punica_npu._dispatch_lora_shrink") as dispatch_shrink,
        patch("vllm_ascend.lora.punica_npu._dispatch_lora_expand") as dispatch_expand,
    ):
        wrapper.add_shrink(
            shrink_output,
            x,
            lora_a_stacked,
            1.0,
            sgmv_metadata=metadata,
        )
        wrapper.add_expand(
            expand_output,
            shrink_output,
            lora_b_stacked,
            (5,),
            sgmv_metadata=metadata,
        )

    shrink_args = wrapper.bgmv_shrink.call_args.args
    assert torch.equal(shrink_args[0], x)
    assert torch.equal(shrink_args[1], lora_a_stacked[0][:, 0])
    assert torch.equal(shrink_args[2], shrink_output[0])
    assert shrink_args[3] is token_lora_indices
    assert shrink_args[4] == 1.0

    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(expand_args[0], shrink_output[0])
    assert not expand_args[0].is_contiguous()
    assert torch.equal(expand_args[1], lora_b_stacked[0][:, 0])
    assert expand_args[2] is expand_output
    assert expand_args[3] is token_lora_indices
    assert expand_args[4:] == (0, 5, True)
    wrapper.bgmv_shrink.assert_called_once()
    wrapper.bgmv_expand_slice.assert_called_once()
    wrapper.sgmv_shrink.assert_not_called()
    wrapper.sgmv_expand_slice.assert_not_called()
    dispatch_shrink.assert_not_called()
    dispatch_expand.assert_not_called()
