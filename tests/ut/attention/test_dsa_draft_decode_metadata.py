# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.attention.test_dsa_metadata_builders import CommonMetadata, make_builder
from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.dsa_attn_kv_plan import DsaAttnKvPlan
from vllm_ascend.ops import rope_dsv4


@pytest.fixture
def draft_case(monkeypatch):
    table = torch.arange(128 * 4, dtype=torch.float32).view(128, 1, 1, 4)
    monkeypatch.setattr(
        rope_dsv4,
        "_ROPE_STATE",
        SimpleNamespace(
            full_rope_cache={"rope": (table, -table)},
            registry_summary={"rope": {"default"}},
            layer_info={"layer": ("rope", ["default"])},
        ),
    )
    monkeypatch.setattr(dsa_v1, "get_cos_and_sin_dsa", rope_dsv4.get_cos_and_sin_dsa)
    monkeypatch.setattr(dsa_v1, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dsa_v1, "_dsa_swa_only_cmp_ratio", lambda *args: 1)
    monkeypatch.setattr(dsa_v1, "split_decodes_and_prefills", lambda *args, **kwargs: (3, 0, 3, 0))

    # Stand in for the native tiling op. Its output must depend on the current
    # device inputs, so stale length preparation or tiling reuse is observable.
    def tile(**kwargs):
        output = torch.zeros(dsa_v1.SAS_METADATA_BUFFER_SIZE, dtype=torch.int32)
        output[:3] = kwargs["seqused_kv"]
        output[3:7] = kwargs["cu_seqlens_q"]
        output[7:11] = kwargs["cu_seqlens_ori_kv"]
        return output

    tiling = Mock(side_effect=tile)
    plan = SimpleNamespace(
        layout_kv="PA_BSND",
        requires_block_offset_slots=True,
        get_dsa_sparse_attn_metadata_op=lambda: tiling,
        get_dsa_sparse_attn_metadata_kwargs=lambda device: {},
    )
    plan.format_dsa_slot_mapping = lambda slots, size: DsaAttnKvPlan.format_dsa_slot_mapping(plan, slots, size)
    monkeypatch.setattr(dsa_v1, "get_dsa_attn_kv_plan", lambda *args: plan)
    lengths = Mock(side_effect=lambda cache, key, seq, count, zero, fallback: torch.cat((zero, seq.cumsum(0).int())))
    monkeypatch.setattr(
        dsa_v1,
        "DeviceOperator",
        SimpleNamespace(
            get_dsa_decode_cu_seqlens_ori_kv=lengths,
            get_dsa_decode_cu_seqlens_cmp_kv=lambda tensor: tensor,
        ),
    )

    def inputs(cls):
        builder = make_builder(cls)
        builder.supports_draft_decode_metadata_update = True
        builder.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=2)
        builder.model_config.max_model_len = 128
        builder.layer_names = ["layer"]
        builder.num_actual_tokens = 3
        builder.slot_mapping = torch.zeros((8, 2), dtype=torch.int32)
        common = CommonMetadata(
            num_reqs=3,
            num_actual_tokens=3,
            num_input_tokens=3,
            query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([3, 7, 11], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([3, 7, 11], dtype=torch.int32),
            block_table_tensor=torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int32),
            slot_mapping=torch.tensor([6, 22, 38], dtype=torch.int64),
            positions=torch.tensor([2, 6, 10]),
            causal=True,
            attn_state=None,
        )
        return builder, common

    return SimpleNamespace(inputs=inputs, tiling=tiling, lengths=lengths)


@pytest.mark.parametrize("cls", [dsa_v1.AscendDSAMetadataBuilder, dsa_v1.AscendDSASWAMetadataBuilder])
def test_draft_refresh_matches_build_and_preserves_buffers(draft_case, cls):
    builder, common = draft_case.inputs(cls)
    metadata = builder.build_for_cudagraph_capture(common, common_ratio_to_sas_metadata={})
    req = metadata.req_metadata
    tensors = (req.cos["layer"], req.sin["layer"], req.slot_mapping, req.sas_metadata)
    pointers = [tensor.data_ptr() for tensor in tensors]

    # Shrink the active batch and grow it again within the captured capacity.
    # The lengths cross both block and SWA window boundaries.
    for seq_lens, positions, slots, query_start in [
        ([4, 0, 0], [3, 0, 0], [7, -1, -1], [0, 1, 1, 1]),
        ([9, 5, 16], [8, 4, 15], [12, 20, 47], [0, 1, 2, 3]),
    ]:
        common.seq_lens.copy_(torch.tensor(seq_lens))
        common._seq_lens_cpu.copy_(torch.tensor(seq_lens))
        common.positions.copy_(torch.tensor(positions))
        common.slot_mapping.copy_(torch.tensor(slots))
        common.query_start_loc.copy_(torch.tensor(query_start))
        common.query_start_loc_cpu.copy_(torch.tensor(query_start))
        draft_case.tiling.reset_mock()
        draft_case.lengths.reset_mock()
        builder.update_draft_decode_metadata(metadata)
        draft_case.tiling.assert_called_once()
        draft_case.lengths.assert_called_once()

        fresh, _ = draft_case.inputs(cls)
        expected = fresh.build_for_cudagraph_capture(common, common_ratio_to_sas_metadata={}).req_metadata
        for actual, reference in zip(
            tensors,
            (expected.cos["layer"], expected.sin["layer"], expected.slot_mapping, expected.sas_metadata),
            strict=True,
        ):
            torch.testing.assert_close(actual, reference)
        assert [tensor.data_ptr() for tensor in tensors] == pointers
        assert req.positions.data_ptr() == common.positions.data_ptr()
        assert req.raw_slot_mapping.data_ptr() == common.slot_mapping.data_ptr()


def test_draft_refresh_does_not_overwrite_other_builder(draft_case):
    target, target_common = draft_case.inputs(dsa_v1.AscendDSASWAMetadataBuilder)
    draft, draft_common = draft_case.inputs(dsa_v1.AscendDSASWAMetadataBuilder)
    target_req = target.build_for_cudagraph_capture(target_common, common_ratio_to_sas_metadata={}).req_metadata
    draft_metadata = draft.build_for_cudagraph_capture(draft_common, common_ratio_to_sas_metadata={})
    target_tensors = (
        target_req.cos["layer"],
        target_req.sin["layer"],
        target_req.slot_mapping,
        target_req.sas_metadata,
    )
    saved = [tensor.clone() for tensor in target_tensors]
    draft_common.positions.add_(1)
    draft_common.seq_lens.add_(1)
    draft_common.slot_mapping.add_(1)
    draft.update_draft_decode_metadata(draft_metadata)
    for actual, reference in zip(target_tensors, saved, strict=True):
        torch.testing.assert_close(actual, reference)


def test_draft_refresh_updates_shared_sas_output(draft_case):
    first, common = draft_case.inputs(dsa_v1.AscendDSASWAMetadataBuilder)
    second, _ = draft_case.inputs(dsa_v1.AscendDSASWAMetadataBuilder)
    shared = {}
    first_req = first.build_for_cudagraph_capture(common, common_ratio_to_sas_metadata=shared).req_metadata
    second_metadata = second.build_for_cudagraph_capture(common, common_ratio_to_sas_metadata=shared)
    second_req = second_metadata.req_metadata
    assert first_req.sas_metadata is second_req.sas_metadata
    pointer = second_req.sas_metadata.data_ptr()
    common.seq_lens.add_(1)
    second.update_draft_decode_metadata(second_metadata)
    torch.testing.assert_close(second_req.sas_metadata[:3], common.seq_lens)
    assert second_req.sas_metadata.data_ptr() == pointer


def test_empty_draft_refresh_does_no_device_work(draft_case):
    builder, _ = draft_case.inputs(dsa_v1.AscendDSASWAMetadataBuilder)
    builder.update_draft_decode_metadata(SimpleNamespace(num_prefills=0, num_decode_tokens=0))
    draft_case.tiling.assert_not_called()
    draft_case.lengths.assert_not_called()
