# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft metadata with a full-history cache and auxiliary cache backends."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

MAIN, INDEXER, STATE = "draft.attention", "draft.indexer", "draft.state"


def _specs(block_size=256):
    return [SimpleNamespace(block_size=n) for n in (block_size, block_size, 8)]


def _table(values, block_size):
    return SimpleNamespace(get_device_tensor=lambda: values, block_size=block_size)


def test_initialize_reuses_runner_groups_without_changing_target_builders():
    main, indexer, state = _specs()
    groups = [
        AttentionGroup(Mock(), [name, "target." + name], spec, gid, [Mock()])
        for name, spec, gid in [(STATE, state, 0), (INDEXER, indexer, 1), (MAIN, main, 1)]
    ]
    target_builders = [group.metadata_builders for group in groups]
    proposer = object.__new__(AscendEagleProposer)
    proposer.vllm_config, proposer.device = object(), torch.device("cpu")
    proposer.num_speculative_tokens = 3
    proposer._draft_attn_layer_names = {MAIN, INDEXER, STATE}
    proposer.runner = SimpleNamespace(
        attn_groups=[[groups[0]], groups[1:]],
        input_batch=SimpleNamespace(block_table=[_table(None, 8), _table(None, 128)]),
    )
    layers = {
        MAIN: SimpleNamespace(),
        INDEXER: SimpleNamespace(cache_role="indexer"),
        STATE: SimpleNamespace(cache_role="indexer_state"),
    }

    def create(group, config, device, **kwargs):
        group.metadata_builders = [Mock() for _ in range(kwargs["num_metadata_builders"])]

    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_layers_from_vllm_config", return_value=layers),
        patch.object(AttentionGroup, "create_metadata_builders", autospec=True, side_effect=create) as builders,
    ):
        proposer.initialize_attn_backend(object(), [128])
    assert proposer.kv_cache_gid == 1
    assert proposer.attn_layer_names[0] == MAIN
    assert proposer.block_size == proposer.kernel_block_size == 128
    assert len(proposer.draft_attn_groups) == 3
    assert all(call.kwargs["kernel_block_size"] is None for call in builders.call_args_list)
    for group in proposer.draft_attn_groups:
        original = next(g for g in groups if group.layer_names[0] in g.layer_names)
        assert group is not original
        assert group.kv_cache_spec is original.kv_cache_spec
        assert group.kv_cache_group_id == original.kv_cache_group_id
        assert len(group.metadata_builders) == (1 if group.layer_names == [MAIN] else 3)
        assert group.metadata_builders is not original.metadata_builders
    assert [group.metadata_builders for group in groups] == target_builders
    assert all(len(group.layer_names) == 2 for group in groups)


def test_regular_draft_keeps_upstream_initialization():
    proposer = object.__new__(AscendEagleProposer)
    proposer.vllm_config = object()
    proposer._draft_attn_layer_names = {MAIN}
    config, sizes = object(), [128]
    with (
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.get_layers_from_vllm_config",
            return_value={MAIN: SimpleNamespace()},
        ),
        patch.object(SpecDecodeBaseProposer, "initialize_attn_backend") as initialize,
    ):
        proposer.initialize_attn_backend(config, sizes)
    initialize.assert_called_once_with(config, sizes)


@pytest.mark.parametrize("positions", [[6, 7, 8, 9, 0], [2, 3, 4, 5, 0]])
def test_state_slots_follow_draft_positions_and_request_blocks_after_rollback(positions):
    _, _, spec = _specs()
    group = AttentionGroup(Mock(), [STATE], spec, 1)
    state_table = torch.tensor([[9, 90], [3, 30]], dtype=torch.int32)
    proposer = object.__new__(AscendEagleProposer)
    proposer.kv_cache_gid = 0
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=[None, _table(state_table, 8)]))
    common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=5,
        num_actual_tokens=4,
        positions=torch.tensor(positions),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        block_table_tensor=torch.tensor([[100, 101], [200, 201]]),
        slot_mapping=torch.tensor([1000, -1, 2000, 2001, -1]),
    )
    state_common = proposer._cache_group_common_metadata(common, group)
    assert state_common is not common
    result = state_common
    torch.testing.assert_close(result.block_table_tensor, state_table)
    torch.testing.assert_close(
        result.slot_mapping,
        torch.tensor(
            [
                9 * 8 + positions[0] % 8,
                -1,
                int(state_table[1, positions[2] // 8]) * 8 + positions[2] % 8,
                int(state_table[1, positions[3] // 8]) * 8 + positions[3] % 8,
                -1,
            ]
        ),
    )
    assert common.slot_mapping.tolist() == [1000, -1, 2000, 2001, -1]
    state_table.copy_(state_table.flip(0))
    reordered = proposer._cache_group_common_metadata(common, group)
    assert reordered.slot_mapping[0] == 3 * 8 + positions[0] % 8
    assert reordered.slot_mapping[2] == state_table[1, positions[2] // 8] * 8 + positions[2] % 8
    assert reordered.slot_mapping.data_ptr() != result.slot_mapping.data_ptr()


@pytest.mark.parametrize("rejected", [[0, 0], [2, 0], [3, 3]])
@pytest.mark.parametrize("num_input_tokens, max_model_len", [(8, 128), (12, 128), (12, 24)])
def test_base_propose_keeps_each_group_and_draft_step_at_the_accepted_endpoint(
    rejected, num_input_tokens, max_model_len
):
    main, indexer, state = _specs()
    groups = [
        AttentionGroup(Mock(), [MAIN], main, 0),
        AttentionGroup(Mock(), [INDEXER], indexer, 0),
        AttentionGroup(Mock(), [STATE], state, 1),
    ]

    def metadata(common, *args, **kwargs):
        return SimpleNamespace(
            num_prefills=0,
            seq_lens=common.seq_lens,
            positions=common.positions,
            slot_mapping=common.slot_mapping,
        )

    main_builder = Mock(
        build=Mock(side_effect=lambda _, common, *args: metadata(common)), build_for_drafting=Mock(side_effect=metadata)
    )
    groups[0].metadata_builders = [main_builder]
    for group in groups[1:]:
        group.metadata_builders = [
            Mock(
                build=Mock(side_effect=lambda _, common, *args: metadata(common)),
                build_for_drafting=Mock(side_effect=metadata),
            )
            for _ in range(3)
        ]
    proposer = object.__new__(AscendEagleProposer)
    proposer.kv_cache_gid = 0
    proposer.draft_attn_groups = groups
    proposer.attn_layer_names = [MAIN, INDEXER, STATE]
    state_table = torch.tensor([[9, 90, 900, 91], [3, 30, 300, 31]], dtype=torch.int32)
    proposer.runner = SimpleNamespace(
        get_model=Mock(return_value=None),
        dcp_manager=None,
        dynamic_eplb=False,
        input_batch=SimpleNamespace(lora_id_to_lora_request={}, block_table=[None, _table(state_table, 8)]),
        _sync_metadata_across_dp=Mock(return_value=(None, torch.tensor([num_input_tokens]), None)),
    )
    proposer.method = "mtp"
    proposer.dcp_size = 1
    proposer.dp_rank = 0
    proposer.use_cuda_graph = proposer.uses_mrope = proposer.supports_mm_inputs = False
    proposer.parallel_drafting = proposer.has_gdn = proposer.use_compress = False
    proposer.draft_window_size = proposer.sliding_window = None
    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(use_mla=True))
    proposer.block_size = 128
    proposer.max_model_len = max_model_len
    proposer.arange = torch.arange(9, dtype=torch.int32)
    proposer.token_arange_np = proposer.arange.numpy()
    proposer.slot_mapping_group = [torch.full((num_input_tokens,), -1, dtype=torch.int32) for _ in range(3)]
    proposer.seq_lens_group = [torch.zeros(2, dtype=torch.int32) for _ in range(3)]
    proposer.query_start_loc_group = [torch.zeros(3, dtype=torch.int32) for _ in range(3)]
    proposer.token_indices_to_sample = torch.zeros(2, dtype=torch.int32)
    proposer._pad_draft_buffers = Mock()
    # Reuse the same proposer for the next batch, with no retained rejection state.
    for rejects in (rejected, [0, 0]):
        for group in groups[1:]:
            for builder in group.metadata_builders:
                builder.build.reset_mock()
                builder.build_for_drafting.reset_mock()
        positions = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23], dtype=torch.int32)
        proposer.positions = torch.zeros(num_input_tokens, dtype=torch.int32)
        proposer.positions[:8].copy_(positions)
        indices = torch.tensor([3, 7]) - torch.tensor(rejects)
        lengths = torch.tensor([12, 24], dtype=torch.int32)
        common = SimpleNamespace(
            batch_size=lambda: 2,
            num_reqs=2,
            num_actual_tokens=8,
            num_input_tokens=8,
            positions=proposer.positions,
            seq_lens=lengths,
            seq_lens_cpu=lengths.clone(),
            _seq_lens_cpu=lengths.clone(),
            num_computed_tokens_cpu=lengths.clone(),
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
            block_table_tensor=torch.tensor([[4], [6]], dtype=torch.int32),
            slot_mapping=torch.tensor([520, 521, 522, 523, 788, 789, 790, 791], dtype=torch.int32),
        )
        proposer.set_inputs_first_pass = Mock(return_value=(8, indices, common, None))
        with (
            patch(
                "vllm_ascend.spec_decode.llm_base_proposer.set_ascend_forward_context",
                side_effect=RuntimeError("metadata ready"),
            ) as context,
            pytest.raises(RuntimeError, match="metadata ready"),
        ):
            AscendSpecDecodeBaseProposer._propose(
                proposer,
                3,
                target_token_ids=torch.ones(8, dtype=torch.int64),
                target_positions=positions,
                target_hidden_states=torch.ones(8, 2),
                next_token_ids=torch.ones(2, dtype=torch.int64),
                token_indices_to_sample=indices,
                common_attn_metadata=common,
                target_model_batch_desc=SimpleNamespace(uniform=True),
                sampling_metadata=None,
            )
        steps = context.call_args.kwargs["draft_attn_metadatas"]
        assert len(steps) == 3
        for step, per_layer in enumerate(steps):
            assert set(per_layer) == {MAIN, INDEXER, STATE}
            assert len({id(value) for value in per_layer.values()}) == 3
            raw_lengths = lengths - torch.tensor(rejects, dtype=lengths.dtype) + step
            expected = lengths if step == 0 else (raw_lengths - 1) % max_model_len + 1
            for value in per_layer.values():
                torch.testing.assert_close(value.seq_lens, expected)
                assert (value.slot_mapping[2 if step else 8 :] == -1).all()
            if step:
                expected_positions = torch.where(raw_lengths > max_model_len, 0, raw_lengths - 1)
                torch.testing.assert_close(per_layer[MAIN].positions[:2], expected_positions)
                for value in per_layer.values():
                    assert (value.slot_mapping[:2][raw_lengths > max_model_len] == -1).all()
        for group in groups[1:]:
            group.metadata_builders[0].build.assert_called_once()
            for builder in group.metadata_builders[1:]:
                builder.build_for_drafting.assert_called_once()
        for name in (MAIN, INDEXER, STATE):
            assert len({step[name].slot_mapping.data_ptr() for step in steps}) == 3
        torch.testing.assert_close(lengths, torch.tensor([12, 24], dtype=torch.int32))
        assert not hasattr(proposer, "_num_rejected_tokens")


def test_glm_mtp_uses_existing_eagle_proposer():
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next_text")),
        speculative_config=SimpleNamespace(use_step3p5_mtp=lambda: False),
    )
    with patch("vllm_ascend.spec_decode.AscendEagleProposer") as proposer:
        assert get_spec_decode_method("mtp", config, "cpu", None) is proposer.return_value
    proposer.assert_called_once_with(config, "cpu", None)


@pytest.mark.parametrize(
    "table, positions, num_reqs",
    [
        ([], [0, 0, 0], 0),
        ([[]], [0, 1, 2], 1),
        ([[5, -1]], [-1, 8, 16], 1),
    ],
)
def test_proposer_state_slots_mask_idle_and_invalid_pages(table, positions, num_reqs):
    _, _, spec = _specs()
    group = AttentionGroup(Mock(), [STATE], spec, 0)
    state_table = torch.tensor(table, dtype=torch.int32).reshape(
        num_reqs, 0 if not table or not table[0] else len(table[0])
    )
    proposer = object.__new__(AscendEagleProposer)
    proposer.kv_cache_gid = 1
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=[_table(state_table, 8)]))
    common = SimpleNamespace(
        num_reqs=num_reqs,
        num_input_tokens=3,
        positions=torch.tensor(positions),
        query_start_loc=torch.tensor([0, 3][: num_reqs + 1]),
        slot_mapping=torch.tensor([0, 1, 2]),
    )
    metadata = proposer._cache_group_common_metadata(common, group)
    assert metadata.slot_mapping.tolist() == [-1, -1, -1]
    assert common.slot_mapping.tolist() == [0, 1, 2]
