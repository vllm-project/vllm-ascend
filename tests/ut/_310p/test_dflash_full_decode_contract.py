# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend._310p.dflash_full_decode_contract import (
    FullDecodeContractInventoryError,
    build_draft_full_decode_contract_sources,
    build_target_full_decode_contract_sources,
)
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _metadata(seed: int = 0):
    return SimpleNamespace(
        attn_mask=torch.full((2, 2), seed, dtype=torch.bool),
        seq_lens=torch.tensor([seed + 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        block_tables=torch.tensor([[seed]], dtype=torch.int32),
        slot_mapping=torch.tensor([seed], dtype=torch.int32),
        decode_meta=None,
        prefill=None,
        kvcomp_metadata=None,
    )


def _graph_params():
    attention_params = (
        torch.ones(1),  # query
        torch.ones(1),  # key cache
        torch.ones(1),  # value cache
        torch.ones(1, dtype=torch.int32),  # block table
        torch.ones(1, dtype=torch.bool),  # attention mask
        128,
        [1],
        [1],
        1,
        1,
        1.0,
        torch.ones(1),  # output
        torch.ones(1),  # softmax lse
        3,
        2**31 - 1,
        2**31 - 1,
        None,
        None,
        None,
        None,
        "model.layers.0.self_attn.attn",
    )
    return SimpleNamespace(
        workspaces={16: torch.ones(32)},
        attn_params={16: [attention_params]},
    )


def test_target_provider_exposes_call_independent_retained_roles():
    linear_metadata = _metadata()
    linear_metadata.spec_query_start_loc = torch.tensor([0, 16], dtype=torch.int32)
    context = SimpleNamespace(
        attn_metadata={
            "model.layers.0.self_attn.attn": _metadata(),
            "model.layers.1.linear_attn": linear_metadata,
        },
        input_ids=torch.ones(16, dtype=torch.int32),
        num_tokens_across_dp=None,
        mc2_mask=None,
    )
    runner_buffers = {
        "positions": torch.arange(16),
        "query_start_loc": torch.tensor([0, 16], dtype=torch.int32),
        "seq_lens": torch.tensor([16], dtype=torch.int32),
    }

    sources = build_target_full_decode_contract_sources(
        forward_context=context,
        graph_params=_graph_params(),
        runner_buffers=runner_buffers,
        descriptor_num_tokens=16,
    )
    roles = {source.role for source in sources}

    assert {
        "target.forward.input_ids",
        "target.attention.model.layers.0.self_attn.attn.attn_mask",
        "target.attention.model.layers.0.self_attn.attn.seq_lens",
        "target.attention.model.layers.0.self_attn.attn.query_start_loc",
        "target.attention.model.layers.0.self_attn.attn.block_tables",
        "target.attention.model.layers.0.self_attn.attn.slot_mapping",
        "target.attention.model.layers.1.linear_attn.spec_query_start_loc",
        "target.runner.positions",
        "target.runner.query_start_loc",
        "target.runner.seq_lens",
        "target.graph.workspace",
        "target.graph.attention.0.query",
        "target.graph.attention.0.key_cache",
        "target.graph.attention.0.value_cache",
        "target.graph.attention.0.block_table",
        "target.graph.attention.0.attention_mask",
        "target.graph.attention.0.output",
        "target.graph.attention.0.softmax_lse",
    } <= roles
    assert all(source.ownership for source in sources)
    assert all(source.alignment_source for source in sources)


def test_draft_provider_exposes_each_step_and_proposer_buffer():
    context = SimpleNamespace(
        attn_metadata={"draft.layers.0.self_attn.attn": _metadata()},
        draft_attn_metadatas=[
            {"draft.layers.0.self_attn.attn": _metadata(0)},
            {"draft.layers.0.self_attn.attn": _metadata(1)},
        ],
        input_ids=None,
        num_tokens_across_dp=None,
        mc2_mask=None,
    )
    proposer_buffers = {
        "input_ids": torch.ones(16, dtype=torch.int32),
        "positions": torch.arange(16, dtype=torch.int32),
        "hidden_states": torch.ones((16, 8)),
        "token_indices_to_sample": torch.tensor([15], dtype=torch.int32),
        "block_table": torch.ones((1, 8), dtype=torch.int32),
    }

    sources = build_draft_full_decode_contract_sources(
        forward_context=context,
        graph_params=_graph_params(),
        proposer_buffers=proposer_buffers,
        descriptor_num_tokens=16,
    )
    roles = {source.role for source in sources}

    assert {
        "draft.attention.step0.draft.layers.0.self_attn.attn.block_tables",
        "draft.attention.step1.draft.layers.0.self_attn.attn.block_tables",
        "draft.proposer.input_ids",
        "draft.proposer.positions",
        "draft.proposer.hidden_states",
        "draft.proposer.token_indices_to_sample",
        "draft.proposer.block_table",
        "draft.graph.workspace",
    } <= roles


def test_provider_rejects_uninventoried_device_tensor():
    metadata = _metadata()
    metadata.new_graph_tensor = torch.ones(1)
    context = SimpleNamespace(
        attn_metadata={"layer": metadata},
        input_ids=torch.ones(16, dtype=torch.int32),
        num_tokens_across_dp=None,
        mc2_mask=None,
    )

    with pytest.raises(
        FullDecodeContractInventoryError,
        match="new_graph_tensor",
    ):
        build_target_full_decode_contract_sources(
            forward_context=context,
            graph_params=_graph_params(),
            runner_buffers={},
            descriptor_num_tokens=16,
        )


def test_provider_does_not_require_workspace_without_graph_attention_tasks():
    context = SimpleNamespace(
        attn_metadata={"linear_attn": _metadata()},
        input_ids=torch.ones(16, dtype=torch.int32),
        num_tokens_across_dp=None,
        mc2_mask=None,
    )
    graph_params = SimpleNamespace(
        workspaces={16: None},
        attn_params={16: []},
    )

    sources = build_target_full_decode_contract_sources(
        forward_context=context,
        graph_params=graph_params,
        runner_buffers={},
        descriptor_num_tokens=16,
    )

    assert all(source.role != "target.graph.workspace" for source in sources)


def test_target_runner_provider_passes_persistent_buffers_and_graph_params():
    tensor = torch.ones(16)
    runner = object.__new__(NPUModelRunner)
    runner.input_ids = SimpleNamespace(gpu=tensor)
    runner.positions = tensor
    runner.query_start_loc = SimpleNamespace(gpu=tensor)
    runner.seq_lens = tensor
    runner.group_len = SimpleNamespace(gpu=tensor)
    runner.group_key_idx = SimpleNamespace(gpu=tensor)
    runner.group_key_cache_idx = SimpleNamespace(gpu=tensor)
    descriptor = SimpleNamespace(num_tokens=16)
    context = SimpleNamespace()
    graph_params = SimpleNamespace()

    with (
        patch(
            "vllm_ascend.worker.model_runner_v1.get_graph_params",
            return_value=graph_params,
            create=True,
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1.build_target_full_decode_contract_sources",
            return_value=("target-contract",),
            create=True,
        ) as build_contract,
    ):
        result = runner._full_decode_target_retained_inputs(
            context,
            descriptor,
        )

    assert result == ("target-contract",)
    kwargs = build_contract.call_args.kwargs
    assert kwargs["forward_context"] is context
    assert kwargs["graph_params"] is graph_params
    assert kwargs["descriptor_num_tokens"] == 16
    assert set(kwargs["runner_buffers"]) == {
        "input_ids",
        "positions",
        "query_start_loc",
        "seq_lens",
        "group_len",
        "group_key_idx",
        "group_key_cache_idx",
    }


def test_draft_proposer_provider_passes_all_persistent_buffers():
    tensor = torch.ones(16)
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.input_ids = tensor
    proposer.positions = tensor
    proposer.hidden_states = tensor
    proposer.token_indices_to_sample = tensor
    proposer.arange = tensor
    proposer.block_table_tensor_clone = tensor
    proposer.is_rejected_token_mask = tensor
    proposer.is_masked_token_mask = tensor
    proposer.inputs_embeds = tensor
    proposer._full_decode_draft_query_rope_cos_310 = tensor
    proposer._full_decode_draft_query_rope_sin_310 = tensor
    proposer._full_decode_draft_context_rope_cos_310 = tensor
    proposer._full_decode_draft_context_rope_sin_310 = tensor
    descriptor = SimpleNamespace(num_tokens=16)
    context = SimpleNamespace()
    graph_params = SimpleNamespace()

    with (
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.get_draft_graph_params",
            return_value=graph_params,
            create=True,
        ),
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.build_draft_full_decode_contract_sources",
            return_value=("draft-contract",),
            create=True,
        ) as build_contract,
    ):
        result = proposer._full_decode_draft_retained_inputs(
            context,
            descriptor,
        )

    assert result == ("draft-contract",)
    kwargs = build_contract.call_args.kwargs
    assert kwargs["forward_context"] is context
    assert kwargs["graph_params"] is graph_params
    assert kwargs["descriptor_num_tokens"] == 16
    assert set(kwargs["proposer_buffers"]) == {
        "input_ids",
        "positions",
        "hidden_states",
        "token_indices_to_sample",
        "arange",
        "block_table",
        "is_rejected_token_mask",
        "is_masked_token_mask",
        "inputs_embeds",
        "draft_query_rope_cos_310",
        "draft_query_rope_sin_310",
        "draft_context_rope_cos_310",
        "draft_context_rope_sin_310",
    }


def test_draft_full_metadata_binds_persistent_step_buffers():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace()
    proposer.seq_lens_group = [torch.zeros(8, dtype=torch.int32)]
    proposer.query_start_loc_group = [torch.zeros(9, dtype=torch.int32)]
    proposer.slot_mapping_group = [torch.zeros(16, dtype=torch.int32)]
    common = SimpleNamespace(
        seq_lens=proposer.seq_lens_group[0][:1],
        query_start_loc=proposer.query_start_loc_group[0][:2],
        slot_mapping=proposer.slot_mapping_group[0],
    )
    metadata = SimpleNamespace(
        seq_lens=torch.ones(1, dtype=torch.int32),
        query_start_loc=torch.ones(2, dtype=torch.int32),
        slot_mapping=torch.ones(16, dtype=torch.int32),
    )

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.is_310p_dflash_full_decode_only",
        return_value=True,
    ):
        proposer._bind_full_decode_draft_attention_buffers(
            metadata,
            common,
            draft_index=0,
            runtime_mode=CUDAGraphMode.FULL,
        )

    assert metadata.seq_lens.data_ptr() == common.seq_lens.data_ptr()
    assert metadata.query_start_loc.data_ptr() == (common.query_start_loc.data_ptr())
    assert metadata.slot_mapping.data_ptr() == common.slot_mapping.data_ptr()


def test_draft_metadata_binding_is_inactive_outside_full_runtime():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace()
    original = torch.ones(1, dtype=torch.int32)
    metadata = SimpleNamespace(seq_lens=original)
    common = SimpleNamespace(seq_lens=torch.zeros(1, dtype=torch.int32))

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.is_310p_dflash_full_decode_only",
        return_value=True,
    ):
        proposer._bind_full_decode_draft_attention_buffers(
            metadata,
            common,
            draft_index=0,
            runtime_mode=CUDAGraphMode.NONE,
        )

    assert metadata.seq_lens is original


def test_dflash_dummy_capture_uses_runtime_persistent_common_buffers():
    proposer = object.__new__(AscendDflashProposer)
    proposer.vllm_config = SimpleNamespace()
    proposer.seq_lens_group = [torch.zeros(8, dtype=torch.int32)]
    proposer.query_start_loc_group = [torch.zeros(9, dtype=torch.int32)]
    proposer.slot_mapping_group = [torch.zeros(32, dtype=torch.int32)]
    original_seq_lens = torch.tensor([17], dtype=torch.int32)
    original_query_start_loc = torch.tensor([0, 16], dtype=torch.int32)
    original_slot_mapping = torch.arange(16, dtype=torch.int32)
    common = SimpleNamespace(
        seq_lens=original_seq_lens,
        query_start_loc=original_query_start_loc,
        slot_mapping=original_slot_mapping,
    )

    with patch(
        "vllm_ascend.spec_decode.dflash_proposer.is_310p_dflash_full_decode_only",
        return_value=True,
        create=True,
    ):
        proposer._bind_full_decode_dummy_common_buffers(
            common,
            num_reqs=1,
            num_query_total=16,
            runtime_mode=CUDAGraphMode.FULL,
        )

    assert common.seq_lens.data_ptr() == (proposer.seq_lens_group[0].data_ptr())
    assert common.query_start_loc.data_ptr() == (proposer.query_start_loc_group[0].data_ptr())
    assert common.slot_mapping.data_ptr() == (proposer.slot_mapping_group[0].data_ptr())
    assert torch.equal(common.seq_lens, original_seq_lens)
    assert torch.equal(common.query_start_loc, original_query_start_loc)
    assert torch.equal(common.slot_mapping, original_slot_mapping)


def test_full_decode_slot_mapping_uses_descriptor_bounded_view():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace()
    buffer = torch.arange(128, dtype=torch.int32)

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.is_310p_dflash_full_decode_only",
        return_value=True,
    ):
        selected = proposer._select_full_decode_slot_mapping(
            buffer,
            logical_tokens=16,
            runtime_mode=CUDAGraphMode.FULL,
        )

    assert selected.data_ptr() == buffer.data_ptr()
    assert selected.shape == (16,)


def test_slot_mapping_view_is_unchanged_outside_full_runtime():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace()
    buffer = torch.arange(128, dtype=torch.int32)

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.is_310p_dflash_full_decode_only",
        return_value=True,
    ):
        selected = proposer._select_full_decode_slot_mapping(
            buffer,
            logical_tokens=16,
            runtime_mode=CUDAGraphMode.NONE,
        )

    assert selected is buffer


def test_draft_sample_indices_keep_descriptor_view_after_batch_shrinks():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace()
    proposer.num_speculative_tokens = 15
    proposer.token_indices_to_sample = torch.full((128,), -1, dtype=torch.int32)
    runtime_indices = torch.arange(45, dtype=torch.int32)

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.is_310p_dflash_full_decode_only",
        return_value=True,
    ):
        selected = proposer._select_full_decode_token_indices_to_sample(
            runtime_indices,
            graph_num_reqs=4,
            runtime_mode=CUDAGraphMode.FULL,
        )

    assert selected.data_ptr() == proposer.token_indices_to_sample.data_ptr()
    assert selected.shape == (60,)
    assert torch.equal(selected[:45], runtime_indices)
    assert torch.count_nonzero(selected[45:]) == 0


def test_draft_sample_indices_are_unchanged_outside_full_runtime():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace()
    proposer.num_speculative_tokens = 15
    proposer.token_indices_to_sample = torch.zeros(128, dtype=torch.int32)
    runtime_indices = torch.arange(45, dtype=torch.int32)

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.is_310p_dflash_full_decode_only",
        return_value=True,
    ):
        selected = proposer._select_full_decode_token_indices_to_sample(
            runtime_indices,
            graph_num_reqs=4,
            runtime_mode=CUDAGraphMode.NONE,
        )

    assert selected is runtime_indices
