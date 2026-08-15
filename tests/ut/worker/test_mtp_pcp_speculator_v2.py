# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.spec_decode.autoregressive import (
    speculator as speculator_module,
)
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import (
    AscendEagleSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
    AscendMTPSpeculator,
)


@pytest.mark.parametrize(
    ("method", "speculator_cls", "parent_cls"),
    [
        ("mtp", AscendMTPSpeculator, MTPSpeculator),
        ("eagle3", AscendEagleSpeculator, EagleSpeculator),
    ],
)
def test_speculator_rebuilds_global_pcp_attention(
    method: str,
    speculator_cls,
    parent_cls,
) -> None:
    speculator = object.__new__(speculator_cls)
    speculator.method = method
    speculator.input_batch = None
    speculator.pcp_manager = MagicMock()
    speculator.model_state = MagicMock()
    speculator.attn_groups = MagicMock()
    speculator.target_attn_groups = MagicMock()
    speculator.kv_cache_config = MagicMock()

    input_batch = MagicMock()
    local_attn_metadata = MagicMock()
    local_slot_mappings = MagicMock()
    global_block_tables = (MagicMock(),)
    global_slot_mapping = torch.arange(4).unsqueeze(0)
    global_attn_metadata = MagicMock()
    global_slot_mappings = MagicMock()
    local_aux_hidden_states = [
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
    ]
    global_aux_hidden_states = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    aux_hidden_states = local_aux_hidden_states if method == "eagle3" else None

    speculator.pcp_manager.prepare_speculator_attn.return_value = (
        global_block_tables,
        global_slot_mapping,
    )
    speculator.pcp_manager.restore_hidden_states.return_value = global_aux_hidden_states
    speculator.model_state.prepare_attn.return_value = global_attn_metadata

    with (
        patch.object(parent_cls, "propose", return_value=MagicMock()) as propose,
        patch.object(
            speculator_module,
            "build_slot_mappings_by_layer",
            return_value=global_slot_mappings,
        ) as build_slot_mappings,
    ):
        speculator_cls.propose(
            speculator,
            input_batch,
            local_attn_metadata,
            local_slot_mappings,
            MagicMock(),
            aux_hidden_states,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    speculator.pcp_manager.prepare_speculator_attn.assert_called_once_with(input_batch)
    speculator.model_state.prepare_attn.assert_called_once_with(
        input_batch,
        CUDAGraphMode.NONE,
        global_block_tables,
        global_slot_mapping,
        speculator.target_attn_groups,
        speculator.kv_cache_config,
    )
    build_slot_mappings.assert_called_once_with(
        global_slot_mapping,
        speculator.kv_cache_config,
    )

    propose_args = propose.call_args.args
    assert propose_args[0] is input_batch
    assert propose_args[1] is global_attn_metadata
    assert propose_args[2] is global_slot_mappings

    if method == "eagle3":
        restore_args = speculator.pcp_manager.restore_hidden_states.call_args.args
        torch.testing.assert_close(
            restore_args[0],
            torch.cat(local_aux_hidden_states, dim=-1),
        )
        assert len(propose_args[4]) == 1
        assert propose_args[4][0] is global_aux_hidden_states
    else:
        speculator.pcp_manager.restore_hidden_states.assert_not_called()
        assert propose_args[4] is None


@pytest.mark.parametrize(
    "speculator_cls",
    [AscendMTPSpeculator, AscendEagleSpeculator],
)
def test_pcp_draft_prefill_rebuilds_metadata_for_selected_full_graph(
    speculator_cls,
) -> None:
    speculator = object.__new__(speculator_cls)
    input_batch = MagicMock()
    input_batch.num_reqs_after_padding = 4
    input_batch.num_tokens_after_padding = 8
    speculator.input_batch = input_batch

    block_tables = (MagicMock(),)
    slot_mappings = torch.arange(16).unsqueeze(0)
    speculator._pcp_draft_prefill_attn_inputs = (
        block_tables,
        slot_mappings,
    )
    speculator.model_state = MagicMock()
    speculator.model_state.attn_metadata = {"target": object()}
    speculator.target_attn_groups = MagicMock()
    speculator.kv_cache_config = MagicMock()
    speculator.draft_attn_layer_names = {"draft"}

    draft_metadata = object()
    speculator.model_state.prepare_attn.return_value = {
        "draft": draft_metadata,
        "target": object(),
    }
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=4,
        uniform_token_count=2,
    )

    result = speculator.build_draft_attn_metadatas(
        batch_desc,
        is_draft_model_prefill=True,
    )

    speculator.model_state.prepare_attn.assert_called_once_with(
        input_batch,
        CUDAGraphMode.FULL,
        block_tables,
        slot_mappings,
        speculator.target_attn_groups,
        speculator.kv_cache_config,
    )
    assert result == [{"draft": draft_metadata}]


def test_pcp_draft_prefill_rejects_mismatched_graph_shape() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    input_batch = MagicMock()
    input_batch.num_reqs_after_padding = 4
    input_batch.num_tokens_after_padding = 8
    speculator.input_batch = input_batch
    speculator._pcp_draft_prefill_attn_inputs = (
        (MagicMock(),),
        MagicMock(),
    )
    speculator.model_state = MagicMock()
    speculator.model_state.attn_metadata = {"draft": object()}
    speculator.draft_attn_layer_names = {"draft"}

    mismatched_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=6,
        num_reqs=3,
        uniform_token_count=2,
    )

    with pytest.raises(
        RuntimeError,
        match="graph shape does not match",
    ):
        speculator.build_draft_attn_metadatas(
            mismatched_desc,
            is_draft_model_prefill=True,
        )

    speculator.model_state.prepare_attn.assert_not_called()


def test_eagle3_pcp_dummy_run_keeps_local_inputs() -> None:
    speculator = object.__new__(AscendEagleSpeculator)
    speculator.method = "eagle3"
    speculator.input_batch = None
    speculator.pcp_manager = MagicMock()

    input_batch = MagicMock()
    local_attn_metadata = MagicMock()
    local_slot_mappings = MagicMock()
    local_aux_hidden_states = [MagicMock(), MagicMock()]

    with (
        patch.object(
            EagleSpeculator,
            "propose",
            return_value=MagicMock(),
        ) as propose,
        patch.object(
            speculator_module,
            "build_slot_mappings_by_layer",
        ) as build_slot_mappings,
    ):
        AscendEagleSpeculator.propose(
            speculator,
            input_batch,
            local_attn_metadata,
            local_slot_mappings,
            MagicMock(),
            local_aux_hidden_states,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            dummy_run=True,
        )

    speculator.pcp_manager.prepare_speculator_attn.assert_not_called()
    speculator.pcp_manager.restore_hidden_states.assert_not_called()
    build_slot_mappings.assert_not_called()

    propose_args = propose.call_args.args
    assert propose_args[1] is local_attn_metadata
    assert propose_args[2] is local_slot_mappings
    assert propose_args[4] is local_aux_hidden_states


def test_eagle3_pcp_requires_aux_hidden_states() -> None:
    speculator = object.__new__(AscendEagleSpeculator)
    speculator.method = "eagle3"
    speculator.input_batch = None
    speculator.pcp_manager = MagicMock()
    speculator.model_state = MagicMock()
    speculator.attn_groups = MagicMock()
    speculator.target_attn_groups = MagicMock()
    speculator.kv_cache_config = MagicMock()

    global_slot_mapping = torch.arange(4).unsqueeze(0)
    speculator.pcp_manager.prepare_speculator_attn.return_value = (
        (MagicMock(),),
        global_slot_mapping,
    )

    with (
        patch.object(EagleSpeculator, "propose") as propose,
        patch.object(speculator_module, "build_slot_mappings_by_layer"),
        pytest.raises(
            RuntimeError,
            match="requires auxiliary target hidden states",
        ),
    ):
        AscendEagleSpeculator.propose(
            speculator,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            None,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    propose.assert_not_called()


def test_mtp_without_pcp_keeps_existing_proposal_inputs() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.method = "mtp"
    speculator.input_batch = None
    speculator.pcp_manager = None

    input_batch = MagicMock()
    local_attn_metadata = MagicMock()
    local_slot_mappings = MagicMock()

    with (
        patch.object(
            MTPSpeculator,
            "propose",
            return_value=MagicMock(),
        ) as propose,
        patch.object(
            speculator_module,
            "build_slot_mappings_by_layer",
        ) as build_slot_mappings,
    ):
        AscendMTPSpeculator.propose(
            speculator,
            input_batch,
            local_attn_metadata,
            local_slot_mappings,
            MagicMock(),
            None,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    build_slot_mappings.assert_not_called()
    propose_args = propose.call_args.args
    assert propose_args[1] is local_attn_metadata
    assert propose_args[2] is local_slot_mappings
