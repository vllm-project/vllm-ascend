# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
)
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.autoregressive import (
    speculator as speculator_module,
)
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import (
    AscendEagleSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
    AscendMTPSpeculator,
)


def _set_draft_pcp_size(speculator, pcp_size: int) -> None:
    config = MagicMock()
    config.parallel_config.prefill_context_parallel_size = pcp_size
    speculator.vllm_config = config


def _make_padded_input_batch() -> MagicMock:
    input_batch = MagicMock(spec=AscendInputBatch)
    input_batch.num_reqs = 2
    input_batch.num_reqs_after_padding = 4
    input_batch.num_tokens = 6
    input_batch.num_tokens_after_padding = 8
    input_batch.query_start_loc = torch.arange(5, dtype=torch.int32)
    input_batch.query_start_loc_np = np.arange(5, dtype=np.int32)
    input_batch.seq_lens = torch.arange(4, dtype=torch.int32)
    input_batch.seq_lens_cpu_upper_bound = torch.arange(4, dtype=torch.int32)
    input_batch.input_ids = torch.arange(8, dtype=torch.int32)
    input_batch.positions = torch.arange(8, dtype=torch.int64)
    input_batch.is_padding = torch.zeros(8, dtype=torch.bool)
    input_batch.seq_lens_np = np.arange(4, dtype=np.int32)
    return input_batch


def test_mtp_uses_draft_groups_only_when_target_pcp_is_active() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.attn_groups = MagicMock()
    speculator.target_attn_groups = MagicMock()
    _set_draft_pcp_size(speculator, 1)

    speculator.pcp_manager = None
    assert speculator.draft_prefill_attn_groups is speculator.target_attn_groups

    speculator.pcp_manager = MagicMock()
    assert speculator.draft_prefill_attn_groups is speculator.attn_groups


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_pcp_draft_uses_pcp1_eager_config(method: str) -> None:
    vllm_config = MagicMock()
    target_compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    oot_compiler = object()
    target_compilation_config.oot_compiler = oot_compiler
    vllm_config.compilation_config = target_compilation_config
    vllm_config.speculative_config.method = method
    parallel_config = vllm_config.parallel_config
    parallel_config.prefill_context_parallel_size = 2
    draft_parallel_config = MagicMock()
    draft_vllm_config = MagicMock()
    device = MagicMock()

    with (
        patch.object(
            speculator_module.AutoRegressiveSpeculator,
            "__init__",
            side_effect=RuntimeError("stop after config"),
        ) as base_init,
        patch.object(
            speculator_module,
            "replace",
            side_effect=(
                draft_parallel_config,
                draft_vllm_config,
            ),
        ) as replace_config,
        pytest.raises(RuntimeError, match="stop after config"),
    ):
        speculator = object.__new__(AscendMTPSpeculator)
        speculator_module.AscendAutoRegressiveSpeculator.__init__(
            speculator,
            vllm_config,
            device,
        )

    base_init.assert_called_once_with(draft_vllm_config, device)
    assert replace_config.call_args_list[0] == call(
        parallel_config,
        prefill_context_parallel_size=1,
    )
    draft_config_call = replace_config.call_args_list[1]
    assert draft_config_call.args == (vllm_config,)
    assert draft_config_call.kwargs["parallel_config"] is draft_parallel_config
    draft_compilation_config = draft_config_call.kwargs["compilation_config"]
    assert draft_compilation_config is not target_compilation_config
    assert draft_compilation_config.mode == CompilationMode.NONE
    assert draft_compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert draft_compilation_config.oot_compiler is oot_compiler

    assert target_compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert (
        target_compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_DECODE_ONLY
    )


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_non_pcp_draft_reuses_target_config(method: str) -> None:
    vllm_config = MagicMock()
    vllm_config.speculative_config.method = method
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    device = MagicMock()

    with (
        patch.object(
            speculator_module.AutoRegressiveSpeculator,
            "__init__",
            side_effect=RuntimeError("stop after config"),
        ) as base_init,
        patch.object(speculator_module, "replace") as replace_config,
        pytest.raises(RuntimeError, match="stop after config"),
    ):
        speculator = object.__new__(AscendMTPSpeculator)
        speculator_module.AscendAutoRegressiveSpeculator.__init__(
            speculator,
            vllm_config,
            device,
        )

    replace_config.assert_not_called()
    base_init.assert_called_once_with(vllm_config, device)


@pytest.mark.parametrize(
    ("draft_mode", "requested_mode", "expected_mode"),
    [
        (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE),
        (
            CUDAGraphMode.FULL_DECODE_ONLY,
            CUDAGraphMode.FULL_DECODE_ONLY,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ),
    ],
)
def test_autoregressive_graph_manager_respects_draft_config(
    draft_mode: CUDAGraphMode,
    requested_mode: CUDAGraphMode,
    expected_mode: CUDAGraphMode,
) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.vllm_config = MagicMock()
    speculator.vllm_config.compilation_config.cudagraph_mode = draft_mode
    speculator.prefill_cudagraph_manager = MagicMock()
    speculator.decode_cudagraph_manager = MagicMock()
    speculator.update_stream = MagicMock()

    with patch.object(
        speculator_module.AutoRegressiveSpeculator,
        "init_cudagraph_manager",
    ) as parent_init:
        speculator.init_cudagraph_manager(requested_mode)

    parent_init.assert_called_once_with(expected_mode)


def test_set_attn_builds_metadata_under_draft_config() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    draft_vllm_config = MagicMock()
    speculator.vllm_config = draft_vllm_config
    speculator.draft_attn_layer_names = set()

    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = []
    config_context = MagicMock()
    model_state = MagicMock()
    block_tables = MagicMock()
    target_input_buffers = MagicMock()
    target_attn_groups = MagicMock()

    with (
        patch.object(
            speculator_module,
            "set_current_vllm_config",
            return_value=config_context,
        ) as set_current_config,
        patch.object(
            speculator_module.AutoRegressiveSpeculator,
            "set_attn",
        ) as parent_set_attn,
    ):
        speculator.set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

    set_current_config.assert_called_once_with(draft_vllm_config)
    config_context.__enter__.assert_called_once_with()
    config_context.__exit__.assert_called_once()
    parent_set_attn.assert_called_once_with(
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    )
    assert speculator.attn_backends == {}


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
    _set_draft_pcp_size(speculator, 1)
    expected_attn_groups = speculator.attn_groups

    input_batch = _make_padded_input_batch()
    eager_batch = MagicMock()
    last_hidden_states = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    expected_input_batch = eager_batch
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
    global_aux_hidden_states = torch.arange(56, dtype=torch.float32).reshape(8, 7)
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
        patch.object(
            speculator_module,
            "replace",
            return_value=eager_batch,
        ) as replace_batch,
    ):
        speculator_cls.propose(
            speculator,
            input_batch,
            local_attn_metadata,
            local_slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    speculator.pcp_manager.prepare_speculator_attn.assert_called_once_with(
        input_batch
    )
    speculator.model_state.prepare_attn.assert_called_once_with(
        expected_input_batch,
        CUDAGraphMode.NONE,
        global_block_tables,
        global_slot_mapping,
        expected_attn_groups,
        speculator.kv_cache_config,
    )
    build_slot_mappings.assert_called_once_with(
        global_slot_mapping,
        speculator.kv_cache_config,
    )

    propose_args = propose.call_args.args
    assert propose_args[0] is expected_input_batch
    assert propose_args[1] is global_attn_metadata
    assert propose_args[2] is global_slot_mappings

    replace_batch.assert_called_once()
    assert replace_batch.call_args.kwargs["num_reqs_after_padding"] == 2
    assert replace_batch.call_args.kwargs["num_tokens_after_padding"] == 6
    torch.testing.assert_close(
        propose_args[3],
        last_hidden_states[:6],
    )

    if method == "eagle3":
        restore_args = speculator.pcp_manager.restore_hidden_states.call_args.args
        torch.testing.assert_close(
            restore_args[0],
            torch.cat(local_aux_hidden_states, dim=-1),
        )
        assert len(propose_args[4]) == 1
        torch.testing.assert_close(
            propose_args[4][0],
            global_aux_hidden_states[:6],
        )
    else:
        speculator.pcp_manager.restore_hidden_states.assert_not_called()
        assert propose_args[4] is None


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

    _set_draft_pcp_size(speculator, 1)
    input_batch = _make_padded_input_batch()
    eager_batch = MagicMock()

    global_slot_mapping = torch.arange(4).unsqueeze(0)
    speculator.pcp_manager.prepare_speculator_attn.return_value = (
        (MagicMock(),),
        global_slot_mapping,
    )

    with (
        patch.object(EagleSpeculator, "propose") as propose,
        patch.object(speculator_module, "build_slot_mappings_by_layer"),
        patch.object(
            speculator_module,
            "replace",
            return_value=eager_batch,
        ),
        pytest.raises(
            RuntimeError,
            match="requires auxiliary target hidden states",
        ),
    ):
        AscendEagleSpeculator.propose(
            speculator,
            input_batch,
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


@pytest.mark.parametrize(
    ("method", "speculator_cls", "parent_cls"),
    [
        ("mtp", AscendMTPSpeculator, MTPSpeculator),
        ("eagle3", AscendEagleSpeculator, EagleSpeculator),
    ],
)
def test_speculator_without_pcp_keeps_existing_proposal_inputs(
    method: str,
    speculator_cls,
    parent_cls,
) -> None:
    speculator = object.__new__(speculator_cls)
    speculator.method = method
    speculator.input_batch = None
    speculator.pcp_manager = None

    input_batch = MagicMock()
    local_attn_metadata = MagicMock()
    local_slot_mappings = MagicMock()
    local_aux_hidden_states = [MagicMock()] if method == "eagle3" else None

    with (
        patch.object(
            parent_cls,
            "propose",
            return_value=MagicMock(),
        ) as propose,
        patch.object(
            speculator_module,
            "build_slot_mappings_by_layer",
        ) as build_slot_mappings,
    ):
        speculator_cls.propose(
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
        )

    build_slot_mappings.assert_not_called()
    propose_args = propose.call_args.args
    assert propose_args[1] is local_attn_metadata
    assert propose_args[2] is local_slot_mappings
    assert propose_args[4] is local_aux_hidden_states
