# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

import vllm_ascend.attention.sfa_v1 as sfa_module
import vllm_ascend.worker.v2.spec_decode.mtp.speculator as mtp_speculator_module
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator


def test_force_non_pcp_sfa_overrides_dynamic_pcp_backend_selection():
    pcp_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    with (
        patch.object(
            sfa_module,
            "get_current_vllm_config",
            return_value=pcp_config,
        ),
        patch.object(
            sfa_module,
            "enable_sfa_dcp_replicated_indexer",
            return_value=False,
        ),
    ):
        with sfa_module.force_non_pcp_sfa():
            assert (
                sfa_module.AscendSFABackend.get_builder_cls()
                is sfa_module.AscendSFAMetadataBuilder
            )
            assert (
                sfa_module.AscendSFABackend.get_impl_cls()
                is sfa_module.AscendSFAImpl
            )

    assert not sfa_module._force_non_pcp_sfa.get()


def test_load_draft_model_keeps_force_non_pcp_sfa_active():
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    target_model = MagicMock()
    expected = MagicMock()

    def load_draft_model(*_args, **_kwargs):
        assert sfa_module._force_non_pcp_sfa.get()
        return expected

    with patch.object(
        MTPSpeculator,
        "load_draft_model",
        side_effect=load_draft_model,
    ):
        result = speculator.load_draft_model(target_model, {"target.layer"})

    assert result is expected
    assert not sfa_module._force_non_pcp_sfa.get()


def test_load_draft_model_does_not_override_sfa_without_pcp():
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1)
    )
    expected = MagicMock()

    def load_draft_model(*_args, **_kwargs):
        assert not sfa_module._force_non_pcp_sfa.get()
        return expected

    with patch.object(
        MTPSpeculator,
        "load_draft_model",
        side_effect=load_draft_model,
    ):
        result = speculator.load_draft_model(MagicMock(), {"target.layer"})

    assert result is expected


def _make_global_batch():
    return SimpleNamespace(
        num_reqs=2,
        num_tokens=4,
        idx_mapping=torch.tensor([3, 7], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        query_start_loc_np=np.array([0, 1, 4], dtype=np.int32),
        positions=torch.tensor([10, 20, 21, 22], dtype=torch.int64),
        num_scheduled_tokens=np.array([1, 3], dtype=np.int32),
        seq_lens=torch.tensor([11, 23], dtype=torch.int32),
        seq_lens_np=np.array([11, 23], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([11, 23], dtype=torch.int32),
        attn_state=object(),
    )


def test_build_global_pcp_draft_inputs_rebuilds_metadata_and_slot_mappings():
    input_batch = _make_global_batch()
    block_tables = MagicMock()
    gathered_block_tables = [
        torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        torch.tensor([[30, 31], [40, 41]], dtype=torch.int32),
    ]
    global_slot_mappings = [
        torch.tensor([101, 201, 202, 203], dtype=torch.int64),
        torch.tensor([301, 401, 402, 403], dtype=torch.int64),
    ]
    block_tables.gather_block_tables.return_value = gathered_block_tables
    block_tables.compute_slot_mappings.return_value = global_slot_mappings

    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["target.layer", "draft.layer.0"]),
            SimpleNamespace(layer_names=["draft.layer.1"]),
        ]
    )
    speculator = SimpleNamespace(
        block_tables=block_tables,
        attn_groups="global-draft-attn-groups",
        kv_cache_config=kv_cache_config,
        draft_attn_layer_names={"draft.layer.0", "draft.layer.1"},
    )
    expected_metadata = {"draft.layer.0": object(), "draft.layer.1": object()}

    with patch.object(
        mtp_speculator_module,
        "build_attn_metadata",
        return_value=expected_metadata,
    ) as build_attn_metadata:
        metadata, slot_mappings = (
            AscendMTPSpeculator._build_global_pcp_draft_inputs(
                speculator,
                input_batch,
            )
        )

    assert metadata is expected_metadata
    assert set(slot_mappings) == {"draft.layer.0", "draft.layer.1"}
    assert slot_mappings["draft.layer.0"] is global_slot_mappings[0]
    assert slot_mappings["draft.layer.1"] is global_slot_mappings[1]
    block_tables.gather_block_tables.assert_called_once_with(
        input_batch.idx_mapping,
        2,
    )
    block_tables.compute_slot_mappings.assert_called_once_with(
        input_batch.idx_mapping,
        input_batch.query_start_loc,
        input_batch.positions,
        4,
    )

    kwargs = build_attn_metadata.call_args.kwargs
    assert kwargs["attn_groups"] == "global-draft-attn-groups"
    assert kwargs["num_reqs"] == 2
    assert kwargs["num_tokens"] == 4
    assert kwargs["max_query_len"] == 3
    assert kwargs["max_seq_len"] == 23
    assert kwargs["block_tables"] is gathered_block_tables
    assert kwargs["slot_mappings"] is global_slot_mappings
    assert kwargs["attn_state"] is input_batch.attn_state
    assert torch.equal(kwargs["query_start_loc_cpu"], torch.tensor([0, 1, 4]))
    assert torch.equal(kwargs["positions"], input_batch.positions)


def _make_uninitialized_speculator(pcp_size: int):
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size)
    )
    return speculator


def test_propose_replaces_pcp_local_metadata_with_global_draft_view():
    speculator = _make_uninitialized_speculator(pcp_size=2)
    input_batch = _make_global_batch()
    local_metadata = {"draft.layer.0": "pcp-local-metadata"}
    local_slots = {"draft.layer.0": torch.tensor([1])}
    global_metadata = {"draft.layer.0": "global-metadata"}
    global_slots = {"draft.layer.0": torch.tensor([10, 20])}
    speculator._build_global_pcp_draft_inputs = MagicMock(
        return_value=(global_metadata, global_slots)
    )
    expected = torch.tensor([[7]])

    with patch.object(
        AscendAutoRegressiveSpeculator,
        "propose",
        return_value=expected,
    ) as parent_propose:
        result = speculator.propose(
            input_batch,
            local_metadata,
            local_slots,
            "remaining-argument",
            dummy_run=False,
        )

    assert result is expected
    speculator._build_global_pcp_draft_inputs.assert_called_once_with(input_batch)
    args = parent_propose.call_args.args
    assert args[0] is input_batch
    assert args[1] is global_metadata
    assert args[2] is global_slots
    assert args[3] == "remaining-argument"


def test_propose_keeps_target_metadata_without_pcp():
    speculator = _make_uninitialized_speculator(pcp_size=1)
    speculator._build_global_pcp_draft_inputs = MagicMock()
    input_batch = _make_global_batch()
    metadata = {"draft.layer.0": "target-metadata"}
    slots = {"draft.layer.0": torch.tensor([1, 2])}

    with patch.object(
        AscendAutoRegressiveSpeculator,
        "propose",
        return_value=torch.tensor([[3]]),
    ) as parent_propose:
        speculator.propose(input_batch, metadata, slots, "remaining-argument")

    speculator._build_global_pcp_draft_inputs.assert_not_called()
    args = parent_propose.call_args.args
    assert args[1] is metadata
    assert args[2] is slots
