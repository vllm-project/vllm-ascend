# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.autoregressive import (
    speculator as speculator_module,
)
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
    AscendMTPSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_target_pcp_for_replicated_draft,
)


def _make_padded_input_batch() -> MagicMock:
    input_batch = MagicMock(spec=AscendInputBatch)
    input_batch.num_reqs = 2
    input_batch.num_reqs_after_padding = 4
    input_batch.num_tokens = 6
    input_batch.num_tokens_after_padding = 8
    input_batch.idx_mapping = torch.tensor([3, 7], dtype=torch.int32)
    input_batch.query_start_loc = torch.tensor([0, 3, 6, 6, 6], dtype=torch.int32)
    input_batch.query_start_loc_np = np.array([0, 3, 6, 6, 6], dtype=np.int32)
    input_batch.seq_lens = torch.arange(4, dtype=torch.int32)
    input_batch.seq_lens_cpu_upper_bound = torch.arange(4, dtype=torch.int32)
    input_batch.input_ids = torch.arange(8, dtype=torch.int32)
    input_batch.positions = torch.arange(8, dtype=torch.int64)
    input_batch.is_padding = torch.zeros(8, dtype=torch.bool)
    input_batch.seq_lens_np = np.arange(4, dtype=np.int32)
    return input_batch


@pytest.mark.parametrize(
    ("target_pcp_size", "expected_execution_pcp_size"),
    [(2, 1), (1, 1)],
)
def test_draft_runtime_config_preserves_target_worker_topology(
    target_pcp_size: int,
    expected_execution_pcp_size: int,
) -> None:
    draft_parallel_config = SimpleNamespace(
        prefill_context_parallel_size=2,
        enable_expert_parallel=False,
        enable_eplb=False,
        rank=0,
    )
    target_parallel_config = SimpleNamespace(
        prefill_context_parallel_size=target_pcp_size,
        enable_expert_parallel=True,
        enable_eplb=True,
        rank=7,
        data_parallel_size=2,
        data_parallel_rank=1,
    )
    target_config = SimpleNamespace(
        parallel_config=target_parallel_config,
        speculative_config=SimpleNamespace(
            draft_parallel_config=draft_parallel_config,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(decode_mode=lambda: None),
        ),
    )
    draft_model_config = object()
    captured: dict[str, SimpleNamespace] = {}

    def fake_replace(config, **changes):
        values = vars(config).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    def fake_parent_init(speculator, execution_config, device):
        captured["execution_config"] = execution_config
        speculator.vllm_config = execution_config
        speculator.speculative_config = execution_config.speculative_config
        speculator.draft_model_config = draft_model_config
        speculator.input_buffers = object()
        speculator.max_num_reqs = 4
        speculator.max_num_tokens = 8
        speculator.num_speculative_steps = 3

    with (
        patch.object(
            speculator_module,
            "replace",
            side_effect=fake_replace,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.pcp_utils.replace",
            side_effect=fake_replace,
        ),
        patch.object(
            speculator_module.AutoRegressiveSpeculator,
            "__init__",
            new=fake_parent_init,
        ),
        patch.object(
            speculator_module,
            "AscendInputBuffers",
            return_value=object(),
        ),
    ):
        speculator = AscendMTPSpeculator(target_config, torch.device("cpu"))

    execution_config = captured["execution_config"]
    execution_parallel_config = execution_config.parallel_config
    assert execution_parallel_config.prefill_context_parallel_size == expected_execution_pcp_size
    assert execution_parallel_config.enable_expert_parallel
    assert execution_parallel_config.enable_eplb
    assert execution_parallel_config.rank == target_parallel_config.rank
    assert execution_parallel_config.data_parallel_size == 2
    assert execution_parallel_config.data_parallel_rank == 1
    assert target_parallel_config.prefill_context_parallel_size == target_pcp_size
    assert target_parallel_config.enable_expert_parallel
    assert target_parallel_config.enable_eplb

    draft_config = speculator.draft_vllm_config
    assert draft_parallel_config.prefill_context_parallel_size == 2
    assert not draft_parallel_config.enable_expert_parallel
    assert not draft_parallel_config.enable_eplb
    assert draft_config.model_config is draft_model_config
    assert draft_config.parallel_config.prefill_context_parallel_size == expected_execution_pcp_size
    assert draft_config.parallel_config.pipeline_parallel_size == 1


@pytest.mark.parametrize(("replicated_pcp", "manager_is_disabled"), [(True, True), (False, False)])
def test_draft_pcp_context_restores_manager_after_error(
    replicated_pcp: bool,
    manager_is_disabled: bool,
) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.pcp_manager = MagicMock()
    speculator.replicated_pcp = replicated_pcp
    speculator.model_state = SimpleNamespace(pcp_manager=speculator.pcp_manager)

    with (
        pytest.raises(RuntimeError, match="proposal failed"),
        disable_target_pcp_for_replicated_draft(speculator),
    ):
        assert (speculator.model_state.pcp_manager is None) is manager_is_disabled
        raise RuntimeError("proposal failed")

    assert speculator.model_state.pcp_manager is speculator.pcp_manager


def test_draft_pcp_context_rejects_mismatched_manager() -> None:
    current_pcp_manager = MagicMock()
    speculator = SimpleNamespace(
        replicated_pcp=True,
        model_state=SimpleNamespace(pcp_manager=current_pcp_manager),
        pcp_manager=MagicMock(),
    )

    with (
        pytest.raises(
            RuntimeError,
            match="requires model_state to use the target PCP manager",
        ),
        disable_target_pcp_for_replicated_draft(speculator),
    ):
        pytest.fail("mismatched manager must fail before draft execution")

    assert speculator.model_state.pcp_manager is current_pcp_manager


@pytest.mark.parametrize(
    ("replicated_pcp", "expected_source"),
    [(True, "draft"), (False, "target")],
)
def test_draft_prefill_attn_groups_follow_draft_topology(
    replicated_pcp: bool,
    expected_source: str,
) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = replicated_pcp
    speculator.attn_groups = [["draft"]]
    speculator.target_attn_groups = [["target"]]

    expected = speculator.attn_groups if expected_source == "draft" else speculator.target_attn_groups
    assert speculator.draft_prefill_attn_groups is expected


def test_prepare_replicated_prefill_attn_uses_global_batch() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.block_tables = MagicMock()
    speculator.kv_cache_config = object()
    speculator._build_draft_attn_metadata = MagicMock(return_value={"draft.layer": object()})
    input_batch = _make_padded_input_batch()
    input_batch.is_dummy = False
    speculator.input_batch = input_batch
    speculator.replicated_pcp = True
    global_slot_mapping = torch.arange(input_batch.num_tokens_after_padding).unsqueeze(0)
    slot_mappings = {"draft.layer": object()}
    speculator.block_tables.compute_slot_mappings.return_value = global_slot_mapping
    original_attn_metadata = {"local.layer": object()}
    original_slot_mappings = MagicMock()

    with patch.object(
        speculator_module,
        "build_slot_mappings_by_layer",
        return_value=slot_mappings,
    ) as build_slot_mappings:
        attn_metadata, actual_slot_mappings = speculator._prepare_replicated_prefill_attn(
            original_attn_metadata,
            original_slot_mappings,
            input_batch.num_reqs_after_padding,
            input_batch.num_tokens_after_padding,
        )

    assert attn_metadata == speculator._build_draft_attn_metadata.return_value
    assert actual_slot_mappings is slot_mappings
    speculator.block_tables.gather_block_tables.assert_called_once_with(
        input_batch.idx_mapping,
        num_reqs_padded=input_batch.num_reqs_after_padding,
    )
    speculator.block_tables.compute_slot_mappings.assert_called_once_with(
        input_batch.idx_mapping,
        input_batch.query_start_loc,
        input_batch.positions,
        num_tokens_padded=input_batch.num_tokens_after_padding,
    )
    build_slot_mappings.assert_called_once_with(
        global_slot_mapping,
        speculator.kv_cache_config,
    )
    speculator._build_draft_attn_metadata.assert_called_once_with(
        num_reqs=input_batch.num_reqs,
        num_reqs_padded=input_batch.num_reqs_after_padding,
        num_tokens_padded=input_batch.num_tokens_after_padding,
        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
        step=0,
        query_start_loc_np=input_batch.query_start_loc_np,
    )


def test_prefill_rebuilds_replicated_pcp_metadata_before_filtering() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = True
    speculator.input_batch = _make_padded_input_batch()
    speculator.input_batch.is_dummy = False
    speculator.draft_attn_layer_names = {"draft.layer"}

    draft_metadata = object()
    global_slot_mappings = MagicMock()
    speculator._prepare_replicated_prefill_attn = MagicMock(
        return_value=(
            {
                "draft.layer": draft_metadata,
                "target.layer": object(),
            },
            global_slot_mappings,
        )
    )
    local_attn_metadata = {"local.layer": object()}
    local_slot_mappings = MagicMock()

    with patch.object(
        speculator_module.AutoRegressiveSpeculator,
        "_prefill",
    ) as parent_prefill:
        speculator._prefill(
            num_reqs=2,
            num_tokens=8,
            attn_metadata=local_attn_metadata,
            slot_mappings=local_slot_mappings,
            num_tokens_across_dp=None,
        )

    speculator._prepare_replicated_prefill_attn.assert_called_once_with(
        local_attn_metadata,
        local_slot_mappings,
        2,
        8,
    )
    parent_prefill.assert_called_once()
    parent_args = parent_prefill.call_args.args
    assert parent_args[:2] == (2, 8)
    assert parent_args[2] == {"draft.layer": draft_metadata}
    assert parent_args[3] is global_slot_mappings


@pytest.mark.parametrize("attn_architecture", ["GQA", "MLA", "DSA", "SFA"])
@pytest.mark.parametrize("replicated_pcp", [False, True])
def test_graph_prefill_builds_draft_metadata(attn_architecture: str, replicated_pcp: bool) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = replicated_pcp
    speculator.attn_architecture = attn_architecture
    speculator.input_batch = _make_padded_input_batch()
    speculator.input_batch.is_dummy = False
    speculator.block_tables = MagicMock()
    speculator.kv_cache_config = object()
    speculator.draft_attn_layer_names = {"draft.layer"}
    local_draft_metadata = SimpleNamespace(decode=SimpleNamespace(actual_seq_lengths_q=[4, 8]))
    global_draft_metadata = object()
    speculator.model_state = SimpleNamespace(
        attn_metadata={"draft.layer": local_draft_metadata, "target.layer": object()},
    )
    speculator._build_draft_attn_metadata = MagicMock(
        return_value={"draft.layer": global_draft_metadata},
    )

    with patch.object(speculator_module, "build_slot_mappings_by_layer", return_value={}) as build_slots:
        actual = speculator.build_draft_attn_metadatas(
            num_reqs_padded=2,
            num_tokens_padded=8,
            is_draft_model_prefill=True,
        )

    rebuild_metadata = replicated_pcp and attn_architecture in ("DSA", "SFA")
    expected_metadata = global_draft_metadata if rebuild_metadata else local_draft_metadata
    assert actual == [{"draft.layer": expected_metadata}]
    assert actual[0]["draft.layer"] is expected_metadata
    assert speculator._build_draft_attn_metadata.call_count == int(rebuild_metadata)
    assert build_slots.call_count == int(rebuild_metadata)
    assert speculator.block_tables.gather_block_tables.call_count == int(replicated_pcp)
    assert speculator.block_tables.compute_slot_mappings.call_count == int(replicated_pcp)
    assert local_draft_metadata.decode.actual_seq_lengths_q[-1] == 8


@pytest.mark.parametrize("attn_architecture", ["GQA", "MLA"])
def test_graph_prefill_refreshes_captured_cache_buffers(attn_architecture: str) -> None:
    speculator = object.__new__(AscendEagleSpeculator)
    speculator.replicated_pcp = True
    speculator.attn_architecture = attn_architecture
    speculator.draft_attn_layer_names = {"draft.layer"}
    metadata = SimpleNamespace(actual_seq_lengths_q=[4, 8])
    speculator.model_state = SimpleNamespace(attn_metadata={"draft.layer": metadata})
    speculator.kv_cache_config = object()
    speculator._build_draft_attn_metadata = MagicMock()

    # These views stand in for the persistent buffers bound during capture.
    captured_blocks = torch.zeros((2, 3), dtype=torch.int32)
    captured_slots = torch.full((1, 8), -1, dtype=torch.int32)
    block_ptr, slot_ptr = captured_blocks.data_ptr(), captured_slots.data_ptr()
    request_blocks = {3: [7, 13, 0], 7: [17, 19, 23]}

    def gather_blocks(idx_mapping, num_reqs_padded):
        captured_blocks.zero_()
        for row, req_idx in enumerate(idx_mapping.tolist()):
            captured_blocks[row] = torch.tensor(request_blocks[req_idx])
        return (captured_blocks[:num_reqs_padded],)

    def compute_slots(idx_mapping, query_start_loc, positions, num_tokens_padded):
        captured_slots.fill_(-1)
        for row, req_idx in enumerate(idx_mapping.tolist()):
            start, end = query_start_loc[row : row + 2].tolist()
            for token in range(start, end):
                position = int(positions[token])
                block = request_blocks[req_idx][position // 128]
                captured_slots[0, token] = block * 128 + position % 128
        return captured_slots[:, :num_tokens_padded]

    speculator.block_tables = SimpleNamespace(
        gather_block_tables=gather_blocks,
        compute_slot_mappings=compute_slots,
    )
    for req_idx, positions, expected_slots in [
        (3, [126, 127, 128, 129], [1022, 1023, 1664, 1665]),
        (7, [254, 255, 256, 257], [2558, 2559, 2944, 2945]),
    ]:
        batch = _make_padded_input_batch()
        batch.is_dummy = False
        batch.num_reqs, batch.num_tokens = 1, 4
        batch.query_start_loc_np = np.array([0, 4, 8], dtype=np.int32)
        batch.idx_mapping = torch.tensor([req_idx], dtype=torch.int32)
        batch.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
        batch.positions = torch.tensor(positions, dtype=torch.int64)
        speculator.input_batch = batch
        # The preceding single-token decode has left a different slot layout.
        captured_slots.fill_(-1)
        captured_slots[0, 0] = 123
        captured_blocks.fill_(-1)

        with patch.object(speculator_module, "build_slot_mappings_by_layer") as build_slots:
            result = speculator.build_draft_attn_metadatas(2, 8, is_draft_model_prefill=True)

        assert result[0]["draft.layer"] is metadata
        speculator._build_draft_attn_metadata.assert_not_called()
        build_slots.assert_not_called()
        assert metadata.actual_seq_lengths_q == [4, 8]
        assert captured_slots.tolist() == [expected_slots + [-1] * 4]
        assert captured_blocks.tolist() == [request_blocks[req_idx], [0, 0, 0]]
        assert (captured_blocks.data_ptr(), captured_slots.data_ptr()) == (block_ptr, slot_ptr)


@pytest.mark.parametrize("guard", ["non_pcp", "no_batch", "dummy", "no_metadata"])
def test_prepare_replicated_prefill_preserves_bypass(guard: str) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = guard != "non_pcp"
    speculator.input_batch = _make_padded_input_batch() if guard != "no_batch" else None
    if speculator.input_batch is not None:
        speculator.input_batch.is_dummy = guard == "dummy"
    speculator.block_tables = MagicMock()
    speculator._build_draft_attn_metadata = MagicMock()
    metadata = None if guard == "no_metadata" else {"draft.layer": object()}
    slots = {"draft.layer": object()}

    actual_metadata, actual_slots = speculator._prepare_replicated_prefill_attn(metadata, slots, 2, 8)

    assert actual_metadata is metadata
    assert actual_slots is slots
    speculator.block_tables.gather_block_tables.assert_not_called()
    speculator.block_tables.compute_slot_mappings.assert_not_called()
    speculator._build_draft_attn_metadata.assert_not_called()


@pytest.mark.parametrize("attn_architecture", ["GQA", "MLA", "DSA", "SFA"])
@pytest.mark.parametrize("guard", ["no_batch", "dummy"])
def test_graph_prefill_without_real_batch_preserves_metadata(attn_architecture: str, guard: str) -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = True
    speculator.attn_architecture = attn_architecture
    speculator.input_batch = _make_padded_input_batch() if guard == "dummy" else None
    if speculator.input_batch is not None:
        speculator.input_batch.is_dummy = True
    speculator.block_tables = MagicMock()
    speculator._build_draft_attn_metadata = MagicMock()
    metadata = object()
    speculator.draft_attn_layer_names = {"draft.layer"}
    speculator.model_state = SimpleNamespace(attn_metadata={"draft.layer": metadata})

    [actual] = speculator.build_draft_attn_metadatas(2, 8, is_draft_model_prefill=True)

    assert actual["draft.layer"] is metadata
    speculator.block_tables.gather_block_tables.assert_not_called()
    speculator.block_tables.compute_slot_mappings.assert_not_called()
    speculator._build_draft_attn_metadata.assert_not_called()


@pytest.mark.skipif(speculator_module.vllm_version_is("0.28.0"), reason="DPSyncState is a main2main interface")
@pytest.mark.parametrize(
    ("speculator_cls", "parent_cls", "replicated_pcp", "batch_kind"),
    [
        (AscendMTPSpeculator, MTPSpeculator, True, "prefill"),
        (AscendMTPSpeculator, MTPSpeculator, True, "decode"),
        (AscendMTPSpeculator, MTPSpeculator, True, "idle"),
        (AscendMTPSpeculator, MTPSpeculator, False, "prefill"),
        (AscendEagleSpeculator, EagleSpeculator, True, "prefill"),
    ],
)
def test_propose_sync_follows_draft_token_layout(speculator_cls, parent_cls, replicated_pcp, batch_kind) -> None:
    speculator = object.__new__(speculator_cls)
    speculator.replicated_pcp = replicated_pcp
    speculator.input_batch = None
    speculator.pcp_manager = MagicMock()
    speculator.model_state = SimpleNamespace(
        pcp_manager=speculator.pcp_manager,
    )
    input_batch = _make_padded_input_batch()
    input_batch.has_prefill = batch_kind == "prefill"
    input_batch.is_dummy = batch_kind == "idle"
    if batch_kind != "prefill":
        input_batch.num_tokens_after_padding = 2
    num_tokens = input_batch.num_tokens_after_padding
    target_num_tokens = num_tokens // 2 if replicated_pcp and input_batch.has_prefill else num_tokens
    target_sync = SimpleNamespace(
        eager=True,
        uniform_token_count=None,
        num_tokens_across_dp=torch.tensor([target_num_tokens, target_num_tokens]),
    )
    expected = object()

    def parent_propose(*args, **kwargs):
        assert args[0] is input_batch
        assert (speculator.model_state.pcp_manager is None) is replicated_pcp
        # Exercise upstream's real reuse checks; only the collective is mocked.
        with patch.object(dp_utils, "sync_cudagraph_and_dp_padding") as sync:
            sync.return_value = (SimpleNamespace(cg_mode=CUDAGraphMode.NONE), object())
            dp_utils.dispatch_cg_and_sync_dp(
                None,
                input_batch.num_reqs,
                num_tokens,
                None,
                dp_size=2,
                dp_rank=0,
                need_eager=True,
                dp_sync=args[11],
            )
        assert sync.call_count == int(replicated_pcp)
        return expected

    with (
        patch.object(
            parent_cls,
            "propose",
            side_effect=parent_propose,
        ),
        patch.object(
            speculator_module,
            "build_attn_metadata_wrapper",
            return_value=nullcontext(),
        ),
        patch.object(
            speculator_module,
            "torch_gather_wrapper",
            return_value=nullcontext(),
        ),
    ):
        actual = speculator.propose(
            input_batch,
            *[MagicMock() for _ in range(10)],
            dp_sync=target_sync,
        )

    assert actual is expected
    assert speculator.input_batch is input_batch
    assert speculator.model_state.pcp_manager is speculator.pcp_manager


def test_propose_preserves_v028_dp_token_counts() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.replicated_pcp = True
    input_batch = object()
    token_counts = torch.tensor([4, 8])
    with (
        patch.object(speculator_module, "vllm_version_is", return_value=True),
        patch.object(speculator_module, "disable_target_pcp_for_replicated_draft", return_value=nullcontext()),
        patch.object(speculator_module, "build_attn_metadata_wrapper", return_value=nullcontext()),
        patch.object(speculator_module, "torch_gather_wrapper", return_value=nullcontext()),
        patch.object(MTPSpeculator, "propose") as parent,
    ):
        speculator.propose(input_batch, *[MagicMock() for _ in range(10)], token_counts, dp_sync=object())
    assert parent.call_args.args[11] is token_counts
