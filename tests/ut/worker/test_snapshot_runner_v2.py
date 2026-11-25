# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import get_drafter_model
from vllm_ascend.snapshot.model_runner_lifecycle.restore import (
    _reset_block_tables,
    _reset_input_buffers,
    _reset_request_state,
    _reset_speculator,
    reset_graph_managers,
)


def _staged():
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    buffer = StagedWriteTensor.__new__(StagedWriteTensor)
    buffer.gpu = torch.ones(4)
    buffer._staged_write_indices = [1]
    buffer._staged_write_starts = [0]
    buffer._staged_write_contents = [7]
    buffer._staged_write_cu_lens = [1]
    return buffer


def _uva():
    return SimpleNamespace(cpu=torch.ones(4), copy_to_uva=MagicMock())


def test_v2_draft_checkpoint_uses_runner_interface():
    model = torch.nn.Linear(2, 2)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        get_draft_model=lambda: model,
    )
    assert get_drafter_model(runner) is model


def test_v2_input_buffers_reset_preserves_storage():
    names = ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens", "dcp_local_seq_lens")
    buffers = SimpleNamespace(**{name: torch.ones(4) for name in names})
    pointers = {name: getattr(buffers, name).data_ptr() for name in names}
    _reset_input_buffers(buffers)
    for name in names:
        assert torch.count_nonzero(getattr(buffers, name)) == 0
        assert getattr(buffers, name).data_ptr() == pointers[name]


def test_v2_block_tables_reset_rebuilds_pointer_metadata():
    table = _staged()
    tables = SimpleNamespace(
        block_tables=[table],
        num_blocks=_uva(),
        input_block_tables=[torch.ones(4)],
        slot_mappings=torch.ones(4),
        init_block_table_layout_tensors=MagicMock(),
    )
    _reset_block_tables(tables)
    assert not table._staged_write_indices
    assert not table._staged_write_contents
    assert torch.count_nonzero(table.gpu) == 0
    assert torch.count_nonzero(tables.input_block_tables[0]) == 0
    assert torch.all(tables.slot_mappings == -1)
    tables.num_blocks.copy_to_uva.assert_called_once()
    tables.init_block_table_layout_tensors.assert_called_once()


def test_v2_request_state_reset_and_idle_contract():
    state = SimpleNamespace(
        num_reqs=1,
        all_token_ids=_staged(),
        total_len=_staged(),
        num_computed_tokens=_staged(),
        prompt_len=_uva(),
        prefill_len=_uva(),
        num_computed_prefill_tokens=np.ones(4),
        num_computed_tokens_np=np.ones(4),
        max_seq_len=np.ones(4),
        last_sampled_tokens=torch.ones(4),
        draft_tokens=torch.ones(4),
        next_prefill_tokens=torch.ones(4),
    )
    with pytest.raises(RuntimeError, match="empty Model Runner V2 request state"):
        _reset_request_state(state)
    state.num_reqs = 0
    _reset_request_state(state)
    _reset_request_state(state)
    assert not np.any(state.num_computed_tokens_np)
    assert torch.count_nonzero(state.next_prefill_tokens) == 0
    assert not state.num_computed_tokens._staged_write_contents


@pytest.mark.parametrize("with_speculator", [False, True])
@pytest.mark.parametrize("with_fast_prefill", [False, True])
def test_v2_graph_managers_recreated_without_loading_or_compiling_model(
    monkeypatch, with_speculator, with_fast_prefill
):
    from vllm_ascend.compilation import acl_graph

    for name in ("_graph_params", "_draft_graph_params", "_draft_graph_prefill_params"):
        monkeypatch.setattr(acl_graph, name, object())
    runner = SimpleNamespace(
        vllm_config=object(),
        device=torch.device("cpu"),
        decode_query_len=4,
        cudagraph_manager=SimpleNamespace(
            cudagraph_mode=1, lora_capture_cases=[0], varlen_decode=False, ubatch_runner=None
        ),
        fast_prefill=SimpleNamespace(cudagraph_manager=None) if with_fast_prefill else None,
        speculator=SimpleNamespace(init_cudagraph_manager=MagicMock()) if with_speculator else None,
    )
    if runner.fast_prefill is not None:
        runner.fast_prefill.cudagraph_manager = runner.cudagraph_manager

    def construct_target(*args, **kwargs):
        assert acl_graph._draft_graph_params is None
        assert acl_graph._draft_graph_prefill_params is None
        # Use the real registration guard that failed on hardware.
        acl_graph.set_graph_params([4, 8])
        return SimpleNamespace(cudagraph_mode=1, lora_capture_cases=[0], varlen_decode=False, ubatch_runner=None)

    def construct_draft(mode):
        assert acl_graph._graph_params is not None
        acl_graph.set_draft_graph_prefill_params([4, 8])
        acl_graph.set_draft_graph_params([1, 2])

    if runner.speculator is not None:
        runner.speculator.init_cudagraph_manager.side_effect = construct_draft
    with patch("vllm_ascend.worker.v2.aclgraph_utils.ModelAclGraphManager", side_effect=construct_target) as manager:
        reset_graph_managers(runner)
        previous_manager = runner.cudagraph_manager
        reset_graph_managers(runner)
    assert runner.cudagraph_manager is not previous_manager
    assert manager.call_args.kwargs["ubatch_runner"] is None
    if runner.fast_prefill is not None:
        assert runner.fast_prefill.cudagraph_manager is runner.cudagraph_manager
    if runner.speculator is not None:
        assert runner.speculator.init_cudagraph_manager.call_count == 2
        runner.speculator.init_cudagraph_manager.assert_called_with(1)
    else:
        assert acl_graph._draft_graph_params is None
        assert acl_graph._draft_graph_prefill_params is None
    with pytest.raises(ValueError, match="already been set"):
        acl_graph.set_graph_params([4, 8])


@pytest.mark.parametrize("adaptive_verification", [False, True])
def test_v2_mtp_resets_step_indices_and_draft_inputs(adaptive_verification):
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator

    class Speculator(AutoRegressiveSpeculator):
        def load_draft_model(self, *args):
            raise AssertionError("Restore must not reload or compile the draft model")

    speculator = Speculator.__new__(Speculator)
    names = (
        "input_ids",
        "positions",
        "is_padding",
        "query_start_loc",
        "seq_lens",
        "dcp_local_seq_lens",
        "seq_lens_cpu",
    )
    speculator.input_buffers = SimpleNamespace(**{name: torch.ones(4) for name in names})
    for name in (
        "idx_mapping",
        "temperature",
        "seeds",
        "draft_tokens",
        "hidden_states",
        "current_draft_step",
        "last_token_indices",
        "sample_src_positions",
    ):
        setattr(speculator, name, torch.ones(4))
    speculator.draft_logits = None
    speculator.draft_is_prefilling = torch.ones(4, dtype=torch.bool)
    speculator.draft_token_confidence_probs = torch.ones(4)
    speculator.acceptance_estimator = object() if adaptive_verification else None
    speculator.max_num_reqs = 4
    speculator.num_speculative_steps = 3
    speculator.device = torch.device("cpu")
    speculator.inputs_embeds = None
    speculator.input_batch = object()
    with patch(
        "vllm.v1.worker.gpu.spec_decode.acceptance_estimator.OnlineAcceptanceEstimator",
        autospec=True,
    ) as estimator:
        _reset_speculator(speculator)
    if adaptive_verification:
        estimator.assert_called_once_with(4, 3, speculator.device)
        assert speculator.acceptance_estimator is estimator.return_value
    else:
        estimator.assert_not_called()
    assert speculator.input_batch is None
    assert torch.count_nonzero(speculator.current_draft_step) == 0
    assert torch.count_nonzero(speculator.last_token_indices) == 0
    assert torch.count_nonzero(speculator.input_buffers.positions) == 0
    assert not speculator.draft_is_prefilling.any()
    assert not speculator.draft_token_confidence_probs.any()


def test_v2_restore_uses_current_pcp_factory_contract():
    from vllm_ascend.snapshot.model_runner_lifecycle import restore

    runner = MagicMock()
    runner.attn_groups = []
    runner.speculator = None
    runner.kvpp.scheduler = None
    with torch.inference_mode():
        runner.input_buffers.seq_lens_cpu = torch.ones(4)
        runner.num_computed_tokens_cpu = torch.ones(4)
        runner.input_buffers.positions = torch.ones(4)
    positions = runner.input_buffers.positions
    pointer = positions.data_ptr()
    with (
        patch.object(restore, "_reset_request_state"),
        patch.object(restore, "_reset_input_buffers", side_effect=lambda buffers: buffers.positions.zero_()),
        patch.object(restore, "_reset_block_tables"),
        patch("vllm.v1.worker.gpu.pcp_manager.maybe_build_pcp_manager", autospec=True) as factory,
    ):
        restore._reset_v2_runner_runtime_state(runner)

    assert not positions.any() and positions.data_ptr() == pointer
    assert not runner.input_buffers.seq_lens_cpu.any()
    assert not runner.num_computed_tokens_cpu.any()

    factory.assert_called_once_with(
        runner.vllm_config,
        runner.device,
        runner.supports_mm_inputs,
        runner.block_tables,
        cls=runner.pcp_manager_cls,
    )
    assert runner.pcp_manager.kv_cache_config is runner.kv_cache_config
    assert runner.model_state.pcp_manager is runner.pcp_manager
