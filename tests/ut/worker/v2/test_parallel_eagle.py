# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-Ascend project
import sys
from contextlib import nullcontext
from copy import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.helpers.golden_copy_and_expand import npu_copy_and_expand_eagle_inputs_stub
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle.parallel import AscendParallelEagleSpeculator


def test_expanded_compile_range_is_installed_before_base_initialization():
    module = sys.modules[AscendParallelEagleSpeculator.__module__]
    compilation = SimpleNamespace(
        compile_ranges_endpoints=[8192],
        static_forward_context={"target.layer": object()},
        cudagraph_mode=module.CUDAGraphMode.FULL,
        pass_config=SimpleNamespace(
            fuse_allreduce_rms=False,
            enable_sp=False,
            fuse_rope_kvcache=False,
            fuse_qk_norm_rope_kvcache=False,
        ),
    )
    target_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=256),
        speculative_config=SimpleNamespace(num_speculative_tokens=8),
        compilation_config=compilation,
    )

    def replace_config(config, **changes):
        result = copy(config)
        result.__dict__.update(changes)
        if "scheduler_config" in changes:
            # Exercise upstream's actual range normalization without loading
            # a model or initializing distributed/NPU state in the CPU test.
            module.VllmConfig._set_compile_ranges(result)
        return result

    def initialize_base(spec, config, device):
        spec.vllm_config = config
        spec.num_speculative_steps = config.speculative_config.num_speculative_tokens
        spec.max_num_reqs = config.scheduler_config.max_num_seqs
        spec.max_num_tokens = config.scheduler_config.max_num_batched_tokens
        spec.draft_model_config = SimpleNamespace(hf_config=object())

    with (
        patch.object(module, "replace", side_effect=replace_config),
        patch.object(module, "get_parallel_drafting_token_id", return_value=99),
        patch.object(AscendParallelEagleSpeculator.__mro__[1], "__init__", initialize_base),
    ):
        spec = AscendParallelEagleSpeculator(target_config, torch.device("cpu"))

    assert spec.max_num_tokens == 9984
    assert spec.vllm_config.compilation_config.compile_ranges_endpoints == [8192, 9984]
    assert target_config.scheduler_config.max_num_batched_tokens == 8192
    assert compilation.compile_ranges_endpoints == [8192]
    assert spec.vllm_config.compilation_config is not compilation
    assert spec.vllm_config.compilation_config.static_forward_context is compilation.static_forward_context
    assert spec.vllm_config.compilation_config.cudagraph_mode == module.CUDAGraphMode.FULL


def make_speculator(steps=3):
    spec = AscendParallelEagleSpeculator.__new__(AscendParallelEagleSpeculator)
    spec.num_speculative_steps = steps
    spec.extra_query_tokens = steps - 1
    spec.max_model_len = 128
    spec.parallel_token_id = 99
    spec.parallel_hidden = torch.tensor([-1.0, -2.0])
    spec.hidden_states = torch.zeros(32, 2)
    spec.parallel_sample_indices = torch.full((4 * steps,), 31, dtype=torch.long)
    spec.parallel_sample_steps = torch.arange(steps).repeat(4)
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(32, dtype=torch.int32),
        positions=torch.zeros(32, dtype=torch.long),
    )
    spec.draft_tokens = torch.zeros(4, steps, dtype=torch.int64)
    spec.replicated_pcp = False
    spec.pcp_manager = None
    return spec


@pytest.mark.parametrize("parallel", [False, True])
def test_factory_selects_parallel_eagle_without_changing_autoregressive_eagle(parallel):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="eagle3",
            parallel_drafting=parallel,
            use_dspark=lambda: False,
            use_dflash=lambda: False,
            use_eagle=lambda: True,
        )
    )
    class_path = (
        "vllm_ascend.worker.v2.spec_decode.eagle.parallel.AscendParallelEagleSpeculator"
        if parallel
        else "vllm_ascend.worker.v2.spec_decode.eagle.speculator.AscendEagleSpeculator"
    )
    with patch(class_path) as constructor:
        assert init_speculator(config, torch.device("cpu")) is constructor.return_value
    constructor.assert_called_once_with(config, torch.device("cpu"))


@pytest.mark.parametrize("rejections", [[0, 0], [2, 1]])
def test_parallel_inputs_preserve_target_hidden_and_replace_mask_hidden(rejections):
    spec = make_speculator()
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=7,
        input_ids=torch.arange(10, 17, dtype=torch.int32),
        positions=torch.tensor([20, 21, 22, 23, 40, 41, 42]),
        query_start_loc=torch.tensor([0, 4, 7], dtype=torch.int32),
    )
    hidden = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    with patch.object(
        torch.ops._C_ascend,
        "npu_copy_and_expand_eagle_inputs",
        npu_copy_and_expand_eagle_inputs_stub,
        create=True,
    ):
        count, rejected = spec._prepare_parallel_inputs(
            batch, hidden, torch.tensor([70, 80], dtype=torch.int32), torch.tensor(rejections)
        )
    assert count == 11
    assert rejected.dtype == torch.bool
    assert rejected.sum().item() == sum(rejections)
    for row, (start, end) in enumerate([(0, 4), (4, 7)]):
        output_start = start + row * 2
        accepted = end - start - rejections[row]
        first_sample = output_start + accepted - 1
        torch.testing.assert_close(
            spec.hidden_states[output_start : first_sample + 1], hidden[start : start + accepted]
        )
        torch.testing.assert_close(
            spec.hidden_states[first_sample + 1 : first_sample + 3], spec.parallel_hidden.expand(2, -1)
        )
        assert spec.input_buffers.input_ids[first_sample : first_sample + 3].tolist() == [70 + row * 10, 99, 99]
        assert spec.parallel_sample_indices[row * 3 : row * 3 + 3].tolist() == list(
            range(first_sample, first_sample + 3)
        )
        assert not rejected[first_sample : first_sample + 3].any()
    assert spec.parallel_sample_indices[6:].tolist() == [0] * 6


def test_parallel_forward_samples_all_steps_in_one_model_call():
    spec = make_speculator()
    spec.parallel_sample_indices[:6] = torch.tensor([1, 2, 3, 7, 8, 9])
    spec.input_buffers.positions[:11] = torch.arange(20, 31)
    hidden = torch.arange(22, dtype=torch.float32).reshape(11, 2)
    spec._run_model = MagicMock(return_value=(hidden, hidden))
    spec.sample_draft = MagicMock(return_value=torch.tensor([10, 11, 12, 20, 21, 22]))
    spec.idx_mapping = torch.tensor([2, 0, -1, -1], dtype=torch.int32)
    spec.temperature = torch.zeros(4)
    spec.seeds = torch.zeros(4, dtype=torch.int64)
    spec.draft_logits = None

    spec._parallel_forward(2, 11, {}, {}, None)

    spec._run_model.assert_called_once()
    spec.sample_draft.assert_called_once()
    args = spec.sample_draft.call_args.args
    torch.testing.assert_close(args[0], hidden[[1, 2, 3, 7, 8, 9]])
    assert args[1].tolist() == [22, 23, 24, 28, 29, 30]
    assert args[2].tolist() == [2, 2, 2, 0, 0, 0]
    assert args[5].tolist() == [0, 1, 2, 0, 1, 2]
    assert spec.draft_tokens[:2].tolist() == [[10, 11, 12], [20, 21, 22]]


def test_parallel_positions_past_context_limit_do_not_write_kv():
    spec = make_speculator()
    spec.max_model_len = 8
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        input_ids=torch.tensor([10], dtype=torch.int32),
        positions=torch.tensor([7]),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    with patch.object(
        torch.ops._C_ascend,
        "npu_copy_and_expand_eagle_inputs",
        npu_copy_and_expand_eagle_inputs_stub,
        create=True,
    ):
        count, rejected = spec._prepare_parallel_inputs(
            batch, torch.ones(1, 2), torch.tensor([70], dtype=torch.int32), torch.tensor([0])
        )
    assert count == 3
    assert spec.input_buffers.positions[:3].tolist() == [7, 7, 7]
    assert rejected.tolist() == [False, True, True]


def test_capture_records_only_parallel_forward_with_draft_inputs():
    spec = make_speculator()
    manager = MagicMock()
    manager.use_breakable_cg = False
    spec.prefill_cudagraph_manager = manager
    spec.idx_mapping = torch.ones(4, dtype=torch.int32)
    spec.model_state = object()
    spec.block_tables = object()
    spec.attn_groups = [[object()]]
    spec.kv_cache_config = object()
    spec.decode_cudagraph_manager = MagicMock()
    with patch(
        "vllm_ascend.worker.v2.spec_decode.eagle.parallel.build_attn_metadata_wrapper",
        return_value=nullcontext(),
    ):
        spec.capture()
    manager.capture.assert_called_once()
    args = manager.capture.call_args.args
    assert args[0] == spec._parallel_forward
    assert args[2] is spec.input_buffers
    assert args[4] is spec.attn_groups
    spec.decode_cudagraph_manager.capture.assert_not_called()
    assert not spec.parallel_sample_indices.any()


@pytest.mark.parametrize("full_graph", [False, True])
def test_propose_expands_metadata_and_masks_rejected_slots_before_forward(full_graph):
    module = sys.modules[AscendParallelEagleSpeculator.__module__]
    spec = make_speculator()
    spec.input_buffers.query_start_loc = torch.zeros(6, dtype=torch.int32)
    spec.input_buffers.seq_lens = torch.zeros(4, dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=8,
        input_ids=torch.arange(10, 18, dtype=torch.int32),
        positions=torch.tensor([20, 21, 22, 23, 40, 41, 42, 43]),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_np=torch.tensor([0, 4, 8], dtype=torch.int32).numpy(),
        seq_lens=torch.tensor([24, 44], dtype=torch.int32),
        seq_lens_np=torch.tensor([24, 44], dtype=torch.int32).numpy(),
        idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
        is_prefilling_np=torch.tensor([False, False]).numpy(),
        has_prefill=False,
    )
    mode = module.CUDAGraphMode.FULL if full_graph else module.CUDAGraphMode.NONE
    desc = SimpleNamespace(num_reqs=3 if full_graph else 2, num_tokens=18 if full_graph else 12, cg_mode=mode)
    slots = torch.zeros(1, desc.num_tokens, dtype=torch.int32)
    slots[:, 12:] = -1
    spec.block_tables = MagicMock()
    spec.block_tables.compute_slot_mappings.return_value = slots
    spec.block_tables.input_block_tables = [torch.zeros(4, 8, dtype=torch.int32)]
    spec.attn_groups = [[object()]]
    spec.kv_cache_config = object()
    spec.prefill_cudagraph_manager = MagicMock()
    spec._copy_request_inputs = MagicMock()
    spec._prepare_eplb_forward = MagicMock()
    spec._parallel_forward = MagicMock()
    spec.dp_size = 1
    spec.dp_rank = 0
    with (
        patch.object(
            torch.ops._C_ascend, "npu_copy_and_expand_eagle_inputs", npu_copy_and_expand_eagle_inputs_stub, create=True
        ),
        patch.object(module, "get_uniform_decode_token_count", return_value=6),
        patch.object(module, "dispatch_cg_and_sync_dp", return_value=(desc, None)),
        patch.object(module, "vllm_version_is", return_value=False),
        patch.object(module, "build_slot_mappings_by_layer", return_value={}),
        patch.object(module, "build_attn_metadata", return_value={"draft": object()}) as metadata,
    ):
        spec.propose(
            batch,
            {},
            {},
            torch.arange(16, dtype=torch.float32).reshape(8, 2),
            None,
            torch.tensor([1, 0]),
            torch.tensor([1, 2]),
            torch.tensor([70, 71, 72]),
            torch.tensor([80, 81, 82]),
            torch.zeros(4),
            torch.zeros(4, dtype=torch.int64),
        )
    # Request order differs from state order; request 2 samples 72, while
    # request 0 is still prefilling and uses its scheduled token 80.
    assert spec.input_buffers.input_ids[[2, 7]].tolist() == [72, 80]
    assert slots[0, [5, 10, 11]].tolist() == [-1, -1, -1]
    kwargs = metadata.call_args.kwargs
    expected_query = [0, 6, 12, 18] if full_graph else [0, 6, 12]
    assert kwargs["query_start_loc_cpu"].tolist() == expected_query
    assert kwargs["query_start_loc_gpu"].tolist() == expected_query
    assert kwargs["num_actual_tokens"] == 12
    assert kwargs["seq_lens"].tolist() == ([26, 46, 0] if full_graph else [26, 46])
    if full_graph:
        spec.prefill_cudagraph_manager.run_fullgraph.assert_called_once_with(desc)
        spec._parallel_forward.assert_not_called()
    else:
        spec._parallel_forward.assert_called_once()
        spec.prefill_cudagraph_manager.run_fullgraph.assert_not_called()
