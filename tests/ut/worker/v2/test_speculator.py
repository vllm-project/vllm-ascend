# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
    gather,
    torch_gather_wrapper,
)


def test_calc_next_seq_lens_cpu():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.max_model_len = 100
    seq_lens_cpu = torch.tensor([10, 20, 98, 66], dtype=torch.int32)
    result = speculator._calc_next_seq_lens_cpu(seq_lens_cpu, num_reqs=3, num_reqs_padded=4, step=3)
    expected = torch.tensor([13, 23, 100, 0], dtype=torch.int32)
    torch.testing.assert_close(result, expected)


def test_get_seq_lens_cpu():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    buffers = AscendInputBuffers.__new__(AscendInputBuffers)
    buffers.seq_lens_cpu = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
    speculator.target_input_buffers = buffers
    result = speculator._get_seq_lens_cpu(num_reqs_padded=4)
    expected = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_gather_without_out():
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    index = torch.tensor([[1, 0], [0, 1]])
    result = gather(input_tensor, 1, index)
    expected = torch.tensor([[2, 1], [3, 4]])
    assert torch.equal(result, expected)


def test_gather_with_out():
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    index = torch.tensor([[1, 0], [0, 1]])
    out = torch.zeros((2, 2), dtype=torch.int64)
    result = gather(input_tensor, 1, index, out=out)
    expected = torch.tensor([[2, 1], [3, 4]])
    assert result is out
    assert torch.equal(out, expected)


def test_torch_gather_wrapper():
    original_gather = torch.gather
    with torch_gather_wrapper():
        assert torch.gather is gather
    assert torch.gather is original_gather


def test_maybe_remove_d2t_for_eagle3_same_vocab():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.method = "eagle3"
    speculator.draft_model_config = SimpleNamespace(get_vocab_size=lambda: 100)
    draft_model = SimpleNamespace(config=SimpleNamespace(draft_vocab_size=100), draft_id_to_target_id=object())
    speculator._maybe_remove_d2t(draft_model)
    assert draft_model.draft_id_to_target_id is None


def test_maybe_remove_d2t_for_eagle3_different_vocab():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.method = "eagle3"
    speculator.draft_model_config = SimpleNamespace(get_vocab_size=lambda: 100)
    mapping = object()
    draft_model = SimpleNamespace(config=SimpleNamespace(draft_vocab_size=80), draft_id_to_target_id=mapping)
    speculator._maybe_remove_d2t(draft_model)
    assert draft_model.draft_id_to_target_id is mapping


def test_maybe_remove_d2t_for_non_eagle3():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.method = "mtp"
    mapping = object()
    draft_model = SimpleNamespace(draft_id_to_target_id=mapping)
    speculator._maybe_remove_d2t(draft_model)
    assert draft_model.draft_id_to_target_id is mapping


def test_draft_prefill_attn_groups():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_groups = object()
    speculator.target_attn_groups = object()
    speculator.replicated_pcp = True
    assert speculator.draft_prefill_attn_groups is speculator.attn_groups
    speculator.replicated_pcp = False
    assert speculator.draft_prefill_attn_groups is speculator.target_attn_groups


def test_load_model_disables_mm_inputs_for_pp():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2))
    speculator.supports_mm_inputs = True
    target_model = MagicMock()
    with patch.object(AutoRegressiveSpeculator, "load_model") as parent_load_model:
        speculator.load_model(target_model)
    parent_load_model.assert_called_once_with(target_model)
    assert speculator.supports_mm_inputs is False


def test_load_draft_model_calls_remove_d2t():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    target_model = MagicMock()
    draft_model = MagicMock()
    target_attn_layer_names = {"layer.0"}
    speculator._maybe_remove_d2t = MagicMock()
    with patch.object(
        AutoRegressiveSpeculator, "load_draft_model", return_value=draft_model
    ) as parent_load_draft_model:
        result = speculator.load_draft_model(target_model, target_attn_layer_names)
    parent_load_draft_model.assert_called_once_with(target_model, target_attn_layer_names)
    speculator._maybe_remove_d2t.assert_called_once_with(draft_model)
    assert result is draft_model


def test_init_cudagraph_manager_enforce_eager():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.speculative_config = SimpleNamespace(enforce_eager=True)
    speculator.prefill_cudagraph_manager = MagicMock()
    speculator.decode_cudagraph_manager = MagicMock()
    speculator.update_stream = object()
    with patch.object(AutoRegressiveSpeculator, "init_cudagraph_manager") as parent_init:
        speculator.init_cudagraph_manager(CUDAGraphMode.FULL)
    parent_init.assert_called_once_with(CUDAGraphMode.NONE)
    assert speculator.prefill_cudagraph_manager.speculator is speculator
    assert speculator.decode_cudagraph_manager.speculator is speculator
    assert speculator.prefill_cudagraph_manager.update_stream is speculator.update_stream
    assert speculator.decode_cudagraph_manager.update_stream is speculator.update_stream


def test_init_cudagraph_manager_keeps_graph_mode():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.speculative_config = SimpleNamespace(enforce_eager=False)
    speculator.prefill_cudagraph_manager = MagicMock()
    speculator.decode_cudagraph_manager = MagicMock()
    speculator.update_stream = object()
    with patch.object(AutoRegressiveSpeculator, "init_cudagraph_manager") as parent_init:
        speculator.init_cudagraph_manager(CUDAGraphMode.FULL)
    parent_init.assert_called_once_with(CUDAGraphMode.FULL)


def test_run_model_broadcasts_replicated_hidden_states():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.replicated_pcp = True
    last_hidden_states = torch.tensor([1])
    hidden_states = torch.tensor([2])
    expected = (torch.tensor([3]), torch.tensor([4]))
    with (
        patch.object(
            AutoRegressiveSpeculator,
            "_run_model",
            return_value=(last_hidden_states, hidden_states),
        ) as parent_run_model,
        patch.object(AscendPCPManager, "broadcast_replicated_hidden_states", return_value=expected) as broadcast,
    ):
        result = speculator._run_model(3, None, None, None)
    parent_run_model.assert_called_once_with(3, None, None, None, CUDAGraphMode.NONE, None)
    broadcast.assert_called_once_with(last_hidden_states, hidden_states, 3, replicated_pcp=True)
    assert result is expected


def test_generate_draft_updates_attn_metadata():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    attn_metadata = {"layer.0": MagicMock()}
    speculator._update_decode_attn_metadata = MagicMock()
    with patch.object(AutoRegressiveSpeculator, "_generate_draft") as parent_generate:
        speculator._generate_draft(2, 4, attn_metadata, None, None)
    parent_generate.assert_called_once_with(2, 4, attn_metadata, None, None, CUDAGraphMode.NONE)
    speculator._update_decode_attn_metadata.assert_called_once_with(attn_metadata, 1, 2)


def test_generate_draft_skips_update_without_attn_metadata():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator._update_decode_attn_metadata = MagicMock()
    with patch.object(AutoRegressiveSpeculator, "_generate_draft"):
        speculator._generate_draft(2, 4, None, None, None)
    speculator._update_decode_attn_metadata.assert_not_called()


def test_multi_step_decode_full_graph():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.decode_cudagraph_manager = MagicMock()
    batch_desc = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)
    with patch.object(AutoRegressiveSpeculator, "_multi_step_decode") as parent_decode:
        speculator._multi_step_decode(2, False, batch_desc, None)
    speculator.decode_cudagraph_manager.run_fullgraph.assert_called_once_with(batch_desc)
    parent_decode.assert_not_called()


def test_multi_step_decode_non_full_graph():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    batch_desc = SimpleNamespace(cg_mode=CUDAGraphMode.NONE)
    seq_lens_cpu_upper_bound = torch.tensor([10, 20], dtype=torch.int32)
    with patch.object(AutoRegressiveSpeculator, "_multi_step_decode") as parent_decode:
        speculator._multi_step_decode(2, False, batch_desc, None, seq_lens_cpu_upper_bound)
    parent_decode.assert_called_once_with(2, False, batch_desc, None, seq_lens_cpu_upper_bound)


def test_prefill_filters_target_only_metadata():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    draft_metadata = object()
    target_metadata = object()
    attn_metadata = {"draft": draft_metadata, "target": target_metadata}
    slot_mappings = {"draft": torch.tensor([1])}
    speculator.draft_attn_layer_names = {"draft"}
    speculator._prepare_replicated_prefill_attn = MagicMock(return_value=(attn_metadata, slot_mappings))
    with patch.object(AutoRegressiveSpeculator, "_prefill") as parent_prefill:
        speculator._prefill(2, 4, attn_metadata, slot_mappings, None)
    parent_prefill.assert_called_once_with(
        2,
        4,
        {"draft": draft_metadata},
        slot_mappings,
        None,
        CUDAGraphMode.NONE,
        None,
    )


def test_build_draft_attn_metadata_sets_decode_only():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    metadata = SimpleNamespace(attn_state=None)
    speculator.input_batch = SimpleNamespace(is_prefilling_np=np.array([True, False]))
    speculator.input_buffers = SimpleNamespace(positions=torch.tensor([0, 1]))
    speculator.use_dcp = False
    speculator.draft_vllm_config = SimpleNamespace(parallel_config=object())
    speculator._build_uniform_attn_metadata = MagicMock(return_value={"draft": metadata, "none": None})
    speculator._update_decode_attn_metadata = MagicMock()
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.build_draft_attn_metadata_factory",
        return_value=nullcontext(),
    ):
        result = speculator._build_draft_attn_metadata(2, 2, 2, seq_lens, 1)
    assert result["draft"] is metadata
    assert metadata.attn_state == AscendAttentionState.DecodeOnly
    speculator._update_decode_attn_metadata.assert_called_once_with({"draft": metadata, "none": None}, 1, 2)


def test_build_uniform_attn_metadata_sets_decode_only():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    metadata = SimpleNamespace(attn_state=None)
    speculator.input_batch = SimpleNamespace(is_prefilling_np=np.array([False, False]))
    speculator.input_buffers = SimpleNamespace(positions=torch.tensor([0, 1]))
    batch_desc = SimpleNamespace(num_tokens=2)
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.build_draft_attn_metadata_factory",
            return_value=nullcontext(),
        ),
        patch.object(
            AutoRegressiveSpeculator,
            "_build_uniform_attn_metadata",
            return_value={"draft": metadata},
            create=True,
        ) as parent_build,
    ):
        result = speculator._build_uniform_attn_metadata(
            batch_desc,
            2,
            1,
            seq_lens,
            1,
        )

    assert result == {"draft": metadata}
    assert metadata.attn_state == AscendAttentionState.DecodeOnly
    parent_build.assert_called_once_with(
        batch_desc,
        2,
        1,
        seq_lens,
        1,
        True,
        None,
    )


def test_build_attn_metadata_sets_decode_only():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    metadata = SimpleNamespace(attn_state=None)
    speculator.input_batch = SimpleNamespace(is_prefilling_np=np.array([False, False]))
    speculator.input_buffers = SimpleNamespace(positions=torch.tensor([0, 1]))
    batch_desc = SimpleNamespace(num_tokens=2)
    query_start_loc_np = np.array([0, 1, 2], dtype=np.int32)
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.build_draft_attn_metadata_factory",
            return_value=nullcontext(),
        ),
        patch.object(
            AutoRegressiveSpeculator,
            "_build_attn_metadata",
            return_value={"draft": metadata},
            create=True,
        ) as parent_build,
    ):
        result = speculator._build_attn_metadata(
            2,
            batch_desc,
            query_start_loc_np,
            seq_lens,
            1,
        )

    assert result == {"draft": metadata}
    assert metadata.attn_state == AscendAttentionState.DecodeOnly
    parent_build.assert_called_once_with(
        2,
        batch_desc,
        query_start_loc_np,
        seq_lens,
        1,
        True,
        None,
    )


def test_build_draft_attn_metadatas_prefill():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    draft_metadata = object()
    speculator.model_state = SimpleNamespace(attn_metadata={"draft": draft_metadata, "target": object()})
    speculator.draft_attn_layer_names = {"draft"}
    prepared = {"draft": object()}
    speculator._prepare_replicated_prefill_attn = MagicMock(return_value=(prepared, None))
    result = speculator.build_draft_attn_metadatas(4, 8, True)
    assert result == [prepared]
    filtered = speculator._prepare_replicated_prefill_attn.call_args.args[0]
    assert filtered == {"draft": draft_metadata}


def test_build_draft_attn_metadatas_decode_updates_each_step():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.model_state = SimpleNamespace(attn_metadata={"draft": object(), "target": object()})
    speculator.draft_attn_layer_names = {"draft"}
    speculator.input_batch = SimpleNamespace(num_reqs=2)
    first = {"draft": object()}
    second = {"draft": object()}
    speculator._init_decode_draft_attn_metadatas = MagicMock(return_value=[first, second])
    speculator._update_decode_attn_metadata = MagicMock()
    result = speculator.build_draft_attn_metadatas(4, 4, False)
    assert result == [first, second]
    speculator._update_decode_attn_metadata.assert_any_call(first, 1, 2)
    speculator._update_decode_attn_metadata.assert_any_call(second, 2, 2)
    assert speculator._update_decode_attn_metadata.call_count == 2


def test_init_decode_draft_attn_metadatas_sparse_attention():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    for architecture in ("DSA", "SFA"):
        speculator.attn_architecture = architecture
        assert speculator._init_decode_draft_attn_metadatas({"draft": object()}, 4) == []


def test_init_decode_draft_attn_metadatas_gqa():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_architecture = "GQA"
    speculator.input_batch = SimpleNamespace(num_reqs=2, seq_lens_cpu_upper_bound=torch.tensor([10, 20]))
    speculator.input_buffers = SimpleNamespace(
        draft_seq_lens_cpus=[
            torch.tensor([11, 21, 0]),
            torch.tensor([12, 22, 0]),
        ]
    )
    metadata = SimpleNamespace(attn_state=None, seq_lens_cpu=None)
    speculator._build_draft_attn_metadata = MagicMock(return_value={"draft": metadata})
    result = speculator._init_decode_draft_attn_metadatas({"draft": metadata}, 3)
    assert len(result) == 2
    assert result[0]["draft"].attn_state == AscendAttentionState.DecodeOnly
    assert torch.equal(result[0]["draft"].seq_lens_cpu, torch.tensor([11, 21, 0]))
    assert torch.equal(result[1]["draft"].seq_lens_cpu, torch.tensor([12, 22, 0]))


def test_update_decode_attn_metadata_gqa():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_architecture = "GQA"
    speculator.max_model_len = 32
    speculator.use_dcp = False
    metadata = SimpleNamespace(
        seq_lens_cpu=torch.zeros(4, dtype=torch.int32),
        seq_lens_list=None,
        actual_seq_lengths_q=None,
    )
    speculator._get_seq_lens_cpu = MagicMock(return_value=torch.tensor([10, 20, 31, 7], dtype=torch.int32))
    speculator._update_decode_attn_metadata({"draft": metadata}, step=2, num_reqs=3)
    assert metadata.seq_lens_list == [12, 22, 32, 0]
    assert metadata.actual_seq_lengths_q == [1, 2, 3, 4]
    assert torch.equal(metadata.seq_lens_cpu, torch.tensor([12, 22, 32, 0], dtype=torch.int32))


def test_build_fia_params_prefill():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    block_table = torch.arange(12).reshape(4, 3)
    metadata = SimpleNamespace(block_tables=block_table, actual_seq_lengths_q=[1, 2], seq_lens_list=[10, 20])
    speculator.model_state = SimpleNamespace(attn_metadata={"draft": metadata})
    speculator.draft_attn_layer_names = {"draft"}
    result = speculator.build_fia_params(3, {"draft": metadata}, True)
    assert len(result) == 1
    assert result[0]["layer_name"] == "draft"
    assert result[0]["actual_seq_lengths"] == [1, 2]
    assert result[0]["actual_seq_lengths_kv"] == [10, 20]
    assert result[0]["block_table"].shape == (4, 3)


def test_build_fia_params_decode():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    metadata = SimpleNamespace(block_tables=torch.arange(12).reshape(4, 3))
    speculator.model_state = SimpleNamespace(attn_metadata={"draft": metadata})
    speculator.draft_attn_layer_names = {"draft"}
    speculator.input_batch = SimpleNamespace(num_reqs=2, seq_lens_np=np.array([10, 29]))
    speculator.num_speculative_steps = 3
    speculator.max_model_len = 30
    result = speculator.build_fia_params(3, {"draft": metadata}, False)
    assert len(result) == 2
    assert result[0]["actual_seq_lengths"] == [1, 2, 3]
    assert result[0]["actual_seq_lengths_kv"] == [11, 30, 0]
    assert result[1]["actual_seq_lengths_kv"] == [12, 30, 0]


def test_set_attn_detects_architecture():
    cases = [
        (AscendDSABackend, "DSA"),
        (AscendMLABackend, "MLA"),
        (AscendSFABackend, "SFA"),
        (AscendAttentionBackend, "GQA"),
    ]
    for backend, expected in cases:
        speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
        speculator.vllm_config = object()
        speculator.attn_groups = object()
        with (
            patch(
                "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.set_current_vllm_config",
                return_value=nullcontext(),
            ),
            patch.object(AutoRegressiveSpeculator, "set_attn"),
            patch(
                "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator._get_graph_update_backend",
                return_value=backend,
            ),
        ):
            speculator.set_attn(object(), object(), object(), object(), object())
        assert speculator.attn_backend is backend
        assert speculator.attn_architecture == expected


def test_capture_single_step_only_captures_prefill():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.last_token_indices = MagicMock()
    speculator.prefill_cudagraph_manager = MagicMock()
    speculator.prefill_cudagraph_manager.use_breakable_cg = True
    speculator.decode_cudagraph_manager = MagicMock()
    speculator.model = object()
    speculator.model_state = object()
    speculator.target_input_buffers = object()
    speculator.block_tables = object()
    speculator.target_attn_groups = object()
    speculator.replicated_pcp = False
    speculator.kv_cache_config = object()
    speculator.num_speculative_steps = 1
    with patch(
        "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.disable_target_pcp_for_replicated_draft",
        return_value=nullcontext(),
    ):
        speculator.capture()
    speculator.last_token_indices.zero_.assert_called_once_with()
    speculator.prefill_cudagraph_manager.init_breakable_cg_runner.assert_called_once_with(speculator.model)
    speculator.prefill_cudagraph_manager.capture.assert_called_once()
    speculator.decode_cudagraph_manager.capture.assert_not_called()


def test_capture_multi_step_captures_decode():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.last_token_indices = MagicMock()
    speculator.prefill_cudagraph_manager = MagicMock()
    speculator.prefill_cudagraph_manager.use_breakable_cg = False
    speculator.decode_cudagraph_manager = MagicMock()
    speculator.model_state = object()
    speculator.target_input_buffers = object()
    speculator.input_buffers = object()
    speculator.block_tables = object()
    speculator.target_attn_groups = object()
    speculator.attn_groups = object()
    speculator.replicated_pcp = False
    speculator.kv_cache_config = object()
    speculator.num_speculative_steps = 3
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.disable_target_pcp_for_replicated_draft",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.build_attn_metadata_wrapper",
            return_value=nullcontext(),
        ),
    ):
        speculator.capture()
    speculator.prefill_cudagraph_manager.capture.assert_called_once()
    speculator.decode_cudagraph_manager.capture.assert_called_once()


def test_propose_replicated_pcp_disables_dp_sync():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.replicated_pcp = True
    input_batch = object()
    args = [object() for _ in range(10)]
    dp_sync = object()
    expected = object()
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.disable_target_pcp_for_replicated_draft",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.build_attn_metadata_wrapper",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.torch_gather_wrapper",
            return_value=nullcontext(),
        ),
        patch.object(AutoRegressiveSpeculator, "propose", return_value=expected) as parent_propose,
    ):
        result = speculator.propose(input_batch, *args, dp_sync=dp_sync)
    assert speculator.input_batch is input_batch
    assert parent_propose.call_args.args[11] is None
    assert result is expected


def test_init_dcp_disabled():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.draft_vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=1))

    speculator._init_dcp()

    assert speculator.use_dcp is False
    assert speculator.dcp_manager is None


def test_init_dcp_enabled():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.draft_vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=2))
    speculator.max_num_tokens = 128
    speculator.max_num_reqs = 16
    speculator.device = torch.device("cpu")
    speculator.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True))
    dcp_group = SimpleNamespace(rank_in_group=1)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.get_dcp_group",
            return_value=dcp_group,
        ),
        patch("vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.DCPManager") as dcp_manager_cls,
    ):
        speculator._init_dcp()

    assert speculator.use_dcp is True
    assert speculator.dcp_manager is dcp_manager_cls.return_value
    dcp_manager_cls.assert_called_once_with(
        dcp_world_size=2,
        dcp_rank=1,
        max_buffer_num_tokens=128,
        max_num_reqs=16,
        device=torch.device("cpu"),
        vllm_config=speculator.draft_vllm_config,
        use_async_scheduling=True,
    )


def test_update_decode_attn_metadata_mla_with_dcp():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_architecture = "MLA"
    speculator.max_model_len = 32
    speculator.use_dcp = True

    local_seq_lens = torch.tensor([6, 11, 0], dtype=torch.int32)
    dcp_manager = SimpleNamespace(
        prepare_dcp_local_seq_lens_cpu=MagicMock(return_value=local_seq_lens),
        dcp_world_rank=1,
    )
    speculator.dcp_manager = dcp_manager
    speculator.draft_vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=1,
        )
    )

    decode_metadata = MagicMock()
    metadata = SimpleNamespace(
        seq_lens_cpu=torch.zeros(3, dtype=torch.int32),
        decode=decode_metadata,
    )
    seq_lens_cpu = torch.tensor([10, 20, 30], dtype=torch.int32)
    speculator._get_seq_lens_cpu = MagicMock(return_value=seq_lens_cpu)

    speculator._update_decode_attn_metadata(
        {"draft": metadata},
        step=1,
        num_reqs=2,
    )

    expected = torch.tensor([11, 21, 0], dtype=torch.int32)

    dcp_manager.prepare_dcp_local_seq_lens_cpu.assert_called_once()
    prepare_args = dcp_manager.prepare_dcp_local_seq_lens_cpu.call_args.args
    torch.testing.assert_close(prepare_args[0], expected)

    assert decode_metadata.seq_lens_list == [11, 21, 0]
    assert decode_metadata.actual_seq_lengths_q == [1, 2, 3]

    decode_metadata.update_dcp_seq_lens_cpu.assert_called_once()
    update_args = decode_metadata.update_dcp_seq_lens_cpu.call_args.args
    update_kwargs = decode_metadata.update_dcp_seq_lens_cpu.call_args.kwargs

    torch.testing.assert_close(update_args[0], expected)
    torch.testing.assert_close(update_args[1], local_seq_lens)
    torch.testing.assert_close(update_args[2], torch.ones_like(expected))
    assert update_kwargs["dcp_size"] == 2
    assert update_kwargs["dcp_rank"] == 1
    assert update_kwargs["cp_kv_cache_interleave_size"] == 1

    assert torch.equal(metadata.seq_lens_cpu, expected)
