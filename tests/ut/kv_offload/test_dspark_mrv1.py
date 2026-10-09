# SPDX-License-Identifier: Apache-2.0
"""MRV1 adapters for the shared GLM MLA DSpark PD protocol."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    DSparkContextChunk,
    DSparkContextDescriptor,
)
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.worker.dspark_pd import send_mrv1_dspark_prefill_kv
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _proposer():
    proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
    # Cross the lexical 9/10 boundary and interleave groups. The kernel's
    # context mapping list must follow model order, not group or string order.
    proposer.model = Mock()
    proposer.model.get_draft_kv_cache_layer_names.return_value = ["draft.9", "draft.10"]
    proposer.draft_attn_groups = [
        SimpleNamespace(kv_cache_group_id=2, layer_names=["draft.10"]),
        SimpleNamespace(kv_cache_group_id=1, layer_names=["draft.9"]),
    ]
    proposer.runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=size)) for size in (128, 256, 512)
            ]
        )
    )
    proposer._per_group_kernel_block_sizes = {1: 128, 2: 128}
    proposer.device = torch.device("cpu")
    proposer.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Glm5DSparkForCausalLM"]))
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        model_config=SimpleNamespace(max_model_len=1024, get_hidden_size=lambda: 4),
    )
    return proposer


def test_loaded_draft_names_available_before_attention_initialization():
    proposer = _proposer()
    del proposer.draft_attn_groups
    assert proposer.draft_attn_layer_names == {"draft.9", "draft.10"}


def test_context_layout_uses_model_order_and_logical_block_sizes():
    assert _proposer().get_draft_context_group_layout() == ((1, 2), (1, 2), {1: 256, 2: 512})


def test_context_layout_rejects_missing_loaded_layer():
    proposer = _proposer()
    proposer.draft_attn_groups.pop()
    with pytest.raises(ValueError, match="every loaded draft"):
        proposer.get_draft_context_group_layout()


def test_local_context_uses_shared_writer_and_actual_group_mapping():
    proposer = _proposer()
    chunk = DSparkContextChunk(DSparkContextDescriptor("r", "g", 16, (2, 22), 4), 0, 3)
    features = torch.zeros((3, 8), dtype=torch.bfloat16)
    module = "vllm_ascend.spec_decode.dspark_proposer"
    with (
        patch(f"{module}.get_dspark_aux_layer_ids", return_value=(2, 22)),
        patch.object(proposer, "_create_draft_vllm_config", return_value=proposer.vllm_config),
        patch(f"{module}.set_current_vllm_config"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context.initialize_draft_context_chunk"
        ) as write,
    ):
        proposer.initialize_local_context(chunk, features, {1: [7], 2: [9]})
    assert write.call_args.args == (proposer.model, chunk, features)
    assert write.call_args.kwargs["layer_group_ids"] == (1, 2)
    assert write.call_args.kwargs["block_sizes_by_group"] == {1: 256, 2: 512}
    assert write.call_args.kwargs["draft_block_ids_by_group"] == {1: (7,), 2: (9,)}


@pytest.mark.parametrize("field", ["prefill_context_parallel_size", "decode_context_parallel_size"])
def test_local_context_rejects_cp_before_writing(field):
    proposer = _proposer()
    setattr(proposer.vllm_config.parallel_config, field, 2)
    with pytest.raises(ValueError, match="context parallelism"):
        proposer.initialize_local_context(None, None, {})


def test_mrv1_batch_order_chunk_offsets_and_paired_prefix_are_preserved():
    runner = SimpleNamespace(
        drafter=Mock(),
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["b", "a"],
            num_computed_tokens_cpu=np.array([4, 0]),
            num_prompt_tokens=np.array([8, 3]),
        ),
        query_start_loc=SimpleNamespace(np=np.array([0, 4, 7])),
        _dspark_prefill_progress={"b": ("g-b", 4)},
    )
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"a": 3, "b": 4}, finished_req_ids={"old"}, kv_connector_metadata=object()
    )
    connector = Mock()
    connector.get_dspark_context_descriptor.side_effect = lambda req_id, length: DSparkContextDescriptor(
        req_id, f"g-{req_id}", length, (2, 22), 4
    )
    requests = {
        req_id: SimpleNamespace(dspark_context_generation=f"g-{req_id}", local_block_ids=([1],))
        for req_id in ("a", "b")
    }
    runner.drafter.get_draft_context_group_layout.return_value = ((0,), (0,), {0: 16})
    prefix = Mock()
    prefix_meta = object()
    # Distinct rows prove we project the reordered batch, not scheduler dict order.
    aux = [torch.arange(28, dtype=torch.bfloat16).reshape(7, 4)] * 2
    module = "vllm_ascend.worker.dspark_pd"
    with (
        patch(f"{module}.get_kv_transfer_group"),
        patch(f"{module}.find_dspark_context_connector", return_value=(connector, SimpleNamespace(requests=requests))),
        patch(f"{module}.find_dspark_prefix_connector", return_value=(prefix, prefix_meta)),
    ):
        send_mrv1_dspark_prefill_kv(runner, scheduler, aux)
    calls = runner.drafter.initialize_local_context.call_args_list
    assert [(c.args[0].descriptor.request_id, c.args[0].token_offset, c.args[0].num_tokens) for c in calls] == [
        ("b", 4, 4),
        ("a", 0, 3),
    ]
    torch.testing.assert_close(calls[1].args[1], torch.cat(aux, dim=-1)[4:7])
    assert connector.send_dspark_draft_kv.call_count == 2
    assert prefix.save_dspark_prefix.call_count == 2
    assert runner._dspark_prefill_progress == {}


def test_missing_aux_fails_before_remote_send():
    connector = Mock()
    metadata = SimpleNamespace(requests={"r": SimpleNamespace(dspark_context_generation="g")})
    module = "vllm_ascend.worker.dspark_pd"
    with (
        patch(f"{module}.get_kv_transfer_group"),
        patch(f"{module}.find_dspark_context_connector", return_value=(connector, metadata)),
        pytest.raises(RuntimeError, match="no target auxiliary"),
    ):
        send_mrv1_dspark_prefill_kv(SimpleNamespace(), SimpleNamespace(kv_connector_metadata=None), None)
    connector.send_dspark_draft_kv.assert_not_called()


@pytest.mark.parametrize("sync_self", [False, True])
def test_pp_aux_and_topk_rows_are_not_sequence_sharded(sync_self):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2))
    runner.pd_dspark_aux_layer_ids = (2, 22)
    source = IntermediateTensors(
        {
            "hidden_states": torch.ones((4, 3)),
            "residual": torch.ones((4, 3)),
            "pp_aux": torch.arange(24).reshape(8, 3),
            "pp_topk": torch.ones((8, 2)),
        }
    )
    runner.intermediate_tensors = IntermediateTensors({k: v.clone() for k, v in source.items()})
    with patch("vllm_ascend.worker.model_runner_v1.enable_sp", return_value=True):
        result = runner.sync_and_slice_intermediate_tensors(8, source, sync_self)
    assert result["hidden_states"].shape[0] == 4
    assert result["residual"].shape[0] == 4
    assert result["pp_aux"].shape[0] == 8
    assert result["pp_topk"].shape[0] == 8
    torch.testing.assert_close(result["pp_aux"], source["pp_aux"])


@pytest.mark.parametrize("bad_input", ["layers", "width", "length", "dtype"])
def test_local_context_rejects_schema_mismatch_before_device_write(bad_input):
    proposer = _proposer()
    descriptor = DSparkContextDescriptor(
        "r",
        "g",
        2048 if bad_input == "length" else 16,
        (3, 22) if bad_input == "layers" else (2, 22),
        4,
    )
    chunk = DSparkContextChunk(descriptor, 0, 3)
    features = torch.zeros(
        (3, 7 if bad_input == "width" else 8),
        dtype=torch.float32 if bad_input == "dtype" else torch.bfloat16,
    )
    module = "vllm_ascend.spec_decode.dspark_proposer"
    with (
        patch(f"{module}.get_dspark_aux_layer_ids", return_value=(2, 22)),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context.initialize_draft_context_chunk"
        ) as write,
        pytest.raises(ValueError, match="schema|BF16"),
    ):
        proposer.initialize_local_context(chunk, features, {1: [7], 2: [9]})
    write.assert_not_called()


def test_local_context_rejects_gqa_draft_before_device_write():
    proposer = _proposer()
    proposer.draft_model_config.hf_config.architectures = ["Qwen3DSparkModel"]
    with pytest.raises(ValueError, match="GLM MLA draft"):
        proposer.initialize_local_context(None, None, {})
