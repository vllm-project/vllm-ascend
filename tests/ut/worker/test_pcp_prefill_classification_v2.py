# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.attention.utils import split_decodes_and_prefills
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


def finalize_batch(global_batch, local_batch, *, is_decode_sharded=False, hf_text_config=None):
    manager = AscendPCPManager.__new__(AscendPCPManager)
    if hf_text_config is None:
        hf_text_config = SimpleNamespace(index_topk=2048)
    manager.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=hf_text_config))
    manager.kv_cache_config = None
    manager._input_buffers = SimpleNamespace(seq_lens_np=np.zeros(local_batch.num_reqs_after_padding, dtype=np.int32))
    with (
        patch.object(AscendPCPManager, "is_decode_sharded", new_callable=PropertyMock, return_value=is_decode_sharded),
        patch.object(manager, "_pad_hidden_restore_idx"),
        patch.object(manager, "_pad_replicated_decode_graph", return_value=local_batch),
        patch("vllm_ascend.worker.v2.pcp_manager.build_attn_state"),
    ):
        return manager._finalize_partition(global_batch, local_batch, None)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("is_global_prefill", [True, False])
@pytest.mark.parametrize("is_decode_sharded", [True, False])
def test_short_prefill_lookahead_keeps_pcp_collective_participation(rank, is_global_prefill, is_decode_sharded):
    # One real prompt token plus MTP5 splits into six one-token shards.
    # Ranks 1..5 begin beyond the prompt; ranks 6,7 carry an empty placeholder.
    num_tokens = int(not is_global_prefill or rank < 6)
    local_start = 1536 + (rank if is_global_prefill and rank < 6 else 0)
    local_prefill = is_global_prefill and local_start < 1537
    global_batch = SimpleNamespace(req_ids=["request"], is_prefilling_np=np.array([is_global_prefill]))
    local_batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        num_reqs_after_padding=2,
        num_tokens=num_tokens,
        num_computed_tokens_np=np.array([local_start]),
        num_scheduled_tokens=np.array([num_tokens]),
        seq_lens_cpu_upper_bound=torch.zeros(1, dtype=torch.int32),
        is_prefilling_np=np.array([local_prefill, False]),
        has_prefill=local_prefill,
        decode_graph_eligible=not local_prefill,
    )
    result = finalize_batch(global_batch, local_batch, is_decode_sharded=is_decode_sharded)
    common = SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=num_tokens,
        num_reqs=1,
        num_actual_tokens=num_tokens,
        query_start_loc_cpu=torch.tensor([0, num_tokens], dtype=torch.int32),
        is_prefilling=torch.from_numpy(result.is_prefilling_np),
    )
    with patch("vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled", return_value=False):
        _, _, num_decode_tokens, _ = split_decodes_and_prefills(
            common, decode_threshold=6, treat_short_extends_as_decodes=False
        )
    group = SimpleNamespace(
        world_size=8,
        all_gather=MagicMock(side_effect=lambda value, dim=0: torch.cat([value] * 8, dim=dim)),
    )
    with patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group):
        _gather_prefill_cache_inputs((torch.zeros(1, 4),), torch.arange(8), num_decode_tokens)

    assert group.all_gather.call_count == int(is_global_prefill)
    assert result.has_prefill == is_global_prefill
    assert result.decode_graph_eligible != is_global_prefill
    assert not result.is_prefilling_np[1]
    assert global_batch.is_prefilling_np.tolist() == [is_global_prefill]


def test_mixed_pcp_batch_restores_flags_by_request_identity():
    global_batch = SimpleNamespace(req_ids=["decode", "prefill"], is_prefilling_np=np.array([False, True]))
    local_batch = SimpleNamespace(
        req_ids=["decode", "prefill", "prefill"],
        num_reqs=3,
        num_reqs_after_padding=4,
        num_tokens=3,
        num_computed_tokens_np=np.array([2000, 1537, 1536]),
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.zeros(3, dtype=torch.int32),
        is_prefilling_np=np.array([False, False, True, False]),
        has_prefill=True,
        decode_graph_eligible=False,
    )
    result = finalize_batch(global_batch, local_batch)
    assert result.is_prefilling_np.tolist() == [False, True, True, False]
    assert result.has_prefill
    assert not result.decode_graph_eligible


@pytest.mark.parametrize("has_graph_eligibility", [True, False])
@pytest.mark.parametrize("is_local_prefill", [True, False])
def test_mixed_batch_shard_preserves_graph_restrictions(has_graph_eligibility, is_local_prefill):
    global_batch = SimpleNamespace(req_ids=["decode", "prefill"], is_prefilling_np=np.array([False, True]))
    local_batch = SimpleNamespace(
        req_ids=["prefill" if is_local_prefill else "decode"],
        num_reqs=1,
        num_reqs_after_padding=1,
        num_tokens=1,
        num_computed_tokens_np=np.array([2000]),
        num_scheduled_tokens=np.ones(1, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.zeros(1, dtype=torch.int32),
        is_prefilling_np=np.array([False]),
        has_prefill=False,
    )
    if has_graph_eligibility:
        local_batch.decode_graph_eligible = False

    result = finalize_batch(global_batch, local_batch)

    assert result.has_prefill == is_local_prefill
    assert result.is_prefilling_np.tolist() == [is_local_prefill]
    if has_graph_eligibility:
        assert not result.decode_graph_eligible
    else:
        assert not hasattr(result, "decode_graph_eligible")


@pytest.mark.parametrize(
    "hf_text_config,expected_prefill",
    [
        (SimpleNamespace(), False),
        (SimpleNamespace(index_topk=2048, compress_ratios=[4]), False),
        (SimpleNamespace(index_topk=2048), True),
        (SimpleNamespace(index_topk=2048, index_kpool=4), True),
    ],
    ids=["dense", "compressed-dsa", "sfa", "glm-kpool-sfa"],
)
def test_prefill_restoration_is_restricted_to_sfa(hf_text_config, expected_prefill):
    global_batch = SimpleNamespace(req_ids=["request"], is_prefilling_np=np.array([True]))
    local_batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        num_reqs_after_padding=1,
        num_tokens=1,
        num_computed_tokens_np=np.array([1537]),
        num_scheduled_tokens=np.ones(1, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.zeros(1, dtype=torch.int32),
        is_prefilling_np=np.array([False]),
        has_prefill=False,
        decode_graph_eligible=True,
    )

    result = finalize_batch(global_batch, local_batch, hf_text_config=hf_text_config)

    assert result.is_prefilling_np.tolist() == [expected_prefill]
    assert result.has_prefill == expected_prefill
    assert result.decode_graph_eligible != expected_prefill


@pytest.mark.parametrize("remaining_prompt_tokens", [1, 2], ids=["one-prompt-token", "two-prompt-tokens"])
@pytest.mark.parametrize("mixed", [False, True], ids=["prefill-only", "mixed"])
@pytest.mark.parametrize(
    "tp_size,pcp_size,dcp_size,global_rank",
    [(1, 8, 1, rank) for rank in range(8)] + [(2, 2, 4, rank) for rank in range(4)],
)
def test_sfa_mtp5_prefill_gathers_main_kv(remaining_prompt_tokens, mixed, tp_size, pcp_size, dcp_size, global_rank):
    """Real partitioning must retain SFA prefill identity and matching KV sends.

    Main shards ordinary SFA decodes; GLM KPool still replicates them. Exercise
    sharded SFA with DCP1 and GLM KPool with DCP4 without depending on #22131.

    Cover sharded and replicated decode policies, empty ranks, and mixed batches.
    Only the device copies, cache writes, and distributed collective are mocked.
    """
    lookahead = 5
    computed_tokens = 1536
    scheduled_tokens = remaining_prompt_tokens + lookahead
    total_tokens = scheduled_tokens + int(mixed)
    num_reqs = 1 + int(mixed)
    pcp_rank, tp_rank = divmod(global_rank, tp_size)
    device = torch.device("cpu")
    buffers = AscendInputBuffers(num_reqs, total_tokens, device)
    base = InputBatch.make_dummy(num_reqs, total_tokens, buffers, max_query_len=scheduled_tokens)
    global_batch = AscendInputBatch(
        **base.__dict__,
        seq_lens_np=np.array(([2001] if mixed else []) + [computed_tokens + scheduled_tokens], dtype=np.int32),
    )
    global_batch.req_ids = (["decode"] if mixed else []) + ["warm-prefill"]
    global_batch.num_scheduled_tokens = np.array(([1] if mixed else []) + [scheduled_tokens], dtype=np.int32)
    global_batch.query_start_loc_np = np.concatenate(([0], global_batch.num_scheduled_tokens.cumsum())).astype(np.int32)
    global_batch.query_start_loc.copy_(torch.from_numpy(global_batch.query_start_loc_np))
    global_batch.num_computed_tokens_np[:] = ([2000] if mixed else []) + [computed_tokens]
    global_batch.num_computed_prefill_tokens_np[:] = global_batch.num_computed_tokens_np
    global_batch.prefill_len_np[:] = ([2000] if mixed else []) + [computed_tokens + remaining_prompt_tokens]
    global_batch.is_prefilling_np[:] = ([False] if mixed else []) + [True]
    global_batch.has_prefill = True
    global_batch.decode_graph_eligible = False
    global_batch.num_draft_tokens = lookahead
    global_batch.num_draft_tokens_per_req = np.array(([0] if mixed else []) + [lookahead], dtype=np.int32)
    global_batch.input_ids.copy_(torch.arange(total_tokens, dtype=torch.int32))
    global_batch.positions.copy_(
        torch.tensor(([2000] if mixed else []) + list(range(computed_tokens, computed_tokens + scheduled_tokens)))
    )
    global_batch.seq_lens.copy_(torch.from_numpy(global_batch.seq_lens_np))
    global_batch.seq_lens_cpu_upper_bound.copy_(global_batch.seq_lens)
    global_batch.is_padding.fill_(False)

    manager = AscendPCPManager(
        pcp_size,
        pcp_rank,
        device,
        shard_decode_requests=dcp_size == 1,
        max_num_reqs=num_reqs,
        max_num_tokens=total_tokens,
        dcp_world_size=dcp_size,
        dcp_rank=tp_rank * pcp_size + pcp_rank,
    )
    manager.bind_config(
        SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(index_topk=2048, **({"index_kpool": 4} if dcp_size > 1 else {}))
            ),
            parallel_config=ParallelConfig(
                tensor_parallel_size=tp_size,
                prefill_context_parallel_size=pcp_size,
                decode_context_parallel_size=dcp_size,
            ),
            speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=lookahead),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        )
    )
    manager.kv_cache_config = None

    def copy_to_cpu(value, out=None, device=None):
        value = torch.as_tensor(value, device="cpu")
        return value if out is None else out.copy_(value)

    with (
        patch("vllm.v1.worker.gpu.pcp_manager.async_tensor_h2d", side_effect=copy_to_cpu),
        patch("vllm_ascend.worker.v2.pcp_manager.build_attn_state"),
    ):
        local_batch = manager.partition_batch(
            global_batch, BatchExecutionDescriptor(CUDAGraphMode.NONE, total_tokens, None)
        )

    if dcp_size > 1 and not mixed:
        expected_offsets = (
            list(range(6)) if remaining_prompt_tokens == 1 else ([0, 1, 6] if pcp_rank == 0 else [2, 3, 4, 5])
        )
        assert local_batch.input_ids[: local_batch.num_tokens].tolist() == expected_offsets
    common = SimpleNamespace(
        max_query_len=int(local_batch.num_scheduled_tokens.max()),
        num_reqs=local_batch.num_reqs,
        num_actual_tokens=local_batch.num_tokens,
        query_start_loc_cpu=torch.from_numpy(local_batch.query_start_loc_np),
        is_prefilling=torch.from_numpy(local_batch.is_prefilling_np),
    )
    with patch("vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled", return_value=False):
        _, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common, decode_threshold=lookahead + 1, treat_short_extends_as_decodes=False
        )

    metadata = SimpleNamespace(
        num_input_tokens=local_batch.num_tokens_after_padding,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        pcp_has_global_prefill=True,
        uses_pcp_sharded_layout=True,
    )
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = True
    impl.qk_rope_head_dim = 0
    impl.is_pcp_decode_sharded = manager.is_decode_sharded
    group = SimpleNamespace(world_size=pcp_size, rank_in_group=pcp_rank)
    kv = torch.zeros(local_batch.num_tokens, 4)
    slots = torch.arange(pcp_size * metadata.num_input_tokens)
    with (
        patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=group),
        patch(
            "vllm_ascend.attention.sfa_v1.all_gather_async",
            side_effect=lambda value, group: (torch.cat([value] * group.world_size), None),
        ) as gather,
        patch.object(AscendSFAImpl, "_exec_kv_local", return_value=(None, None)) as write,
    ):
        finish = impl.exec_kv(kv, None, None, (), slots, metadata, defer_wait=True)
        finish()

    gather.assert_called_once()
    expected_rows = (1 + int(mixed)) if dcp_size == 1 else (6 if remaining_prompt_tokens == 1 else 4)
    assert gather.call_args.args[0].shape == (expected_rows, kv.shape[1])
    write.assert_called_once()
    replicated_decode_tokens = int(mixed and dcp_size > 1)
    assert write.call_args.args[0].shape == (pcp_size * expected_rows + replicated_decode_tokens, kv.shape[1])
    expected_slots = slots
    if replicated_decode_tokens:
        rank_slots = slots.view(pcp_size, metadata.num_input_tokens)
        expected_slots = torch.cat((rank_slots[0, :1], rank_slots[:, 1:].flatten()))
    torch.testing.assert_close(write.call_args.args[4], expected_slots)
    assert num_decode_tokens == int(mixed and (dcp_size > 1 or pcp_rank == 0))
    expected_flags = [req_id == "warm-prefill" for req_id in local_batch.req_ids]
    assert local_batch.is_prefilling_np[: local_batch.num_reqs].tolist() == expected_flags
    assert local_batch.has_prefill == any(expected_flags)
    if any(expected_flags):
        assert not local_batch.decode_graph_eligible
    assert global_batch.is_prefilling_np.tolist() == ([False] if mixed else []) + [True]
