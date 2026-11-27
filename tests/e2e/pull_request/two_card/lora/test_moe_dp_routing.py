# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""HCCL routing plus native LoRA math, using GLM's expert dimensions.

The cache fixture deliberately starts ranks with different slot histories.
It is not a checkpoint loader; real cache loading is covered by the full model.
No communication or numerical operator is mocked.
"""

import json
import socket
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("MoE LoRA routing acceptance requires NPU hardware", allow_module_level=True)
import vllm.lora.utils  # noqa: F401 - initialize the framework wrapper registry
from vllm.config import LoRAConfig
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest

import vllm_ascend.ops  # noqa: F401 - register production NPU operators
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.lora.fused_moe import (
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    prepare_lora_indices,
)
from vllm_ascend.lora.moe_dp import MoELoRACoordinator
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.ops.fused_moe import token_dispatcher
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op

HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS, EXPERTS_PER_TOKEN, LORA_RANK = 4096, 2048, 288, 8, 8
ROUTED_EXPERT_IDS = (0, 1, 142, 143, 144, 145, 286, 287)


class _InMemoryAdapterManager:
    def __init__(self, rank):
        self.device = torch.device(f"npu:{rank}")
        self._adapter_manager = SimpleNamespace(lora_slots=2, lora_index_to_id=[11 if rank == 0 else 22, None])

    def set_active_adapters(self, requests, mapping):
        required = {request.lora_int_id for request in requests}
        slots = self._adapter_manager.lora_index_to_id
        for adapter_id in sorted(required):
            if adapter_id not in slots:
                free = next(i for i, old in enumerate(slots) if old not in required)
                slots[free] = adapter_id


class _HcclDataParallelGroup:
    def all_gather(self, tensor, dim):
        assert dim == 0
        peers = [torch.empty_like(tensor) for _ in range(2)]
        dist.all_gather(peers, tensor)
        return torch.cat(peers)


def _make_adapter_factors(cache, *, ep_rank, local_experts):
    device = cache.device
    aa = tuple(
        torch.zeros(2, local_experts, LORA_RANK, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) for _ in range(2)
    )
    bb = tuple(
        torch.zeros(2, local_experts, INTERMEDIATE_SIZE, LORA_RANK, device=device, dtype=torch.bfloat16)
        for _ in range(2)
    )
    for slot, adapter_id in enumerate(cache._adapter_manager.lora_index_to_id):
        if adapter_id is None:
            continue
        coefficient = {11: 0.125, 22: 0.25, 33: 0.5}[adapter_id]
        aa[0][slot, :, 0, 0] = coefficient
        aa[1][slot, :, 0, 0] = coefficient * 2
        for expert in range(local_experts):
            global_expert = ep_rank * local_experts + expert
            bb[0][slot, expert, :, 0] = (global_expert // 8 + 1) / 64
            bb[1][slot, expert, :, 0] = -(global_expert // 8 + 1) / 64
    return aa, bb


def _check_expert_delta(
    punica, cache, hidden, experts, slots, expected_ids, expected_experts, *, ep_rank, local_experts
):
    torch.testing.assert_close(experts.cpu(), expected_experts, atol=0, rtol=0)
    markers = hidden[:, 0].cpu().double()
    expected_adapter_ids = [expected_ids.get(round(marker * 8), -1) for marker in markers.tolist()]
    actual_adapter_ids = [
        cache._adapter_manager.lora_index_to_id[slot] if slot >= 0 else -1 for slot in slots.cpu().tolist()
    ]
    assert actual_adapter_ids == expected_adapter_ids
    aa, bb = _make_adapter_factors(cache, ep_rank=ep_rank, local_experts=local_experts)
    output = torch.zeros(hidden.shape[0], INTERMEDIATE_SIZE * 2, device=hidden.device, dtype=torch.bfloat16)
    punica.add_lora_fused_moe(
        output,
        hidden,
        aa,
        bb,
        expert_ids=experts,
        adapter_enabled=torch.ones(2, device=hidden.device, dtype=torch.int32),
        token_lora_mapping=slots,
    )
    torch.npu.synchronize()
    expected = torch.zeros(output.shape, dtype=torch.float64)
    global_experts = expected_experts + ep_rank * local_experts
    for row, marker in enumerate(markers.tolist()):
        adapter_id = expected_ids.get(round(marker * 8), -1)
        coefficient = {11: 0.125, 22: 0.25, 33: 0.5}.get(adapter_id, 0)
        value = marker * coefficient * (int(global_experts[row]) // 8 + 1) / 64
        expected[row, :INTERMEDIATE_SIZE] = value
        expected[row, INTERMEDIATE_SIZE:] = -2 * value
    expected = expected.to(torch.bfloat16).double()
    actual = output.cpu().double()
    error = float((actual - expected).abs().max()) if output.numel() else 0
    assert torch.equal(actual, expected), (error, actual[:, 0].tolist(), expected[:, 0].tolist())
    assert expected.abs().max() > 0
    down_inputs = torch.zeros(hidden.shape[0], INTERMEDIATE_SIZE, device=hidden.device, dtype=torch.bfloat16)
    down_inputs[:, 0] = hidden[:, 0]
    down = torch.zeros(hidden.shape[0], HIDDEN_SIZE, device=hidden.device, dtype=torch.bfloat16)
    punica.add_lora_fused_moe(
        down,
        down_inputs,
        (aa[0][..., :INTERMEDIATE_SIZE].contiguous(),),
        (torch.cat([bb[0], bb[0]], dim=-2),),
        expert_ids=experts,
        adapter_enabled=torch.ones(2, device=hidden.device, dtype=torch.int32),
        token_lora_mapping=slots,
    )
    torch.npu.synchronize()
    down_reference = torch.cat([expected[:, :INTERMEDIATE_SIZE], expected[:, :INTERMEDIATE_SIZE]], dim=-1)
    down_error = float((down.cpu().double() - down_reference).abs().max())
    assert torch.equal(down.cpu().double(), down_reference), down_error
    return {
        "rows": hidden.shape[0],
        "max_error": error,
        "max_down_error": down_error,
        "nonzero": int(torch.count_nonzero(expected)),
        "received_ids": sorted(set(expected_ids.values())),
    }


def _dp_worker(rank, init_method, output_directory):
    # Use device collectives for token exchange and a separate CPU group for
    # adapter residency, just as the production coordinator does.
    torch.npu.set_device(rank)
    assert enable_custom_op(), "Must execute the native Ascend LoRA kernels"
    dist.init_process_group("hccl", rank=rank, world_size=2, init_method=init_method, timeout=timedelta(seconds=90))
    cpu = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
    checks = []
    stack = ExitStack()
    try:
        stack.enter_context(override_forward_context(ForwardContext({}, {}, {})))
        cache = _InMemoryAdapterManager(rank)
        punica = PunicaWrapperNPU(
            32, 8, cache.device, lora_config=LoRAConfig(max_loras=2, max_lora_rank=LORA_RANK, lora_dtype=torch.bfloat16)
        )
        coordinator = MoELoRACoordinator(cache, [punica], cpu)
        cases = [([11], [22]), ([11], [-1]), ([-1], [22]), ([11, 11, 11], [22]), ([33], [-1]), ([11], [22])]
        for case_index, rank_ids in enumerate(cases):
            # Start with divergent slot histories, then exercise base-only
            # peers, unequal token counts, eviction and restoration.
            requests = {
                LoRARequest(f"adapter-{adapter_id}", adapter_id, f"/in-memory/{adapter_id}")
                for adapter_id in rank_ids[rank]
                if adapter_id > 0
            }
            coordinator.synchronize(requests)
            local_ids = rank_ids[rank]
            slots_table = cache._adapter_manager.lora_index_to_id
            punica.update_metadata(
                LoRAMapping(tuple(max(value, 0) for value in local_ids), (max(local_ids[0], 0),), is_prefill=True),
                slots_table,
                3,
                154880,
            )
            hidden = torch.zeros(len(local_ids), HIDDEN_SIZE, device=cache.device, dtype=torch.bfloat16)
            hidden[:, 0] = (torch.arange(len(local_ids), device=cache.device) + 1 + rank * 32) / 8
            expected_ids = {
                1 + source * 32 + row: adapter_id
                for source, ids in enumerate(rank_ids)
                for row, adapter_id in enumerate(ids)
            }
            ctx = SimpleNamespace(punica_wrapper=punica, top_k=EXPERTS_PER_TOKEN, local_num_experts=NUM_EXPERTS)
            # AllGather metadata must follow the same padding and unpadding
            # as the activations. Validate the expert order independently.
            prepare = object.__new__(PrepareAndFinalizeWithAllGather)
            prepare.lora_context = ctx
            prepare.moe_config = SimpleNamespace(
                dp_size=2, pcp_size=1, is_sequence_parallel=False, dp_group=_HcclDataParallelGroup()
            )
            _EXTRA_CTX.max_tokens_across_dp = max(map(len, rank_ids))
            prepared = prepare.prepare(hidden.clone(), torch.zeros(len(local_ids), NUM_EXPERTS, device=cache.device))
            ids = torch.tensor(ROUTED_EXPERT_IDS, device=cache.device, dtype=torch.int32).repeat(
                prepared.hidden_states.shape[0], 1
            )
            gather = token_dispatcher.TokenDispatcherWithAllGather(
                num_experts=NUM_EXPERTS, num_local_experts=NUM_EXPERTS, top_k=EXPERTS_PER_TOKEN
            )
            gather.set_lora_context(ctx)
            params = MoETokenDispatchInput(
                prepared.hidden_states,
                torch.ones_like(ids, dtype=torch.float32) / EXPERTS_PER_TOKEN,
                ids,
                MoeRouterInput(None, 0, None, False),
                MoEQuantParams(),
            )
            dispatched = gather.token_dispatch(params)
            expert, slots = _recover_moe_lora_routing_allgather(ctx, dispatched.combine_metadata.expanded_row_idx, ids)
            result = _check_expert_delta(
                punica,
                cache,
                dispatched.hidden_states,
                expert,
                slots,
                expected_ids,
                torch.tensor(sorted(ROUTED_EXPERT_IDS * prepared.hidden_states.shape[0]), dtype=torch.long),
                ep_rank=0,
                local_experts=NUM_EXPERTS,
            )
            checks.append({"case": case_index, "path": "allgather", **result})

            # A quantized AlltoAll dispatcher must preserve BF16 activations
            # on every peer while expert LoRA is active anywhere in the group.
            ctx.local_num_experts = NUM_EXPERTS // 2
            prepare_lora_indices(ctx, num_tokens=len(local_ids), pad_size=0, tp_size=1, tp_rank=0)
            group = SimpleNamespace(device_group=dist.group.WORLD, rank_in_group=rank, world_size=2)
            with patch.object(token_dispatcher, "get_ep_group", return_value=group):
                exchange = token_dispatcher.TokenDispatcherWithAll2AllV(
                    num_experts=NUM_EXPERTS, num_local_experts=NUM_EXPERTS // 2, top_k=EXPERTS_PER_TOKEN
                )
                exchange.set_lora_context(ctx)
                ids = torch.tensor(ROUTED_EXPERT_IDS, device=cache.device, dtype=torch.int32).repeat(len(local_ids), 1)
                params = MoETokenDispatchInput(
                    hidden.clone(),
                    torch.ones_like(ids, dtype=torch.float32) / EXPERTS_PER_TOKEN,
                    ids,
                    MoeRouterInput(None, 0, None, False),
                    MoEQuantParams(quant_type=QuantType.W8A8),
                )
                dispatched = exchange.token_dispatch(params)
                assert dispatched.hidden_states.dtype == torch.bfloat16
                assert dispatched.dynamic_scale is None
                expert, slots = _recover_moe_lora_routing_all2all(ctx, dispatched.group_list)
                result = _check_expert_delta(
                    punica,
                    cache,
                    dispatched.hidden_states,
                    expert,
                    slots,
                    expected_ids,
                    torch.tensor(
                        sorted(
                            [
                                item % (NUM_EXPERTS // 2)
                                for item in ROUTED_EXPERT_IDS
                                if item // (NUM_EXPERTS // 2) == rank
                            ]
                            * sum(map(len, rank_ids))
                        ),
                        dtype=torch.long,
                    ),
                    ep_rank=rank,
                    local_experts=NUM_EXPERTS // 2,
                )
                checks.append({"case": case_index, "path": "quantized-alltoall-dispatch", **result})
            coordinator.step_active = False
        (Path(output_directory) / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "checks": checks,
                    "status": "PASSED",
                    "cache_loading": "in-memory fixture; requires separate full-model validation",
                },
                indent=2,
            )
        )
        dist.barrier()
    finally:
        stack.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="requires two logical NPUs")
def test_glm_shaped_expert_lora_routing_and_native_deltas(tmp_path):
    """Exercise real HCCL/CANN routing with different worker-local slot orders."""
    if not get_current_hardware_profile().supports(HardwareCapability.LORA_CUSTOM_OPS):
        pytest.skip("requires the native Ascend LoRA backend")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_dp_worker, args=(f"tcp://127.0.0.1:{port}", str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["status"] == "PASSED"
        assert len(result["checks"]) == 12
        assert all(row["max_error"] == 0 and row["max_down_error"] == 0 for row in result["checks"])
