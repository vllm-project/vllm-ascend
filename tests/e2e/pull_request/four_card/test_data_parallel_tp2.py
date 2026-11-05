import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_dp_group,
    get_ep_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import create_forward_context, override_forward_context
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import wait_until_npu_memory_free
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState, refresh_model_routing_tables
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.patch.platform.patch_fused_moe import _ascend_apply_eplb_mapping
from vllm_ascend.utils import is_950

MODELS = ["Qwen/Qwen3-30B-A3B"]
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PARALLEL_SCRIPT = REPO_ROOT / "examples" / "offline_data_parallel.py"
EPLB_WORLD_SIZE = 4
EPLB_LOGICAL_EXPERTS = 8
EPLB_LOCAL_EXPERTS = 3
EPLB_ROWS = 4
EPLB_HIDDEN_SIZE = 128
EPLB_TIMEOUT = 300


@pytest.mark.skip(reason="broken, fix me")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"})
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen3_inference_dp2_tp2(model, max_tokens):
    env = os.environ.copy()

    cmd = [
        sys.executable,
        str(DATA_PARALLEL_SCRIPT),
        "--model",
        model,
        "--dp-size",
        "2",
        "--tp-size",
        "2",
        "--node-size",
        "1",
        "--node-rank",
        "0",
        "--trust-remote-code",
    ]

    print(f"Running subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    output = proc.stdout.decode(errors="ignore")

    print(output)

    assert "DP rank 0 needs to process" in output
    assert "DP rank 1 needs to process" in output
    assert "Generated text:" in output
    assert proc.returncode == 0


@torch.inference_mode()
def _run_a5_allgather_eplb(rank, port, graph_mode):
    torch.npu.set_device(rank)
    # Initialize the global world before enabling DP rank offsets.
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=EPLB_WORLD_SIZE,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="hccl",
            timeout=timedelta(seconds=EPLB_TIMEOUT),
        )
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=2,
            data_parallel_rank=rank // 2,
            enable_expert_parallel=True,
        )
    )
    try:
        with set_current_vllm_config(config), override_forward_context(create_forward_context(None, config)):
            initialize_model_parallel(tensor_model_parallel_size=2, backend="hccl")
            _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
            _EXTRA_CTX.max_tokens_across_dp = EPLB_ROWS
            device = torch.device("npu", rank)
            physical_count = EPLB_WORLD_SIZE * EPLB_LOCAL_EXPERTS
            placement = torch.cat([torch.arange(EPLB_LOGICAL_EXPERTS), torch.arange(4)])
            logical_map = torch.full((1, EPLB_LOGICAL_EXPERTS, 2), -1, dtype=torch.int64, device=device)
            replicas = torch.ones((1, EPLB_LOGICAL_EXPERTS), dtype=torch.int64, device=device)

            def set_placement():
                for logical in range(EPLB_LOGICAL_EXPERTS):
                    slots = torch.where(placement == logical)[0]
                    logical_map[0, logical].fill_(-1)
                    logical_map[0, logical, : slots.numel()].copy_(slots)
                    replicas[0, logical] = slots.numel()

            set_placement()
            loads = torch.zeros((1, physical_count), dtype=torch.int32, device=device)
            state = AscendEplbLayerState()
            state.set_layer_state(0, loads, logical_map, replicas)
            state.should_record_tensor = torch.tensor(True, device=device)
            state.allgather_record_tensor = torch.zeros((), dtype=torch.int32, device=device)
            router = SimpleNamespace(eplb_state=state, _validate_eplb_state=lambda: None)
            model_state = SimpleNamespace(
                model=SimpleNamespace(moe_layers=[router]),
                physical_to_logical_map=placement.unsqueeze(0).to(device),
                _allgather_physical_map=placement.unsqueeze(0).to(device),
                expert_load_pass=loads,
                expert_load_window=torch.zeros((2, 1, physical_count), dtype=torch.int32, device=device),
                _load_mapping_generation=0,
            )
            table_pointer = state.allgather_expert_replica_routing_table.data_ptr()
            prepare = PrepareAndFinalizeWithAllGather(
                SimpleNamespace(is_sequence_parallel=False, dp_size=2, pcp_size=1, dp_group=get_dp_group())
            )
            dispatcher = TokenDispatcherWithAllGather(
                top_k=1, num_experts=physical_count, num_local_experts=EPLB_LOCAL_EXPERTS
            )
            expert_map = torch.full((physical_count,), -1, dtype=torch.int32, device=device)
            start = rank * EPLB_LOCAL_EXPERTS
            expert_map[start : start + EPLB_LOCAL_EXPERTS] = torch.arange(EPLB_LOCAL_EXPERTS, device=device)
            hidden = (
                torch.arange(EPLB_ROWS, device=device, dtype=torch.float32)[:, None]
                .expand(-1, EPLB_HIDDEN_SIZE)
                .contiguous()
                .to(torch.bfloat16)
            )
            hidden.add_(1 + rank // 2 * EPLB_ROWS)
            logits = torch.zeros((EPLB_ROWS, EPLB_LOGICAL_EXPERTS), device=device)
            padding = torch.ones(EPLB_ROWS, dtype=torch.bool, device=device)
            logical_ids = torch.ones((EPLB_ROWS * 2, 1), dtype=torch.int32, device=device)
            weights = torch.ones_like(logical_ids, dtype=torch.float32)
            num_tokens = torch.tensor(EPLB_ROWS, device=device)
            state.num_unpadded_tokens_tensors = [num_tokens]
            state.allgather_source_is_padding = padding
            gather_mask = prepare.all_gather_eplb_mask
            mask = None

            def capture_mask(source_mask):
                nonlocal mask
                mask = gather_mask(source_mask)
                return mask

            prepare.all_gather_eplb_mask = capture_mask

            def forward():
                prepared = prepare.prepare(hidden, logits)
                physical = _ascend_apply_eplb_mapping(router, logical_ids)
                dispatched = dispatcher.token_dispatch(
                    MoETokenDispatchInput(
                        prepared.hidden_states,
                        weights,
                        physical,
                        MoeRouterInput(expert_map, 0, None, False),
                        MoEQuantParams(),
                    )
                )
                # Identity experts isolate dispatch/combine ownership from FFN numerics.
                output = dispatcher.token_combine(dispatched.hidden_states, dispatched.combine_metadata)
                state.record_load(
                    dispatched.group_list,
                    dispatched.group_list_type,
                    topk_ids=physical,
                    num_source_tokens=EPLB_ROWS,
                    is_sequence_parallel=False,
                    prepare_finalize=prepare,
                )
                assert mask is not None
                owners = ((physical >= start) & (physical < start + EPLB_LOCAL_EXPERTS)).to(torch.int32)
                return output, owners, physical, mask, prepared.hidden_states

            # Compile kernels and initialize HCCL before capture.
            forward()
            torch.npu.synchronize()
            graph = None
            if graph_mode == "full_graph":
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    result = forward()

            for valid_rows, eligible in [((3, 2), (True, False)), ((0, 2), (False, True)), ((0, 0), (False, False))]:
                dp_rank = rank // 2
                padding.fill_(True)
                padding[: valid_rows[dp_rank]] = False
                state.should_record_tensor.fill_(eligible[dp_rank])
                state.allgather_record_tensor.fill_(-1)
                loads.zero_()
                if graph is None:
                    result = forward()
                else:
                    graph.replay()
                output, owners, physical, mask, gathered = result
                torch.distributed.all_reduce(output, group=get_ep_group().device_group)
                torch.distributed.all_reduce(owners, group=get_ep_group().device_group)
                torch.testing.assert_close(output, gathered)
                torch.testing.assert_close(owners, torch.ones_like(owners))
                expected_mask = torch.tensor(
                    [row < valid_rows[dp] and eligible[dp] for dp in range(2) for row in range(EPLB_ROWS)]
                )
                torch.testing.assert_close(mask.cpu(), expected_mask)
                assert state.allgather_record_tensor.item() == int(expected_mask.any())
                slots = torch.where(placement == 1)[0]
                expected_physical = slots[(torch.arange(EPLB_ROWS * 2) + 1) % slots.numel()][:, None]
                torch.testing.assert_close(physical.cpu().long(), expected_physical)
                expected_load = torch.bincount(expected_physical[expected_mask].flatten(), minlength=physical_count)
                torch.distributed.all_reduce(loads, group=get_ep_group().device_group)
                torch.testing.assert_close(loads.cpu().flatten().long(), expected_load)

                # A changed placement must preserve captured pointers and discard old history.
                placement = placement.roll(1)
                set_placement()
                model_state.physical_to_logical_map.copy_(placement.unsqueeze(0))
                model_state.expert_load_window.fill_(1)
                refresh_model_routing_tables(model_state)
                assert state.allgather_expert_replica_routing_table.data_ptr() == table_pointer
                assert not loads.any()
                assert not model_state.expert_load_window.any()
                generation = model_state._load_mapping_generation
                model_state.expert_load_window.fill_(1)
                refresh_model_routing_tables(model_state)
                assert model_state.expert_load_window.all()
                assert model_state._load_mapping_generation == generation
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    "graph_mode",
    [
        pytest.param(
            mode,
            marks=pytest.mark.e2e_coverage(
                arch="moe",
                feature="eplb",
                parallel="DP,TP,EP",
                deploy="",
                hardware="A5",
                quantization="BF16",
                graph_mode=mode,
            ),
        )
        for mode in ("eager", "full_graph")
    ],
)
def test_a5_dp2_tp2_allgather_eplb(graph_mode):
    if not is_950():
        pytest.skip("A5 AllGather EPLB regression")
    assert torch.npu.device_count() >= EPLB_WORLD_SIZE
    workers = mp.spawn(_run_a5_allgather_eplb, args=(get_open_port(), graph_mode), nprocs=EPLB_WORLD_SIZE, join=False)
    deadline = time.monotonic() + EPLB_TIMEOUT
    try:
        while not workers.join(timeout=10):
            assert time.monotonic() < deadline, "A5 AllGather EPLB workers timed out"
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
