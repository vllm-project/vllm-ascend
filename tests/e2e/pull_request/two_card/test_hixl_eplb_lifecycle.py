# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Small deterministic MoE: real HIXL, layerwise reload, and CaMem unmap."""

import importlib.util
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from vllm.distributed.eplb import eplb_state as upstream_state
from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.distributed.eplb import eplb_communicator, eplb_state, hixl_compat
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState, AscendEplbState, refresh_model_routing_tables
from vllm_ascend.ops.fused_moe.eplb import map_to_physical
from vllm_ascend.utils import adapt_patch
from vllm_ascend.worker import worker as worker_module
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


class _ExpertLayer(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        initial = upstream_state.EplbState.build_initial_global_physical_to_logical_map(4, 2)
        values = torch.tensor(initial[rank * 3 : (rank + 1) * 3], dtype=torch.float32)
        self.weight = torch.nn.Parameter((values[:, None] + 1).expand(3, 32).contiguous(), requires_grad=False)
        self.eplb_state = AscendEplbLayerState()

    def get_expert_weights(self):
        return [self.weight]

    def set_eplb_state(self, **kwargs):
        self.eplb_state.set_layer_state(**kwargs)


class _ExpertModel(torch.nn.Module):
    set_eplb_state = MixtureOfExperts.set_eplb_state
    num_moe_layers = 1
    num_routed_experts = 4
    num_logical_experts = 4
    num_redundant_experts = 2
    num_physical_experts = 6
    num_local_physical_experts = 3
    num_expert_groups = 1

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.moe_layers = torch.nn.ModuleList([_ExpertLayer(rank)])

    def load_weights(self, weights):
        # Checkpoint experts have logical identities; use the same initial
        # layout contract as RoutedExperts.make_expert_params_mapping.
        checkpoint = dict(weights)
        initial = upstream_state.EplbState.build_initial_global_physical_to_logical_map(4, 2)
        loaded = torch.stack([checkpoint[f"expert.{i}"] for i in initial[self.rank * 3 : (self.rank + 1) * 3]])
        parameter = self.moe_layers[0].weight
        parameter.weight_loader(parameter, loaded)
        return {"moe_layers.0.weight"}

    def forward(self, logical_ids):
        layer = self.moe_layers[0]
        physical_ids = map_to_physical(logical_ids, layer.eplb_state.expert_replica_routing_table).reshape(-1)
        local_ids = physical_ids - self.rank * self.num_local_physical_experts
        owned = (local_ids >= 0) & (local_ids < self.num_local_physical_experts)
        values = layer.weight.index_select(0, local_ids.clamp(0, self.num_local_physical_experts - 1)).mean(-1)
        return values, owned


def _assert_identity(model, model_state, logical_ids, output=None):
    local_map = model_state.physical_to_logical_map[0, model.rank * 3 : (model.rank + 1) * 3]
    torch.testing.assert_close(model.moe_layers[0].weight[:, 0], (local_map + 1).float())
    values, owned = model(logical_ids) if output is None else output
    expected = (logical_ids.reshape(-1) + 1).float()
    torch.testing.assert_close(values[owned], expected[owned])


def _migrate_last_slot(state, rank):
    ms = state.model_states["model"]
    communicator = ms.communicator
    communicator.set_stream(None)
    placement = ms.physical_to_logical_map[0].cpu().numpy()
    communicator.set_transfer_context(placement, layer_idx=0)
    communicator.add_recv([ms.expert_buffer[0][0]], src_rank=1 - rank, expert_id=int(placement[5 - 3 * rank]))
    communicator.execute()
    # This is the existing receive-buffer -> final-slot commit, not an added
    # registration workaround or staging allocation.
    ms.model.moe_layers[0].weight[2].copy_(ms.expert_buffer[0][0])
    mapping = ms.physical_to_logical_map.cpu()
    mapping[:, [2, 5]] = mapping[:, [5, 2]]
    upstream_state._commit_eplb_maps(ms, mapping)
    refresh_model_routing_tables(ms)
    torch.npu.synchronize()
    dist.barrier()


def _lifecycle_worker(rank, port, binding):
    # Match NPUWorker initialization, including the upstream event adaptation.
    adapt_patch()
    torch.npu.set_device(rank)
    if binding == "ctypes":
        eplb_communicator._resolve_hixl_module = lambda: hixl_compat
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    group = SimpleNamespace(cpu_group=dist.group.WORLD, world_size=2, rank_in_group=rank)
    config = SimpleNamespace(
        model="lifecycle-regression", quantization=None, dtype=torch.float32, compute_hash=lambda: "model"
    )
    parallel = SimpleNamespace(
        enable_elastic_ep=False,
        num_ubatches=0,
        eplb_config=SimpleNamespace(
            use_async=False, window_size=2, step_interval=4, policy="default", communicator="hixl"
        ),
    )
    ascend_config = SimpleNamespace(
        weight_nz_mode=0, rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False)
    )
    allocator = CaMemAllocator.get_instance()
    with (
        torch_cuda_wrapper(),
        patch.object(eplb_state, "get_ep_group", return_value=group),
        patch.object(eplb_state, "get_eplb_group", return_value=group),
        patch.object(upstream_state, "get_ep_group", return_value=group),
        patch.object(upstream_state, "get_eplb_group", return_value=group),
        patch.object(worker_module, "get_ascend_config", return_value=ascend_config),
        torch.device(f"npu:{rank}"),
    ):
        token = eplb_state.EXPERT_MAPPING_EP_SIZE.set(2)
        try:
            with allocator.use_memory_pool("weights"):
                model = _ExpertModel(rank)
                record_metadata_for_reloading(model)
                state = AscendEplbState(parallel, torch.device(f"npu:{rank}"))
                state.add_model(model, config)
            with allocator.use_memory_pool("kv_cache"):
                unused_cache = torch.ones(1024)
            ms = state.model_states["model"]
            worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
            runner = SimpleNamespace(
                eplb_state=state,
                model=model,
                get_model=lambda: model,
                lora_config=None,
                model_config=config,
                reset_lora_state=lambda: None,
                reset_encoder_cache=lambda: None,
                reset_mm_cache=lambda: None,
            )
            runner.reload_weights = partial(GPUModelRunner.reload_weights, runner)
            worker.model_runner = runner
            logical_ids = torch.arange(4, dtype=torch.int64).repeat(4).reshape(-1, 1)
            _assert_identity(model, ms, logical_ids)
            pointers = [tensor.data_ptr() for tensor in state.lifecycle_tensors()]
            weight_ptr = model.moe_layers[0].weight.data_ptr()
            buffer_ptr = ms.expert_buffer[0].data_ptr()
            graph = torch.npu.NPUGraph()
            capture_stream = torch.npu.Stream()
            capture_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(capture_stream):
                for _ in range(3):
                    model(logical_ids)
            torch.npu.current_stream().wait_stream(capture_stream)
            with torch.npu.graph(graph, stream=capture_stream):
                graph_output = model(logical_ids)

            _migrate_last_slot(state, rank)
            _assert_identity(model, ms, logical_ids)
            # Exercise the real upstream checkpoint and kernel reload paths.
            checkpoint = [(f"expert.{i}", torch.full((32,), i + 1.0, device="cpu")) for i in range(4)]
            worker.reload_weights(weights_iterator=iter(checkpoint))
            _assert_identity(model, ms, logical_ids)
            assert ms.physical_to_logical_map[0].cpu().tolist() == list(
                upstream_state.EplbState.build_initial_global_physical_to_logical_map(4, 2)
            )
            _migrate_last_slot(state, rank)
            placement = ms.physical_to_logical_map.cpu().clone()
            kernel_weights = model.moe_layers[0].weight.detach().cpu().clone()
            worker.reload_weights(
                weights_iterator=iter([("moe_layers.0.weight", kernel_weights)]), is_checkpoint_format=False
            )
            torch.testing.assert_close(ms.physical_to_logical_map.cpu(), placement)
            _assert_identity(model, ms, logical_ids)

            for level in (1, 2):
                placement = ms.physical_to_logical_map.cpu().clone()
                physical_weights = model.moe_layers[0].weight.detach().cpu().clone()
                old = ms.communicator
                worker.sleep(level=level)
                assert old._engine is None
                worker.wake_up(tags=["kv_cache"])
                assert state._suspended
                worker.wake_up(tags=["weights"])
                assert not state._suspended
                torch.testing.assert_close(ms.physical_to_logical_map.cpu(), placement)
                if level == 2:
                    # Level 2 deliberately discards weights. Restore kernel
                    # format first to preserve the saved placement contract.
                    worker.reload_weights(
                        weights_iterator=iter([("moe_layers.0.weight", physical_weights)]), is_checkpoint_format=False
                    )
                _assert_identity(model, ms, logical_ids)
                graph.replay()
                torch.npu.synchronize()
                _assert_identity(model, ms, logical_ids, graph_output)
                assert [tensor.data_ptr() for tensor in state.lifecycle_tensors()] == pointers
                assert model.moe_layers[0].weight.data_ptr() == weight_ptr
                assert ms.expert_buffer[0].data_ptr() == buffer_ptr
                _migrate_last_slot(state, rank)
                _assert_identity(model, ms, logical_ids)
            state.close()
            assert unused_cache.numel() == 1024
        finally:
            eplb_state.EXPERT_MAPPING_EP_SIZE.reset(token)
    dist.destroy_process_group()


@pytest.mark.parametrize("binding", ["ctypes", "official"])
def test_migration_reload_sleep_wake_and_graph_identity(monkeypatch, binding):
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    if binding == "official" and importlib.util.find_spec("hixl") is None:
        pytest.skip("CANN does not provide the official Python binding")
    # Existing CaMem pools require expandable_segments=False. The separate
    # registration regression continues to test True in fresh subprocesses.
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")
    mp.spawn(_lifecycle_worker, args=(get_open_port(), binding), nprocs=2, join=True)
