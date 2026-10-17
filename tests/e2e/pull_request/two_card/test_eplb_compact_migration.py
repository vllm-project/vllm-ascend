# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from datetime import timedelta
from inspect import signature

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from vllm.distributed.eplb.rebalance_execute import move_from_buffer, transfer_layer
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling
from vllm.utils.network_utils import get_open_port

import vllm_ascend.patch.platform  # noqa: F401
from tests.e2e.pull_request.eplb_compact_utils import nz_tensor, require_native_nz_copy
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile
from vllm_ascend.distributed.eplb.eplb_communicator import AscendHixlEplbCommunicator, _HixlSession
from vllm_ascend.ops.fused_moe.routed_experts import EplbExpertTensorList
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

pytestmark = pytest.mark.skipif(
    get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED,
    reason="NZ MXFP8/MXFP4 GMM requires A5",
)


def _migration_worker(rank, port, fp4):
    torch.set_num_threads(4)
    torch.npu.set_device(rank)
    with torch_cuda_wrapper():
        pass
    device = torch.device(f"npu:{rank}")
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=300)
    )
    baseline = MemorySnapshot(device=device)
    shapes = ((512, 512), (512, 256))
    quant_dtype = torch_npu.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn

    def expert(projection, logical_id):
        n, k = shapes[projection]
        generator = torch.Generator().manual_seed(93000 + projection * 100 + logical_id)
        if fp4:
            raw = torch.randint(0, 256, (n, k // 2), generator=generator, dtype=torch.uint8)
        else:
            raw = torch.randint(-3, 4, (n, k), generator=generator, dtype=torch.int8).to(torch.float8_e4m3fn)
        scale = torch.randint(124, 128, (n, k // 64, 2), generator=generator, dtype=torch.uint8)
        options = {"customize_dtype": torch.float8_e4m3fn}
        if fp4:
            options["input_dtype"] = torch_npu.float4_e2m1fn_x2
        return torch_npu.npu_format_cast(raw.to(device), 29, **options), scale.to(device)

    weights: list[EplbExpertTensorList] = []
    for projection in range(2):
        pairs = [expert(projection, rank * 2 + slot) for slot in range(2)]
        weights.extend(EplbExpertTensorList(pair[index] for pair in pairs) for index in range(2))
    # Exercise the same format-preserving buffer allocator used by add_model.
    buffers = [torch.empty_like(view) for view in weights]
    for view, buffer in zip(weights, buffers):
        for source, destination in zip(view, buffer):
            assert source.untyped_storage().nbytes() == destination.untyped_storage().nbytes()
            assert torch_npu.get_npu_format(source) == torch_npu.get_npu_format(destination)
    parameter_aliases = [torch.nn.Parameter(tensor, requires_grad=False) for view in weights for tensor in view]
    communicator = AscendHixlEplbCommunicator(dist.group.WORLD, [weights], buffers, compact_weights=True)
    session = _HixlSession(dist.group.WORLD)
    session.initialize({"model": communicator})
    assert len(session._registered_handles) == 1
    pointers = [tensor.data_ptr() for view in weights + buffers for tensor in view]
    assert [tensor.data_ptr() for tensor in parameter_aliases] == pointers[:8]

    inputs = []
    for _, k in shapes:
        raw = torch.randn(4, k, dtype=torch.bfloat16, device=device) * 0.125
        inputs.append(torch_npu.npu_dynamic_mx_quant(raw, dst_type=torch.float8_e4m3fn))
    groups = torch.tensor([2, 4], dtype=torch.int64, device=device)

    def gmm(projection, weight, scale):
        x, x_scale = inputs[projection]
        options = dict(
            x=[x],
            weight=[tensor.T for tensor in weight],
            per_token_scale=[x_scale],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=groups,
            group_list_type=0,
            output_dtype=torch.bfloat16,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=quant_dtype,
        )
        options["antiquant_scale" if fp4 else "scale"] = [tensor.transpose(0, 1) for tensor in scale]
        if not fp4:
            options["scale_dtype"] = torch_npu.float8_e8m0fnu
        return torch_npu.npu_grouped_matmul(**options)[0]

    for projection in range(2):
        gmm(projection, *weights[projection * 2 : projection * 2 + 2])
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [gmm(projection, *weights[projection * 2 : projection * 2 + 2]) for projection in range(2)]
    old = torch.arange(4, dtype=torch.int64)
    stream = torch.npu.Stream()
    for target in (torch.tensor([1, 0, 3, 2]), torch.tensor([3, 2, 1, 0]), torch.arange(4)):
        with memory_profiling(baseline, weights_memory=0):
            temporary = torch.empty(32 * 1024**2, dtype=torch.uint8, device=device)
            del temporary
        communicator.set_stream(stream)
        metadata = transfer_layer(
            old_layer_indices=old,
            new_layer_indices=target,
            expert_weights=weights,
            expert_weights_buffer=buffers,
            ep_group=dist.group.WORLD,
            communicator=communicator,
            **{"stream" if "stream" in signature(transfer_layer).parameters else "cuda_stream": stream},
        )
        stream.synchronize()
        move_from_buffer(weights, buffers, metadata, target.numpy(), rank)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        for projection in range(2):
            pairs = [expert(projection, int(target[rank * 2 + slot])) for slot in range(2)]
            reference = gmm(projection, [pair[0] for pair in pairs], [pair[1] for pair in pairs])
            torch.testing.assert_close(outputs[projection].cpu(), reference.cpu(), rtol=0, atol=0)
            for slot, pair in enumerate(pairs):
                for index in range(2):
                    torch.testing.assert_close(
                        weights[projection * 2 + index][slot].cpu().view(torch.uint8),
                        pair[index].cpu().view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
        assert pointers == [tensor.data_ptr() for view in weights + buffers for tensor in view]
        old = target
    assert sum(timing.transfer_bytes for timing in communicator._eplb_hixl_phase_timings) > 0
    session.close()
    dist.destroy_process_group()


@pytest.mark.parametrize("fp4", [False, True])
def test_compact_eplb_nz_migration_replays_gmm_graph(monkeypatch, fp4):
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    torch.npu.set_device(0)
    # Match both worker projections, including the uint8-packed FP4 carrier.
    # Skip in the parent only for the ordinary allocator's known copy_ gap.
    for n, k in ((512, 512), (512, 256)):
        if fp4:
            raw = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
        else:
            raw = torch.randint(-3, 4, (n, k), dtype=torch.int8).to(torch.float8_e4m3fn)
        require_native_nz_copy(nz_tensor(raw, fp4=fp4))
    mp.spawn(_migration_worker, args=(get_open_port(), fp4), nprocs=2, join=True)
