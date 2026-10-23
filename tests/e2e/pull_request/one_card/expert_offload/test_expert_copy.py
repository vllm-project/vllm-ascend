# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import acl  # type: ignore[import-not-found]
import pytest
import torch
import torch_npu

from vllm_ascend.expert_offload.host_store import HostExpertStore
from vllm_ascend.expert_offload.prefill import PrefillExpertPool
from vllm_ascend.expert_offload.quantization import MXFP4ExpertAdapter
from vllm_ascend.expert_offload.transfer import TorchExpertTransfer
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
    maybe_normalize_mxfp_scale_layout,
)
from vllm_ascend.utils import is_950

pytestmark = pytest.mark.skipif(not is_950(), reason="950DT MXFP4 NZ regression")
ACL_MEMCPY_DEVICE_TO_HOST = 2
NUM_EXPERTS = 6
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256
GROUP_SIZE = 32


@pytest.fixture
def expert_bundle():
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "w13_weight": torch.randint(
            256, (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2), dtype=torch.uint8, generator=generator
        ),
        "w2_weight": torch.randint(
            256, (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2), dtype=torch.uint8, generator=generator
        ),
        "w13_weight_scale": torch.full(
            (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // GROUP_SIZE), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "w2_weight_scale": torch.full(
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // GROUP_SIZE), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
    }
    layer = SimpleNamespace(
        local_num_experts=NUM_EXPERTS,
        layer_name="test.experts",
        weight_loader=lambda *args, **kwargs: True,
        **{name: torch.nn.Parameter(tensor, requires_grad=False) for name, tensor in tensors.items()},
    )
    store = HostExpertStore(layer, MXFP4ExpertAdapter.tensor_names)
    for name in store.names:
        shards = ("w1", "w3") if name.startswith("w13") else ("w2",)
        store.loaded[name] = {(eid, shard) for eid in range(NUM_EXPERTS) for shard in shards}
    store.finalize(layer)
    # Exercise the real post-processing without initializing a model/runtime config.
    scheme = object.__new__(AscendW4A8MXFPDSDynamicFusedMoEMethod)
    scheme.group_size = GROUP_SIZE
    return MXFP4ExpertAdapter(SimpleNamespace(activation="silu"), scheme), store


def _physical_bytes(tensor):
    torch.npu.current_stream().synchronize()
    size = torch_npu.get_storage_size(tensor) * tensor.element_size()
    result = torch.empty(size, dtype=torch.uint8, device="cpu")
    error = acl.rt.memcpy(result.data_ptr(), size, tensor.untyped_storage().data_ptr(), size, ACL_MEMCPY_DEVICE_TO_HOST)
    assert error == 0
    return result


def _gemm(x, weight, scale, counts):
    quantized, token_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
    token_scale = maybe_normalize_mxfp_scale_layout(token_scale)
    return torch_npu.npu_grouped_matmul(
        x=[quantized],
        weight=[weight],
        scale=None,
        antiquant_scale=[scale],
        per_token_scale=[token_scale],
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        split_item=2,
        group_type=0,
        group_list=torch.tensor(counts, dtype=torch.int64, device=x.device),
        group_list_type=0,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        output_dtype=torch.bfloat16,
    )[0]


def test_nz_slot_replacement_preserves_storage_and_both_gemms(expert_bundle, monkeypatch):
    adapter, store = expert_bundle
    device = torch.device("npu:0")
    resident = adapter.materialize(store, [0, 1, 2], device)
    pointers = {name: tensor.data_ptr() for name, tensor in resident.items()}
    expected = adapter.materialize(store, [3, 4, 5], device)
    full = adapter.materialize(store, range(NUM_EXPERTS), device)
    store.prepare_native(adapter, resident, device)

    def forbidden_conversion(*args):
        raise AssertionError("runtime expert replacement must not format-cast")

    monkeypatch.setattr(adapter.scheme, "process_weights_after_loading", forbidden_conversion)
    allocated = torch.npu.memory_allocated(device)
    torch.npu.reset_peak_memory_stats(device)
    TorchExpertTransfer.copy_experts(adapter, store, resident, [(3, 0), (4, 1), (5, 2)], device)
    assert torch.npu.memory_allocated(device) == allocated
    assert torch.npu.max_memory_allocated(device) == allocated
    for name in resident:
        assert resident[name].data_ptr() == pointers[name]
        assert torch.equal(_physical_bytes(resident[name]), _physical_bytes(expected[name]))
    x = torch.randn(6, HIDDEN_SIZE, generator=torch.Generator().manual_seed(1)).to(device, dtype=torch.bfloat16)
    actual = _gemm(x, resident["w13_weight"], resident["w13_weight_scale"], [2, 4, 6])
    reference = _gemm(x, full["w13_weight"], full["w13_weight_scale"], [0, 0, 0, 2, 4, 6])
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    gate, up = actual.chunk(2, dim=-1)
    hidden = torch.nn.functional.silu(gate) * up
    actual = _gemm(hidden, resident["w2_weight"], resident["w2_weight_scale"], [2, 4, 6])
    reference = _gemm(hidden, full["w2_weight"], full["w2_weight_scale"], [0, 0, 0, 2, 4, 6])
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_nz_prefill_pool_refill_preserves_addresses_and_complete_bundle(expert_bundle, monkeypatch):
    adapter, store = expert_bundle
    indices = torch.arange(NUM_EXPERTS - 1, -1, -1)
    store.tensors = {name: tensor.index_select(0, indices) for name, tensor in store.tensors.items()}
    binding = SimpleNamespace(adapter=adapter, host_store=store, num_experts=NUM_EXPERTS, device=torch.device("npu:0"))
    expected = adapter.materialize(store, range(NUM_EXPERTS), binding.device)
    resident = adapter.materialize(store, [0, 1], binding.device)
    store.prepare_native(adapter, resident, binding.device)
    pool = PrefillExpertPool()
    pool.initialize(binding)
    pointers = {name: tensor.data_ptr() for name, tensor in pool.tensors.items()}

    def forbidden_conversion(*args):
        raise AssertionError("runtime prefill must not format-cast")

    monkeypatch.setattr(adapter.scheme, "process_weights_after_loading", forbidden_conversion)
    actual = pool.fill(binding)
    for name in actual:
        assert actual[name].data_ptr() == pointers[name]
        assert torch.equal(_physical_bytes(actual[name]), _physical_bytes(expected[name]))
    pool.close()


@torch.inference_mode()
def test_acl_graph_replay_updates_nz_slots_and_rejects_callback_failure(expert_bundle, monkeypatch):
    from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager, OffloadState
    from vllm_ascend.expert_offload.graph import GraphExpertStaging
    from vllm_ascend.expert_offload.residency import ExpertResidency

    adapter, store = expert_bundle
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    manager = ExpertOffloadManager()
    manager.state = OffloadState.READY
    manager.graph_load_stream = torch.npu.Stream(device=device)
    resident = adapter.materialize(store, [0, 1], device)
    binding = SimpleNamespace(
        manager=manager,
        adapter=adapter,
        host_store=store,
        resident=resident,
        residency=ExpertResidency(NUM_EXPERTS, 2),
        num_experts=NUM_EXPERTS,
        device=device,
        close=lambda: None,
    )
    binding.residency.commit(binding.residency.plan([0, 1]))
    manager.bindings["test"] = binding
    staging = GraphExpertStaging(binding, (1, 1), torch.int32)
    ids = torch.zeros((1, 1), dtype=torch.int32, device=device)
    x = torch.randn(1, HIDDEN_SIZE, generator=torch.Generator().manual_seed(7)).to(device, dtype=torch.bfloat16)
    full = adapter.materialize(store, range(NUM_EXPERTS), device)
    store.prepare_native(adapter, resident, device)

    def forbidden_conversion(*args):
        raise AssertionError("graph replay must not format-cast")

    monkeypatch.setattr(adapter.scheme, "process_weights_after_loading", forbidden_conversion)
    pointers = {name: tensor.data_ptr() for name, tensor in resident.items()}
    capture_stream = torch.npu.Stream(device=device)
    manager.subscribe_graph_stream(capture_stream)

    def forward(weights, scale, hidden, counts):
        quantized, token_scale = torch_npu.npu_dynamic_mx_quant(hidden, dst_type=torch.float8_e4m3fn)
        return torch_npu.npu_grouped_matmul(
            x=[quantized],
            weight=[weights],
            scale=None,
            antiquant_scale=[scale],
            per_token_scale=[maybe_normalize_mxfp_scale_layout(token_scale)],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=counts,
            group_list_type=0,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]

    try:
        # Initialize kernel/workspace paths before capture.
        _gemm(x, resident["w13_weight"], resident["w13_weight_scale"], [1, 1])
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with manager.graph_capture(), torch.npu.graph(graph, stream=capture_stream):
            physical = staging.prepare(ids)
            counts = (torch.arange(2, device=device) >= physical.reshape(-1)[0]).to(torch.int64)
            gate, up = forward(resident["w13_weight"], resident["w13_weight_scale"], x, counts).chunk(2, -1)
            hidden = torch.nn.functional.silu(gate) * up
            output = forward(resident["w2_weight"], resident["w2_weight_scale"], hidden, counts)
        for expert_id in (5, 3, 1, 4, 5):
            manager.begin_execution()
            try:
                ids.fill_(expert_id)
                graph.replay()  # Replay on the default stream, not capture_stream.
                torch.npu.current_stream().synchronize()
                manager.check_graph_errors()
                expected_counts = [int(i >= expert_id) for i in range(NUM_EXPERTS)]
                gate, up = _gemm(x, full["w13_weight"], full["w13_weight_scale"], expected_counts).chunk(2, -1)
                expected = _gemm(
                    torch.nn.functional.silu(gate) * up, full["w2_weight"], full["w2_weight_scale"], expected_counts
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                assert {name: tensor.data_ptr() for name, tensor in resident.items()} == pointers
            finally:
                manager.finish_execution(failed=manager.graph_error is not None)
        manager.begin_execution()
        before = binding.residency.logical_to_physical()

        def fail_copy(*_args, **_kwargs):
            raise RuntimeError("injected graph transfer failure")

        monkeypatch.setattr(TorchExpertTransfer, "copy_experts", fail_copy)
        ids.fill_(0)
        graph.replay()
        torch.npu.current_stream().synchronize()
        with pytest.raises(RuntimeError, match="rejecting execution output"):
            manager.check_graph_errors()
        assert binding.residency.logical_to_physical() == before
        assert manager.state is OffloadState.FAILED
    finally:
        manager.close()
