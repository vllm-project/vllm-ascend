# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.attention.context_parallel.sfa_cp as pcp_module
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


@pytest.mark.parametrize("pcp,tp,dcp", [(2, 1, 1), (4, 2, 1), (2, 4, 2), (8, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "rope,rows,replicated", [(False, 0, 0), (False, 2, 0), (False, 4, 1), (True, 4, 0), (True, 4, 1)]
)
@pytest.mark.parametrize("quantized", [False, True])
def test_main_kv_matches_synchronous_inputs_and_slots(
    monkeypatch, pcp, tp, dcp, dtype, rope, rows, replicated, quantized
):
    impl_cls = pcp_module.AscendSFAPCPDCPImpl if dcp > 1 else pcp_module.AscendSFAPCPImpl
    impl = impl_cls.__new__(impl_cls)
    # A service configured for graphs may execute this prefill eagerly.
    impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False))
    impl.qk_rope_head_dim = 2 if rope else 0
    impl.enable_sparse_sfa_c8 = quantized
    impl.is_pcp_decode_sharded = False
    metadata = pcp_module.AscendSFADCPMetadata.__new__(pcp_module.AscendSFADCPMetadata)
    metadata.num_input_tokens = 4
    metadata.num_decode_tokens = replicated
    metadata.num_prefills = 0  # This rank must join another rank's prefill.
    metadata.pcp_has_global_prefill = True
    metadata.uses_pcp_sharded_layout = True
    slots = torch.arange(pcp * 4)
    metadata.dcp_context = SimpleNamespace(slot_mapping=slots.flip(0))
    inputs = torch.arange(rows * 3, dtype=dtype).reshape(rows, 3)
    cos, sin = (torch.randn(rows, 2), torch.randn(rows, 2)) if rope else (None, None)
    cache = (torch.zeros(pcp * 4, 3, dtype=torch.int8 if quantized else torch.float32),)
    events, writes = [], []

    def gather(value, dim=0):
        return torch.cat([value + rank for rank in range(pcp)], dim=dim)

    group = SimpleNamespace(world_size=pcp, all_gather=gather)
    monkeypatch.setattr(pcp_module, "get_pcp_group", lambda: group)
    monkeypatch.setattr("vllm.v1.attention.ops.pcp.get_pcp_group", lambda: group)
    monkeypatch.setattr(pcp_module, "get_tp_group", lambda: SimpleNamespace(world_size=tp))
    monkeypatch.setattr(pcp_module, "get_dcp_group", lambda: SimpleNamespace(world_size=dcp))
    monkeypatch.setattr(
        pcp_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    )

    def write(self, values, actual_cos, actual_sin, actual_cache, actual_slots, actual_metadata):
        assert actual_cache is cache and actual_metadata is metadata
        if events:
            count = 3 if rope else 1
            assert events == ["launch"] * count + ["query"] + ["wait"] * count
        writes.append((values.clone(), actual_cos, actual_sin, actual_slots.clone()))
        return None, None

    monkeypatch.setattr(AscendSFAImpl, "exec_kv", write)
    expected_result = impl.exec_kv(inputs, cos, sin, cache, slots, metadata)

    def launch(send, actual_group):
        assert actual_group is group and send.is_contiguous()
        events.append("launch")
        output = send.new_full((send.shape[0] * pcp, send.shape[1]), float("nan"))

        def wait():
            events.append("wait")
            output.copy_(gather(send))

        return output, SimpleNamespace(wait=wait)

    monkeypatch.setattr(pcp_module, "all_gather_async", launch)
    finish = impl._start_main_kv_gather(inputs, cos, sin, cache, slots, metadata)
    assert finish is not None and len(writes) == 1
    events.append("query")
    assert finish() == expected_result
    for expected, actual in zip(*writes):
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(expected, actual)


@pytest.mark.parametrize("input_dtype,cache_dtype", [(torch.float16, torch.float16), (torch.bfloat16, torch.float32)])
def test_main_kv_completion_uses_original_cache_writer(monkeypatch, input_dtype, cache_dtype):
    impl = pcp_module.AscendSFAPCPImpl.__new__(pcp_module.AscendSFAPCPImpl)
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_c8 = False
    impl.is_pcp_decode_sharded = True
    impl.kv_lora_rank = 3
    impl.kv_a_layernorm = lambda values: values * 2
    metadata = SimpleNamespace(
        num_input_tokens=4,
        num_decode_tokens=1,
        num_prefills=1,
        pcp_has_global_prefill=True,
        uses_pcp_sharded_layout=True,
    )
    group = SimpleNamespace(world_size=2, all_gather=lambda values, dim: torch.cat((values, values + 10), dim))
    monkeypatch.setattr(pcp_module, "get_pcp_group", lambda: group)
    monkeypatch.setattr("vllm.v1.attention.ops.pcp.get_pcp_group", lambda: group)
    monkeypatch.setattr(
        pcp_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    )
    monkeypatch.setattr(
        pcp_module, "all_gather_async", lambda send, _: (group.all_gather(send, 0), SimpleNamespace(wait=lambda: None))
    )

    def scatter(cache, slots, values):
        valid = slots[:, 0] >= 0
        cache[slots[valid, 0]] = values[valid]

    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    values = torch.arange(6, dtype=input_dtype).reshape(2, 3)
    slots = torch.tensor([0, 1, -1, -1, 2, 3, -1, -1])
    expected, actual = torch.zeros(4, 3, dtype=cache_dtype), torch.zeros(4, 3, dtype=cache_dtype)
    impl.exec_kv(values, None, None, (expected,), slots, metadata)
    finish = impl._start_main_kv_gather(values, None, None, (actual,), slots, metadata)
    finish()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mode", ["graph", "one_rank", "decode", "unsharded", "unexpanded", "replicated_only"])
def test_main_kv_non_collective_paths_do_not_submit_work(monkeypatch, mode):
    impl = pcp_module.AscendSFAPCPImpl.__new__(pcp_module.AscendSFAPCPImpl)
    impl.qk_rope_head_dim = 0
    impl.is_pcp_decode_sharded = False
    metadata = SimpleNamespace(
        num_input_tokens=2,
        num_decode_tokens=2 if mode == "replicated_only" else 0,
        num_prefills=0 if mode == "decode" else 1,
        pcp_has_global_prefill=False,
        uses_pcp_sharded_layout=mode != "unsharded",
    )
    monkeypatch.setattr(pcp_module, "get_pcp_group", lambda: SimpleNamespace(world_size=1 if mode == "one_rank" else 2))
    monkeypatch.setattr(
        pcp_module,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL if mode == "graph" else CUDAGraphMode.NONE),
    )
    values = torch.zeros(2, 3)
    slots = torch.arange(2 if mode == "unexpanded" else 4)
    gather = Mock()
    monkeypatch.setattr(pcp_module, "all_gather_async", gather)
    assert impl._start_main_kv_gather(values, None, None, (values,), slots, metadata) is None
    gather.assert_not_called()
