# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.lora.uno import UnoPackedLoRA, uno_lora_weight_key


@pytest.mark.parametrize("inputs,sizes", [(4096, (4096, 1024, 1024)), (4096, (12288, 12288)), (12288, (4096,))])
def test_packed_lora_graph_retains_gating_and_reloaded_adapter(inputs, sizes):
    torch.npu.set_device(0)
    torch.manual_seed(71)
    with torch.inference_mode():
        aa = tuple(torch.randn(1, 1, 128, inputs, device="npu", dtype=torch.bfloat16) * 0.01 for _ in sizes)
        bb = tuple(torch.randn(1, 1, size, 128, device="npu", dtype=torch.bfloat16) * 0.01 for size in sizes)
        packed = UnoPackedLoRA(aa, bb, sizes)
        pointers = packed.a.data_ptr(), packed.b.data_ptr()
        x = torch.randn(16, inputs, device="npu", dtype=torch.bfloat16)
        base = torch.randn(16, sum(sizes), device="npu", dtype=torch.bfloat16)
        actual = torch.empty_like(base)
        mask = torch.ones(16, 1, device="npu", dtype=torch.float32)
        wrapper = object.__new__(PunicaWrapperNPU)
        wrapper._dense_lora_slot = 0
        wrapper._dense_row_mask = mask
        wrapper._uno_packed_lora = {uno_lora_weight_key(aa, bb): packed}
        graph = torch.npu.NPUGraph()
        actual.copy_(base)
        wrapper.add_lora_linear(actual, x, aa, bb, 1.0, sizes)
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            actual.copy_(base)
            wrapper.add_lora_linear(actual, x, aa, bb, 1.0, sizes)
        for reload in range(3):
            for a, b in zip(aa, bb):
                a.normal_(std=0.01)
                b.normal_(std=0.01)
            packed.refresh()
            assert (packed.a.data_ptr(), packed.b.data_ptr()) == pointers
            x.normal_()
            base.normal_()
            mask.fill_(1)
            mask[:: 2 + reload] = 0
            buffers = tuple(torch.empty(16, 128, device="npu", dtype=torch.float32) for _ in sizes)
            reference = base.clone()
            wrapper._dense_shrink(buffers, x, aa, 1.0)
            wrapper._dense_expand(reference, buffers, bb, sizes, 0, True)
            graph.replay()
            torch.testing.assert_close(actual, reference, atol=0.005, rtol=0.02)
            base_rows = (mask[:, 0] == 0).nonzero().flatten()
            assert torch.equal(actual[base_rows], base[base_rows])
