# SPDX-License-Identifier: Apache-2.0
"""Check scratch ownership across graph nodes with different launch sizes."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

from .test_sparse_flash_attention_active_cores import check_outputs, make_inputs

pytestmark = pytest.mark.skipif("910" not in torch.npu.get_device_name(0), reason="Requires an A2/A3 NPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", [0, 3])
@pytest.mark.parametrize("query_counts", [(1, 20, 3), (20, 1, 3)])
@pytest.mark.parametrize("page_size", [48, 128])
@torch.inference_mode()
def test_active_workspace_mixed_graph_nodes(dtype, mode, query_counts, page_size):
    assert enable_custom_op()
    fixtures = [make_inputs(dtype, (count,), 17, page_size, mode=mode) for count in query_counts]
    for inputs, cpu in fixtures:
        check_outputs(torch.ops._C_ascend.npu_sparse_flash_attention(**inputs), cpu)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [torch.ops._C_ascend.npu_sparse_flash_attention(**inputs) for inputs, _ in fixtures]
    for _ in range(3):
        for inputs, cpu in fixtures:
            inputs["query"].mul_(0.75)
            cpu["query"] = inputs["query"].cpu()
        graph.replay()
        torch.npu.synchronize()
        for output, (_, cpu) in zip(outputs, fixtures, strict=True):
            check_outputs(output, cpu)
