# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_ascend.models.common.ops import sequence_parallel as sp
from vllm_ascend.models.qwen4_exp import model, mtp, ple


def test_qwen_target_mtp_and_ple_use_opaque_ascend_sp_helpers():
    from vllm.models.qwen4_exp.amd import ple_layer

    ple.patch_upstream_ple_short_conv()
    assert model.upstream_model.sp_shard is sp.sp_shard
    assert model.upstream_model.sp_reduce_scatter is sp.sp_reduce_scatter
    assert mtp.upstream_mtp.sp_shard is sp.sp_shard
    assert ple_layer.sp_shard is sp.sp_shard
    assert ple_layer.sp_reduce_scatter is sp.sp_reduce_scatter


@pytest.mark.parametrize("tokens", [1, 4, 7, 8, 128])
@pytest.mark.parametrize("operation", ["sp_shard", "sp_reduce_scatter"])
def test_sp_padding_remains_opaque_to_dynamic_tracing(monkeypatch, tokens, operation):
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: 8)
    graph = make_fx(getattr(sp, operation), tracing_mode="symbolic")(torch.empty(tokens, 4))
    calls = [node for node in graph.graph.nodes if node.op == "call_function"]
    assert len(calls) == 1
    assert str(calls[0].target) == f"vllm.ascend_{operation}_impl.default"
    result = calls[0].meta["val"]
    assert tuple(result.shape) == ((tokens + 7) // 8, 4)
