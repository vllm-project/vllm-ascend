# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe import (
    routed_experts_capturer as routed_experts_capturer_module,
)

from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import (
    AscendGroupedTopKRouter,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_upstream_routed_experts_binder_supports_ascend_router():
    assert "_bind_routed_experts_capturer" not in NPUModelRunner.__dict__

    router = AscendGroupedTopKRouter(
        top_k=2,
        global_num_experts=4,
        num_expert_group=None,
        topk_group=None,
    )
    routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_monolithic=False),
    )
    moe_runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(moe_runner)
    moe_runner.layer_name = "model.layers.3.mlp.experts"
    moe_runner.router = router
    moe_runner.routed_experts = routed_experts

    model = nn.Module()
    model.add_module("moe", moe_runner)
    model_runner = NPUModelRunner.__new__(NPUModelRunner)
    model_runner.model = model
    capturer = MagicMock()

    binder = getattr(model_runner, "_bind_routed_experts_capturer", None)
    if binder is not None:
        binder(capturer)
    else:
        bind_routed_experts_capturer = vars(routed_experts_capturer_module).get("bind_routed_experts_capturer")
        assert bind_routed_experts_capturer is not None
        bind_routed_experts_capturer(model, capturer)

    assert callable(router.capture_fn)
    topk_ids = torch.tensor([[1, 2]], dtype=torch.int32)
    router.capture_fn(topk_ids)
    capturer.capture.assert_called_once_with(3, topk_ids)
