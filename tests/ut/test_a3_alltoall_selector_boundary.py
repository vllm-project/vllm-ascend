# SPDX-License-Identifier: Apache-2.0
"""Exercise actual selector source without initializing the NPU runtime."""

import ast
import logging
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS


def test_a2_override_capacity_and_ep_guards():
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ascend_forward_context.py"
    names = {"MoECommType", "_select_a2_moe_comm_method", "_select_a3_moe_comm_method", "select_moe_comm_method"}
    tree = ast.parse(source.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    device = NS(A2=2, A3=3, A5=5, _310P=310)
    state = NS(platform=device.A2, ep=4, capacity=32, override=False, fused=0, moe=True)
    namespace = dict(
        Enum=Enum,
        VllmConfig=object,
        AscendDeviceType=device,
        logger=logging.getLogger(__name__),
        envs=NS(VLLM_ASCEND_FXRT_TEST_A3_ALLTOALL=False),
        get_ep_group=lambda: NS(world_size=state.ep),
        get_ascend_device_type=lambda: state.platform,
        get_mc2_tokens_capacity=lambda: state.capacity,
        get_ascend_config=lambda: NS(enable_fused_mc2=state.fused),
        is_moe_model=lambda config: state.moe,
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    config = NS(
        model_config=NS(get_num_experts=lambda: 256, hf_text_config=NS(moe_quantize="w8a8_dynamic")),
        parallel_config=NS(enable_expert_parallel=True, world_size_across_dp=4, pipeline_parallel_size=1),
    )
    choose = namespace["select_moe_comm_method"]
    kind = namespace["MoECommType"]
    for capacity in (32, 1024):
        state.capacity = capacity
        for override in (False, True):
            namespace["envs"].VLLM_ASCEND_FXRT_TEST_A3_ALLTOALL = override
            for tokens in (1, capacity, capacity + 1, 8192):
                expected = kind.ALLTOALL if override and tokens > capacity else kind.ALLGATHER
                assert choose(tokens, config) == expected
    state.ep = 1
    assert choose(8192, config) == kind.ALLGATHER
    state.ep = 4
    config.parallel_config.enable_expert_parallel = False
    assert choose(8192, config) == kind.ALLGATHER
    config.parallel_config.enable_expert_parallel = True
    # The test override must not alter production A3/fused selection.
    state.platform, state.fused = device.A3, 1
    assert choose(8192, config) == kind.FUSED_MC2
    state.moe = False
    assert choose(8192, config) is None


if __name__ == "__main__":
    test_a2_override_capacity_and_ep_guards()
    print("PASS: A2 capacity boundaries, EP guards and unchanged A3 selection")
