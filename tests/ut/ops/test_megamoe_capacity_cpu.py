# SPDX-License-Identifier: Apache-2.0
"""Exercise production buffer and selector methods without the NPU runtime."""

import ast
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec("from __future__ import annotations\n" + ast.unparse(method), scope)
    return scope[name]


@pytest.mark.parametrize("a5", [False, True])
@pytest.mark.parametrize("decode_only", [False, True])
@pytest.mark.parametrize("capacity", [65536, 131072])
@pytest.mark.parametrize("global_bs", [0, 128])
def test_receive_capacity_reaches_cann(a5, decode_only, capacity, global_bs):
    dispatcher = SimpleNamespace(
        need_shared_expert_args=a5, global_bs=global_bs, ep_world_size=8, max_num_tokens_per_rank=16
    )
    allocate = Mock(return_value=object())
    impl = SimpleNamespace(
        token_dispatcher=dispatcher,
        moe_config=SimpleNamespace(
            experts_per_token=4, num_experts=64, hidden_dim=3584, intermediate_size_per_partition=3072
        ),
        get_symm_buffer_for_mega_moe=allocate,
    )
    fn = load(
        "vllm_ascend/ops/fused_moe/moe_comm_method.py",
        "_init_mega_moe_symm_buffer",
        {
            "TokenDispatcherWithMC2": SimpleNamespace,
            "get_mc2_group": lambda: SimpleNamespace(device_group="ep"),
            "get_ascend_config": lambda: SimpleNamespace(mega_moe_max_tokens=capacity),
            "logger": Mock(),
        },
    )
    assert fn(impl, is_decode_only_node=decode_only) is allocate.return_value
    assert allocate.call_args.kwargs["max_recv_token_num"] == (capacity if a5 or not decode_only else 16 * 8 * 4)


class Comm(Enum):
    FUSED_MC2 = auto()
    MC2 = auto()
    ALLGATHER = auto()
    ALLTOALL = auto()


@pytest.mark.parametrize("dp", [1, 2, 4])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("tokens", [1, 8192])
def test_local_prefill_does_not_select_megamoe_across_dp(dp, prefill, tokens):
    fn = load(
        "vllm_ascend/ascend_forward_context.py",
        "_select_capacity_and_world_size_moe_comm_method",
        {
            "get_ascend_config": lambda: SimpleNamespace(enable_fused_mc2=1),
            "is_mega_moe_supported": lambda: True,
            "MoECommType": Comm,
        },
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(top_k_experts=1)),
        parallel_config=SimpleNamespace(data_parallel_size=dp, world_size_across_dp=8),
    )
    result = fn(tokens, config, 8, cann_mega_moe_supported=True, is_pure_prefill=prefill)
    expected = Comm.FUSED_MC2 if prefill and dp == 1 else (Comm.MC2 if tokens <= 8 else Comm.ALLTOALL)
    assert result == expected
