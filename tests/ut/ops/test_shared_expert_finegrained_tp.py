# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.nn import functional as F
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.ascend_config import FinegrainedTPConfig
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops import linear_op
from vllm_ascend.ops.fused_moe import shared_experts as shared_module
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.fused_moe import _ascend_moe_forward_shared_sp_fake
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts, SharedExpertMLPPath
from vllm_ascend.quantization.quant_type import QuantType


def valid_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=8,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=64,
            cudagraph_capture_sizes=[4, 16, 64],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=768),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        use_v2_model_runner=True,
        additional_config={},
        lora_config=None,
        model_config=SimpleNamespace(is_moe=True),
    )


@pytest.mark.parametrize("size", [0, 1, 2, 4, 8])
def test_shared_size_is_independent_of_dense_mlp(size):
    config = FinegrainedTPConfig(shared_expert_tensor_parallel_size=size)
    config._validate_preconditions(valid_config())
    assert config.mlp_tensor_parallel_size == 1
    assert config.shared_expert_tensor_parallel_size == size


@pytest.mark.parametrize(
    ("section", "key", "value", "error"),
    [
        ("parallel_config", "tensor_parallel_size", 2, "TP=PCP=DCP=PP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "TP=PCP=DCP=PP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "TP=PCP=DCP=PP=1"),
        ("parallel_config", "pipeline_parallel_size", 2, "TP=PCP=DCP=PP=1"),
        ("parallel_config", "data_parallel_size", 3, "divide"),
        ("compilation_config", "cudagraph_mode", CUDAGraphMode.NONE, "graph mode"),
        ("compilation_config", "max_cudagraph_capture_size", 32, "capture capacity"),
        ("kv_transfer_config", "is_kv_consumer", False, "PD decode"),
        ("model_config", "is_moe", False, "MOE models"),
        (None, "use_v2_model_runner", False, "V2 model runner"),
        (None, "lora_config", object(), "LoRA"),
        (None, "additional_config", {"enable_shared_expert_dp": True}, "replicate"),
    ],
)
def test_shared_tp_rejects_unsafe_config(section, key, value, error):
    vc = valid_config()
    setattr(getattr(vc, section) if section else vc, key, value)
    with pytest.raises((ValueError, AssertionError), match=error):
        FinegrainedTPConfig(shared_expert_tensor_parallel_size=2)._validate_preconditions(vc)


def test_shared_tp_rejects_negative_size():
    with pytest.raises(ValueError, match="shared_expert_tensor_parallel_size"):
        FinegrainedTPConfig(shared_expert_tensor_parallel_size=-1)


@pytest.mark.parametrize("gather_output,input_is_parallel", [(False, True), (True, True), (False, False)])
def test_shared_runner_checks_linear_layout(monkeypatch, gather_output, input_is_parallel):
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    gate_up = SimpleNamespace(input_size=4, gather_output=gather_output)
    down = SimpleNamespace(input_is_parallel=input_is_parallel)
    gate_up.custom_op = linear_op.SharedExpertColumnParallelOp(gate_up)
    down.custom_op = linear_op.SharedExpertRowParallelOp(down)
    layer = SimpleNamespace(gate_up_proj=gate_up, down_proj=down, act_fn=torch.nn.SiLU())
    config = SimpleNamespace(
        finegrained_tp_config=FinegrainedTPConfig(shared_expert_tensor_parallel_size=2),
        multistream_overlap_shared_expert=False,
        enable_shared_expert_dp=False,
    )
    moe = SimpleNamespace(
        hidden_dim=4,
        in_dtype=torch.float32,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        is_sequence_parallel=False,
    )
    monkeypatch.setattr(shared_module, "get_ascend_config", lambda: config)
    monkeypatch.setattr(shared_module, "get_shared_expert_tp_group", lambda: group)
    if gather_output or not input_is_parallel:
        with pytest.raises(ValueError, match="partitioned down input"):
            AscendSharedExperts(layer, moe, QuantType.NONE, SimpleNamespace())
    else:
        executor = AscendSharedExperts(layer, moe, QuantType.NONE, SimpleNamespace())
        assert executor.finegrained_tp_group is group
        assert not executor.megamoe_shared_weights_ready


@pytest.mark.parametrize("prefix", ["shared_experts", "shared_expert", "share_expert"])
@pytest.mark.parametrize("direct", ["column", "row"])
def test_shared_weight_layout_uses_own_group_not_dense_group(monkeypatch, prefix, direct):
    group = SimpleNamespace(world_size=4, rank_in_group=2)
    monkeypatch.setattr(linear_op, "shared_expert_tp_enable", lambda: True)
    monkeypatch.setattr(linear_op, "get_shared_expert_tp_group", lambda: group)
    monkeypatch.setattr(linear_op, "get_mlp_tp_group", MagicMock(side_effect=AssertionError("dense group used")))
    op, rank, size = linear_op.get_parallel_op(True, f"model.layers.4.mlp.{prefix}.gate_up_proj", object(), direct)
    expected = linear_op.SharedExpertColumnParallelOp if direct == "column" else linear_op.SharedExpertRowParallelOp
    assert isinstance(op, expected)
    assert (rank, size) == (2, 4)


@pytest.mark.parametrize("rank", [0, 1])
def test_shared_down_adds_bias_once_and_does_not_reduce_in_linear(monkeypatch, rank):
    group = SimpleNamespace(world_size=2, rank_in_group=rank)
    monkeypatch.setattr(linear_op, "get_shared_expert_tp_group", lambda: group)
    layer = SimpleNamespace(bias=torch.ones(4))
    op = linear_op.SharedExpertRowParallelOp(layer)
    op.skip_bias_add = False
    op.input_is_parallel = True
    op.quant_method = SimpleNamespace(apply=lambda layer, x, bias: F.linear(x, torch.ones(4, 2), bias))
    output, _ = op.apply_impl(torch.ones(3, 2))
    torch.testing.assert_close(output, torch.full((3, 4), 3.0 if rank == 0 else 2.0))


@pytest.mark.parametrize("num_tokens", [0, 1, 3])
@pytest.mark.parametrize("with_gate", [False, True])
def test_shared_shards_restore_dp_owners_without_latent_projection(monkeypatch, num_tokens, with_gate):
    # A CPU collective rendezvous exercises the actual runner AG/RS boundary,
    # including an idle DP owner and token padding, against an unsharded MLP.
    torch.manual_seed(7)
    counts = [num_tokens, 0]
    capacity = max(num_tokens, 1)
    hidden = 4
    intermediate = 6
    inputs = [torch.randn(n, hidden) for n in counts]
    padded = [F.pad(x, (0, 0, 0, capacity - x.shape[0])) for x in inputs]
    gate = torch.randn(intermediate, hidden)
    up = torch.randn_like(gate)
    down = torch.randn(hidden, intermediate)
    expert_gate = torch.randn(1, hidden)
    parts: list[torch.Tensor | None] = [None, None]
    barrier = Barrier(2)
    monkeypatch.setattr(shared_module, "npu_stream_switch", lambda *a, **k: nullcontext())
    monkeypatch.setattr(shared_module, "shared_experts_calculation_stream", lambda: None)

    def rank_forward(rank):
        def reduce_scatter(partial, dim):
            parts[rank] = partial
            barrier.wait(timeout=10)
            first, second = parts
            assert first is not None and second is not None
            return (first + second).chunk(2, dim=dim)[rank]

        executor = AscendSharedExperts.__new__(AscendSharedExperts)
        shard = slice(rank * 3, (rank + 1) * 3)
        executor.layer = SimpleNamespace(
            gate_up_proj=lambda x: (F.linear(x, torch.cat([gate[shard], up[shard]])), None),
            down_proj=lambda x: (F.linear(x, down[:, shard]), None),
            act_fn=lambda x: F.silu(x.chunk(2, dim=-1)[0]) * x.chunk(2, dim=-1)[1],
            expert_gate=(lambda x: (F.linear(x, expert_gate), None)) if with_gate else None,
        )
        executor.finegrained_tp_group = SimpleNamespace(
            all_gather=lambda x, dim: torch.cat(padded, dim=dim), reduce_scatter=reduce_scatter
        )
        executor.multistream_overlap = False
        executor.quant_type = QuantType.NONE
        executor.lora_context = None
        executor._select_mlp_path = lambda: SharedExpertMLPPath.LINEAR_WRAPPER
        out = executor.forward(PreparedSharedExpertInput(padded[rank]), RoutedMoEMilestones())
        return out[: counts[rank]]

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(rank_forward, [0, 1]))
    for x, out in zip(inputs, outputs):
        reference = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
        if with_gate:
            reference *= F.sigmoid(F.linear(x, expert_gate))
        torch.testing.assert_close(out, reference)


def test_shared_graph_fake_accepts_model_without_latent_input():
    hidden = torch.empty(4, 8)
    shared, routed = _ascend_moe_forward_shared_sp_fake(hidden, hidden, None, None, "layer", 0)
    assert shared.shape == routed.shape == hidden.shape


def test_reused_shared_group_is_destroyed_once(monkeypatch):
    group = MagicMock()
    for name in ["_MLP_TP", "_LMTP", "_EMBED_TP", "_OTP", "_SHARED_EXPERT_TP"]:
        monkeypatch.setattr(parallel_state, name, group)
    for name in ["_KVPP", "_MC2", "_P_TP", "_DYNAMIC_EPLB"]:
        monkeypatch.setattr(parallel_state, name, None)
    parallel_state.destroy_ascend_model_parallel()
    group.destroy.assert_called_once()
    with pytest.raises(AssertionError, match="not initialized"):
        parallel_state.get_shared_expert_tp_group()
