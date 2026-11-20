#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Behavioral unit tests for declarative A3/A5 MegaMoe shared-expert fusion."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx
from vllm.model_executor.layers.activation import SiluAndMul

import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_mod
import vllm_ascend.ops.fused_moe.moe_comm_method as comm_mod
import vllm_ascend.ops.fused_moe.moe_utils as moe_utils
import vllm_ascend.ops.fused_moe.shared_experts as shared_mod
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import (
    MegamoeSharedExpertFusion,
    MegamoeSharedWeights,
    MoEWeights,
    build_fused_experts_input,
    shared_moe_weight_kwargs,
)
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl, TokenDispatcherWithMC2
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts, SharedExpertParallelMode
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


class _LinearStub(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        quant_type: QuantType = QuantType.NONE,
        scale: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
        tp_size: int = 1,
        output_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.tp_size = tp_size
        if output_sizes is not None:
            self.output_sizes = output_sizes
        if scale is not None:
            self.weight_scale = nn.Parameter(scale, requires_grad=False)
        if offset is not None:
            self.weight_offset = nn.Parameter(offset, requires_grad=False)
        self.quant_method = SimpleNamespace(
            quant_method=SimpleNamespace(quant_type=quant_type),
            process_weights_after_loading=lambda _layer: None,
        )


_A3_PROFILE = get_hardware_profile(AscendDeviceType.A3)
_A5_PROFILE = get_hardware_profile(AscendDeviceType.A5)


def _patch_shared_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile=_A3_PROFILE,
    enable_shared_dp: bool = True,
    is_draft: bool = False,
    offload_config=None,
) -> list[bool]:
    vllm_cfg = SimpleNamespace(offload_config=offload_config)
    ascend_cfg = SimpleNamespace(
        multistream_overlap_shared_expert=False,
        enable_shared_expert_dp=enable_shared_dp,
        megamoe_shared_expert_fusion=True,
    )
    monkeypatch.setattr(shared_mod, "get_current_vllm_config", lambda: vllm_cfg)
    monkeypatch.setattr(shared_mod, "is_draft_vllm_config", lambda _c: is_draft)
    monkeypatch.setattr(shared_mod, "use_cann_megamoe", lambda _c: True)
    monkeypatch.setattr(shared_mod, "get_ascend_config", lambda: ascend_cfg)
    monkeypatch.setattr(shared_mod, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(shared_mod, "_is_npu_runtime_available", lambda: False)
    allow_log: list[bool] = []
    state = {"flag": True}
    monkeypatch.setattr(shared_mod, "_get_allow_internal_format", lambda: state["flag"])

    def _set_flag(enabled: bool) -> None:
        state["flag"] = bool(enabled)
        allow_log.append(bool(enabled))

    monkeypatch.setattr(shared_mod, "_set_allow_internal_format", _set_flag)
    return allow_log


def _build_shared_executor(
    w1: _LinearStub,
    w2: _LinearStub,
    *,
    routed_quant: QuantType = QuantType.W4A8,
    hidden_dim: int = 1024,
    routed_inter: int = 256,
    tp_group=None,
) -> AscendSharedExperts:
    layer = SimpleNamespace(gate_up_proj=w1, down_proj=w2, act_fn=SiluAndMul.__new__(SiluAndMul), expert_gate=None)
    moe_cfg = SimpleNamespace(
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=routed_inter,
        in_dtype=torch.bfloat16,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        is_sequence_parallel=False,
        activation="silu",
        tp_group=tp_group or SimpleNamespace(world_size=1, rank_in_group=0),
    )
    routed_qm = SimpleNamespace(process_weights_after_loading=lambda *_a, **_k: None)
    return AscendSharedExperts(layer, moe_cfg, routed_quant, routed_qm)


def test_hardware_profile_and_write_only_npu_config_restore(monkeypatch):
    assert _A3_PROFILE.supports(HardwareCapability.CANN_MEGAMOE_SHARED_EXPERT_FUSION)
    assert not _A3_PROFILE.supports(HardwareCapability.CANN_MEGAMOE_MXFP)
    assert _A5_PROFILE.supports(HardwareCapability.CANN_MEGAMOE_SHARED_EXPERT_FUSION)
    assert _A5_PROFILE.supports(HardwareCapability.CANN_MEGAMOE_MXFP)
    for dt in (AscendDeviceType.A2, AscendDeviceType._310P):
        assert not get_hardware_profile(dt).supports(HardwareCapability.CANN_MEGAMOE_SHARED_EXPERT_FUSION)

    state = {"raw": b"enable", "writes": []}

    class _WriteOnlyNpuConfig:
        def __getattr__(self, name: str):
            raise AttributeError(name)

        def __setattr__(self, name: str, value: object) -> None:
            if name == "allow_internal_format":
                state["writes"].append(bool(value))
                state["raw"] = b"enable" if value else b"disable"
                return
            super().__setattr__(name, value)

    monkeypatch.setattr(shared_mod.torch, "npu", SimpleNamespace(config=_WriteOnlyNpuConfig()), raising=False)
    monkeypatch.setattr(
        shared_mod.torch_npu,
        "_C",
        SimpleNamespace(_npu_getOption=lambda _k: state["raw"]),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="boom"):
        with shared_mod._preserve_allow_internal_format():
            shared_mod._set_allow_internal_format(False)
            assert shared_mod._get_allow_internal_format() is False
            raise RuntimeError("boom")
    assert shared_mod._get_allow_internal_format() is True
    assert state["writes"] == [False, True]


def test_a3_declarative_shared_collection_and_zero_copy_nz(monkeypatch):
    _patch_shared_env(monkeypatch, profile=_A3_PROFILE)
    cast_calls: list[tuple] = []
    monkeypatch.setattr(
        shared_mod.torch_npu,
        "npu_format_cast",
        lambda t, fmt, **kw: (cast_calls.append((fmt, kw)) or t.contiguous()),
        raising=False,
    )
    # Square (1024, 1024) W8A8 weights already in FRACTAL_NZ must be reused zero-copy without transposing.
    monkeypatch.setattr(shared_mod, "_get_tensor_acl_format", lambda _t: ACL_FORMAT_FRACTAL_NZ)
    w1_raw = torch.randint(-8, 8, (1024, 1024), dtype=torch.int8)
    w2_raw = torch.randint(-8, 8, (512, 1024), dtype=torch.int8)
    s1_fp32 = torch.linspace(0.01, 0.5, 1024, dtype=torch.float32)
    s2_fp32 = torch.linspace(0.02, 0.4, 1024, dtype=torch.float32)
    w1 = _LinearStub(w1_raw, quant_type=QuantType.W8A8, scale=s1_fp32)
    w2 = _LinearStub(w2_raw, quant_type=QuantType.W8A8, scale=s2_fp32)

    executor = _build_shared_executor(w1, w2, routed_quant=QuantType.W4A8, hidden_dim=1024, routed_inter=256)
    w1.quant_method.process_weights_after_loading(w1)
    w2.quant_method.process_weights_after_loading(w2)

    fusion = executor.megamoe_shared_fusion()
    assert executor.megamoe_shared_weights_ready is True
    assert cast_calls == []
    assert fusion.weights is not None
    assert fusion.weights.w1[0].data_ptr() == w1_raw.data_ptr()
    assert fusion.weights.w2[0].data_ptr() == w2_raw.data_ptr()
    assert fusion.weight_type == moe_utils._CANN_ACL_INT8
    assert fusion.quant_out_dtype == moe_utils._CANN_ACL_INT8
    assert fusion.weights.w1_scale[0].dtype == torch.int64
    unpacked = (fusion.weights.w1_scale[0] & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    torch.testing.assert_close(unpacked, s1_fp32, rtol=0, atol=0)

    # Homogeneous (NONE, NONE) BF16 transposes PyTorch (out, in) -> kernel (in, out) in FRACTAL_ND.
    monkeypatch.setattr(shared_mod, "_get_tensor_acl_format", lambda _t: ACL_FORMAT_FRACTAL_ND)
    bf_w1 = _LinearStub(torch.randn(1024, 1024, dtype=torch.bfloat16))
    bf_w2 = _LinearStub(torch.randn(1024, 512, dtype=torch.bfloat16))
    bf_exec = _build_shared_executor(bf_w1, bf_w2, routed_quant=QuantType.NONE, hidden_dim=1024, routed_inter=256)
    bf_w1.quant_method.process_weights_after_loading(bf_w1)
    bf_w2.quant_method.process_weights_after_loading(bf_w2)
    bf_fusion = bf_exec.megamoe_shared_fusion()
    assert bf_fusion.weights is not None and bf_fusion.weights.w1_scale is None
    assert bf_fusion.weight_type is None and bf_fusion.quant_out_dtype is None
    torch.testing.assert_close(bf_fusion.weights.w2[0], bf_w2.weight.T)


def test_a5_mxfp_declarative_shared_collection(monkeypatch):
    allow_log = _patch_shared_env(monkeypatch, profile=_A5_PROFILE)
    cast_calls: list[tuple] = []
    monkeypatch.setattr(shared_mod.torch_npu, "float4_e2m1fn_x2", "fp4x2_sentinel", raising=False)
    monkeypatch.setattr(
        shared_mod.torch_npu,
        "npu_format_cast",
        lambda t, fmt, **kw: (cast_calls.append((fmt, kw)) or t.contiguous()),
        raising=False,
    )
    # A5 W4A8MXFP: post_load_transposed=True (in_packed, out), kernel_transposed=False -> transpose to (out, in_packed).
    w1 = _LinearStub(
        torch.randint(0, 255, (512, 1024), dtype=torch.uint8),
        quant_type=QuantType.W4A8MXFP,
        scale=torch.randint(1, 200, (32, 1024), dtype=torch.uint8),
    )
    w2 = _LinearStub(
        torch.randint(0, 255, (256, 1024), dtype=torch.uint8),
        quant_type=QuantType.W4A8MXFP,
        scale=torch.randint(1, 200, (16, 1024), dtype=torch.uint8),
    )
    executor = _build_shared_executor(w1, w2, routed_quant=QuantType.W8A8MXFP, hidden_dim=1024, routed_inter=512)
    w1.quant_method.process_weights_after_loading(w1)
    w2.quant_method.process_weights_after_loading(w2)

    fusion = executor.megamoe_shared_fusion()
    assert fusion.weights is not None
    assert fusion.weights.w1[0].shape == (1024, 512)
    assert fusion.weights.w2[0].shape == (1024, 256)
    assert fusion.weights.w1_scale[0].dtype == torch.float8_e8m0fnu
    assert fusion.weight_type == moe_utils._CANN_ACL_FLOAT4_E2M1
    assert fusion.quant_out_dtype == moe_utils._CANN_ACL_FLOAT8_E4M3FN
    assert any(kw.get("input_dtype") == "fp4x2_sentinel" for _fmt, kw in cast_calls)
    assert allow_log[-1] is True


@pytest.mark.parametrize(
    "case",
    ["draft", "uva_offload", "mixed_bf16_on_a3", "bad_scale", "nonzero_offset", "unaligned_inter"],
)
def test_shared_fusion_fail_closed_guards(monkeypatch, case):
    offload = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=4.0), prefetch=None) if case == "uva_offload" else None
    _patch_shared_env(monkeypatch, profile=_A3_PROFILE, is_draft=(case == "draft"), offload_config=offload)
    monkeypatch.setattr(shared_mod.torch_npu, "npu_format_cast", lambda t, *_a, **_k: t.contiguous(), raising=False)

    if case == "mixed_bf16_on_a3":
        w1 = _LinearStub(torch.randn(1024, 1024, dtype=torch.bfloat16))
        w2 = _LinearStub(torch.randn(1024, 512, dtype=torch.bfloat16))
    elif case == "unaligned_inter":
        w1 = _LinearStub(torch.ones(1024, 1280, dtype=torch.int8), quant_type=QuantType.W8A8, scale=torch.ones(1280))
        w2 = _LinearStub(torch.ones(640, 1024, dtype=torch.int8), quant_type=QuantType.W8A8, scale=torch.ones(1024))
    else:
        bad_scale = torch.zeros(1024) if case == "bad_scale" else torch.ones(1024)
        offset = torch.ones(1024, dtype=torch.int8) if case == "nonzero_offset" else None
        w1 = _LinearStub(
            torch.ones(1024, 1024, dtype=torch.int8), quant_type=QuantType.W8A8, scale=bad_scale, offset=offset
        )
        w2 = _LinearStub(torch.ones(512, 1024, dtype=torch.int8), quant_type=QuantType.W8A8, scale=torch.ones(1024))

    executor = _build_shared_executor(w1, w2, routed_quant=QuantType.W4A8, hidden_dim=1024, routed_inter=256)
    w1.quant_method.process_weights_after_loading(w1)
    w2.quant_method.process_weights_after_loading(w2)
    assert executor.megamoe_shared_weights_ready is False
    assert executor.megamoe_shared_fusion().weights is None


def test_tp4_sharded_shared_experts_reassembly_and_numerical_equivalence(monkeypatch):
    _patch_shared_env(monkeypatch, profile=_A3_PROFILE, enable_shared_dp=False)
    monkeypatch.setattr(shared_mod.torch_npu, "npu_format_cast", lambda t, *_a, **_k: t.contiguous(), raising=False)

    torch.manual_seed(7)
    tp_size, hidden_dim, shared_inter = 4, 1024, 512
    full_gate = torch.randint(-16, 16, (shared_inter, hidden_dim), dtype=torch.int8)
    full_up = torch.randint(-16, 16, (shared_inter, hidden_dim), dtype=torch.int8)
    full_down = torch.randint(-16, 16, (hidden_dim, shared_inter), dtype=torch.int8)
    s_gate = torch.linspace(0.02, 0.20, shared_inter, dtype=torch.float32)
    s_up = torch.linspace(0.03, 0.25, shared_inter, dtype=torch.float32)
    s_down = torch.linspace(0.01, 0.15, hidden_dim, dtype=torch.float32)

    # Simulate 4 TP ranks in lockstep (including out-of-order w2 -> w1 loading on odd ranks).
    pending_ag: list[torch.Tensor] = []

    class _SyncTPGroup:
        def __init__(self, rank: int):
            self.world_size = tp_size
            self.rank_in_group = rank

    # Stage step-by-step using lockstep replay over all 4 ranks.
    executors: list[AscendSharedExperts] = []
    linears: list[tuple[_LinearStub, _LinearStub]] = []
    part_i = shared_inter // tp_size
    for r in range(tp_size):
        w1_shard = torch.cat([full_gate[r * part_i : (r + 1) * part_i], full_up[r * part_i : (r + 1) * part_i]], dim=0)
        s1_shard = torch.cat([s_gate[r * part_i : (r + 1) * part_i], s_up[r * part_i : (r + 1) * part_i]], dim=0)
        w2_shard = full_down[:, r * part_i : (r + 1) * part_i]
        # AscendW8A8DynamicLinearMethod stores transposed (in, out).
        l1 = _LinearStub(
            w1_shard.T.contiguous(),
            quant_type=QuantType.W8A8,
            scale=s1_shard,
            tp_size=tp_size,
            output_sizes=[shared_inter, shared_inter],
        )
        l2 = _LinearStub(w2_shard.T.contiguous(), quant_type=QuantType.W8A8, scale=s_down, tp_size=tp_size)
        linears.append((l1, l2))
        executors.append(
            _build_shared_executor(l1, l2, routed_quant=QuantType.W4A8, hidden_dim=hidden_dim, tp_group=_SyncTPGroup(r))
        )

    # First callback on each rank stages locally with zero collectives (even when w2 fires before w1).
    for r, (l1, l2) in enumerate(linears):
        first = l2 if r % 2 else l1
        first.quant_method.process_weights_after_loading(first)
        assert executors[r].megamoe_shared_weights_ready is False

    # Run second callback across all 4 ranks using lockstep threads so tp_group.all_gather pairs cleanly.
    import threading

    barrier = threading.Barrier(tp_size)
    buckets: list[torch.Tensor | None] = [None] * tp_size

    def _make_ag(rank: int):
        def _ag(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
            buckets[rank] = tensor
            barrier.wait()
            out = torch.cat([b for b in buckets if b is not None], dim=dim)
            barrier.wait()
            return out

        return _ag

    for r, ex in enumerate(executors):
        ex.moe_config.tp_group.all_gather = _make_ag(r)

    def _finish_rank(r: int) -> None:
        l1, l2 = linears[r]
        second = l1 if r % 2 else l2
        second.quant_method.process_weights_after_loading(second)

    threads = [threading.Thread(target=_finish_rank, args=(r,)) for r in range(tp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    del pending_ag

    # Verify exact float64 SwiGLU equivalence between assembled full MegaMoe weights and summed TP shards.
    x = torch.randn(5, hidden_dim, dtype=torch.float64)
    tp_partial_sum = torch.zeros(5, hidden_dim, dtype=torch.float64)
    for r in range(tp_size):
        assert executors[r].megamoe_shared_weights_ready is True
        l1, l2 = linears[r]
        s1_r = l1.weight_scale.to(torch.float64)
        gu_r = (x @ l1.weight.to(torch.float64)) * s1_r
        g_r, u_r = gu_r.chunk(2, dim=-1)
        act_r = F.silu(g_r) * u_r
        tp_partial_sum += (act_r @ l2.weight.to(torch.float64)) * s_down.to(torch.float64)

    fw = executors[0].megamoe_shared_fusion().weights
    assert fw is not None
    s1_full = (fw.w1_scale[0] & 0xFFFFFFFF).to(torch.int32).view(torch.float32).to(torch.float64)
    s2_full = (fw.w2_scale[0] & 0xFFFFFFFF).to(torch.int32).view(torch.float32).to(torch.float64)
    gu_full = (x @ fw.w1[0].to(torch.float64)) * s1_full
    g_f, u_f = gu_full.chunk(2, dim=-1)
    fused_ref = ((F.silu(g_f) * u_f) @ fw.w2[0].to(torch.float64)) * s2_full
    torch.testing.assert_close(fused_ref, tp_partial_sum, rtol=1e-12, atol=1e-12)

    # Asymmetric single-rank failure disables all 4 ranks via status_t all_gather without deadlock.
    bad_execs: list[AscendSharedExperts] = []
    bad_linears: list[tuple[_LinearStub, _LinearStub]] = []
    for r in range(tp_size):
        bad_s = torch.zeros_like(s_down) if r == 2 else s_down
        l1 = _LinearStub(
            linears[r][0].weight.data.clone(),
            quant_type=QuantType.W8A8,
            scale=linears[r][0].weight_scale.data.clone(),
            tp_size=tp_size,
        )
        l2 = _LinearStub(linears[r][1].weight.data.clone(), quant_type=QuantType.W8A8, scale=bad_s, tp_size=tp_size)
        g = _SyncTPGroup(r)
        g.all_gather = _make_ag(r)
        bad_linears.append((l1, l2))
        bad_execs.append(_build_shared_executor(l1, l2, routed_quant=QuantType.W4A8, hidden_dim=hidden_dim, tp_group=g))

    def _run_bad(r: int) -> None:
        l1, l2 = bad_linears[r]
        l1.quant_method.process_weights_after_loading(l1)
        l2.quant_method.process_weights_after_loading(l2)

    threads = [threading.Thread(target=_run_bad, args=(r,)) for r in range(tp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(not ex.megamoe_shared_weights_ready and ex.megamoe_shared_fusion().weights is None for ex in bad_execs)


_CPU_ARMED_LIB = None
if not torch._C._dispatch_has_kernel_for_dispatch_key("vllm::ascend_moe_forward_megamoe_armed", "CPU"):
    _CPU_ARMED_LIB = torch.library.Library("vllm", "IMPL", "CPU")
    _CPU_ARMED_LIB.impl("ascend_moe_forward_megamoe_armed", fused_moe_mod._ascend_moe_forward_megamoe_armed)


def test_armed_custom_op_opcheck_and_inductor_stride(monkeypatch):
    layer_name = "ut.megamoe.opcheck_layer"

    class _FakeLayer:
        def _forward_impl(self, hidden_states, router_logits, shared_experts_input, input_ids):
            del router_logits, shared_experts_input, input_ids
            # Return padded width (16 > 12) with non-contiguous view to verify contiguous unpadding.
            base = torch.cat([hidden_states * 2.0, hidden_states[:, :4]], dim=-1)
            return base.transpose(0, 1).transpose(0, 1)

    monkeypatch.setattr(fused_moe_mod, "_resolve_layer_name", lambda name: str(name))
    monkeypatch.setattr(fused_moe_mod, "get_layer_from_name", lambda _name: _FakeLayer())

    x = torch.randn(4, 12, dtype=torch.float32)
    logits = torch.randn(4, 8, dtype=torch.float32)
    torch.library.opcheck(
        torch.ops.vllm.ascend_moe_forward_megamoe_armed.default,
        (x, logits, x, None, layer_name, 12),
        test_utils=("test_schema", "test_faketensor"),
    )
    compiled = torch.compile(
        lambda h, r: torch.ops.vllm.ascend_moe_forward_megamoe_armed(h, r, h, None, layer_name, 12),
        backend="inductor",
        fullgraph=True,
    )
    out = compiled(x, logits)
    assert out.shape == (4, 12) and out.is_contiguous() and out.stride() == (12, 1)
    torch.testing.assert_close(out, x * 2.0)


@pytest.mark.parametrize(
    "parallel_mode",
    [
        SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY,
        SharedExpertParallelMode.TENSOR_PARALLEL,
        SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY,
    ],
)
def test_armed_runner_single_fx_graph_across_parallel_and_comm_modes(monkeypatch, parallel_mode):
    tp_size = 4
    is_sp = parallel_mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
    ctx = SimpleNamespace(moe_comm_type=MoECommType.FUSED_MC2, use_mega_moe=True)
    monkeypatch.setattr(fused_moe_mod, "_EXTRA_CTX", ctx)
    reduce_calls = {"n": 0}
    monkeypatch.setattr(
        fused_moe_mod,
        "tensor_model_parallel_all_reduce",
        lambda t: (reduce_calls.__setitem__("n", reduce_calls["n"] + 1) or t * tp_size),
    )
    monkeypatch.setattr(fused_moe_mod.torch, "npu", SimpleNamespace(current_stream=lambda: MagicMock()), raising=False)

    fusion = MegamoeSharedExpertFusion(
        weights=MegamoeSharedWeights(w1=[torch.ones(1)], w2=[torch.ones(1)]),
        quant_out_dtype=258,
        weight_type=258,
    )
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    runner.layer_name = f"ut.armed_runner.{parallel_mode.name}"
    runner.moe_config = SimpleNamespace(
        hidden_dim=8,
        hidden_dim_unpadded=0,
        tp_size=tp_size,
        ep_size=tp_size,
        is_sequence_parallel=is_sp,
        in_dtype=torch.bfloat16,
        tp_group=SimpleNamespace(world_size=tp_size, rank_in_group=0),
    )
    qm = SimpleNamespace(has_unpadded_output=False)
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 1.5
    runner.divide_shared_output_in_fp16 = False
    runner._gate = None
    runner.gate = None
    runner._shared_experts = object()
    runner._sequence_parallel_context = lambda: nullcontext()
    runner._maybe_pad_hidden_states = lambda s, h: (h, None, None)
    runner._maybe_add_zero_expert_output = lambda t: t
    runner._encode_layer_name = lambda: runner.layer_name

    # Reference full-rank contributions: routed = 2 * x * 1.5 = 3 * x; shared = 4 * x; total = 7 * x.
    def _routed_forward(hidden_states, router_logits, input_ids=None):
        del router_logits, input_ids
        active_shared = runner.routed_experts.ascend_megamoe_shared_fusion.weights is not None
        if ctx.moe_comm_type is MoECommType.FUSED_MC2 and ctx.use_mega_moe and active_shared:
            scale = runner.routed_experts.ascend_megamoe_fold_scale
            return hidden_states * (2.0 * scale + 4.0), RoutedMoEMilestones()
        # ALLGATHER returns unreduced 1/TP routed shard on non-SP; MC2/ALLTOALL/SP return reduced routed output.
        routed_full = hidden_states * 2.0
        out = routed_full if (is_sp or ctx.moe_comm_type is not MoECommType.ALLGATHER) else routed_full / tp_size
        return out, RoutedMoEMilestones()

    def _shared_forward(prepared: PreparedSharedExpertInput, milestones, defer_output_wait=False):
        del milestones, defer_output_wait
        assert runner.routed_experts.ascend_megamoe_shared_fusion.weights is None
        h = prepared.hidden_states
        # TP returns 1/TP partial sum; SEDP and SP-only (after internal reduce-scatter) return full shard result.
        return h * (4.0 / tp_size if parallel_mode is SharedExpertParallelMode.TENSOR_PARALLEL else 4.0)

    runner.routed_experts = SimpleNamespace(
        ascend_megamoe_shared_fusion=fusion,
        ascend_megamoe_fold_scale=1.0,
        quant_method=qm,
        forward_impl=_routed_forward,
    )
    runner.ascend_shared_experts = SimpleNamespace(
        megamoe_shared_weights_ready=True,
        _megamoe_shared_fusion_disabled=False,
        multistream_overlap=False,
        parallel_mode=lambda: parallel_mode,
        prepare_input_before_routed=lambda h: PreparedSharedExpertInput(h),
        forward=_shared_forward,
    )
    monkeypatch.setattr(fused_moe_mod, "_resolve_layer_name", lambda name: str(name))
    monkeypatch.setattr(fused_moe_mod, "get_layer_from_name", lambda _name: runner)

    x = torch.linspace(0.1, 1.6, 16, dtype=torch.float32).reshape(2, 8)
    logits = torch.ones(2, 4, dtype=torch.float32)
    graph = make_fx(lambda h, r: runner.forward(h, r))(x, logits)

    for comm, use_mm in (
        (MoECommType.FUSED_MC2, True),
        (MoECommType.MC2, False),
        (MoECommType.ALLTOALL, False),
        (MoECommType.ALLGATHER, False),
    ):
        ctx.moe_comm_type = comm
        ctx.use_mega_moe = use_mm
        reduce_calls["n"] = 0
        out = graph(x, logits)
        torch.testing.assert_close(out, x * 7.0)
        assert fusion.weights is not None
        if is_sp:
            assert reduce_calls["n"] == 0

    # Distinct shared_experts_input object must run standalone without double-fusing inside MegaMoe.
    ctx.moe_comm_type = MoECommType.FUSED_MC2
    ctx.use_mega_moe = True
    torch.testing.assert_close(runner.forward(x, logits, shared_experts_input=x.clone()), runner.forward(x, logits, x))


def test_fused_mc2_comm_shared_kwargs_and_multi_quant_symm_buffer(monkeypatch):
    fusion = MegamoeSharedExpertFusion(
        weights=MegamoeSharedWeights(
            w1=[torch.ones(4, 4, dtype=torch.int8)],
            w2=[torch.ones(4, 4, dtype=torch.int8)],
            w1_scale=[torch.ones(4, dtype=torch.int64)],
            w2_scale=[torch.ones(4, dtype=torch.int64)],
        ),
        quant_out_dtype=258,
        weight_type=258,
    )
    layer = SimpleNamespace(ascend_megamoe_shared_fusion=fusion, ascend_megamoe_fold_scale=2.0)
    assert set(shared_moe_weight_kwargs(layer)) == {"shared_w1", "shared_w2", "shared_w1_scale", "shared_w2_scale"}
    assert shared_moe_weight_kwargs(layer, use_mega_moe=False) == {}

    comm = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    comm.moe_config = SimpleNamespace(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=16,
        intermediate_size_per_partition=8,
        tp_size=1,
        ep_size=4,
    )
    td = TokenDispatcherWithMC2.__new__(TokenDispatcherWithMC2)
    td.global_bs = 0
    td.max_num_tokens_per_rank = 32
    td.ep_world_size = 4
    td.ep_rank_id = 0
    comm.token_dispatcher = td
    comm._mega_moe_hccl_state_stale = False
    comm._mega_moe_symm_buffers = {}
    comm.mega_moe_symm_buffer = None
    comm._mega_moe_supported_kwargs = frozenset(
        {
            "shared_l1_weights",
            "shared_l2_weights",
            "shared_l1_weights_sf",
            "shared_l2_weights_sf",
            "shared_weight1_type",
            "shared_weight2_type",
        }
    )

    created_buffers: list[SimpleNamespace] = []

    def _get_symm(*_pos, **kw):
        buf = SimpleNamespace(
            combine_comm_mode=1,
            quant_mode=kw["dispatch_quant_mode"],
            _ctx_manager=SimpleNamespace(update_group=MagicMock(), ccl_buffer_size=1024),
            context=torch.zeros(1),
        )
        created_buffers.append(buf)
        return buf

    seen_calls: list[dict] = []

    def _mega_moe(hidden_states, _ids, topk_weights, _w1, _w2, symm, **kw):
        seen_calls.append({"symm_mode": symm.combine_comm_mode, "topk": topk_weights.clone(), **kw})
        return hidden_states, torch.ones(8, dtype=torch.int32)

    comm.get_symm_buffer_for_mega_moe = _get_symm
    comm.mega_moe = _mega_moe
    monkeypatch.setattr(comm_mod, "get_current_hardware_profile", lambda: _A3_PROFILE)
    monkeypatch.setattr(comm_mod, "get_mc2_group", lambda: SimpleNamespace(device_group="ep_grp"))
    monkeypatch.setattr(comm_mod, "get_ascend_config", lambda: SimpleNamespace(mega_moe_max_tokens=0))
    monkeypatch.setattr(comm_mod.torch.distributed, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(comm_mod.torch.distributed, "get_rank", lambda group=None: 0)
    barrier_calls = {"n": 0}
    monkeypatch.setattr(
        comm_mod.torch.distributed,
        "barrier",
        lambda **_kw: barrier_calls.__setitem__("n", barrier_calls["n"] + 1),
    )
    monkeypatch.setattr(comm_mod.torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)

    def _run_comm(qt: QuantType, target_layer) -> None:
        kw = shared_moe_weight_kwargs(target_layer, use_mega_moe=(qt != QuantType.NONE))
        weights = MoEWeights(
            w1=[torch.ones(4, 4, dtype=torch.int8 if qt != QuantType.NONE else torch.bfloat16)],
            w2=[torch.ones(4, 4, dtype=torch.int8 if qt != QuantType.NONE else torch.bfloat16)],
            w1_scale=[torch.ones(4)] if qt != QuantType.NONE else None,
            w2_scale=[torch.ones(4)] if qt != QuantType.NONE else None,
            **kw,
        )
        inp = build_fused_experts_input(
            hidden_states=torch.ones(2, 16, dtype=torch.bfloat16),
            topk_weights=torch.ones(2, 2, dtype=torch.float32),
            topk_ids=torch.zeros(2, 2, dtype=torch.int32),
            layer=target_layer,
            quant_type=qt,
            dynamic_eplb=False,
            mc2_mask=torch.ones(2, dtype=torch.bool),
        )
        comm._apply_cann_mega_moe(inp, weights)

    # 1) Mixed A3 (W4A8 routed=268, W8A8 shared=258): forces combine_comm_mode=0 during call, restores 1 after.
    _run_comm(QuantType.W4A8, layer)
    assert seen_calls[-1]["weight1_type"] == moe_utils._CANN_ACL_INT4
    assert seen_calls[-1]["shared_weight1_type"] == moe_utils._CANN_ACL_INT8
    assert seen_calls[-1]["symm_mode"] == 0
    assert created_buffers[0].combine_comm_mode == 1
    torch.testing.assert_close(seen_calls[-1]["topk"], torch.full((2, 2), 2.0))

    # 2) BF16 routed layer allocates a separate symm_buffer instead of colliding with INT8 buffer.
    _run_comm(QuantType.NONE, SimpleNamespace())
    assert len(created_buffers) == 2

    # 3) HCCL teardown + refresh updates every cached symm_buffer with a single deduplicated barrier.
    assert comm.prepare_hccl_teardown() is True
    backend = SimpleNamespace(get_hccl_comm_name=lambda _r: "hccl_restored")
    monkeypatch.setattr(
        comm_mod,
        "get_mc2_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(_get_backend=lambda _d: backend)),
    )
    assert comm.refresh_hccl_runtime_state() is True
    assert barrier_calls["n"] == 1
    assert all(b._ctx_manager.update_group.call_count == 1 for b in created_buffers)

    # 4) Missing required shared_weight1_type on legacy CANN build fails fast instead of mis-interpreting INT8 as INT4.
    comm._mega_moe_supported_kwargs = frozenset(
        {"shared_l1_weights", "shared_l2_weights", "shared_l1_weights_sf", "shared_l2_weights_sf"}
    )
    with pytest.raises(ValueError, match="shared_weight1_type"):
        _run_comm(QuantType.W4A8, layer)
