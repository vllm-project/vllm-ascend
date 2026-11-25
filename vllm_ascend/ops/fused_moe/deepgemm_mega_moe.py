# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""DeepSeek V4 family FP8 x FP4 MegaMoE on Ascend 950.

Like the CUDA adapter, load checkpoint weights once into the kernel layout,
stage local tokens and global expert IDs into symmetric memory, then run the
fused dispatch/GEMM/SwiGLU/combine operation, including the shared FP8 experts.
Each EP rank owns a contiguous range of physical expert slots.
"""

import importlib
import os
import sys
from functools import cache
from pathlib import Path

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import logger

from vllm_ascend.utils import is_deepseek_v41

# The pinned kernel launches this many Cube blocks and waits for all of them.
_DEEPGEMM_CUBE_CORES = 32


_BUNDLED = "vllm_ascend.third_party.deep_gemm"


@cache
def load_deepgemm():
    """Load once per worker; the build pins the native API and JIT headers."""
    import torch_npu  # noqa: F401 -- load NPU shared libraries before the extension
    from vllm import envs as vllm_envs

    # Both forks register global pybind11 types. Do not load two extensions
    # into one process, even if the other package also targets NPU.
    if "deep_gemm" in sys.modules:
        raise RuntimeError("An external deep_gemm is already loaded; MegaMoE requires the bundled Ascend backend")
    os.environ.setdefault("DG_JIT_CACHE_DIR", str(Path(vllm_envs.VLLM_CACHE_ROOT) / "deep_gemm" / "ascend"))
    try:
        module = importlib.import_module(_BUNDLED)
    except ModuleNotFoundError as exc:
        if exc.name is None or not _BUNDLED.startswith(exc.name):
            raise
        raise RuntimeError(
            "NPU MegaMoE was not built. Reinstall vLLM Ascend with its Ascend 950 native backend."
        ) from exc
    logger.info("Loaded bundled NPU DeepGEMM %s from %s", module.__version__, module.__file__)
    return module


def deepgemm_device_supported(group):
    """Resolve the native collective once per EP group, before packing weights."""
    supported = getattr(group, "_deepgemm_supported", None)
    if supported is not None:
        return supported

    try:
        cores = getattr(torch.npu.get_device_properties(torch.npu.current_device()), "cube_core_num", None)
    except (AttributeError, RuntimeError):
        cores = None
    # All peers must select the same collective, including peers whose driver
    # does not expose the Cube count. Query only the device bound to this worker.
    counts = [None] * group.world_size
    torch.distributed.all_gather_object(counts, cores, group=group.cpu_group)
    supported = not any(n and n > 0 and n != _DEEPGEMM_CUBE_CORES for n in counts)
    if not supported:
        logger.warning_once(
            "Disabling deepgemm_mega_moe: the kernel expects %d Cube cores, EP ranks report %s. "
            "Using the standard enable_fused_mc2=3 selection.",
            _DEEPGEMM_CUBE_CORES,
            tuple(counts),
        )
    group._deepgemm_supported = supported
    return supported


class DeepGemmExperts:
    """One layer's weights over process-group-owned symmetric scratch."""

    def __init__(self, layer):
        config = get_current_vllm_config()
        text = config.model_config.hf_text_config
        if not (
            text.model_type == "deepseek_v4"
            or getattr(type(text), "model_type", None) == "deepseek_v4"
            or is_deepseek_v41(text)
        ):
            raise ValueError("deepgemm_mega_moe requires a DeepSeek V4 family model")
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("DeepGEMM MegaMoE requires TP1")
        group = get_ep_group()
        if getattr(config.parallel_config, "expert_placement_strategy", "linear") != "linear":
            raise ValueError("DeepGEMM requires linear expert placement")
        if config.parallel_config.use_ubatching:
            raise ValueError("DeepGEMM symmetric scratch does not support overlapping microbatches")
        if config.lora_config is not None:
            raise ValueError("DeepGEMM fused experts do not support MoE LoRA")
        if layer.apply_router_weight_on_input:
            raise ValueError("MegaMoE expects output routing weights")
        moe = layer.moe_config
        if getattr(moe.activation, "value", moe.activation) != "silu":
            raise ValueError("MegaMoE requires SwiGLU")
        eplb_enabled = layer.router is not None and layer.router.eplb_state is not None
        if moe.num_experts != moe.num_logical_experts and not eplb_enabled:
            raise ValueError("MegaMoE expert replicas require an EPLB router")

        self.hidden = layer.w2_weight.shape[1]
        self.intermediate = layer.w13_weight.shape[1] // 2
        self.num_experts = moe.num_experts
        self.topk = moe.experts_per_token
        self.activation_clamp = moe.swiglu_limit
        self.max_tokens = config.scheduler_config.max_num_batched_tokens
        self.expert_tokens = (
            torch.empty(moe.num_local_experts, dtype=torch.int32, device=layer.w2_weight.device)
            if eplb_enabled
            else None
        )
        self.zero_expert_tokens = torch.zeros_like(self.expert_tokens) if eplb_enabled else None
        deep_gemm = load_deepgemm()
        self.kernel = deep_gemm.fp8_fp4_mega_moe
        self.num_shared_experts = text.n_shared_experts or 0
        shared, shared_l1, shared_l2 = self._prepare_shared_weights(layer, deep_gemm.transform_weights_for_mega_moe)
        # Byte-paired UE8M0 is packed along K, not the NZ-transposed layout
        # used by the existing CANN grouped matmul backend.
        w13, w2 = deep_gemm.transform_weights_for_mega_moe(
            (
                layer.w13_weight.data.contiguous().view(torch.int8),
                layer.w13_weight_scale.data.contiguous().view(torch.uint8).view(torch.int16),
            ),
            (
                layer.w2_weight.data.contiguous().view(torch.int8),
                layer.w2_weight_scale.data.contiguous().view(torch.uint8).view(torch.int16),
            ),
        )
        # The target loads before DSpark. All layers with the same shape use
        # one view; draft layouts reuse the target allocation through the
        # dependency's public base= API, which checks group and capacity.
        key = (self.num_experts, self.topk, self.hidden, self.intermediate, self.max_tokens, self.num_shared_experts)
        buffers = getattr(group, "_deepgemm_mega_moe_buffers", None)
        if buffers is None:
            buffers = {}
            group._deepgemm_mega_moe_buffers = buffers
        if key not in buffers:
            base = next(iter(buffers.values()), None)
            buffer = deep_gemm.SymmBuffer(
                group.device_group,
                self.num_experts,
                self.max_tokens,
                self.topk,
                self.hidden,
                self.intermediate,
                num_shared_experts=self.num_shared_experts,
                base=base,
            )
            self._compile_kernel(w13, w2, deep_gemm, group, buffer, shared_l1, shared_l2)
            buffers[key] = buffer
            logger.info(
                "DeepGEMM MegaMoE: experts=%d topk=%d shared=%d H=%d I=%d capacity=%d bytes=%d shared_base=%s",
                self.num_experts,
                self.topk,
                self.num_shared_experts,
                self.hidden,
                self.intermediate,
                self.max_tokens,
                buffers[key].num_bytes,
                base is not None,
            )
        self.buffer = buffers[key]
        # Publish only after the kernel and all packed weights are usable.
        layer.w13_weight.data, layer.w13_weight_scale.data = w13
        layer.w2_weight.data, layer.w2_weight_scale.data = w2
        self.shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self.shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        if shared is not None:
            gate_up, down = shared.layer.gate_up_proj, shared.layer.down_proj
            gate_up.weight.data, gate_up.weight_scale.data = shared_l1
            down.weight.data, down.weight_scale.data = shared_l2
            self.shared_l1_weights = (gate_up.weight, gate_up.weight_scale)
            self.shared_l2_weights = (down.weight, down.weight_scale)
            shared.megamoe_shared_weights_ready = True

    def _compile_kernel(self, w13, w2, deep_gemm, group, buffer, shared_l1, shared_l2):
        # Cold JIT completion can differ across ranks by more than the device
        # barrier watchdog. Compile each layout before publishing it, then join
        # on the CPU group before any rank launches the distributed kernel.
        dry_run = deep_gemm._C.get_dry_run()
        try:
            deep_gemm._C.set_dry_run(True)
            out = torch.empty((0, self.hidden), dtype=torch.bfloat16, device=w2[0].device)
            self.kernel(
                out,
                w13,
                w2,
                buffer,
                shared_l1_weights=shared_l1,
                shared_l2_weights=shared_l2,
                activation_clamp=self.activation_clamp,
            )
        finally:
            deep_gemm._C.set_dry_run(dry_run)
        torch.npu.synchronize()
        torch.distributed.barrier(group=group.cpu_group)

    @staticmethod
    def _shared_weight_pair(linear):
        """Expand checkpoint block scales to per-output, per-32 UE8M0."""
        weight, scale = linear.weight.data, linear.weight_scale.data
        rows, columns = weight.shape
        scale_rows, scale_cols = scale.shape
        row_repeat, row_tail = divmod(rows, scale_rows)
        block_k, col_tail = divmod(columns, scale_cols)
        if row_tail or col_tail or not block_k or block_k % 32:
            raise ValueError("Shared checkpoint scales must tile the weight in groups of 32 columns")
        if scale.dtype == torch.float32:
            # Match the DS checkpoint conversion: FP32 powers of two to UE8M0.
            scale = ((scale.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        elif scale.dtype in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)):
            scale = scale.contiguous().view(torch.uint8)
        else:
            raise ValueError("Shared checkpoint scales must contain FP32 or UE8M0 bytes")
        scale = scale.repeat_interleave(row_repeat, dim=0).repeat_interleave(block_k // 32, dim=1)
        return weight.contiguous(), scale.contiguous().view(torch.int16)

    def _prepare_shared_weights(self, layer, transform):
        reference = getattr(layer, "_deepgemm_mega_moe_shared_experts", None)
        shared = reference() if reference is not None else None
        if shared is None:
            if self.num_shared_experts:
                raise RuntimeError(
                    "DeepGEMM shared experts are configured but their module reference is missing or expired"
                )
            return None, None, None
        gate_up, down = shared.layer.gate_up_proj, shared.layer.down_proj
        packed_l1, packed_l2 = transform(self._shared_weight_pair(gate_up), self._shared_weight_pair(down))
        return shared, packed_l1, packed_l2

    def apply(self, layer, x, topk_weights, topk_ids):
        from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

        count = x.shape[0]
        # Pass raw scale bytes through the graph; the native binding reinterprets
        # them as E8M0. No device cast or staging copy is needed.
        scale = self.buffer.x_sf[:count].view(torch.uint8).view(count, self.hidden // 64, 2)
        torch.ops._C_ascend.dynamic_mx_quant_out(x, self.buffer.x[:count], scale)
        self.buffer.topk_idx[:count].copy_(topk_ids)
        # DeepInfer scales routed weights before SwiGLU/requantization. Shared
        # experts have their own kernel path and must not receive this factor.
        routed_weights = self.buffer.topk_weights[:count]
        routed_scale = getattr(layer, "ascend_megamoe_fold_scale", 1.0)
        if routed_scale == 1.0:
            routed_weights.copy_(topk_weights)
        else:
            torch.mul(topk_weights, routed_scale, out=routed_weights)
        out = torch.empty((count, self.hidden), dtype=torch.bfloat16, device=x.device)
        if self.expert_tokens is not None:
            # The native kernel accumulates. Reset each forward with a same-device DMA copy.
            self.expert_tokens.copy_(self.zero_expert_tokens, non_blocking=True)
        self.kernel(
            out,
            (layer.w13_weight, layer.w13_weight_scale),
            (layer.w2_weight, layer.w2_weight_scale),
            self.buffer,
            shared_l1_weights=self.shared_l1_weights,
            shared_l2_weights=self.shared_l2_weights,
            cumulative_local_expert_recv_stats=self.expert_tokens,
            activation_clamp=self.activation_clamp,
        )
        # Shared experts execute inside this kernel, so no side-stream
        # milestones are needed by the framework.
        return FusedExpertsResult(routed_out=out, expert_tokens=self.expert_tokens)
