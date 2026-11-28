# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel row linear with a single RS output contract."""

# TODO: The scenarios where this operator pays off are not fully determined
# yet, so this is a temporary standalone implementation. Once the benefits
# are confirmed, it will be moved into RowParallelLinear.
import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm_ascend.ops.linear_op import CustomRowParallelOp
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND


class SequenceParallelMMReduceScatterOp(CustomRowParallelOp):
    """Unquantized BF16 projection owning the post-attention ReduceScatter.

    The model installs the op on the O-projection when sequence parallelism
    is active (without SP the projection keeps its plain GEMM forward). It
    replaces the whole forward: fused MMRS when possible, otherwise plain
    MM + ReduceScatter. Both paths return the SP token shard, so the caller
    never branches.

    Every gate is evaluated dynamically in _can_fuse, keeping the only
    fused/fallback branch in run(). Facts that cannot change after loading
    (config, hardware capability, layer layout) are cached once at
    construction; the weight layout is re-checked per forward (a cheap
    metadata read) and passed as a fresh transposed view. The backend
    targets BF16 MMRS with BF16 output and requires ND weights (the
    production BF16 configuration, weight_nz_mode=1); any other weight
    format stays on the fallback.
    """

    SUPPORTED_TP_SIZES = (2, 4, 8)
    # GEMM inner-dimension range supported by the fused kernel, same bounds
    # as the Kimi O-projection backend.
    MIN_K = 256
    MAX_K = 65535

    def __init__(self, layer, attention_layer_name: str, *, min_tokens: int, max_tokens: int):
        super().__init__(layer)
        self.update_attrs()
        self.attention_layer_name = attention_layer_name
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        # A custom op already owning the layer's communication (e.g.
        # fine-grained OTP) keeps ownership; this op only delegates to it.
        self._previous_op = layer.custom_op
        backend = self.comm_group.device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(self.tp_rank)
        self._static_reason = self._find_static_reason()
        if self._static_reason is None:
            logger.info_once("Layer MMRS enabled for %s", self.prefix)
        else:
            logger.info_once("Layer MMRS disabled for %s: %s", self.prefix, self._static_reason)

    def _find_static_reason(self) -> str | None:
        """Collect the facts fixed after loading; None leaves fusion possible."""
        layer = self.layer
        if self._previous_op is not None:
            return "an existing custom linear op owns the communication"
        if layer.reduce_results or not layer.input_is_parallel or layer.disable_tp:
            return "requires a TP-partitioned input and an external ReduceScatter"
        if layer.bias is not None:
            return "only bias-free projections are supported"
        if self.tp_size not in self.SUPPORTED_TP_SIZES:
            return "requires TP size 2, 4 or 8"
        if not callable(getattr(torch_npu, "npu_mm_reduce_scatter_base", None)):
            return "requires the torch_npu fused MM+ReduceScatter interface"
        if not isinstance(layer.quant_method, UnquantizedLinearMethod):
            return "requires an unquantized linear method"
        weight = getattr(layer, "weight", None)
        if weight is None or weight.ndim != 2 or weight.dtype != torch.bfloat16:
            return "requires unquantized BF16 projection weights"
        config = get_current_vllm_config()
        # Pure decode nodes (PD-disaggregated kv_consumer) never see a
        # prefill batch, so the fused path could never fire.
        transfer = config.kv_transfer_config
        if transfer is not None and transfer.is_kv_consumer and not transfer.is_kv_producer:
            return "a pure decode node never runs a prefill batch"
        if config.lora_config is not None:
            return "LoRA owns the projection weights"
        # Context-parallel weight/group adaptations have a separate
        # ownership contract. Only the original TP group is supported.
        if (
            config.parallel_config.prefill_context_parallel_size != 1
            or config.parallel_config.decode_context_parallel_size != 1
        ):
            return "context parallelism owns the TP group"
        if not get_current_hardware_profile().supports(HardwareCapability.MM_REDUCE_SCATTER_BF16):
            return "hardware does not support the BF16 MM ReduceScatter kernel"
        return None

    def apply_impl(self, input_: torch.Tensor):
        if self._previous_op is not None:
            return self._previous_op.apply_impl(input_)
        return self.run(input_), self.bias if self.skip_bias_add else None

    def run(self, input_: torch.Tensor) -> torch.Tensor:
        # The only fused/fallback branch; every gate lives in _can_fuse and
        # is evaluated per forward, before any collective.
        if not self._can_fuse(input_):
            return self._run_fallback(input_)
        return self._run_fused(input_)

    def _can_fuse(self, x: torch.Tensor) -> bool:
        if self._static_reason is not None:
            return False
        if not is_forward_context_available():
            return False
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            metadata = metadata.get(self.attention_layer_name)
        # Missing metadata includes dummy/profile runs. Decode and mixed
        # batches stay on the existing path, including FULL_DECODE_ONLY capture.
        if metadata is None or getattr(metadata, "num_prefills", 0) <= 0 or getattr(metadata, "num_decodes", None) != 0:
            return False
        if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[0] == 0:
            return False
        if not self.MIN_K <= x.shape[1] < self.MAX_K:
            return False
        # The model gates the token count to the range where fusion pays
        # off; TP ranks see the same count after the SP all-gather.
        if not self.min_tokens <= x.shape[0] < self.max_tokens:
            return False
        return self._weight_supported(x)

    def _weight_supported(self, x: torch.Tensor) -> bool:
        """Check the weight layout for the fused kernel, per forward.

        The loaded weight is [N, K]; the fused operator takes a fresh
        transposed [K, N] view of it (no copy, nothing cached), so a
        replaced or in-place-mutated weight is seen on the very next call.
        """
        weight = self.layer.weight
        basic = (
            weight.ndim == 2
            and weight.dtype == torch.bfloat16
            and weight.shape[0] == self.layer.output_size
            and weight.shape[1] == x.shape[1]
            and self.layer.output_size > 0
            and weight.is_contiguous()
        )
        return basic and torch_npu.get_npu_format(weight) == ACL_FORMAT_FRACTAL_ND

    def _run_fallback(self, input_: torch.Tensor) -> torch.Tensor:
        """Plain MM + ReduceScatter fallback returning the SP token shard."""
        assert self.quant_method is not None
        input_parallel = self.get_input_parallel(input_)
        # Only rank 0 adds the bias, so the ReduceScatter does not duplicate it.
        bias = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output = self.quant_method.apply(self.layer, input_parallel, bias)
        # The caller dropped its external ReduceScatter; this op returns
        # the SP token shard on every path.
        return sp_reduce_scatter(output)

    def _run_fused(self, input_: torch.Tensor) -> torch.Tensor:
        """BF16 MM + ReduceScatter through the fused path.

        sp_reduce_scatter zero-pads the GEMM result to the TP multiple, so
        padding the bias-free input rows beforehand yields the same shard.
        _can_fuse has already validated the weight layout; the [K, N]
        transposed view is created here per call (no copy).
        """
        logger.info_once("Layer MMRS fused path active")
        padding = (-input_.shape[0]) % self.tp_size
        if padding:
            input_ = F.pad(input_, (0, 0, 0, padding))
        # comm_mode is left to the device adaptor's per-platform default
        # (AIV on A2/A3, AI CPU or CCU on A5).
        return DeviceOperator.npu_mm_reduce_scatter_base(
            input_.contiguous(),
            self.layer.weight.t(),
            self.hcom,
            self.tp_size,
            reduce_op="sum",
        )
