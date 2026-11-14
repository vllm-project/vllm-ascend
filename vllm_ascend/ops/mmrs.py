# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel row linear with a single RS output contract."""

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm_ascend.ops.linear_op import CustomRowParallelOp
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND


def _sequence_parallel_mmrs(x: torch.Tensor, layer_name: str, output_size: int, tp_size: int) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer.custom_op.run(x)


def _sequence_parallel_mmrs_fake(x: torch.Tensor, layer_name: str, output_size: int, tp_size: int) -> torch.Tensor:
    return x.new_empty(((x.shape[0] + tp_size - 1) // tp_size, output_size))


# Keep runtime metadata, NZ inspection and modulo padding out of Dynamo. In
# particular, a decode/dummy trace must not permanently disable prefill fusion.
direct_register_custom_op(
    op_name="ascend_sequence_parallel_mmrs",
    op_func=_sequence_parallel_mmrs,
    fake_impl=_sequence_parallel_mmrs_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


class SequenceParallelMMReduceScatterOp(CustomRowParallelOp):
    """Unquantized BF16 projection returning a TP token shard on every path.

    The backend targets A3 BF16 AIV MMRS with BF16 output and requires ND
    weights (the production BF16 configuration, weight_nz_mode=1); any other
    weight format stays on the MM + RS fallback. Additional dtype backends
    extend this class by overriding unsupported_reason, _static_weight_ok
    and _run_fused without touching the BF16 path.
    """

    SUPPORTED_TP_SIZES = (2, 4, 8)
    MIN_K = 256
    MAX_K = 65535

    @staticmethod
    def unsupported_reason(layer) -> str | None:
        if layer.custom_op is not None:
            return "an existing custom linear op owns the communication"
        if layer.reduce_results or not layer.input_is_parallel or layer.disable_tp:
            return "requires a TP-partitioned input and an external ReduceScatter"
        if layer.bias is not None:
            return "only bias-free projections are supported"
        if layer.tp_size not in SequenceParallelMMReduceScatterOp.SUPPORTED_TP_SIZES:
            return "requires TP size 2, 4 or 8"
        if not callable(getattr(torch_npu, "npu_mm_reduce_scatter_base", None)):
            return "requires the torch_npu fused MM+ReduceScatter interface"
        if not isinstance(layer.quant_method, UnquantizedLinearMethod):
            return "requires an unquantized linear method"
        weight = getattr(layer, "weight", None)
        if weight is None or weight.ndim != 2 or weight.dtype != torch.bfloat16:
            return "requires unquantized BF16 projection weights"
        return None

    def __init__(self, layer, attention_layer_name: str):
        super().__init__(layer)
        if reason := self.unsupported_reason(layer):
            raise ValueError(reason)
        self.update_attrs()
        self.attention_layer_name = attention_layer_name
        backend = self.comm_group.device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(self.tp_rank)
        self._static_ok_weight: torch.Tensor | None = None
        self._static_ok_key: tuple | None = None
        self._x2_weight: torch.Tensor | None = None

    def apply_impl(self, input_: torch.Tensor) -> tuple[torch.Tensor, None]:
        output = torch.ops.vllm.ascend_sequence_parallel_mmrs(input_, self.prefix, self.layer.output_size, self.tp_size)
        return output, None

    def _can_fuse(self, x: torch.Tensor) -> bool:
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
        # No token-count gate: the fused/fallback choice must stay measurable
        # end to end; a cap would hide the fusion effect in A/B comparisons.
        return self._static_weight_ok(x.shape[1])

    def _static_weight_ok(self, k: int) -> bool:
        """Validate the unquantized weight and cache the fused x2 operand.

        Loading is complete before the first forward, so only a replaced or
        in-place-mutated weight is revalidated. The loaded weight is [N, K];
        the fused operator takes the transposed [K, N] view directly.
        """
        weight = self.layer.weight
        if self._static_ok_weight is weight and self._static_ok_key == (weight._version, k):
            return True
        basic = (
            weight.ndim == 2
            and weight.dtype == torch.bfloat16
            and weight.shape[0] == self.layer.output_size
            and weight.shape[1] == k
            and self.layer.output_size > 0
            and weight.is_contiguous()
        )
        if not basic:
            return False
        if torch_npu.get_npu_format(weight) != ACL_FORMAT_FRACTAL_ND:
            return False
        self._static_ok_weight = weight
        self._static_ok_key = (weight._version, k)
        self._x2_weight = weight.t()
        return True

    def run(self, input_: torch.Tensor) -> torch.Tensor:
        # All gates use TP-consistent metadata before any collective. Never
        # attempt an exception-based fallback after a fused collective starts.
        if not self._can_fuse(input_):
            assert self.quant_method is not None
            output = self.quant_method.apply(self.layer, input_, None)
            return sp_reduce_scatter(output)
        return self._run_fused(input_)

    def _run_fused(self, input_: torch.Tensor) -> torch.Tensor:
        """BF16 MM + ReduceScatter through the AIV fusion path.

        sp_reduce_scatter zero-pads the GEMM result to the TP multiple, so
        padding the bias-free input rows beforehand yields the same shard.
        _can_fuse has already cached the [K, N] x2 view.
        """
        logger.info_once("Layer MMRS fused path active")
        padding = (-input_.shape[0]) % self.tp_size
        if padding:
            input_ = F.pad(input_, (0, 0, 0, padding))
        return torch_npu.npu_mm_reduce_scatter_base(
            input_.contiguous(),
            self._x2_weight,
            self.hcom,
            self.tp_size,
            reduce_op="sum",
            comm_mode="aiv",
        )


def install_sequence_parallel_mmrs(layer, *, sequence_parallel: bool, attention_layer_name: str) -> bool:
    """Transfer the caller's RS to a supported linear; False changes nothing."""
    if not sequence_parallel:
        return False
    config = get_current_vllm_config()
    # Pure decode nodes (PD-disaggregated kv_consumer) never see a
    # prefill batch, so the fused path could never fire; keep the
    # plain MM + RS path there. Every other node (mixed deployment
    # without a kv connector, pure prefill, kv_both) installs it.
    transfer = config.kv_transfer_config
    if transfer is not None and transfer.is_kv_consumer and not transfer.is_kv_producer:
        return False
    if config.lora_config is not None:
        return False
    # Context-parallel weight/group adaptations have a separate ownership
    # contract. Only the original TP group is supported in this first backend.
    if (
        config.parallel_config.prefill_context_parallel_size != 1
        or config.parallel_config.decode_context_parallel_size != 1
    ):
        return False
    if not get_current_hardware_profile().supports(HardwareCapability.MM_REDUCE_SCATTER_BF16):
        return False
    if reason := SequenceParallelMMReduceScatterOp.unsupported_reason(layer):
        logger.info_once("Layer MMRS disabled for %s: %s", layer.prefix, reason)
        return False
    context = config.compilation_config.static_forward_context
    if layer.prefix in context:
        raise ValueError(f"Duplicate layer name: {layer.prefix}")
    op = SequenceParallelMMReduceScatterOp(layer, attention_layer_name)
    context[layer.prefix] = layer
    layer.custom_op = op
    logger.info_once("Layer MMRS enabled for %s", layer.prefix)
    return True
