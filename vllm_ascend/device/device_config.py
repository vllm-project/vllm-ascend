# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from enum import Enum

import torch
import torch_npu


class AscendDeviceType(Enum):
    A2 = 0
    A3 = 1
    _310P = 2
    A5 = 3


def _soc_version_to_device_type(soc_version: int) -> AscendDeviceType:
    if 220 <= soc_version <= 225:
        return AscendDeviceType.A2
    if 250 <= soc_version <= 255:
        return AscendDeviceType.A3
    if 200 <= soc_version <= 205:
        return AscendDeviceType._310P
    if soc_version == 260:
        return AscendDeviceType.A5
    raise RuntimeError(f"Can not support soc_version: {soc_version}.")


def check_ascend_device_type():
    """Verify the runtime device matches the built-in device type.

    Call this at worker startup, after `torch_npu` is usable, to detect
    mismatches between the installed wheel and the actual NPU.
    """
    soc_version = torch_npu.npu.get_soc_version()
    cur_device_type = _soc_version_to_device_type(soc_version)
    assert DeviceConfig._device_type == cur_device_type, (
        f"Current device type: {cur_device_type} does not match the installed version's device type: "
        f"{DeviceConfig._device_type}, please check your installation package."
    )


class _DeviceConfig:
    """Device configuration that abstracts away device-type differences.

    External modules should import the singleton :data:`device_config` and access
    its properties instead of inspecting :class:`AscendDeviceType` directly.
    Private identity checks (``_is_*``) are reserved for internal use; external
    callers use only the semantic properties below.
    """

    HEAD_SIZE_PADDING = 128

    _DSV4_BLOCK_SIZES = {
        128: [[128, 128, 8, 32], [16640, 131072]],
        64: [[64, 64, 4, 16], [8320, 65536]],
        32: [[32, 32, 2, 8], [4160, 32768]],
    }
    _DSV4_BLOCK_SIZES_A5 = {
        128: [[128, 128, 8, 16], [16896, 81920]],
        64: [[64, 64, 4, 8], [8448, 40960]],
        32: [[32, 32, 2, 4], [4224, 20480]],
    }

    def __init__(self):
        from vllm_ascend import _build_info  # type: ignore

        device_type = getattr(_build_info, "__device_type__", None)
        if device_type is None:
            self._device_type = _soc_version_to_device_type(torch_npu.npu.get_soc_version())
        else:
            self._device_type = AscendDeviceType[device_type]

    # ==== private identity (for internal use only) ===========================

    @property
    def _is_a2(self) -> bool:
        return self._device_type == AscendDeviceType.A2

    @property
    def _is_a3(self) -> bool:
        return self._device_type == AscendDeviceType.A3

    @property
    def _is_310p(self) -> bool:
        return self._device_type == AscendDeviceType._310P

    @property
    def _is_a5(self) -> bool:
        return self._device_type == AscendDeviceType.A5

    # ==== dtype choices ======================================================

    @property
    def c8_k_cache_dtype(self) -> torch.dtype:
        """KV cache dtype for sparse C8 indexer (k_cache)."""
        return torch.float8_e4m3fn if self._is_a5 else torch.int8

    @property
    def c8_k_scale_cache_dtype(self) -> torch.dtype:
        """KV cache scale dtype for sparse C8 indexer (k_scale_cache)."""
        return torch.float32 if self._is_a5 else torch.float16

    @property
    def fa_quant_act_dtype(self) -> torch.dtype:
        """Activation quantization dtype for fused-attention quant."""
        return torch.float8_e4m3fn if self._is_a5 else torch.int8

    @property
    def kv_cache_dtype(self) -> torch.dtype:
        """Default KV cache dtype."""
        return torch.float8_e4m3fn if self._is_a5 else torch.int8

    @property
    def indexer_k_dtype(self) -> torch.dtype:
        """Indexer k_cache dtype."""
        return torch.float8_e4m3fn if self._is_a5 else torch.int8

    @property
    def mla_scale_dtype(self) -> torch.dtype:
        """Scale dtype for MLA attention spec."""
        return torch.float if self._is_a5 else torch.float16

    @property
    def swa_cache_dtype(self) -> torch.dtype:
        """SWA cache dtype."""
        return torch.float8_e4m3fn if self._is_a5 else torch.bfloat16

    @property
    def kv_quant_dtype(self) -> torch.dtype:
        """KV quantization dtype."""
        return torch.float8_e4m3fn if self._is_a5 else torch.int8

    @property
    def norm_dtype(self) -> torch.dtype | None:
        """RMSNorm dtype for Compressor layers."""
        return torch.float32 if self._is_a5 else None

    # ==== size / shape =======================================================

    def pd_cache_head_size(self, head_dim: int) -> int:
        """Get the padded cache head size for a given head dimension.

        On A5, the head size is padded by HEAD_SIZE_PADDING to accommodate
        quant-scale metadata alongside the KV data.
        """
        return head_dim + self.HEAD_SIZE_PADDING if self._is_a5 else head_dim

    @property
    def dsv4_block_sizes(self) -> dict:
        """DeepSeek V4 block-size tables (per head_size)."""
        return self._DSV4_BLOCK_SIZES_A5 if self._is_a5 else self._DSV4_BLOCK_SIZES

    @property
    def slot_mapping_2d(self) -> bool:
        """Whether slot_mapping uses 2D [block_idx, offset] format.

        A5 uses 1D flat indices; other devices use 2D.
        """
        return not self._is_a5

    @property
    def kv_cache_has_v_tensor(self) -> bool:
        """Whether sparse C8 KV cache has a separate v_tensor.

        On A5, kv_lora and k_rope are merged into a single CKV tensor.
        """
        return not self._is_a5

    @property
    def indexer_has_full_cache(self) -> bool:
        """Whether indexer KV cache includes a full_cache slot."""
        return self._is_a5

    @property
    def sparse_c8_cache_tuple_len(self) -> int:
        """Number of elements in sparse C8 KV cache tuple."""
        return 3 if self._is_a5 else 4

    # ==== feature flags ======================================================

    @property
    def enable_atb(self) -> bool:
        """Whether ATB extensions should be registered/warmed-up."""
        return not self._is_a5

    @property
    def enable_local_comm_res(self) -> bool:
        """Whether to set up ascend local communication resources."""
        return self._is_a5

    @property
    def use_paged_attention(self) -> bool:
        """Whether paged attention is supported."""
        return not self._is_a5

    @property
    def mlapo_decode_only(self) -> bool:
        """Whether MLAPO is restricted to decode-only instances.

        On A5, MLAPO can be enabled unconditionally; on other devices
        it is restricted to decode-only instances.
        """
        return not self._is_a5

    @property
    def mlapo_requires_quantization(self) -> bool:
        """Whether MLAPO requires W8A8 quantization.

        On A5, MLAPO supports unquantized (float) paths.
        """
        return not self._is_a5

    @property
    def mlapo_uses_a5_weight_processing(self) -> bool:
        """Whether MLAPO uses A5-specific weight processing path."""
        return self._is_a5

    @property
    def sparse_c8_supports_unquantized_mlapo(self) -> bool:
        """Whether sparse C8 indexer supports unquantized MLAPO."""
        return self._is_a5

    @property
    def fa_quant_decode_only(self) -> bool:
        """Whether FA-quant is restricted to decode-only instances.

        On A5, FA-quant can be enabled on any instance.
        """
        return not self._is_a5

    @property
    def fa_quant_uses_nz_format(self) -> bool:
        """Whether FA-quant uses NZ (non-zero) layout for K cache."""
        return not self._is_a5

    @property
    def supports_mxfp(self) -> bool:
        """Whether MXFP quantization (MXFP4/MXFP8) is supported."""
        return self._is_a5

    @property
    def enable_custom_ops(self) -> bool:
        """Whether custom ops are enabled."""
        return not self._is_a5

    @property
    def supports_compilation_custom_ops(self) -> bool:
        """Whether vLLM compilation custom ops are supported.

        Currently disabled on 310P.
        """
        return self._device_type != AscendDeviceType._310P

    @property
    def supports_triton(self) -> bool:
        """Whether Triton is available on this device.

        Triton is required on A2/A3/A5, only 310P cannot install it.
        Additionally verifies that Triton is actually functional
        (importable with backends).
        """
        if self._is_310p:
            return False
        from importlib.util import find_spec

        if find_spec("triton") is None:
            return False
        try:
            from triton.backends import backends  # type: ignore[import-untyped]  # noqa: F401
        except Exception:
            return False
        return True

    @property
    def enable_npu_graph_ex(self) -> bool:
        """Whether npugraph_ex is enabled."""
        return not self._is_a5

    @property
    def enable_irq_binding(self) -> bool:
        """Whether IRQ CPU binding is enabled."""
        return not self._is_a5

    @property
    def use_ascendc_sampler(self) -> bool:
        """Whether to use AscendC top-k/top-p sampler kernel."""
        return self._device_type in {AscendDeviceType.A2, AscendDeviceType.A3}

    @property
    def use_ascend_lora_ops(self) -> bool:
        """Whether to use ascend custom LoRA ops vs torch native LoRA ops."""
        return not self._is_310p

    @property
    def use_310p_comm_adaptation(self) -> bool:
        """Whether to activate 310P-specific communication adaptation."""
        return self._is_310p

    # ==== 310P-specific semantic flags ====================================

    @property
    def force_nz_weight_format(self) -> bool:
        """Whether weights should always be converted to NZ format.

        310P always converts supported weights to FRACTAL_NZ regardless
        of the weight_nz_mode config.
        """
        return self._is_310p

    @property
    def use_310p_op_implementations(self) -> bool:
        """Whether to register 310P-specific operator implementations.

        310P uses dedicated kernel implementations for ops such as
        FusedMoE, RMSNorm, RotaryEmbedding, etc.
        """
        return self._is_310p

    @property
    def supports_advanced_quantization(self) -> bool:
        """Whether compressed-tensors and fp8 quantization are supported.

        Disabled on 310P which only supports modelslim quantization.
        """
        return not self._is_310p

    @property
    def use_310p_worker(self) -> bool:
        """Whether to use the 310P-specific worker implementation."""
        return self._is_310p

    @property
    def supports_advanced_attention_backends(self) -> bool:
        """Whether MLA, SFA, DSA attention backends are supported.

        310P only supports the basic attention backend.
        """
        return not self._is_310p

    @property
    def use_310p_mamba_config(self) -> bool:
        """Whether to use the 310P-specific mamba cache configuration."""
        return self._is_310p

    @property
    def supports_advanced_model_patches(self) -> bool:
        """Whether advanced model patches (qwen3_5, dflash, vl) are supported.

        Disabled on 310P which uses IDEX-based model patches instead.
        """
        return not self._is_310p

    @property
    def use_310p_gdn_attention(self) -> bool:
        """Whether to use the 310P-specific GDN attention backend."""
        return self._is_310p

    @property
    def wo_a_transpose_enabled(self) -> bool:
        """Whether to transpose wo_a weight during weight loading."""
        return not self._is_a5

    @property
    def compressor_uses_quant_config(self) -> bool:
        """Whether Compressor layers should receive quant_config."""
        return not self._is_a5

    @property
    def fal_descale_reciprocal_uses_float(self) -> bool:
        """Whether FAK descale reciprocal should be computed in float32."""
        return self._is_a5

    @property
    def indexer_uses_quant_matmul(self) -> bool:
        """Whether indexer wq_b uses NPU quant matmul (with per-token scale)."""
        return not self._is_a5

    @property
    def vision_tower_preserves_quantized_dtype(self) -> bool:
        """Whether vision tower weights are pre-quantized and must not be
        overwritten by model dtype conversion."""
        return self._is_a5

    # ==== distributed / comm =================================================

    @property
    def memcache_lazy_init(self) -> bool:
        """Whether KV transfer memcache supports lazy initialization.

        Disabled on A2 due to known issues.
        """
        return not self._is_a2

    # ==== MoE ================================================================

    @property
    def moe_dispatch_requires_extra_args(self) -> bool:
        """Whether MoE token dispatch requires extra_args for v2 API."""
        return self._device_type in {AscendDeviceType.A3, AscendDeviceType.A5}

    @property
    def moe_dispatch_uses_a5_specific(self) -> bool:
        """Whether MoE token dispatch uses A5-specific params."""
        return self._is_a5

    # ==== CPU binding ========================================================

    @property
    def cpu_binding_mode(self) -> str:
        """CPU binding strategy string."""
        return {
            AscendDeviceType.A2: "topo_affinity",
            AscendDeviceType.A3: "global_slice",
            AscendDeviceType._310P: "topo_affinity",
            AscendDeviceType.A5: "global_slice",
        }.get(self._device_type, "topo_affinity")

    @property
    def npu_smi_uses_logical_id(self) -> bool:
        """Whether npu-smi uses logical chip-level IDs (card*2+chip)."""
        return self._is_a3

    # ==== fa_quant properties ================================================

    @property
    def fa_quant_uses_c_kv_scale(self) -> bool:
        """Whether FA-quant decode passes c_kv_scale to npu_kv_rmsnorm_rope_cache."""
        return self._is_a5

    @property
    def fa_quant_uses_fak_descale(self) -> bool:
        """Whether FA-quant prefill applies fak_descale_float multiplication."""
        return self._is_a5

    @property
    def fa_quant_dynamic_quant_decode(self) -> bool:
        """Whether to apply dynamic quantization to decode_ql_nope."""
        return self._is_a5

    @property
    def fa_quant_layout(self) -> str:
        """Input layout string for FA-quant decode attention."""
        return "BNSD" if self._is_a5 else "BSND_NBSD"

    # ==== Device adaptor =====================================================

    @property
    def device_operator(self):
        """Returns the device-specific adaptor class.

        Internal use by DeviceOperator singleton.
        """
        from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor

        return A5DeviceAdaptor if self._is_a5 else BaseDeviceAdaptor


DeviceConfig = _DeviceConfig()
