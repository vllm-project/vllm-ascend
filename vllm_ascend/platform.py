#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import math
import os
from importlib import import_module, util
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch
import vllm.envs as envs_vllm
from vllm.logger import logger
from vllm.platforms import Platform, PlatformEnum

# todo: please remove it when solve cuda hard code in vllm
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"


from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.device.hardware_profile import (
    AttentionBackendFamily,
    HardwareCapability,
    QuantizationBackendFamily,
    get_current_hardware_profile,
)

# isort: off
from vllm_ascend.utils import (
    ASCEND_QUANTIZATION_METHOD,
    COMPILATION_PASS_KEY,
    COMPRESSED_TENSORS_METHOD,
    FP8_METHOD,
    bootstrap_custom_op_env,
    check_kv_extra_config,
    enable_sfa_dcp_replicated_indexer,
    is_moe_model,
    model_uses_sfa_sparse,
    refresh_block_size,
    update_cudagraph_capture_sizes,
    enable_sp,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    ModelConfig = None
    VllmConfig = None
    FlexibleArgumentParser = None

# Keep Breakable CUDAGraph opt-in on Ascend. Upstream may auto-enable it
# for selected architectures when the environment variable is absent.
value = os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
logger.info_once(
    "Breakable CUDAGraph on Ascend is opt-in; using VLLM_USE_BREAKABLE_CUDAGRAPH=%s.",
    value,
    scope="process",
)

_CUSTOM_OP_REGISTERED = False
# Delete after the driver is released; temporarily hard-coded to 4
MAX_REDUCED_CAPTURE_SIZES = 4


DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset(
    {
        "Qwen2MoeForCausalLM",
    }
)


def _is_default_v2_model_runner_model(vllm_config: VllmConfig) -> bool:
    """Model whitelist: enable V2 for default-V2 architectures and non-MoE models."""
    model_config = vllm_config.model_config
    if model_config is None:
        return False

    if model_config.runner_type != "generate":
        return False

    if getattr(model_config, "is_hybrid", False):
        return False

    if getattr(model_config, "is_attention_free", False):
        return False
    architectures = getattr(model_config, "architectures", [])
    return any(arch in DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES for arch in architectures) or not model_config.is_moe


def _is_supported_v2_model_runner_feature(vllm_config: VllmConfig) -> bool:
    """Feature whitelist: only whitelisted features may be enabled with a whitelisted model."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return True
    if speculative_config.method in ("eagle", "mtp", "dflash"):
        logger.info_once(
            "Model Runner V2 is enabled by default for speculative method '%s'.",
            speculative_config.method,
        )
        return True
    return False


def _use_v2_model_runner(vllm_config: VllmConfig) -> bool:
    use_v2_model_runner = envs_vllm.VLLM_USE_V2_MODEL_RUNNER
    if use_v2_model_runner is not None:
        return use_v2_model_runner

    if not _is_default_v2_model_runner_model(vllm_config):
        logger.warning_once(
            "Model Runner V2 model whitelist does not include this model; using the V1 model runner instead."
        )
        return False

    if not _is_supported_v2_model_runner_feature(vllm_config):
        return False

    from vllm.triton_utils import HAS_TRITON  # noqa: E402

    if not HAS_TRITON:
        logger.warning_once("Model Runner V2 requires Triton; using the V1 model runner instead.")
        return False

    return True


class NPUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    simple_compile_backend: str = "eager"  # Disable torch.compile()
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    ]
    dispatch_key: str = "PrivateUse1"

    supported_quantization: list[str] = [
        ASCEND_QUANTIZATION_METHOD,
        COMPRESSED_TENSORS_METHOD,
        FP8_METHOD,
        "deepseek_v4_fp8",
        "modelopt_mxfp8",
        "mxfp8",
    ]

    @property
    def pass_key(self) -> str:
        """
        Inductor config key for the PassManager custom pass, for example 'post_grad_custom_post_pass'.
        It is a parameter of inductor_config used to register custom passes.
        Currently, we only use Inductor's 'pattern matcher' functionality, so we define our own pass_key.
        """
        return COMPILATION_PASS_KEY

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        pass

    def is_sleep_mode_available(self) -> bool:
        return True

    def is_cumem_allocator_available(self) -> bool:
        # vLLM main gates sleep mode on the platform reporting a
        # usable cumem allocator. NPU provides its own ``CaMemAllocator``
        # (vllm_ascend.device_allocator.camem), so report availability here.
        # ModelConfig validation runs before custom-op init, so avoid importing
        # the extension and just declare support.
        return True

    @classmethod
    def is_pin_memory_available(cls):
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return True

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode()

    @classmethod
    def set_device(cls, device: torch.device):
        torch.npu.set_device(device)

    @classmethod
    def register_custom_kv_cache_specs(cls, vllm_config: VllmConfig) -> None:
        from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs

        register_ascend_kv_cache_specs()

    @classmethod
    def get_pass_manager_cls(cls) -> str:
        """
        Get the pass manager class for this platform.
        It will be registered as a custom pass under the current_platform.pass_key.
        """
        return "vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager"

    @classmethod
    def get_compile_backend(self) -> str:
        """
        Get the custom compile backend. Previously, we used EagerAdaptor by default.
        To use graph fusion operations, we defined our own backend compiler.
        """
        return "vllm_ascend.compilation.compiler_interface.AscendCompiler"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_ascend.lora.punica_npu.PunicaWrapperNPU"

    @classmethod
    def get_current_memory_usage(cls, device: torch.types.Device | None = None) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator"

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        """
        Get piecewise backend class for piecewise graph.
        """
        return "vllm_ascend.compilation.acl_graph.ACLGraphWrapper"  # noqa

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        device_props = torch.npu.get_device_properties(device_id)
        if not hasattr(device_props, "uuid") or device_props.uuid is None:
            raise RuntimeError(f"Device {device_id} does not have a valid UUID.")
        return device_props.uuid

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """
        Get the total memory of the NPU device in bytes.
        DO NOT IMPLEMENT: Implementing it calls get_device_name() in advance and initializes torch_npu too early.
        torch_npu allows global initialization only once; duplicate initialization causes errors.
        """
        raise NotImplementedError

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads: int | None = None):
        use_compress = getattr(attn_selector_config, "use_compress", False)
        use_mla = attn_selector_config.use_mla
        use_sparse = attn_selector_config.use_sparse
        # index_kpool GLM is not DeepSeek SFA; keep MLA backend.
        try:
            from vllm.config import get_current_vllm_config
            from vllm_ascend.utils import enable_sfa

            if use_sparse and not enable_sfa(get_current_vllm_config()):
                use_sparse = False
        except Exception:
            pass
        key = (use_mla, use_sparse)
        backend_key = (*key, use_compress)

        if attn_selector_config.use_pcp and attn_selector_config.use_dcp:
            raise NotImplementedError("Ascend MRV2 does not support PCP and DCP simultaneously yet.")

        if not attn_selector_config.use_pcp and _validate_fa3_backend(key, attn_selector_config):
            return "vllm_ascend.attention.fa3_v1.AscendFABackend"

        backend_map = {
            (True, False, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",
            (False, False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
            (True, True, False): "vllm_ascend.attention.sfa_v1.AscendSFABackend",
            (True, False, True): "vllm_ascend.attention.dsa_v1.AscendDSABackend",
        }
        compatibility_backend_map = {
            (
                False,
                False,
            ): "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310",
            # TODO If MLA/SFA is supported in the future, consider implementing the logic described in these comments.
            # (True, False): "...AscendMLABackend310",
            # (True, True):  "...AscendSFABackend310",
        }

        if get_current_hardware_profile().attention_backend_family is AttentionBackendFamily.COMPATIBILITY:
            return compatibility_backend_map.get(key, compatibility_backend_map[(False, False)])

        if attn_selector_config.use_pcp:
            pcp_backend_map = {
                (True, False, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",
                (False, False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
                (True, True, False): "vllm_ascend.attention.sfa_v1.AscendSFABackend",
                (True, False, True): "vllm_ascend.attention.dsa_v1.AscendDSABackend",
            }
            pcp_backend = pcp_backend_map.get(backend_key)
            if pcp_backend is None:
                raise NotImplementedError(f"Ascend MRV2 PCP does not support attention backend {backend_key}.")
            return pcp_backend

        return backend_map[backend_key]

    @classmethod
    def import_kernels(cls) -> None:
        # Directly importing vllm_ascend_C prevents ASCEND_RT_VISIBLE_DEVICES
        # from being applied during runtime initialization, which causes bugs
        # in the RL module. Therefore, we currently use lazy initialization
        # to avoid this issue. See https://github.com/vllm-project/vllm-ascend/pull/884.
        # TODO: when the above issue is fixed, we can uncomment the following lines.
        # from vllm_ascend.utils import enable_custom_op
        # enable_custom_op()
        # set custom ops path
        global _CUSTOM_OP_REGISTERED
        if _CUSTOM_OP_REGISTERED:
            return
        bootstrap_custom_op_env()
        _CUSTOM_OP_REGISTERED = True

    @classmethod
    def pre_register_and_update(cls, parser: FlexibleArgumentParser | None = None) -> None:
        from vllm_ascend.utils import adapt_patch

        adapt_patch(is_global_patch=True)

        # For online serving, "ascend" quantization method is not a choice natively,
        # so we need to add "ascend" quantization method to quantization methods list
        # and the user can enable quantization using "vllm serve --quantization ascend".
        if parser is not None:
            quant_action = parser._option_string_actions.get("--quantization")
            if quant_action and hasattr(quant_action, "choices") and quant_action.choices:
                if ASCEND_QUANTIZATION_METHOD not in quant_action.choices:
                    quant_action.choices.append(ASCEND_QUANTIZATION_METHOD)

        if get_current_hardware_profile().quantization_backend_family is QuantizationBackendFamily.STANDARD:
            from vllm_ascend.quantization import (  # noqa: F401
                AscendCompressedTensorsConfig,
                AscendFp8Config,
                AscendModelOptMxFp8Config,
                AscendModelSlimConfig,
            )
        else:
            from vllm_ascend._310p.quantization import AscendModelSlimConfig310  # noqa: F401

        _config_deprecated_logging()

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: VllmConfig) -> None:
        """Apply Ascend-specific defaults."""

        default_max_cg_capture_size = _get_default_max_cudagraph_capture_size(vllm_config)
        if default_max_cg_capture_size is not None:
            vllm_config.compilation_config.max_cudagraph_capture_size = default_max_cg_capture_size

    def num_compute_units(cls, device_id: int = 0) -> int:
        """Return the number of Cube Cores on the NPU device.
        This is the NPU equivalent of CUDA's ``multi_processor_count``
        (SM count).  On Ascend hardware the closest concept is
        ``cube_core_num`` exposed by ``torch.npu.get_device_properties``,
        which represents the matrix-compute units (analogous to CUDA SMs).
        This value is consumed by vLLM's
        ``layernorm_guard.calc_rows_per_block`` to size the Triton kernel
        launch grid.  Note that the result is clamped to 4 by that
        function, so the exact value has minimal impact on correctness;
        it only affects kernel occupancy heuristics.
        """
        props = torch.npu.get_device_properties(device_id)
        # cube_core_num is the matrix-compute unit count, semantically
        # closest to CUDA's multi_processor_count (SM count).
        cube_core_num = getattr(props, "cube_core_num", None)
        if cube_core_num is not None and cube_core_num > 0:
            return int(cube_core_num)
        # Fallback for older torch-npu versions that may not expose cube_core_num
        vector_core_num = getattr(props, "vector_core_num", None)
        if vector_core_num is not None and vector_core_num > 0:
            return int(vector_core_num)
        return 24  # safe default (24 Cube Cores)

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None:
        # TODO: NPU still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        using_kv_transfer_with_hybrid = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and vllm_config.kv_transfer_config
        )
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if (
            not cache_config.enable_prefix_caching
            and using_kv_transfer_with_hybrid
            and cache_config.mamba_cache_mode == "align"
        ):
            if cache_config.mamba_block_size is None or cache_config.mamba_block_size == model_config.max_model_len:
                cache_config.mamba_block_size = cache_config.block_size
            else:
                # mamba_block_size must be a multiple of block_size, so that it can hand the block hash
                assert cache_config.mamba_block_size % cache_config.block_size == 0, (
                    f"mamba_block_size must be a multiple of block_size: {cache_config.block_size}"
                )

    @classmethod
    def _validate_indexer_pp_config(cls, vllm_config: VllmConfig) -> None:
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size <= 1:
            return

        config = getattr(vllm_config.model_config, "hf_text_config", None)
        if config is None:
            return

        indexer_types = getattr(config, "indexer_types", None)
        use_index_cache = getattr(config, "use_index_cache", False)
        if indexer_types is None and not use_index_cache:
            return

        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if not isinstance(num_hidden_layers, int):
            return

        from vllm.distributed.utils import get_pp_indices

        for pp_rank in range(pp_size):
            start_layer, end_layer = get_pp_indices(
                num_hidden_layers,
                pp_rank,
                pp_size,
            )
            if start_layer >= end_layer:
                continue

            if use_index_cache:
                index_topk_pattern = getattr(config, "index_topk_pattern", None)
                if index_topk_pattern is None:
                    index_topk_freq = getattr(config, "index_topk_freq", 1)
                    index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
                    skip_topk = max(start_layer - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
                else:
                    skip_topk = start_layer < len(index_topk_pattern) and index_topk_pattern[start_layer] == "S"
                if skip_topk:
                    raise ValueError(
                        "Index cache dependency crosses a pipeline-parallel stage boundary: "
                        f"PP rank {pp_rank}/{pp_size} owns layers [{start_layer}, {end_layer}), "
                        f"but layer {start_layer} skips Top-K computation without a preceding "
                        "Top-K recomputation in the same PP stage. "
                        "Cross-PP Top-K index propagation is not supported."
                    )

            if indexer_types is None:
                continue

            has_full_indexer = False
            for layer_id in range(start_layer, end_layer):
                indexer_type = indexer_types[layer_id] if layer_id < len(indexer_types) else None
                if isinstance(indexer_type, str):
                    indexer_type = indexer_type.lower()
                if indexer_type == "full":
                    has_full_indexer = True
                elif indexer_type == "shared" and not has_full_indexer:
                    raise ValueError(
                        "IndexShare group crosses a pipeline-parallel stage boundary: "
                        f"PP rank {pp_rank}/{pp_size} owns layers [{start_layer}, {end_layer}), "
                        f"but layer {layer_id} uses a shared Indexer without a preceding "
                        "full Indexer in the same PP stage. "
                        "Cross-PP Top-K index propagation is not supported."
                    )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config.vllm import VllmConfig as VllmConfig

        VllmConfig.use_v2_model_runner = property(_use_v2_model_runner)

        from vllm_ascend.quantization.utils import maybe_auto_detect_quantization
        from vllm_ascend.logger import configure_ascend_file_logging, configure_ascend_logging

        # 1.Configure logging
        configure_ascend_file_logging()
        configure_ascend_logging()

        # 2.Early exit checks and validate parallel config
        device_config = getattr(vllm_config, "device_config", None)
        if device_config is not None and getattr(device_config, "device_type", cls.device_type) != cls.device_type:
            logger.debug("Skipping Ascend-specific config updates for device type %s.", device_config.device_type)
            return

        if vllm_config.model_config is None:
            logger.warning("Model config is missing. Skipping Ascend-specific config updates.")
            return

        cls._validate_indexer_pp_config(vllm_config)

        _validate_draft_decode_context_parallel_config(vllm_config)
        _validate_parallel_config(vllm_config)

        # 3.Auto detect quantization method
        maybe_auto_detect_quantization(vllm_config)

        # 4.Make sure the config is compatible with Ascend
        _fix_incompatible_config(vllm_config)

        # 5.Initialize Ascend config and validate Ascend-specific options
        # (fused MC2 exclusivity + scheduler extension policies)
        # ascend_config is only used for verification, this object must NOT be modified here
        ascend_config = init_ascend_config(vllm_config)
        _check_ascend_config(vllm_config, ascend_config)

        # 6.Update compilation / cudagraph modes (ascend_config -> vllm_config).
        _update_compilation_modes(vllm_config, ascend_config)

        # 7.Recompute cudagraph sizes and setup compile backend (vllm_config).
        _setup_compile_backend(
            vllm_config,
            compile_backend=cls.get_compile_backend(),
            enable_shared_expert_dp=ascend_config.enable_shared_expert_dp,
            enable_dsa_cp=ascend_config.enable_dsa_cp,
        )

        # 8.Setup worker class, custom ops and scheduler (ascend_config -> vllm_config).
        _setup_worker_and_scheduler(vllm_config, ascend_config)

        # 9.Validate SFA / DCP / KV and SP consistency (vllm_config)
        _validate_sfa_dcp_kv_sp(vllm_config)

        # 10.Set pytorch NPU allocator env (vllm_config)
        _set_pytorch_npu_alloc_env(vllm_config)

    @classmethod
    def set_additional_forward_context(
        cls,
        attn_metadata: dict[str, Any],
        vllm_config: VllmConfig,
        dp_metadata,
        num_tokens: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        cudagraph_runtime_mode=None,
        batch_descriptor=None,
        ubatch_slices=None,
    ) -> dict[str, Any]:
        """set additional forward context for ascend npus.

        Args:
            attn_metadata (dict[str, Any]): attention metadata for all layers.
            vllm_config (VllmConfig): configuration of vllm.
            dp_metadata (Dpmetadata): metadata for data parallelism.
                lack of typehint because of circular import.
            num_tokens (int | None, optional): number of tokens. Defaults to None.
            num_tokens_across_dp (torch.Tensor | None, optional): number of tokens
                across data parallelism.Defaults to None.
            cudagraph_runtime_mode (CUDAGraphMode, optional): mode of cudagraph runtime.
                Defaults to None.lack of typehint because of circular import.
            batch_descriptor (BatchDescriptor, optional): descriptor of batch.
                Defaults to None.
            ubatch_slices (UBatchSlices, optional): slice info for dual batch.
                Defaults to None. lack of typehint because of circular import

        Returns:
            dict[str, Any]: _description_
        """
        # NOTE(Ronald1995): avoid circular import.
        from vllm_ascend.ascend_forward_context import (
            get_mc2_mask,
            get_mrv2_in_profile_run,
            select_moe_comm_method,
        )
        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method
        from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg
        from vllm.distributed import get_dp_group, get_tensor_model_parallel_world_size

        # NOTE(Ronald1995): avoid circular import, cudagraph_runtime_mode is
        # CUDAGraphMode.NONE in vllm, but we can't set CUDAGraphMode.NONE in
        # argument default value, so we set it to None first, then set it to
        # CUDAGraphMode.NONE here.
        from vllm.config import CUDAGraphMode

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        # TODO(Ronald1995): model runner v1 still use ascend_forward_context,
        # when v1's forward context is refactored, we can remove this branch.
        # Currently, model runner v2 use the new forward context.
        # compared to v1, v2's forward context lacks some fields, such as:
        # is_first_layer, prefetch_mlp_gate_up_proj, prefetch_mlp_gate_down_proj,
        # prefetch_mlp_enabled, model_instance, is_draft_model.
        dynamic_mx_quant_scale_alg = get_dynamic_mx_quant_scale_alg(vllm_config)
        if not vllm_config.use_v2_model_runner:
            return {"dynamic_mx_quant_scale_alg": dynamic_mx_quant_scale_alg}

        # is_draft_model will be removed later, so we set it to False temporarily.
        is_draft_model = False
        # v2 has 2 graphs in eager, one for prefill, the other for decodes, this flag is aimed to distinguish them.
        is_draft_model_prefill = False
        sinks = False
        in_profile_run = get_mrv2_in_profile_run()

        tp_world_size = get_tensor_model_parallel_world_size()

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing.
        capturing = False

        mmrs_fusion = True
        if is_moe_model(vllm_config):
            mmrs_fusion = False
        padded_length = None

        if num_tokens is None and attn_metadata is not None:
            num_tokens = list(attn_metadata.values())[0].num_actual_tokens
        pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        max_tokens_across_pcp = num_tokens if pcp_size > 1 else 0
        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and dp_metadata is not None:
            max_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu.max().item()
            padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
        else:
            max_tokens_across_dp = num_tokens

        # NOTE: Must use max_tokens_across_dp instead of num_tokens for MoE comm method selection
        # to ensure consistent communication method across all DP ranks
        moe_comm_type = select_moe_comm_method(
            max_tokens_across_dp,
            vllm_config,
        )
        moe_comm_method = get_moe_comm_method(moe_comm_type)

        mc2_mask = None
        padded_num_tokens = None
        if num_tokens is not None:
            num_actual_tokens = num_tokens
            # NOTE: token num which need to pad to when mc2
            padded_num_tokens = math.ceil(max_tokens_across_dp / tp_world_size) * tp_world_size
            reserved_mc2_mask = get_mc2_mask()
            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[:padded_num_tokens]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
        return {
            "moe_comm_type": moe_comm_type,
            "moe_comm_method": moe_comm_method,
            "capturing": capturing,
            "mmrs_fusion": mmrs_fusion,
            "num_tokens": num_tokens,
            "padded_length": padded_length,
            "max_tokens_across_dp": max_tokens_across_dp,
            "max_tokens_across_pcp": max_tokens_across_pcp,
            "mc2_mask": mc2_mask,
            "is_draft_model": is_draft_model,
            "is_draft_model_prefill": is_draft_model_prefill,
            "in_profile_run": in_profile_run,
            "padded_num_tokens": padded_num_tokens,
            "sinks": sinks,
            "dynamic_mx_quant_scale_alg": dynamic_mx_quant_scale_alg,
        }


def _fix_incompatible_config(vllm_config: VllmConfig) -> None:
    """
    Check and correct parameters in VllmConfig that are incompatible with Ascend NPU.
    If GPU-specific or currently unsupported parameters are set by the user,
    log a warning and reset them to safe values.
    """
    _validate_eplb_config(vllm_config)
    model_config = vllm_config.model_config
    # ==================== 1. Model Config ====================
    if model_config:
        # Disable Cascade Attention (GPU feature)
        if getattr(model_config, "disable_cascade_attn", False):
            logger.warning(
                "GPU-specific parameter is not supported on Ascend. "
                "parameter=disable_cascade_attn, value=True, action: resetting to False."
            )
            model_config.disable_cascade_attn = False

    # ==================== 2. Cache Config ====================
    if vllm_config.cache_config:
        # Check and reset cpu_kvcache_space_bytes
        if getattr(vllm_config.cache_config, "cpu_kvcache_space_bytes", False):
            logger.warning(
                "Parameter is tied to incompatible backend. "
                "parameter=cpu_kvcache_space_bytes, action: resetting to None for Ascend."
            )
            vllm_config.cache_config.cpu_kvcache_space_bytes = None

        if getattr(vllm_config.cache_config, "calculate_kv_scales", False):
            logger.warning(
                "Parameter is not supported on Ascend NPU. parameter=calculate_kv_scales, action: resetting to False."
            )
            vllm_config.cache_config.calculate_kv_scales = False

    # ==================== 3. MultiModal Config ====================
    multimodal_config = getattr(model_config, "multimodal_config", None) if model_config else None
    if multimodal_config:
        # Ascend uses a different mechanism for Multi-Modal attention
        if getattr(multimodal_config, "mm_encoder_attn_backend", None) is not None:
            logger.warning(
                "Parameter is set but Ascend uses different mechanism. "
                "parameter=mm_encoder_attn_backend, action: resetting to None."
            )
            multimodal_config.mm_encoder_attn_backend = None

    # ==================== 4. Observability Config ====================
    if vllm_config.observability_config:
        # NVTX tracing is NVIDIA specific
        if getattr(vllm_config.observability_config, "enable_layerwise_nvtx_tracing", False):
            logger.warning(
                "Parameter relies on NVIDIA-specific tools. "
                "parameter=enable_layerwise_nvtx_tracing, action: resetting to False."
            )
            vllm_config.observability_config.enable_layerwise_nvtx_tracing = False

    # ==================== 5. Scheduler Config ====================
    if vllm_config.scheduler_config:
        # Partial prefills are specific to ROCm optimization
        if getattr(vllm_config.scheduler_config, "max_num_partial_prefills", 1) != 1:
            logger.warning(
                "Parameter is optimized for incompatible platform. "
                "parameter=max_num_partial_prefills, action: resetting to default (1). "
            )
            vllm_config.scheduler_config.max_num_partial_prefills = 1

    # ==================== 6. Speculative Config ====================
    if vllm_config.speculative_config:
        # Ascend automatically inherits main model quantization
        if getattr(vllm_config.speculative_config, "quantization", None) is not None:
            logger.warning(
                "Speculative quantization is set but Ascend automatically uses "
                "the main model's quantization method. "
                "parameter=quantization, action: resetting to None. "
            )
            vllm_config.speculative_config.quantization = None

    # ==================== 7. KV Transfer Config ====================
    if vllm_config.kv_transfer_config:
        # Buffer size is primarily tied to NCCL (GPU) backends
        current_buffer_size = getattr(vllm_config.kv_transfer_config, "kv_buffer_size", 1e9)
        if current_buffer_size != 1e9:
            logger.warning(
                "Parameter is optimized for incompatible backend. "
                "parameter=kv_buffer_size, value=%s, action: resetting to default (1e9). ",
                current_buffer_size,
            )
            # Use setattr to safely assign the value
            vllm_config.kv_transfer_config.kv_buffer_size = 1e9

        # Check and reset enable_permute_local_kv
        if getattr(vllm_config.kv_transfer_config, "enable_permute_local_kv", False):
            logger.warning(
                "Parameter is tied to incompatible backend. "
                "parameter=enable_permute_local_kv, action: resetting to False. "
            )
            vllm_config.kv_transfer_config.enable_permute_local_kv = False

        # Validate KV transfer parallelism (tp/dp) and make engine_id unique so
        # P/D nodes or restarts never collide; _engine_id_patched keeps it idempotent.
        check_kv_extra_config(vllm_config)
        if not getattr(vllm_config.kv_transfer_config, "_engine_id_patched", False):
            vllm_config.kv_transfer_config.engine_id = f"{vllm_config.kv_transfer_config.engine_id}-{uuid4().hex}"
            vllm_config.kv_transfer_config._engine_id_patched = True

    # ==================== 8. Attention Config ====================
    if vllm_config.attention_config:
        att_config = vllm_config.attention_config

        # Boolean flags that must be False on Ascend (typically NVIDIA-specific)
        force_false_flags = [
            "use_prefill_decode_attention",
            "use_cudnn_prefill",
            "use_trtllm_ragged_deepseek_prefill",
            "use_trtllm_attention",
            "disable_flashinfer_prefill",
            "disable_flashinfer_q_quantization",
        ]
        for flag in force_false_flags:
            if getattr(att_config, flag, False):
                logger.warning(
                    "Ignored GPU-specific parameter. parameter=%s, action: resetting to False. ",
                    flag,
                )
                setattr(att_config, flag, False)

        # Reset specific values to None as Ascend uses its own internal logic
        if getattr(att_config, "flash_attn_version", None) is not None:
            logger.warning(
                "Ignored parameter. Ascend uses its own attention backend. "
                "parameter=flash_attn_version, action: resetting to None. "
            )
            att_config.flash_attn_version = None

        # Notify the user that the backend will be managed by Ascend plugins.
        # FA3 is selected directly in get_attn_backend_cls when RL training
        # consistency is enabled, so it does not depend on this field.
        if getattr(att_config, "backend", None) is not None:
            logger.info(
                "User specified attention backend '%s'. Note that Ascend NPU "
                "will use its registered plugin backend instead. Resetting to None.",
                att_config.backend,
            )
            att_config.backend = None

        # CUDA Graph specific split points are not applicable
        if getattr(att_config, "flash_attn_max_num_splits_for_cuda_graph", 32) != 32:
            logger.warning(
                "Parameter is ignored on Ascend. "
                "parameter=flash_attn_max_num_splits_for_cuda_graph, action: resetting to default (32). "
            )
            att_config.flash_attn_max_num_splits_for_cuda_graph = 32

    # ==================== 9. Parallel Config ====================
    if vllm_config.parallel_config:
        # ray_workers_use_nsight requires NVIDIA Nsight which is not
        # available on Ascend NPU
        if getattr(vllm_config.parallel_config, "ray_workers_use_nsight", False):
            logger.warning(
                "Parameter requires NVIDIA-specific tools. "
                "parameter=ray_workers_use_nsight, action: resetting to False. "
            )
            vllm_config.parallel_config.ray_workers_use_nsight = False

        # --numa-bind relies on GPU-to-NUMA topology detection which is
        # not supported on Ascend NPU.  Seamlessly replace with the
        # Ascend-native CPU binding via additional_config.
        # --numa-bind-nodes and --numa-bind-cpus are also ignored because
        # the Ascend NPU implementation performs automatic topo-affinity
        # CPU binding internally.
        if getattr(vllm_config.parallel_config, "numa_bind", False):
            vllm_config.parallel_config.numa_bind = False
            if vllm_config.additional_config is None:
                vllm_config.additional_config = {}
            vllm_config.additional_config.setdefault("enable_cpu_binding", True)
            logger.info(
                "'--numa-bind' is not supported on Ascend NPU (GPU-to-"
                "NUMA topology detection unavailable). Automatically "
                "converted to --additional-config "
                "'{\"enable_cpu_binding\": true}' for Ascend-native "
                "CPU-core binding."
            )

        if getattr(vllm_config.parallel_config, "numa_bind_nodes", None):
            logger.info(
                "'--numa-bind-nodes' is ignored on Ascend NPU. The "
                "Ascend-native CPU binding automatically performs "
                "topo-affinity core allocation."
            )
            vllm_config.parallel_config.numa_bind_nodes = None

        if getattr(vllm_config.parallel_config, "numa_bind_cpus", None):
            logger.info(
                "'--numa-bind-cpus' is ignored on Ascend NPU. The "
                "Ascend-native CPU binding automatically performs "
                "topo-affinity core allocation."
            )
            vllm_config.parallel_config.numa_bind_cpus = None

        if getattr(vllm_config.parallel_config, "enable_dbo", False):
            logger.warning(
                "Parameter is currently ignored on Ascend. parameter=enable_dbo, action: resetting to False. "
            )
            vllm_config.parallel_config.enable_dbo = False

        ubatch_size = getattr(vllm_config.parallel_config, "ubatch_size", 0)
        if ubatch_size != 0:
            logger.warning(
                "Parameter is currently ignored on Ascend. parameter=ubatch_size, value=%d, action: resetting to 0. ",
                ubatch_size,
            )
            vllm_config.parallel_config.ubatch_size = 0

    # ==================== 10. Compilation Config ====================
    if vllm_config.compilation_config:
        if getattr(vllm_config.compilation_config, "use_inductor_graph_partition", False):
            logger.warning(
                "Parameter is not supported on Ascend NPU (use_inductor is False). "
                "parameter=use_inductor_graph_partition, action: resetting to False."
            )
            vllm_config.compilation_config.use_inductor_graph_partition = False

    # ==================== 11. VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS ====================
    if envs_vllm.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS < 1836:
        envs_vllm.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS = 3000
        logger.info(
            "The timeout interval of the HCCL operator is 1836s. Timeout in "
            "seconds for execute_model RPC calls in multiprocessing must be "
            "greater than 1836s, Set VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000"
        )


def _validate_eplb_config(vllm_config: VllmConfig) -> None:
    additional_config = vllm_config.additional_config or {}
    eplb_config = additional_config.get("eplb_config", {})
    if not isinstance(eplb_config, dict):
        raise TypeError("additional_config.eplb_config must be a dictionary.")

    use_v2_model_runner = bool(getattr(vllm_config, "use_v2_model_runner", False))
    if use_v2_model_runner:
        legacy_eplb_fields = sorted(set(eplb_config) - {"load_collection_phase"})
        if legacy_eplb_fields:
            raise ValueError(
                "Model Runner V2 only accepts 'load_collection_phase' in "
                "additional_config.eplb_config; legacy fields are not supported: "
                f"{', '.join(legacy_eplb_fields)}."
            )
        if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv(
            "EXPERT_MAP_RECORD", "false"
        ).lower() in ("true", "1"):
            raise ValueError(
                "DYNAMIC_EPLB and EXPERT_MAP_RECORD are Model Runner V1 controls. "
                "Unset them and use --enable-eplb with Model Runner V2."
            )
        load_collection_phase = eplb_config.get("load_collection_phase", "all")
        if load_collection_phase != "all" and not vllm_config.parallel_config.enable_eplb:
            raise ValueError("additional_config.eplb_config.load_collection_phase requires --enable-eplb.")
        if vllm_config.parallel_config.enable_eplb:
            upstream_eplb_config = vllm_config.parallel_config.eplb_config
            if upstream_eplb_config.communicator not in (None, "torch_gloo"):
                raise ValueError(
                    "Async EPLB on Ascend requires the torch_gloo communicator "
                    f"(CPU staging), but got {upstream_eplb_config.communicator!r}. "
                    "Set eplb_config.communicator to 'torch_gloo'."
                )
            if not upstream_eplb_config.use_async:
                logger.warning(
                    "Synchronous EPLB is not supported on Ascend; "
                    "parameter=eplb_config.use_async, value=False, "
                    "action: forcing asynchronous EPLB."
                )
                upstream_eplb_config.use_async = True
                upstream_eplb_config.communicator = "torch_gloo"
            if vllm_config.parallel_config.enable_elastic_ep:
                raise ValueError("Async EPLB is not supported with elastic EP on Ascend.")
    elif "load_collection_phase" in eplb_config:
        raise ValueError(
            "additional_config.eplb_config.load_collection_phase is only supported by "
            "Model Runner V2; use eplb_heat_collection_stage with Model Runner V1."
        )
    elif vllm_config.parallel_config.enable_eplb:
        raise ValueError("Upstream EPLB is only supported by Model Runner V2 on Ascend.")


def _check_ascend_config(vllm_config: VllmConfig, ascend_config) -> None:
    """Validate Ascend-specific options.

    Covers the scheduler extension policies (enable_balance_scheduling / short_request_first_config /
    dyntra_lb_config / recompute_scheduler_enable). Reads from the AscendConfig singleton
    initialized from vllm_config; env fallbacks are handled inside AscendConfig.
    """
    # Validate scheduler extension policies (read ascend_config.scheduler_config)
    from vllm_ascend.core.recompute_scheduler import RecomputeSchedulerConfig

    additional_config = vllm_config.additional_config
    if additional_config is None:
        vllm_config.additional_config = {}
        additional_config = vllm_config.additional_config
    scheduler_extension_config = ascend_config.scheduler_config

    # enable_balance_scheduling: only supported in PD-mixed mode
    if scheduler_extension_config.enable_balance_scheduling:
        kv_transfer_config = vllm_config.kv_transfer_config
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        if kv_transfer_config is not None and kv_role != "kv_both":
            raise ValueError(
                "enable_balance_scheduling only supports PD-mixed mode "
                "(kv_role='kv_both' or no kv_transfer_config), and is not supported in "
                "PD-disaggregated mode (kv_role='kv_producer'/'kv_consumer')."
            )

    _validate_kv_load_failure_policy(vllm_config)

    # short_request_first_config: requires fcfs policy and excludes
    # batch_job_sched_config / profiling_chunk_config / kv_consumer
    if scheduler_extension_config.short_request_first_config.enabled:
        kv_transfer_config = vllm_config.kv_transfer_config
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        if vllm_config.scheduler_config.policy != "fcfs":
            raise ValueError(
                "ShortRequestFirst scheduling requires scheduler_config.policy='fcfs', "
                f"but got {vllm_config.scheduler_config.policy!r}."
            )
        if scheduler_extension_config.batch_job_sched_config.enabled:
            raise ValueError(
                "ShortRequestFirst scheduling cannot be enabled with batch_job_sched_config. "
                "Please disable one of them."
            )
        if scheduler_extension_config.profiling_chunk_config.enabled:
            raise ValueError(
                "ShortRequestFirst scheduling cannot be enabled with profiling_chunk_config. "
                "Please disable one of them."
            )
        if kv_role == "kv_consumer":
            raise ValueError(
                "ShortRequestFirst scheduling is supported only on prefill or PD-mixed nodes, "
                "not PD-disaggregated D nodes (kv_role='kv_consumer')."
            )
        if vllm_config.scheduler_config.async_scheduling:
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_ascend.core.short_request_first_scheduler.ShortRequestFirstAsyncScheduler"
            )

    dyntra_lb_config = scheduler_extension_config.dyntra_lb_config
    if dyntra_lb_config.enabled:
        # DyntraLB targets decoder-only inference on the decode side of a
        # PD-disaggregated deployment. It balances attention KV-cache load
        # across multiple data-parallel ranks within one node.
        # It is not available on prefill or PD-mixed nodes, across multiple
        # DP nodes, or together with the other scheduling modes rejected
        # below.
        # Verified models: Qwen3-30B, Qwen3.5-35B
        if scheduler_extension_config.enable_balance_scheduling:
            raise ValueError("DyntraLB cannot be used with enable_balance_scheduling.")
        kv_transfer_config = vllm_config.kv_transfer_config
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        if kv_transfer_config is None or kv_role != "kv_consumer":
            raise ValueError(
                "DyntraLB is only supported on PD-disaggregated D nodes "
                f"(kv_role='kv_consumer', but got kv_role={kv_role!r})."
            )
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size <= 1:
            raise ValueError("DyntraLB requires data_parallel_size > 1.")
        if parallel_config.nnodes_within_dp > 1:
            raise ValueError(
                "DyntraLB only supports decoder instances that run on a "
                "single node; got "
                f"nnodes_within_dp={parallel_config.nnodes_within_dp}."
            )
        if scheduler_extension_config.profiling_chunk_config.enabled:
            raise ValueError("DyntraLB cannot be used with profiling_chunk_config.")

    # recompute_scheduler_enable: only supported on PD-disaggregated D nodes
    if scheduler_extension_config.recompute_scheduler_enable:
        kv_transfer_config = vllm_config.kv_transfer_config
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        if kv_role == "kv_producer":
            logger.warning(
                "recompute_scheduler_enable is ignored on PD-disaggregated P nodes "
                "(kv_role='kv_producer') and will be deprecated on P nodes in a future release. "
                "Please remove it from P-node configs and keep it only on PD-disaggregated D nodes "
                "(kv_role='kv_consumer')."
            )
            # Write the fix back into vllm_config.additional_config so worker
            # processes (which re-create AscendConfig) observe recompute disabled.
            additional_scheduler_config = additional_config.get("scheduler_config")
            if additional_scheduler_config is None:
                additional_scheduler_config = {}
                additional_config["scheduler_config"] = additional_scheduler_config
            additional_scheduler_config["recompute_scheduler_enable"] = False
        elif kv_transfer_config is None or kv_role != "kv_consumer":
            raise ValueError(
                "recompute_scheduler_enable can only be enabled on PD-disaggregated D nodes "
                f"(kv_role='kv_consumer', but got kv_role={kv_role!r}), and is not supported in PD-mixed mode."
            )
        else:
            async_scheduling = vllm_config.scheduler_config.async_scheduling
            recompute_scheduler_config = RecomputeSchedulerConfig.initialize_from_config(vllm_config)
            recompute_scheduler_config.scheduler_cls = _get_recompute_scheduler_cls(
                async_scheduling=async_scheduling,
                dyntra_lb_enabled=dyntra_lb_config.enabled,
            )
            vllm_config.scheduler_config = recompute_scheduler_config


def _validate_kv_load_failure_policy(vllm_config: VllmConfig) -> None:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return
    if getattr(kv_transfer_config, "kv_load_failure_policy", "fail") == "recompute":
        if getattr(vllm_config.model_config, "is_hybrid", False):
            raise AssertionError("Hybrid models do not support recompute mode kv load failure policy now.")


def _update_compilation_modes(vllm_config: VllmConfig, ascend_config) -> None:
    """Update compilation / cudagraph modes.

    Syncs the Ascend compilation config into additional_config, then derives
    CompilationMode / CUDAGraphMode from enforce_eager and the xlite graph config.
    """
    from vllm.config import CompilationMode
    from vllm.config.compilation import CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    model_config = vllm_config.model_config
    additional_config = vllm_config.additional_config
    if additional_config is None:
        vllm_config.additional_config = {}
        additional_config = vllm_config.additional_config

    # Sync ascend compilation config and derived cache settings
    ascend_compilation_config = ascend_config.ascend_compilation_config
    if ascend_compilation_config:
        additional_config.setdefault("ascend_compilation_config", {}).update(
            vars(ascend_compilation_config)
            if not isinstance(ascend_compilation_config, dict)
            else ascend_compilation_config
        )

    if model_config and hasattr(model_config.hf_text_config, "index_topk"):
        from vllm_ascend.attention.dsa_attn_kv_plan import resolve_dsv4_cache_dtype

        vllm_config.cache_config.cache_dtype = resolve_dsv4_cache_dtype(
            vllm_config.cache_config.cache_dtype,
            str(model_config.dtype).replace("torch.", ""),
        )

    # Update compilation mode in some cases
    enforce_eager = getattr(model_config, "enforce_eager", False)

    if enforce_eager:
        logger.info("Compilation disabled, using eager mode by default")
        compilation_config.mode = CompilationMode.NONE
        if compilation_config.splitting_ops is None:
            compilation_config.splitting_ops = []

    if compilation_config.mode not in [CompilationMode.NONE, CompilationMode.VLLM_COMPILE]:
        logger.warning(
            "NPU does not support compilation mode. mode=%s, action: setting CUDAGraphMode to NONE.",
            compilation_config.mode,
        )
        compilation_config.mode = CompilationMode.NONE

    # Update cudagraph_mode in some cases (read ascend_config.xlite_graph_config)
    xlite_graph_config = ascend_config.xlite_graph_config
    if xlite_graph_config.enabled:
        if xlite_graph_config.full_mode and vllm_config.speculative_config is None:
            logger.info("ACLGraph has been disabled when speculation is disabled in xlite full mode")
            enforce_eager = True
            model_config.enforce_eager = True
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        else:
            logger.info("Falling back to FULL_DECODE_ONLY under xlite decode-only mode")
            compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

    # Encoder-decoder models currently only support PIECEWISE mode
    # TODO(Jian Li): Confirm this behavior and explain why
    if (
        model_config
        and model_config.is_encoder_decoder
        and compilation_config.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
    ):
        cudagraph_mode = (
            CUDAGraphMode.PIECEWISE if compilation_config.mode == CompilationMode.VLLM_COMPILE else CUDAGraphMode.NONE
        )
        logger.info_once(
            "Encoder-decoder models don't support %s, fallback to %s.",
            compilation_config.cudagraph_mode,
            cudagraph_mode,
        )
        compilation_config.cudagraph_mode = cudagraph_mode


def _setup_compile_backend(
    vllm_config: VllmConfig,
    compile_backend: str,
    *,
    enable_shared_expert_dp: bool = False,
    enable_dsa_cp: bool = False,
) -> None:
    """Recompute cudagraph sizes and setup the compile backend.

    Recomputes cudagraph capture sizes (TP token-layout-aware), then configures
    the oot compiler and disables npugraph_ex / static kernel when the
    cudagraph mode does not support them. Writes back into additional_config
    for workers.
    """
    from vllm.config import CompilationMode
    from vllm.config.compilation import CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    additional_config = vllm_config.additional_config
    if additional_config is None:
        vllm_config.additional_config = {}
        additional_config = vllm_config.additional_config

    # Recompute cudagraph sizes before extending splitting_ops (honors the
    # current max / size inputs after the mode adjustments above).
    compilation_config.cudagraph_num_of_warmups = 1
    vllm_config._set_cudagraph_sizes()
    additional_config = vllm_config.additional_config or {}
    if (
        not additional_config.get("enable_flashcomm1", False)
        and int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0")) == 0
    ):
        vllm_config.parallel_config.all2all_backend = (
            "flashinfer_all2allv"  # TODO: a tricky way to disable SP moe. Disable this when SP is supported.
        )
        logger.info_once("FlashComm1 is disabled. Using flashinfer_all2allv as the all2all backend.")
    requires_tp_aligned_capture_sizes = enable_sp(vllm_config) or enable_shared_expert_dp or enable_dsa_cp
    if (
        vllm_config.parallel_config.tensor_parallel_size > 1
        and compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and not vllm_config.model_config.enforce_eager
        and requires_tp_aligned_capture_sizes
    ):
        original_sizes = compilation_config.cudagraph_capture_sizes
        sp_aclgraph_sizes = vllm_config.update_sizes_for_sequence_parallelism(original_sizes)
        if not sp_aclgraph_sizes:
            raise AssertionError(
                f"cudagraph_capture_sizes {original_sizes} does not contain"
                f"values that are multiples of tp_size "
                f"{vllm_config.parallel_config.tensor_parallel_size}"
            )

        if len(sp_aclgraph_sizes) != len(original_sizes):
            # Match max_cudagraph_capture_size with the valid max size to avoid
            # initialization error of vllm server.
            compilation_config.max_cudagraph_capture_size = sp_aclgraph_sizes[-1]
            compilation_config.cudagraph_capture_sizes = sp_aclgraph_sizes
            update_cudagraph_capture_sizes(vllm_config, sp_aclgraph_sizes)

    # Get custom compile backend for graph fusion
    compilation_config.oot_compiler = compile_backend
    compilation_config.use_inductor = False
    if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        compilation_config.mode = CompilationMode.NONE
        additional_config["ascend_compilation_config"]["enable_npugraph_ex"] = False
        additional_config["ascend_compilation_config"]["enable_static_kernel"] = False
    elif compilation_config.cudagraph_mode.requires_piecewise_compilation():
        # Our is_cuda_alike is False so we cannot reuse the assertion of upstream
        if compilation_config.mode != CompilationMode.VLLM_COMPILE and not envs_vllm.VLLM_USE_BREAKABLE_CUDAGRAPH:
            raise AssertionError(
                "Compilation mode should be CompilationMode.VLLM_COMPILE "
                "when cudagraph_mode piecewise cudagraphs is used, "
                "cudagraph_mode=%s",
                compilation_config.cudagraph_mode,
            )
        compilation_config.set_splitting_ops_for_v1(
            all2all_backend=vllm_config.parallel_config.all2all_backend,
            data_parallel_size=vllm_config.parallel_config.data_parallel_size,
        )
        # NOTE: Theoretically, we should also add this in the attention ops; the
        # class attribute may still hold the pre-modification value after spawn.
        compilation_config.splitting_ops.extend(
            [
                "vllm::mla_forward",
                "vllm::dsa_forward",
            ]
        )
        # TODO(2026/7/15): Delete the reduced gear after the new driver is released.
        if get_current_hardware_profile().supports(HardwareCapability.REDUCED_CUDAGRAPH_CAPTURE_SIZES):
            _prune_reduced_capture_sizes(vllm_config)
        additional_config["ascend_compilation_config"]["enable_npugraph_ex"] = False
        additional_config["ascend_compilation_config"]["enable_static_kernel"] = False
    elif compilation_config.cudagraph_mode.has_full_cudagraphs():
        # Don't split the FX graph for static kernel; it would compile multiple times.
        compilation_config.splitting_ops = []
    else:
        logger.info("%s cudagraph_mode is not support on NPU. falling back to NONE", compilation_config.cudagraph_mode)
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.mode = CompilationMode.NONE
        additional_config["ascend_compilation_config"]["enable_npugraph_ex"] = False
        additional_config["ascend_compilation_config"]["enable_static_kernel"] = False

    # TODO: Remove this check when ACL Graph supports ASCEND_LAUNCH_BLOCKING=1
    if compilation_config.cudagraph_mode != CUDAGraphMode.NONE and os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "1":
        raise ValueError(
            "ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. "
            "Please unset ASCEND_LAUNCH_BLOCKING or set it to 0. If you "
            "need ASCEND_LAUNCH_BLOCKING for debugging, consider other methods — "
            "for example, check the plog files (default: $HOME/ascend/log/debug) "
            "for more information about runtime errors."
        )


def _setup_worker_and_scheduler(
    vllm_config: VllmConfig,
    ascend_config,
) -> None:
    # Select worker class and refresh block size
    parallel_config = vllm_config.parallel_config
    if parallel_config and parallel_config.worker_cls == "auto":
        additional_config = vllm_config.additional_config or {}
        if (
            not additional_config.get("enable_flashcomm1", False)
            and int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0")) == 0
        ):
            parallel_config.all2all_backend = (
                "flashinfer_all2allv"  # TODO: a tricky way to disable SP moe. Disable this when SP is supported.
            )
            logger.info_once("FlashComm1 is disabled. Using flashinfer_all2allv as the all2all backend.")
        hardware_profile = get_current_hardware_profile()
        if ascend_config.xlite_graph_config.enabled and hardware_profile.supports(
            HardwareCapability.STANDARD_WORKER_PATCHES
        ):
            logger.info("openEuler Xlite enabled. See: https://atomgit.com/openeuler/GVirt/tree/master/xlite")
            parallel_config.worker_cls = "vllm_ascend.xlite.xlite_worker.XliteWorker"
        else:
            parallel_config.worker_cls = hardware_profile.default_worker_cls

    refresh_block_size(vllm_config)

    # Automatically activate all custom ops on profiles using the standard path.
    if get_current_hardware_profile().supports(HardwareCapability.AUTO_ENABLE_CUSTOM_OPS):
        vllm_config.compilation_config.custom_ops = ["all"]

    # Select specialized scheduler class
    scheduler_config = ascend_config.scheduler_config
    if scheduler_config.dyntra_lb_config.enabled and not scheduler_config.recompute_scheduler_enable:
        vllm_config.scheduler_config.scheduler_cls = _get_dyntra_lb_scheduler_cls(
            async_scheduling=vllm_config.scheduler_config.async_scheduling
        )

    # Use ProfilingChunkScheduler when profiling-based chunk sizing is on.
    if scheduler_config.profiling_chunk_config.enabled:
        vllm_config.scheduler_config.scheduler_cls = (
            "vllm_ascend.core.scheduler_profiling_chunk.ProfilingChunkScheduler"
        )
        import vllm_ascend.patch.platform.patch_profiling_chunk  # noqa

    # Extend original scheduler_config to use BatchJobAwareScheduler.
    if scheduler_config.batch_job_sched_config.enabled:
        if vllm_config.scheduler_config.async_scheduling:
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_ascend.core.batch_job_aware_scheduler.BatchJobAwareAsyncScheduler"
            )
        else:
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_ascend.core.batch_job_aware_scheduler.BatchJobAwareScheduler"
            )


def _validate_sfa_dcp_kv_sp(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config

    cp_size = parallel_config.prefill_context_parallel_size * parallel_config.decode_context_parallel_size
    use_sparse = model_uses_sfa_sparse(model_config)
    if (
        vllm_config.kv_transfer_config is not None
        and cache_config.block_size != parallel_config.cp_kv_cache_interleave_size
        and cp_size > 1
    ):
        raise AssertionError(
            f"cp_kv_cache_interleave_size({parallel_config.cp_kv_cache_interleave_size}) "
            f"and block_size({cache_config.block_size}) "
            "needs to be equal if PCP or DCP is enabled in P/D disaggregate and kv pool scenario."
        )

    if use_sparse and cp_size > 1 and parallel_config.cp_kv_cache_interleave_size != cache_config.block_size:
        logger.warning_once(
            "The current SFA context-parallel implementation requires "
            f"cp_kv_cache_interleave_size({parallel_config.cp_kv_cache_interleave_size})"
            f" == block_size({cache_config.block_size}). "
            f"Override cp_kv_cache_interleave_size to {cache_config.block_size}."
        )
        vllm_config.parallel_config.cp_kv_cache_interleave_size = cache_config.block_size

    if enable_sp(vllm_config):
        if vllm_config.parallel_config.tensor_parallel_size <= 1:
            raise AssertionError("Sequence parallelism is only supported when tp_size > 1.")

        if is_moe_model(vllm_config) and not vllm_config.parallel_config.enable_expert_parallel:
            raise AssertionError("Sequence parallelism requires enable_expert_parallel=True for MoE models.")


def _set_pytorch_npu_alloc_env(vllm_config: VllmConfig) -> None:
    # Set "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" by default to optimize NPU memory management.
    # Find more details at https://docs.vllm.ai/projects/ascend/en/latest/faqs.html#how-to-handle-the-out-of-memory-issue
    # NOTE: We should not set this environment variable in RL (sleep mode) scenarios.
    # Find more details about how to configure this environment variable at https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_012.html
    if vllm_config.model_config and not vllm_config.model_config.enable_sleep_mode:
        npu_alloc_configs = os.getenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
        # This environment variable may have more than one key-value pairs.
        # We should append ",expandable_segments:True" to the current configs.
        # For example: "page_size:1g" + ",expandable_segments:True".
        # NOTE: `max_split_size_mb` or `garbage_collection_threshold` cannot
        # be enabled together with `expandable_segments=True`.
        if (
            "expandable_segments" not in npu_alloc_configs
            and "max_split_size_mb" not in npu_alloc_configs
            and "garbage_collection_threshold" not in npu_alloc_configs
        ):
            npu_alloc_configs += ",expandable_segments:True"
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = npu_alloc_configs
        logger.info("Set PYTORCH_NPU_ALLOC_CONF=%s", npu_alloc_configs)


def _disable_expandable_segments() -> None:
    """Remove the allocator option that conflicts with sleep mode."""
    npu_alloc_configs = os.getenv("PYTORCH_NPU_ALLOC_CONF", "")
    if not npu_alloc_configs:
        return

    filtered_configs = [
        config.strip()
        for config in npu_alloc_configs.split(",")
        if config.strip() and not config.strip().startswith("expandable_segments:")
    ]
    updated_configs = ",".join(filtered_configs)
    if updated_configs != npu_alloc_configs:
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = updated_configs
        logger.info("Removed expandable_segments from PYTORCH_NPU_ALLOC_CONF: %s", updated_configs)


def _validate_fa3_backend(key, _attn_selector_config):
    rl_config = get_ascend_config().rl_config
    if not (rl_config.enabled and rl_config.enable_training_consistency):
        logger.info(
            "FA3 will not be enabled when rl_config.enable_training_consistency is false. "
            "Note that Ascend NPU will use its registered plugin backend instead."
        )
        return False
    if key != (False, False):
        raise ValueError("FA3 backend does not support MLA and SFA.")
    if util.find_spec("flash_attn_npu_v3") is None:
        raise ValueError(
            "flash_attn_npu_v3 is not installed but FA3 backend is requested. "
            "Please install flash_attn_npu_v3 to enable FA3."
        )
    mod = import_module("flash_attn_npu_v3")
    if not hasattr(mod, "flash_attn_with_kvcache"):
        raise ValueError(
            "flash_attn_npu_v3 is installed but does not provide "
            "flash_attn_with_kvcache. Please check flash_attn_npu_v3 "
            "whether it supports flash_attn_with_kvcache."
        )
    logger.info(
        "In training-inference consistency scenario, FA3 will be enabled, which may cause performance degradation."
    )
    return True


def _get_default_max_cudagraph_capture_size(vllm_config: VllmConfig) -> int | None:
    """Mirror the default-max branch in vLLM's `_set_cudagraph_sizes()`.

    This helper corresponds to the upstream block under
    "determine the initial max_cudagraph_capture_size" when
    `compilation_config.max_cudagraph_capture_size is None`.

    Ascend injects this default earlier via `apply_config_platform_defaults()`
    so the rest of `_set_cudagraph_sizes()` can keep using upstream logic for
    size-list generation, token-cap clipping, SP filtering, and later
    post-processing. The only intentional difference from upstream is removing
    the CUDA-oriented trailing `* 2`: Ascend wants the default capture upper
    bound to track `max_num_seqs * decode_query_len`, capped at 512.

    Returning `None` means the platform should not inject a default. This
    covers the cases where the user has already provided either
    `max_cudagraph_capture_size` or `cudagraph_capture_sizes`.
    """
    compilation_config = vllm_config.compilation_config
    if compilation_config.max_cudagraph_capture_size is not None:
        return None
    if compilation_config.cudagraph_capture_sizes is not None:
        return None

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
    if max_num_seqs is None:
        return None

    decode_query_len = 1
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config and speculative_config.num_speculative_tokens:
        decode_query_len += speculative_config.num_speculative_tokens

    return min(max_num_seqs * decode_query_len, 512)


def _config_deprecated_logging():
    """Configure deprecated logging format, when used deprecated codes
    in vllm-ascend.
    """
    import logging
    import warnings

    # Customize warning format to be one line
    def one_line_formatwarning(message, category, filename, lineno, line=None):
        return f"{filename}:{lineno}: {category.__name__}: {message}"

    warnings.formatwarning = one_line_formatwarning

    logging.captureWarnings(True)
    warnings.simplefilter("once", DeprecationWarning)

    vllm_logger = logging.getLogger("vllm")
    warnings_logger = logging.getLogger("py.warnings")

    # Propagate vllm logger handlers to warnings logger, to keep the same
    # format with vllm
    if vllm_logger.handlers:
        warnings_logger.handlers = []

        for handler in vllm_logger.handlers:
            warnings_logger.addHandler(handler)

    warnings_logger.propagate = False


def _prune_reduced_capture_sizes(vllm_config):
    original_sizes = vllm_config.compilation_config.cudagraph_capture_sizes
    if not original_sizes:
        return
    if len(original_sizes) <= MAX_REDUCED_CAPTURE_SIZES:
        return
    step = (len(original_sizes) - 1) / (MAX_REDUCED_CAPTURE_SIZES - 1)
    indices = [round(i * step) for i in range(MAX_REDUCED_CAPTURE_SIZES)]
    indices[0], indices[-1] = 0, len(original_sizes) - 1
    sampled_sizes = [original_sizes[i] for i in indices]
    update_cudagraph_capture_sizes(vllm_config, sampled_sizes)
    logger.warning(
        "Adjusted ACL graph batch sizes for model: %d → %d sizes due to HDK incompatibility"
        "and this warning will be cleared soon.",
        len(original_sizes),
        MAX_REDUCED_CAPTURE_SIZES,
    )


def _get_recompute_scheduler_cls(
    *,
    async_scheduling: bool,
    dyntra_lb_enabled: bool,
) -> str:
    if dyntra_lb_enabled:
        if async_scheduling:
            return "vllm_ascend.core.recompute_scheduler.AsyncDyntraLBRecomputeScheduler"
        return "vllm_ascend.core.recompute_scheduler.DyntraLBRecomputeScheduler"
    if async_scheduling:
        return "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler"
    return "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"


def _get_dyntra_lb_scheduler_cls(*, async_scheduling: bool) -> str:
    if async_scheduling:
        return "vllm_ascend.core.dyntra_lb_scheduler.AsyncDyntraLBScheduler"
    return "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBScheduler"


def _validate_parallel_config(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    if not vllm_config.use_v2_model_runner and parallel_config.prefill_context_parallel_size > 1:
        raise ValueError(
            "PCP (Prefill Context Parallelism) is not supported by vLLM Ascend. "
            "Please set --prefill-context-parallel-size to 1. "
            f"Got prefill_context_parallel_size={parallel_config.prefill_context_parallel_size}."
        )

    sfa_dcp_replicated_indexer = enable_sfa_dcp_replicated_indexer(vllm_config)
    if sfa_dcp_replicated_indexer:
        if parallel_config.decode_context_parallel_size != parallel_config.tensor_parallel_size:
            raise AssertionError(
                f"DCP for SFA is only supported when dcp_size({parallel_config.decode_context_parallel_size}) "
                f"== tp_size({parallel_config.tensor_parallel_size})."
            )
        if not get_current_hardware_profile().supports(HardwareCapability.SFA_DCP_REPLICATED_INDEXER):
            raise NotImplementedError(
                "SFA DCP with replicated indexer is not supported by the current hardware profile."
            )


def _validate_draft_decode_context_parallel_config(vllm_config: VllmConfig) -> None:
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return

    parallel_config = vllm_config.parallel_config
    decode_context_parallel_size = parallel_config.decode_context_parallel_size
    if decode_context_parallel_size <= 1:
        return

    if speculative_config.num_speculative_tokens_per_batch_size:
        raise ValueError(
            "Dynamic speculative decoding and decode context "
            "parallelism is not supported by vLLM Ascend. Please set "
            "--decode-context-parallel-size to 1 or remove "
            "num_speculative_tokens_per_batch_size from "
            "--speculative-config."
        )

    draft_model_config = speculative_config.draft_model_config
    if draft_model_config is None:
        return

    # MLA draft models do not use the GQA/MQA DCP head-sharding rule.
    if draft_model_config.use_mla:
        return

    draft_parallel_config = speculative_config.draft_parallel_config
    if draft_parallel_config is not None:
        draft_tensor_parallel_size = draft_parallel_config.tensor_parallel_size
    elif speculative_config.draft_tensor_parallel_size is not None:
        draft_tensor_parallel_size = speculative_config.draft_tensor_parallel_size
    else:
        draft_tensor_parallel_size = parallel_config.tensor_parallel_size

    total_num_attention_heads = draft_model_config.model_arch_config.total_num_attention_heads
    total_num_kv_heads = draft_model_config.get_total_num_kv_heads()

    if draft_tensor_parallel_size <= total_num_kv_heads:
        raise ValueError(
            "Invalid draft model parallel config for speculative decoding: "
            f"tensor parallel size {draft_tensor_parallel_size} must be "
            f"greater than total num kv heads {total_num_kv_heads} when "
            "enable decode context parallel for GQA/MQA draft model"
        )

    max_dcp_size = draft_tensor_parallel_size // total_num_kv_heads
    if decode_context_parallel_size > max_dcp_size:
        raise ValueError(
            "Invalid draft model parallel config for speculative decoding: "
            "decode context parallel size must less than or equal to "
            f"(draft tensor parallel size {draft_tensor_parallel_size} // "
            f"draft total num kv heads {total_num_kv_heads}) = "
            f"{max_dcp_size}, but got {decode_context_parallel_size}"
        )

    num_q_per_kv = total_num_attention_heads // total_num_kv_heads
    if num_q_per_kv % decode_context_parallel_size != 0:
        raise ValueError(
            "Invalid draft model parallel config for speculative decoding: "
            f"total number of q per kv attn heads ({num_q_per_kv}) must "
            "be divisible by dcp world size when enable decode context "
            f"parallel for GQA draft model "
            f"({decode_context_parallel_size})."
        )
