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
from typing import TYPE_CHECKING, Any
from uuid import uuid4
from importlib import import_module, util

import torch
import vllm.envs as envs_vllm
from vllm.logger import logger
from vllm.platforms import Platform, PlatformEnum

# todo: please remove it when solve cuda hard code in vllm
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import init_ascend_config

# isort: off
from vllm_ascend.utils import (
    ASCEND_QUANTIZATION_METHOD,
    COMPILATION_PASS_KEY,
    COMPRESSED_TENSORS_METHOD,
    AscendDeviceType,
    bootstrap_custom_op_env,
    check_kv_extra_config,
    flashcomm2_enable,
    get_ascend_device_type,
    is_moe_model,
    refresh_block_size,
    update_aclgraph_sizes,
    update_cudagraph_capture_sizes,
    is_310p,
    enable_sp,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    ModelConfig = None
    VllmConfig = None
    FlexibleArgumentParser = None

_CUSTOM_OP_REGISTERED = False


def config_deprecated_logging():
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


class NPUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    simple_compile_backend: str = "eager"  # Disable torch.compile()
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    dispatch_key: str = "PrivateUse1"

    supported_quantization: list[str] = [ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD]

    def is_sleep_mode_available(self) -> bool:
        return True

    @property
    def pass_key(self) -> str:
        """
        Inductor config key for the PassManager custom pass, for example 'post_grad_custom_post_pass'.
        It is a parameter of inductor_config used to register custom passes.
        Currently, we only use Inductor's 'pattern matcher' functionality, so we define our own pass_key.
        """
        return COMPILATION_PASS_KEY

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
    def pre_register_and_update(cls, parser: FlexibleArgumentParser | None = None) -> None:
        # Adapt the global patch here.
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

        if not is_310p():
            from vllm_ascend.quantization import AscendCompressedTensorsConfig, AscendModelSlimConfig  # noqa: F401
        else:
            from vllm_ascend._310p.quantization import AscendModelSlimConfig310  # noqa: F401

        config_deprecated_logging()

    @classmethod
    def _get_default_max_cudagraph_capture_size(cls, vllm_config: VllmConfig) -> int | None:
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

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: VllmConfig) -> None:
        """Apply Ascend-specific defaults. Set sp_min_token_num=1 when enable_sp and not set."""
        pass_config = vllm_config.compilation_config.pass_config
        if pass_config.enable_sp and pass_config.sp_min_token_num is None:
            from vllm_ascend.compilation.passes.sequence_parallelism import get_sp_min_token_num

            pass_config.sp_min_token_num = get_sp_min_token_num(vllm_config)
            logger.info("set sp_min_token_num to %s", pass_config.sp_min_token_num)

        default_max_cg_capture_size = cls._get_default_max_cudagraph_capture_size(vllm_config)
        if default_max_cg_capture_size is not None:
            vllm_config.compilation_config.max_cudagraph_capture_size = default_max_cg_capture_size

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        device_props = torch.npu.get_device_properties(device_id)
        if not hasattr(device_props, "uuid") or device_props.uuid is None:
            raise RuntimeError(f"Device {device_id} does not have a valid UUID.")
        return device_props.uuid

    @classmethod
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
    def inference_mode(cls):
        return torch.inference_mode()

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None:
        # TODO: NPU still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        pass

    @classmethod
    def set_device(cls, device: torch.device):
        torch.npu.set_device(device)

    @classmethod
    def _validate_layer_sharding_config(cls, vllm_config: VllmConfig) -> None:
        additional_config = vllm_config.additional_config or {}
        layer_sharding = additional_config.get("layer_sharding") or []
        if not layer_sharding:
            return

        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is None or kv_transfer_config.kv_role != "kv_producer":
            raise ValueError("additional_config.layer_sharding can only be enabled in PD-disaggregated's P node.")

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm_ascend.quantization.utils import maybe_auto_detect_quantization

        if vllm_config.model_config is not None:
            maybe_auto_detect_quantization(vllm_config)

        cls._validate_layer_sharding_config(vllm_config)

        # initialize ascend config from vllm additional_config
        cls._fix_incompatible_config(vllm_config)

        ascend_config = init_ascend_config(vllm_config)

        if vllm_config.kv_transfer_config is not None:
            check_kv_extra_config(vllm_config)
            if not getattr(vllm_config.kv_transfer_config, "_engine_id_patched", False):
                vllm_config.kv_transfer_config.engine_id = f"{vllm_config.kv_transfer_config.engine_id}-{uuid4().hex}"
                vllm_config.kv_transfer_config._engine_id_patched = True
        from vllm.config import CompilationMode  # noqa: E402

        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        ascend_compilation_config = ascend_config.ascend_compilation_config
        if ascend_compilation_config:
            vllm_config.additional_config.setdefault("ascend_compilation_config", {}).update(
                vars(ascend_compilation_config)
                if not isinstance(ascend_compilation_config, dict)
                else ascend_compilation_config
            )

        ascend_config.update_compile_ranges_split_points()

        if model_config and hasattr(model_config.hf_text_config, "index_topk"):
            vllm_config.cache_config.cache_dtype = str(model_config.dtype).replace("torch.", "")

        ascend_fusion_config = ascend_config.ascend_fusion_config
        if ascend_fusion_config:
            vllm_config.additional_config.setdefault("ascend_fusion_config", {}).update(
                vars(ascend_fusion_config) if not isinstance(ascend_fusion_config, dict) else ascend_fusion_config
            )

        if model_config is None:
            logger.warning("Model config is missing. This may indicate that we are running a test case")
            enforce_eager = False
        else:
            enforce_eager = getattr(model_config, "enforce_eager", False)

        from vllm.config.compilation import CUDAGraphMode

        if ascend_config.xlite_graph_config.enabled:
            if ascend_config.xlite_graph_config.full_mode:
                logger.info("ACLGraph is disabled under xlite full mode")
                enforce_eager = True
                model_config.enforce_eager = True
                compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            else:
                logger.info("Falling back to FULL_DECODE_ONLY under xlite decode-only mode")
                compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

        if enforce_eager:
            logger.info("Compilation disabled, using eager mode by default")
            compilation_config.mode = CompilationMode.NONE
            if compilation_config.splitting_ops is None:
                compilation_config.splitting_ops = []

        compilation_config.cudagraph_num_of_warmups = 1

        if compilation_config.mode not in [CompilationMode.NONE, CompilationMode.VLLM_COMPILE]:
            logger.warning(
                "NPU does not support %s compilation mode. Setting CUDAGraphMode to NONE", compilation_config.mode
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # Recompute cudagraph sizes after Ascend-specific compatibility updates.
        # The platform default max is injected earlier via
        # `apply_config_platform_defaults`, so this late pass should only honor
        # the current max / size inputs after the mode adjustments above.
        vllm_config._set_cudagraph_sizes()
        # TODO delete graph size update here when compilation_config.pass_config.enable_sp
        # is supported by vllm-ascend.
        if (
            vllm_config.parallel_config.tensor_parallel_size > 1
            and not vllm_config.model_config.enforce_eager
            and enable_sp(vllm_config)
        ):
            original_sizes = compilation_config.cudagraph_capture_sizes
            sp_aclgraph_sizes = vllm_config.update_sizes_for_sequence_parallelism(original_sizes)
            assert sp_aclgraph_sizes, (
                f"cudagraph_capture_sizes {original_sizes} does not contain"
                f"values that are multiples of tp_size "
                f"{vllm_config.parallel_config.tensor_parallel_size}"
            )
            if len(sp_aclgraph_sizes) != len(original_sizes):
                # If user set the max_num_seqs miss fit the multiple of tp_size,
                # we need to match the max_cudagraph_capture_size with the valid max size,
                # so we can avoid initialization error of vllm server.
                compilation_config.max_cudagraph_capture_size = sp_aclgraph_sizes[-1]
                compilation_config.cudagraph_capture_sizes = sp_aclgraph_sizes
                update_cudagraph_capture_sizes(vllm_config, sp_aclgraph_sizes)

        # TODO: Full graph is fully supported later, and the default value will be set to full graph.
        if compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
            compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

        # encoder-decoder models currently only support piecewise mode
        if model_config and model_config.is_encoder_decoder is True:
            if compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
                logger.warning("encoder-decoder model doesn't support FULL_DECODE_ONLY, fallback to PIECEWISE ")
            compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

        # get custom compile backend for graph fusion
        compilation_config.oot_compiler = cls.get_compile_backend()

        if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            compilation_config.mode = CompilationMode.NONE
            ascend_config.ascend_compilation_config.enable_npugraph_ex = False
        elif compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE:
            logger.info("PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode")
            assert compilation_config.mode == CompilationMode.VLLM_COMPILE, (
                "When enabling VLLM_COMPILE aclgraph, please make sure compilation_config.mode == "
                "CompilationMode.VLLM_COMPILE and compilation_config.cudagraph_mode == CUDAGraphMode.VLLM_COMPILE"
            )
            compilation_config.set_splitting_ops_for_v1(
                all2all_backend=vllm_config.parallel_config.all2all_backend,
                data_parallel_size=vllm_config.parallel_config.data_parallel_size,
            )
            compilation_config.use_inductor = False
            # NOTE: Theoretically, we should also add vllm::mla_forward in the attention ops.
            # Since the process is created in the spawn mode, the value of the class attribute
            # attention ops transmitted is still the one before modification, so it has not been modified.
            # This will cause in scenarios where both piecewise and splitting ops are configured simultaneously,
            # If splitting ops does not contain the vllm::mla forward value, this configuration issue will
            # not be detected in advance assert.
            compilation_config.splitting_ops.extend(["vllm::mla_forward"])
            update_aclgraph_sizes(vllm_config)
            ascend_config.ascend_compilation_config.enable_npugraph_ex = False
        elif (
            compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
            or compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        ):
            logger.info(
                "FULL_DECODE_ONLY compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode"
            )
            compilation_config.use_inductor = False
            compilation_config.splitting_ops = []
            warning_message = """\033[91m
            **********************************************************************************
            * WARNING: You have enabled the *full graph* feature.
            * This is an early experimental stage and may involve various unknown issues.
            * A known problem is that capturing too many batch sizes can lead to OOM
            * (Out of Memory) errors or inference hangs. If you encounter such issues,
            * consider reducing `gpu_memory_utilization` or manually specifying a smaller
            * batch size for graph capture.
            * For more details, please refer to:
            * https://docs.vllm.ai/en/stable/configuration/conserving_memory.html#reduce-cuda-graphs
            **********************************************************************************\033[0m
            """
            logger.warning(warning_message)
        else:
            logger.info(
                "%s cudagraph_mode is not support on NPU. falling back to NONE", compilation_config.cudagraph_mode
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            compilation_config.mode = CompilationMode.NONE
            ascend_config.ascend_compilation_config.enable_npugraph_ex = False

        # TODO: Remove this check when ACL Graph supports ASCEND_LAUNCH_BLOCKING=1
        # Then, we will have to discuss the error handling strategy and user experience
        if (
            compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "1"
        ):
            raise ValueError(
                "ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. "
                "Please unset ASCEND_LAUNCH_BLOCKING or set it to 0. If you "
                "need ASCEND_LAUNCH_BLOCKING for debugging, consider other methods — "
                "for example, check the plog files (default: $HOME/ascend/log/debug) "
                "for more information about runtime errors."
            )

        if parallel_config and parallel_config.worker_cls == "auto":
            # TODO: this is a tricky way to disable `use_sequence_parallel_moe` in vllm.
            if not vllm_config.compilation_config.pass_config.enable_sp:
                parallel_config.all2all_backend = "flashinfer_all2allv"
            if is_310p():
                parallel_config.worker_cls = "vllm_ascend._310p.worker_310p.NPUWorker310"
            elif ascend_config.xlite_graph_config.enabled:
                logger.info("openEuler Xlite enabled. See: https://atomgit.com/openeuler/GVirt/tree/master/xlite")
                parallel_config.worker_cls = "vllm_ascend.xlite.xlite_worker.XliteWorker"
            else:
                parallel_config.worker_cls = "vllm_ascend.worker.worker.NPUWorker"

        refresh_block_size(vllm_config)

        # Activate custom ops for v1, except on 310P
        if get_ascend_device_type() != AscendDeviceType._310P:
            compilation_config.custom_ops = ["all"]

        if envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING:
            kv_transfer_config = vllm_config.kv_transfer_config
            kv_role = getattr(kv_transfer_config, "kv_role", None)
            if kv_transfer_config is not None and kv_role != "kv_both":
                raise ValueError(
                    "VLLM_ASCEND_BALANCE_SCHEDULING (balance scheduling) only supports PD-mixed mode "
                    "(kv_role='kv_both' or no kv_transfer_config), and is not supported in "
                    "PD-disaggregated mode (kv_role='kv_producer'/'kv_consumer')."
                )

        if ascend_config.recompute_scheduler_enable:
            kv_transfer_config = vllm_config.kv_transfer_config
            kv_role = getattr(kv_transfer_config, "kv_role", None)
            if kv_transfer_config is None or kv_role == "kv_both":
                raise ValueError(
                    "recompute_scheduler_enable can only be enabled in PD-disaggregated mode "
                    "(kv_role='kv_producer' or 'kv_consumer'), and is not supported in PD-mixed mode."
                )

            from vllm_ascend.core.recompute_scheduler import RecomputeSchedulerConfig

            recompute_scheduler_config = RecomputeSchedulerConfig.initialize_from_config(vllm_config)
            vllm_config.scheduler_config = recompute_scheduler_config

        # Extend original scheduler_config to use SchedulerDynamicBatch.
        if ascend_config.SLO_limits_for_dynamic_batch != -1:
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_ascend.core.scheduler_dynamic_batch.SchedulerDynamicBatch"
            )
            vllm_config.scheduler_config.enable_chunked_prefill = True
            vllm_config.scheduler_config.SLO_limits_for_dynamic_batch = ascend_config.SLO_limits_for_dynamic_batch

        # Use ProfilingChunkScheduler when profiling-based chunk sizing is on.
        if ascend_config.profiling_chunk_config.enabled:
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_ascend.core.scheduler_profiling_chunk.ProfilingChunkScheduler"
            )
            import vllm_ascend.patch.platform.patch_profiling_chunk  # noqa

        cp_size = parallel_config.decode_context_parallel_size * parallel_config.prefill_context_parallel_size
        if (
            vllm_config.kv_transfer_config is not None
            and cache_config.block_size != parallel_config.cp_kv_cache_interleave_size
            and cp_size > 1
        ):
            raise AssertionError(
                f"cp_kv_cache_interleave_size({parallel_config.cp_kv_cache_interleave_size}) "
                f"and block_size({cache_config.block_size}) "
                "needs to be equal if use pcp or dcp > 1 in P/D disaggregate and kv pool scenario."
            )

        use_sparse = (
            model_config is not None
            and model_config.hf_text_config is not None
            and hasattr(model_config.hf_text_config, "index_topk")
        )
        if use_sparse and cp_size > 1 and parallel_config.cp_kv_cache_interleave_size != cache_config.block_size:
            logger.warning_once(
                "The current SFA's PCP&DCP implementation requires"
                f"cp_kv_cache_interleave_size({parallel_config.cp_kv_cache_interleave_size})"
                f" == block_size({cache_config.block_size}). "
                f"Override cp_kv_cache_interleave_size to {cache_config.block_size}."
            )
            vllm_config.parallel_config.cp_kv_cache_interleave_size = cache_config.block_size

        if enable_sp(vllm_config):
            assert vllm_config.parallel_config.tensor_parallel_size > 1, (
                "Flash Comm v1 is only supported when tp_size > 1."
            )

            assert not is_moe_model(vllm_config) or vllm_config.parallel_config.enable_expert_parallel, (
                "Flash Comm v1 requires enable_expert_parallel=True for MoE models."
            )

        # Set "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" by default to optimize NPU memory management.
        # Find more details at https://docs.vllm.ai/projects/ascend/en/latest/faqs.html#how-to-handle-the-out-of-memory-issue
        # NOTE: We should not set this environment variable in RL (sleep mode) scenarios.
        # Find more details about how to configure this environment variable at https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_012.html
        if model_config and not model_config.enable_sleep_mode:
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

        if ascend_config.enable_mc2_hierarchy_comm and envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2:
            raise ValueError(
                "fused mc2 op cannot be used with hierarchy communication."
                "Please disable VLLM_ASCEND_ENABLE_FUSED_MC2 by setting it to 0."
            )

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
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads: int | None = None):
        key = (attn_selector_config.use_mla, attn_selector_config.use_sparse)

        backend_map = {
            (True, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",
            (False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
            (True, True): "vllm_ascend.attention.sfa_v1.AscendSFABackend",
        }
        backend_map_310 = {
            (
                False,
                False,
            ): "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310",
            # TODO If MLA/SFA is supported in the future, consider implementing the logic described in these comments.
            # (True, False): "...AscendMLABackend310",
            # (True, True):  "...AscendSFABackend310",
        }

        if is_310p():
            return backend_map_310.get(key, backend_map_310[(False, False)])

        backend_cls_path = backend_map[key]

        if selected_backend == AttentionBackendEnum.FLASH_ATTN:
            backend_cls_path = cls._resolve_fa3_backend(key, attn_selector_config)

        return backend_cls_path

    @classmethod
    def _resolve_fa3_backend(cls, key, attn_selector_config):
        if not attn_selector_config.use_batch_invariant:
            raise ValueError(
                "FA3 is not enabled on Ascend without batch invariant mode. "
                "For training-inference consistency, please set "
                "VLLM_BATCH_INVARIANT=1."
            )
        if key != (False, False):
            raise ValueError("FA3 backend does not support MLA and SFA.")
        try:
            if util.find_spec("flash_attn_v3") is None:
                raise ValueError(
                    "flash_attn_v3 is not installed but FA3 backend is requested. "
                    "Please install flash_attn_v3 to enable FA3."
                )
            mod = import_module("flash_attn_v3")
            if not hasattr(mod, "flash_attn_with_kvcache"):
                raise ValueError(
                    "flash_attn_v3 is installed but does not provide "
                    "flash_attn_with_kvcache. Please check flash_attn_v3 "
                    "whether it supports flash_attn_with_kvcache."
                )
        except ValueError:
            raise
        except ImportError:
            raise ValueError(
                "flash_attn_v3 is not installed but FA3 backend is requested. "
                "Please install flash_attn_v3 to enable FA3."
            )
        return "vllm_ascend.attention.fa3_v1.AscendFABackend"

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
    def is_pin_memory_available(cls):
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        """
        Get piecewise backend class for piecewise graph.
        """
        return "vllm_ascend.compilation.acl_graph.ACLGraphWrapper"  # noqa

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

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
        if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            return {}

        # is_draft_model will be removed later, so we set it to False temporarily.
        is_draft_model = False
        # v2 has 2 graphs in eager, one for prefill, the other for decodes, this flag is aimed to distinguish them.
        is_draft_model_prefill = False

        in_profile_run = get_mrv2_in_profile_run()
        moe_comm_type = select_moe_comm_method(
            num_tokens,
            vllm_config,
            is_draft_model=is_draft_model,
        )
        moe_comm_method = get_moe_comm_method(moe_comm_type)

        tp_world_size = get_tensor_model_parallel_world_size()

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing.
        capturing = False

        # set for sequence parallelism, 1000 is the batch size concurrency
        # threshold for enabling the flashcomm_v1 or sequence_parallelism feature.
        # Currently, it is an empirical value. In normal scenarios,
        # if the concurrency exceeds this threshold,
        # the performance benefits can be maximized. Conversely,
        # if the concurrency is below the threshold,
        # the performance may degrade due to the switching of
        # communication methods.
        mmrs_fusion = True
        if is_moe_model(vllm_config):
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None
            mmrs_fusion = False
        else:
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000

        # TODO(Levi-JQ): another PR to normalize the enabling logic for sp/fc2
        flashcomm_v2_enabled = flashcomm2_enable() and tp_world_size > 1 and num_tokens is not None
        pad_size = 0
        padded_length = None
        if flash_comm_v1_enabled or flashcomm_v2_enabled:
            pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size

        if num_tokens is None and attn_metadata is not None:
            num_tokens = list(attn_metadata.values())[0].num_actual_tokens
        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and dp_metadata is not None:
            max_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu.max().item()
            if flash_comm_v1_enabled or flashcomm_v2_enabled:
                padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
                pad_size = padded_length - num_tokens
        else:
            max_tokens_across_dp = num_tokens
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
            "flash_comm_v1_enabled": flash_comm_v1_enabled,
            "flashcomm_v2_enabled": flashcomm_v2_enabled,
            "pad_size": pad_size,
            "padded_length": padded_length,
            "max_tokens_across_dp": max_tokens_across_dp,
            "mc2_mask": mc2_mask,
            "is_draft_model": is_draft_model,
            "is_draft_model_prefill": is_draft_model_prefill,
            "in_profile_run": in_profile_run,
            "padded_num_tokens": padded_num_tokens,
        }

    @staticmethod
    def _fix_incompatible_config(vllm_config: VllmConfig) -> None:
        """
        Check and correct parameters in VllmConfig that are incompatible with Ascend NPU.
        If GPU-specific or currently unsupported parameters are set by the user,
        log a warning and reset them to safe values.
        """
        model_config = vllm_config.model_config
        # ==================== 1. Model Config ====================
        if model_config:
            # Disable Cascade Attention (GPU feature)
            if getattr(model_config, "disable_cascade_attn", False):
                logger.warning(
                    "Parameter '--disable-cascade-attn' is a GPU-specific feature. Resetting to False for Ascend."
                )
                model_config.disable_cascade_attn = False

        # ==================== 2. Cache Config ====================
        if vllm_config.cache_config:
            # Check and reset cpu_kvcache_space_bytes
            if getattr(vllm_config.cache_config, "cpu_kvcache_space_bytes", False):
                logger.warning(
                    "Parameter 'cpu_kvcache_space_bytes' is tied to cpu backend. Resetting to None for Ascend."
                )
                vllm_config.cache_config.cpu_kvcache_space_bytes = None

        # ==================== 3. MultiModal Config ====================
        multimodal_config = getattr(model_config, "multimodal_config", None) if model_config else None
        if multimodal_config:
            # Ascend uses a different mechanism for Multi-Modal attention
            if getattr(multimodal_config, "mm_encoder_attn_backend", None) is not None:
                logger.warning(
                    "Parameter '--mm-encoder-attn-backend' is set but Ascend uses "
                    "a plugin mechanism for multi-modal attention. Resetting to None."
                )
                multimodal_config.mm_encoder_attn_backend = None

        # ==================== 4. Observability Config ====================
        if vllm_config.observability_config:
            # NVTX tracing is NVIDIA specific
            if getattr(vllm_config.observability_config, "enable_layerwise_nvtx_tracing", False):
                logger.warning(
                    "Parameter '--enable-layerwise-nvtx-tracing' relies on NVTX "
                    "(NVIDIA Tools) and is not supported on Ascend. Resetting to False."
                )
                vllm_config.observability_config.enable_layerwise_nvtx_tracing = False

        # ==================== 5. Scheduler Config ====================
        if vllm_config.scheduler_config:
            # Partial prefills are specific to ROCm optimization
            if getattr(vllm_config.scheduler_config, "max_num_partial_prefills", 1) != 1:
                logger.warning(
                    "Parameter '--max-num-partial-prefills' is optimized for ROCm. Resetting to default (1) for Ascend."
                )
                vllm_config.scheduler_config.max_num_partial_prefills = 1

        # ==================== 6. Speculative Config ====================
        if vllm_config.speculative_config:
            # Ascend automatically inherits main model quantization
            if getattr(vllm_config.speculative_config, "quantization", None) is not None:
                logger.warning(
                    "Speculative quantization is set but Ascend automatically uses "
                    "the main model's quantization method. Resetting to None."
                )
                vllm_config.speculative_config.quantization = None

        # ==================== 7. KV Transfer Config ====================
        if vllm_config.kv_transfer_config:
            # Buffer size is primarily tied to NCCL (GPU) backends
            current_buffer_size = getattr(vllm_config.kv_transfer_config, "kv_buffer_size", 1e9)
            if current_buffer_size != 1e9:
                logger.warning(
                    "Parameter 'kv_buffer_size' is optimized for NCCL and may be "
                    "incompatible with current Ascend KV transfer status. Resetting to default (1e9)."
                )
                # Use setattr to safely assign the value
                vllm_config.kv_transfer_config.kv_buffer_size = 1e9

            # Check and reset enable_permute_local_kv
            if getattr(vllm_config.kv_transfer_config, "enable_permute_local_kv", False):
                logger.warning(
                    "Parameter 'enable_permute_local_kv' is tied to NIXL backend. "
                    "Resetting to False for Ascend stability."
                )
                vllm_config.kv_transfer_config.enable_permute_local_kv = False

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
                        "Ignored parameter '%s'. This is a GPU-specific feature "
                        "not supported on Ascend. Resetting to False.",
                        flag,
                    )
                    setattr(att_config, flag, False)

            # Reset specific values to None as Ascend uses its own internal logic
            if getattr(att_config, "flash_attn_version", None) is not None:
                logger.warning(
                    "Ignored parameter 'flash_attn_version'. Ascend uses its own attention backend. Resetting to None."
                )
                att_config.flash_attn_version = None

            # Notify user that the backend will be managed by Ascend plugins
            if (
                getattr(att_config, "backend", None) is not None
                and att_config.backend != AttentionBackendEnum.FLASH_ATTN
            ):
                logger.info(
                    "User specified attention backend '%s'. Note that Ascend NPU "
                    "will use its registered plugin backend instead. Resetting to None.",
                    att_config.backend,
                )
                att_config.backend = None

            # CUDA Graph specific split points are not applicable
            if getattr(att_config, "flash_attn_max_num_splits_for_cuda_graph", 32) != 32:
                logger.warning(
                    "Parameter 'flash_attn_max_num_splits_for_cuda_graph' is "
                    "ignored on Ascend. Resetting to default (32)."
                )
                att_config.flash_attn_max_num_splits_for_cuda_graph = 32

        # ==================== 9. Parallel Config ====================
        if vllm_config.parallel_config:
            # ray_workers_use_nsight requires NVIDIA Nsight which is not
            # available on Ascend NPU
            if getattr(vllm_config.parallel_config, "ray_workers_use_nsight", False):
                logger.warning(
                    "'--ray-workers-use-nsight' requires NVIDIA Nsight which is "
                    "not available on Ascend NPU. Resetting to False."
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
                    "'--enable-dbo' is currently ignored on Ascend NPU because the "
                    "upstream generic DBO path has not yet aligned with the Ascend "
                    "backend/runtime model. Ascend may use separate backend/model-"
                    "specific overlap optimizations, and this may converge as the "
                    "generic overlap framework evolves. Resetting to False."
                )
                vllm_config.parallel_config.enable_dbo = False

            ubatch_size = getattr(vllm_config.parallel_config, "ubatch_size", 0)
            if ubatch_size != 0:
                logger.warning(
                    "'--ubatch-size' is currently ignored on Ascend NPU because it "
                    "depends on the generic DBO path, which is not yet aligned with "
                    "the current Ascend backend/runtime model. Resetting to 0."
                )
                vllm_config.parallel_config.ubatch_size = 0

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return True

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        pass
