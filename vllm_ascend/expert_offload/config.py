# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field, StrictBool, StrictInt, model_validator

from vllm_ascend.config_utils import config
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@config(config=ConfigDict(frozen=True))
class ExpertOffloadConfig:
    """Exact single-worker offload. Unknown keys and ambiguous types fail fast."""

    enabled: StrictBool = False
    num_device_experts: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_capacity(self):
        if self.enabled and self.num_device_experts is None:
            raise ValueError("expert_offload_config.num_device_experts is required when enabled")
        return self

    def validate_layer(self, num_experts: int, top_k: int) -> None:
        if not self.enabled:
            return
        num_device_experts = self.num_device_experts
        if num_device_experts is None or not top_k <= num_device_experts < num_experts:
            raise ValueError(
                "expert offload requires top_k <= num_device_experts < num_experts; "
                f"got top_k={top_k}, slots={num_device_experts}, experts={num_experts}"
            )

    def validate(self, vllm_config: VllmConfig, ascend_config) -> None:
        if not self.enabled:
            return
        if get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED:
            raise ValueError("expert offload currently requires Ascend 950 (A5) hardware")
        if not vllm_config.use_v2_model_runner:
            raise ValueError("expert offload requires Model Runner V2")
        if not vllm_config.model_config.enforce_eager:
            from vllm.config.compilation import CompilationMode, CUDAGraphMode

            compilation_config = vllm_config.compilation_config
            if compilation_config.mode != CompilationMode.NONE or compilation_config.cudagraph_mode not in (
                CUDAGraphMode.NONE,
                CUDAGraphMode.FULL_DECODE_ONLY,
            ):
                raise ValueError(
                    "expert offload graph requires compilation mode=NONE and cudagraph_mode=FULL_DECODE_ONLY "
                    "(or NONE for eager); compiled/piecewise and mixed graphs are unsupported"
                )
        if vllm_config.model_config.enable_sleep_mode:
            raise ValueError(
                "expert offload has no sleep/wake protocol for resident slots and graph buffers; disable sleep mode"
            )
        parallel_config = vllm_config.parallel_config
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel_config, name) != 1:
                raise ValueError(f"expert offload uses worker-local expert ownership and requires {name}=1")
        conflicts = {
            "enable_expert_parallel": "expert slots do not implement distributed expert ownership",
            "enable_eplb": "external expert remapping is not coordinated with the residency map",
            "enable_dbo": "overlapping forwards cannot share the serialized expert pool",
        }
        for name, reason in conflicts.items():
            if getattr(parallel_config, name, False):
                raise ValueError(f"expert offload does not support {name}: {reason}; disable {name}")
        for name, reason in {
            "lora_config": "LoRA composition with prepared expert weights is not implemented",
            "kv_transfer_config": "split-worker loading and offload lifecycle coordination is not implemented",
        }.items():
            if getattr(vllm_config, name, None) is not None:
                raise ValueError(f"expert offload does not support {name}: {reason}; remove {name}")
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            # Speculative decoding is optional. This guard only limits the supported
            # composition; ordinary expert offload uses per-layer capability checks.
            draft_model_config = getattr(speculative_config, "draft_model_config", None)
            draft_hf_config = getattr(draft_model_config, "hf_config", None)
            speculative_method = getattr(speculative_config, "method", None)
            is_dspark = (
                speculative_method == "dspark"
                and getattr(draft_hf_config, "model_type", None) == "deepseek_v4"
                and getattr(draft_hf_config, "architectures", None) == ["DSparkDraftModel"]
            )
            # vLLM normalizes the documented deepseek_mtp alias to mtp and
            # selects GLM's own MTP head from the checkpoint configuration.
            is_glm_mtp = (
                speculative_method in ("mtp", "deepseek_mtp")
                and getattr(vllm_config.model_config.hf_text_config, "model_type", None) == "glm5_next_text"
                and getattr(draft_hf_config, "model_type", None) == "glm5_next_mtp"
                and getattr(draft_hf_config, "architectures", None) == ["Glm5NextMTPModel"]
            )
            if (
                draft_model_config is None
                or draft_model_config.model != vllm_config.model_config.model
                or not speculative_config.enforce_eager
                or speculative_config.enable_adaptive_verification
                or not (is_dspark or is_glm_mtp)
            ):
                raise ValueError(
                    "expert offload speculative_config supports only checkpoint-native DeepSeek V4 DSpark "
                    "or GLM-5.3-Flash MTP, with eager drafting and fixed verification"
                )
        weight_offload_config = vllm_config.offload_config
        if weight_offload_config.uva.cpu_offload_gb or weight_offload_config.prefetch.offload_group_size:
            raise ValueError("expert offload cannot be combined with generic CPU weight offload")
        if str(vllm_config.load_config.load_format) not in ("auto", "safetensors", "pt"):
            raise ValueError(
                "expert offload requires per-expert checkpoint loading and completeness tracking; "
                "use auto, safetensors, or pt checkpoint loading"
            )
        for name in ("mix_placement", "enable_force_eplb", "enable_fused_mc2", "multistream_overlap_shared_expert"):
            if getattr(ascend_config, name):
                raise ValueError(f"expert offload has no slot/pool integration for {name}; disable {name}")
        if ascend_config.eplb_config.dynamic_eplb:
            raise ValueError(
                "expert offload does not coordinate dynamic EPLB with resident slots; disable dynamic EPLB"
            )
        if ascend_config.rl_config.enabled:
            raise ValueError("expert offload cannot rebuild prepared host bytes and graphs during RL weight reloading")
