# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field, StrictBool, StrictInt, model_validator

from vllm_ascend.config_utils import config

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
        slots = self.num_device_experts
        if slots is None or not 1 <= top_k <= slots < num_experts:
            raise ValueError(
                "expert offload requires 1 <= top_k <= num_device_experts < num_experts; "
                f"got top_k={top_k}, slots={slots}, experts={num_experts}"
            )

    def validate(self, vllm_config: VllmConfig, ascend_config) -> None:
        if not self.enabled:
            return
        if not vllm_config.use_v2_model_runner:
            raise ValueError("expert offload requires Model Runner V2")
        if not vllm_config.model_config.enforce_eager:
            from vllm.config.compilation import CompilationMode, CUDAGraphMode

            compilation = vllm_config.compilation_config
            if compilation.mode != CompilationMode.NONE or compilation.cudagraph_mode not in (
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
        parallel = vllm_config.parallel_config
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel, name) != 1:
                raise ValueError(f"expert offload uses worker-local expert ownership and requires {name}=1")
        conflicts = {
            "enable_expert_parallel": "expert slots do not implement distributed expert ownership",
            "enable_eplb": "external expert remapping is not coordinated with the residency map",
            "enable_dbo": "overlapping forwards cannot share the serialized expert pool",
        }
        for name, reason in conflicts.items():
            if getattr(parallel, name, False):
                raise ValueError(f"expert offload does not support {name}: {reason}; disable {name}")
        for name, reason in {
            "lora_config": "LoRA composition with prepared expert weights is not implemented",
            "kv_transfer_config": "split-worker loading and offload lifecycle coordination is not implemented",
        }.items():
            if getattr(vllm_config, name, None) is not None:
                raise ValueError(f"expert offload does not support {name}: {reason}; remove {name}")
        speculative = vllm_config.speculative_config
        if speculative is not None:
            draft = getattr(speculative, "draft_model_config", None)
            if (
                getattr(speculative, "method", None) != "dspark"
                or draft is None
                or draft.model != vllm_config.model_config.model
                or draft.hf_config.model_type != "deepseek_v4"
                or draft.hf_config.architectures != ["DSparkDraftModel"]
                or not speculative.enforce_eager
                or speculative.enable_adaptive_verification
                or vllm_config.scheduler_config.max_num_seqs != 1
            ):
                raise ValueError(
                    "expert offload speculative_config supports only the checkpoint's DeepSeek V4 DSpark "
                    "with eager drafting, fixed verification and max_num_seqs=1"
                )
        offload = vllm_config.offload_config
        if offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size:
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
