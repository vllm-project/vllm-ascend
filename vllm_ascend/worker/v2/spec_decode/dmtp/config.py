# SPDX-License-Identifier: Apache-2.0
from importlib.util import find_spec
from typing import TYPE_CHECKING

import torch
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig

from vllm_ascend.device.hardware_profile import FlashAttentionNpu3Model, get_current_hardware_profile

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig

DMTP_ARCHITECTURE = "DFlash2T1ResidualDraftModel"
DMTP_PROPOSAL_COUNT = 7


class AscendDMTPSpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = "vllm_ascend.worker.v2.spec_decode.dmtp.speculator.AscendDMTPSpeculator"
    supports_watermarking = False

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        if spec.custom_method != "dmtp":
            raise ValueError("Specify method='dmtp' for a DMTP checkpoint.")
        if spec.draft_model_config is None or DMTP_ARCHITECTURE not in spec.draft_model_config.architectures:
            raise ValueError(f"DMTP requires a {DMTP_ARCHITECTURE} draft checkpoint.")
        if spec.num_speculative_tokens != DMTP_PROPOSAL_COUNT:
            raise ValueError("DMTP requires num_speculative_tokens=7 (one native MTP and six DFlash2 proposals).")
        if spec.draft_sample_method != "greedy" or spec.rejection_sample_method != "standard":
            raise ValueError("DMTP requires greedy drafting and standard verification.")
        if spec.enable_adaptive_verification or spec.num_speculative_tokens_per_batch_size is not None:
            raise ValueError("DMTP does not support adaptive verification or dynamic draft lengths.")
        target = spec.target_model_config
        cfg = target.hf_text_config
        if not (
            cfg.model_type == "qwen3_5_text"
            and getattr(cfg, "mtp_num_hidden_layers", 0) == 1
            and cfg.hidden_size == 5120
            and cfg.num_hidden_layers == 64
            and not getattr(cfg, "mtp_use_dedicated_embeddings", False)
        ):
            raise ValueError("DMTP requires a Qwen3.6-27B target with one native MTP layer and shared embeddings.")
        layout = spec.draft_model_config.hf_config.dflash_config
        required = {
            "block_size": 8,
            "known_prefix_tokens": 2,
            "mtp_t1_residual": True,
            "mtp_t1_input_slot": 1,
            "mtp_prefix_rank": 256,
            "mtp_proposal_token_count": 1,
            "dflash_proposal_token_count": 6,
            "target_layer_ids": [0, 15, 30, 45, 60],
            "block_attention": "bidirectional",
            "context_kv_boundary": "strictly_before_anchor",
            "draft_layer_context_windows": [4096] * 5,
            "hidden_norm_zero_centered_gamma": True,
        }
        for name, expected in required.items():
            if layout.get(name) != expected:
                raise ValueError(f"Unsupported DMTP checkpoint: expected dflash_config.{name}={expected!r}.")
        if target.dtype != torch.bfloat16 or target.quantization or spec.draft_model_config.quantization:
            raise ValueError("DMTP currently requires unquantized BF16 target and draft weights.")


def validate_dmtp_engine_config(config: "VllmConfig") -> None:
    spec = config.speculative_config
    if getattr(spec, "custom_method", None) != "dmtp":
        if (
            spec is not None
            and spec.draft_model_config is not None
            and DMTP_ARCHITECTURE in spec.draft_model_config.architectures
        ):
            raise ValueError("Use method='dmtp' with a DMTP draft checkpoint.")
        return
    if config.scheduler_config.async_scheduling:
        raise ValueError("DMTP requires --no-async-scheduling.")
    if not config.model_config.enforce_eager and find_spec("flash_attn_npu_3") is None:
        raise ValueError("DMTP without --enforce-eager requires flash_attn_npu_3 for batched native MTP attention.")
    if (
        not config.model_config.enforce_eager
        and FlashAttentionNpu3Model.QWEN not in get_current_hardware_profile().flash_attn_npu_3_models
    ):
        raise ValueError("DMTP without --enforce-eager requires hardware supported by the Qwen FA3 backend.")
    if config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("DMTP currently requires a BF16 KV cache.")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("DMTP requires --no-enable-prefix-caching.")
    parallel = config.parallel_config
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or spec.draft_parallel_config.tensor_parallel_size != parallel.tensor_parallel_size
        or parallel.tensor_parallel_size != 2
    ):
        raise ValueError("DMTP currently requires TP2 for both models and PP/DP/PCP/DCP of 1.")
    if config.model_config.max_model_len > 3072:
        raise ValueError("DMTP currently requires --max-model-len <= 3072.")
    mm_config = config.model_config.multimodal_config
    if mm_config is not None and not mm_config.language_model_only:
        raise ValueError("DMTP accepts text inputs only. Set --language-model-only.")
    if getattr(parallel, "enable_batch_sharded_sampling", False):
        raise ValueError("DMTP does not support batch-sharded sampling.")
    if config.kv_transfer_config is not None or config.lora_config is not None:
        raise ValueError("DMTP does not support KV transfer or LoRA.")
