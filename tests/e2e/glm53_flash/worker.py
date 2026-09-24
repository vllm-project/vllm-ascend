# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only worker: assert the effective configuration inside both ranks."""

import json

import torch

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

RUNTIME_PREFIX = "GLM53_CI_RUNTIME="


class PrecisionWorker(NPUWorker):
    def load_model(self) -> None:
        super().load_model()
        config = self.vllm_config
        assert type(self.model_runner) is NPUModelRunner, "Expected model_runner_v1, no fallback allowed"
        assert self.use_v2_model_runner is False
        assert config.scheduler_config.disable_hybrid_kv_cache_manager is False
        assert config.scheduler_config.async_scheduling is False
        assert config.scheduler_config.enable_chunked_prefill is True
        assert config.cache_config.enable_prefix_caching is False
        assert config.cache_config.mamba_cache_mode == "none"
        assert config.cache_config.kv_cache_memory_bytes == 1024**3
        assert config.parallel_config.tensor_parallel_size == 2
        assert config.parallel_config.data_parallel_size == 1
        assert config.parallel_config.enable_expert_parallel is True
        assert config.speculative_config is None
        assert config.model_config.enforce_eager is True
        assert config.model_config.logprobs_mode == "raw_logprobs"
        assert config.model_config.quantization == "ascend"
        assert config.model_config.dtype == torch.bfloat16
        assert config.model_config.seed == 0
        assert config.model_config.max_model_len == 8192
        assert config.scheduler_config.max_num_seqs == 1
        assert config.scheduler_config.max_num_batched_tokens == 512
        assert config.model_config.hf_text_config.num_hidden_layers == 5
        assert config.model_config.hf_text_config.num_nextn_predict_layers == 0
        runtime = {
            "rank": self.rank,
            "runner": type(self.model_runner).__module__,
            "hybrid": True,
            "device": torch.npu.get_device_name(self.device),
            "total_memory": torch.npu.get_device_properties(self.device).total_memory,
            "weight_memory_bytes": int(self.model_runner.model_memory_usage),
            "allocated_after_model_load_bytes": torch.npu.memory_allocated(self.device),
        }
        print(RUNTIME_PREFIX + json.dumps(runtime, sort_keys=True), flush=True)

    def execute_model(self, *args, **kwargs):
        result = super().execute_model(*args, **kwargs)
        peak = torch.npu.max_memory_allocated(self.device)
        if peak > getattr(self, "_glm53_reported_peak", 0):
            self._glm53_reported_peak = peak
            print("GLM53_CI_PEAK=" + json.dumps({"rank": self.rank, "allocated_bytes": peak}), flush=True)
        return result
