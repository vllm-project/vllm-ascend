#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Select the Ascend config before upstream's CUDA-only Engram validation.

The pinned vLLM resolves Engram before calling the platform config hook.
Keep this single entry-point adapter until upstream provides that hook.
"""

import importlib.util

# Older vLLM builds have no Engram config or CLI entry to patch.
if importlib.util.find_spec("vllm.config.engram") is not None:
    from dataclasses import asdict

    from pydantic import model_validator
    from vllm.config.engram import EngramConfig
    from vllm.config.utils import config
    from vllm.config.vllm import VllmConfig

    @config
    class AscendEngramConfig(EngramConfig):
        @model_validator(mode="after")
        def _validate_shared_memory(self):
            super()._validate_shared_memory()
            if self.dp_shared_memory and self.embedding_across_dp:
                raise ValueError("dp_shared_memory cannot be combined with embedding_across_dp")
            return self

        def verify_model_config(self, model_config) -> None:
            if (
                model_config is None
                or model_config.architecture != "DeepseekV41ForCausalLM"
                or not getattr(model_config.hf_text_config, "engram_layer_ids", None)
            ):
                raise ValueError("Ascend Engram requires DeepSeek V4.1 with non-empty engram_layer_ids.")

        def verify_parallel_config(self, parallel_config) -> None:
            super().verify_parallel_config(parallel_config)
            if self.embedding_across_dp:
                raise ValueError("Ascend Engram does not support embedding_across_dp")
            tp = parallel_config.tensor_parallel_size
            # The DP dimension may span nodes: Engram shards and exchanges only
            # inside the node-local EDP group that the distributed init derives
            # from the physical placement, and it falls back to plain TP shards
            # where a node holds a single replica. Whether those groups really
            # are node-local is checked against the initialized groups when the
            # table is built.
            if (
                parallel_config.enable_elastic_ep
                or tp not in (1, 2, 4, 8)
                or parallel_config.data_parallel_size < 1
                or parallel_config.pipeline_parallel_size != 1
                or parallel_config.prefill_context_parallel_size != 1
                or parallel_config.decode_context_parallel_size != 1
            ):
                raise ValueError("Ascend Engram requires TP=1/2/4/8 with PP=PCP=DCP=1.")

        def verify_load_config(self, load_config) -> None:
            if self.dp_shared_memory and load_config.load_format not in ("auto", "safetensors"):
                raise ValueError("dp_shared_memory requires load_format auto or safetensors")
            if load_config.load_format not in ("auto", "safetensors", "dummy"):
                raise ValueError("Ascend Engram requires indexed safetensors (auto/safetensors), or dummy weights.")

    _resolve_engram_config = VllmConfig._resolve_and_verify_engram_config

    def _resolve_and_verify_engram_config(self) -> None:
        model_config = self.model_config
        spec = self.speculative_config
        if spec is not None and model_config is spec.draft_model_config:
            model_config = spec.target_model_config
        if model_config is not None and model_config.architecture == "DeepseekV41ForCausalLM":
            if self.engram_config is not None:
                if not isinstance(self.engram_config, AscendEngramConfig):
                    self.engram_config = AscendEngramConfig(**asdict(self.engram_config))
            elif getattr(model_config.hf_text_config, "engram_layer_ids", None):
                self.engram_config = AscendEngramConfig()
        _resolve_engram_config(self)

    VllmConfig._resolve_and_verify_engram_config = _resolve_and_verify_engram_config
