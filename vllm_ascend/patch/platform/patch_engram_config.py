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

"""Allow Engram on Ascend and keep Qwen4Exp PLE host offload opt-in."""

import importlib.util

# Older vLLM builds have no Engram config to patch.
if importlib.util.find_spec("vllm.config.engram") is not None:
    from vllm.config.engram import EngramConfig, model_has_engram_layers

    _QWEN4EXP_ARCHITECTURES = frozenset({"Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"})

    def _is_qwen4exp(model_config) -> bool:
        return model_config is not None and getattr(model_config, "architecture", None) in _QWEN4EXP_ARCHITECTURES

    def _effective_model_config(self):
        """Mirror upstream's draft-to-target switch so the Qwen4Exp check sees
        the target architecture, not the draft's."""
        model_config = self.model_config
        speculative_config = getattr(self, "speculative_config", None)
        if speculative_config is not None and model_config is speculative_config.draft_model_config:
            model_config = speculative_config.target_model_config
        return model_config

    # Keep upstream's model/layer checks; only its CUDA requirement is lifted.
    def verify_model_config(self, model_config) -> None:
        if not model_has_engram_layers(model_config):
            raise ValueError("EngramConfig requires a supported model with non-empty n-gram layer ids.")

    # Accept PCP>1 as a substitute for DP>1 so shared tables work in PCP-only
    # topologies (Ascend MRV2).
    def verify_parallel_config(self, parallel_config) -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1 and parallel_config.prefill_context_parallel_size <= 1:
                raise ValueError(
                    "dp_shared_memory requires data_parallel_size > 1 or prefill_context_parallel_size > 1."
                )
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1 and parallel_config.enable_elastic_ep:
            raise ValueError("Engram embedding_across_dp is not supported with elastic EP yet.")

    EngramConfig.verify_model_config = verify_model_config
    EngramConfig.verify_parallel_config = verify_parallel_config

    # Upstream's _NGRAM_LAYER_FIELDS includes Qwen4Exp, so
    # _resolve_and_verify_engram_config auto-creates EngramConfig(cpu_offload=True)
    # for Qwen4Exp even when the user did not pass --engram-config.  PLE host
    # offload must remain opt-in on Ascend, so skip the auto-creation for
    # Qwen4Exp and let _validate_engram_config in platform.py handle it.
    from vllm.config import VllmConfig

    _upstream_resolve = VllmConfig._resolve_and_verify_engram_config

    def _resolve_and_verify_engram_config(self) -> None:
        if self.engram_config is None and _is_qwen4exp(_effective_model_config(self)):
            # Defer to _validate_engram_config in platform.py, which only
            # enables offload when the user explicitly passes --engram-config.
            return
        _upstream_resolve(self)

    VllmConfig._resolve_and_verify_engram_config = _resolve_and_verify_engram_config
