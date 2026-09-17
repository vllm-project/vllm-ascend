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

import importlib.util

# vLLM only grew EngramConfig after 0.28. Older versions have no
# --engram-config flag at all, which already means "no offload", so there is
# nothing to accept and nothing to relax.
if importlib.util.find_spec("vllm.config.engram") is not None:
    from vllm.config.engram import EngramConfig

    def _verify_model_config(self, model_config) -> None:
        """Require Engram layers, without upstream's CUDA-only restriction.

        Upstream rejects every non-CUDA platform because Engram only existed
        there. The Ascend tables hash the same checkpoint with the same layout
        and only differ in where the rows are stored, so the checkpoint is the
        whole contract here.
        """
        text_config = None if model_config is None else model_config.hf_text_config
        if not getattr(text_config, "engram_layer_ids", None):
            raise ValueError("EngramConfig requires non-empty engram_layer_ids.")
        if getattr(self, "dp_shared_memory", False) or getattr(self, "embedding_across_dp", False):
            raise ValueError("Ascend Engram does not support DP-sharded embeddings.")

    EngramConfig.verify_model_config = _verify_model_config  # type: ignore[method-assign]
