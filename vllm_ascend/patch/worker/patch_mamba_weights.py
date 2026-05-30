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
#

from collections.abc import Iterable

import torch
from vllm.model_executor.models.mamba2 import Mamba2ForCausalLM

# Mamba-Codestral checkpoints use "model." prefix on weight names,
# but vllm's Mamba2ForCausalLM expects names without this prefix.
_MODEL_PREFIX = "model."

# AutoWeightsLoader may drop the trailing 's' from "embeddings"
# during prefix stripping, producing "embedding.weight" instead of
# the expected "embeddings.weight".
_EMBEDDING_KEY = "embedding.weight"
_EMBEDDINGS_KEY = "embeddings.weight"

_orig_mamba2_load_weights = Mamba2ForCausalLM.load_weights


def _normalize_mamba_weight_name(name: str) -> str:
    if name.startswith(_MODEL_PREFIX):
        name = name[len(_MODEL_PREFIX) :]
    if name == _EMBEDDING_KEY:
        return _EMBEDDINGS_KEY
    if name.endswith(f".{_EMBEDDING_KEY}"):
        return name[: -len(_EMBEDDING_KEY)] + _EMBEDDINGS_KEY
    return name


def _strip_model_prefix(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, loaded_weight in weights:
        yield _normalize_mamba_weight_name(name), loaded_weight


def _patched_mamba2_load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    return _orig_mamba2_load_weights(self, _strip_model_prefix(weights))


Mamba2ForCausalLM.load_weights = _patched_mamba2_load_weights
