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

"""Shared lazy model-tokenizer loading for DFX detectors / report decode.

Both the report tokenizer (``DfxProcessor``) and the output-substring detector
need the model tokenizer. This single helper keeps the ``cached_tokenizer_from_config``
load in one place so config/runner access stays consistent.
"""

from __future__ import annotations

from typing import Any


def load_model_tokenizer(runner: Any) -> Any | None:
    """Return the model tokenizer via ``cached_tokenizer_from_config``.

    Returns ``None`` when ``runner`` or its ``model_config`` is unavailable
    (caller may retry later). Raises the underlying exception when the load
    itself fails so callers decide retry-vs-fail semantics.
    """
    if runner is None:
        return None
    from vllm.tokenizers import cached_tokenizer_from_config

    vllm_config = getattr(runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None) if vllm_config is not None else None
    if model_config is None:
        return None
    return cached_tokenizer_from_config(model_config)
