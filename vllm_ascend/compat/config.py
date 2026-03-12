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
"""Compatibility wrappers for vllm.config module.

vLLM's config module changes between versions. This module provides
stable imports that work across different vLLM versions.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Try to import CUDAGraphMode from vLLM
try:
    from vllm.config import CUDAGraphMode
except ImportError:
    try:
        # Older versions might have it in a different place
        from vllm.config import CompilationConfig as CUDAGraphMode
    except ImportError:
        warnings.warn(
            "CUDAGraphMode not found in vllm.config. "
            "Using mock implementation. Graph mode functionality may be limited.",
            RuntimeWarning
        )

        class CUDAGraphMode:
            """Mock CUDAGraphMode when vLLM doesn't provide it."""
            NO_GRAPH = "no_graph"
            CAPTURE = "capture"
            REPLAY = "replay"

# Try to import VllmConfig from vLLM
try:
    from vllm.config import VllmConfig
except ImportError:
    warnings.warn(
        "VllmConfig not found in vllm.config. "
        "Using fallback implementation.",
        RuntimeWarning
    )

    @dataclass
    class VllmConfig:
        """Fallback VllmConfig for older vLLM versions."""
        model_config: Dict[str, Any] = field(default_factory=dict)
        cache_config: Dict[str, Any] = field(default_factory=dict)
        parallel_config: Dict[str, Any] = field(default_factory=dict)
        scheduler_config: Dict[str, Any] = field(default_factory=dict)
        device_config: Dict[str, Any] = field(default_factory=dict)
        load_config: Dict[str, Any] = field(default_factory=dict)
        lora_config: Optional[Dict[str, Any]] = None
        speculative_config: Optional[Dict[str, Any]] = None
        decoding_config: Optional[Dict[str, Any]] = None
        observability_config: Optional[Dict[str, Any]] = None
        prompt_adapter_config: Optional[Dict[str, Any]] = None


# Try to import get_current_vllm_config from vLLM
try:
    from vllm.config import get_current_vllm_config
except ImportError:
    # Create a simple mock if not available
    _current_config: Optional[VllmConfig] = None

    def get_current_vllm_config() -> Optional[VllmConfig]:
        """Get current vLLM config (mock implementation)."""
        global _current_config
        if _current_config is None:
            _current_config = VllmConfig()
        return _current_config


__all__ = [
    'CUDAGraphMode',
    'VllmConfig',
    'get_current_vllm_config',
]