#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
# isort: skip_file
from contextlib import contextmanager

import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.xlite.xlite import XliteWrapper


class XliteModelRunner(NPUModelRunner):
    fallback_model: nn.Module
    """The fallback model from the native :class:`NPUModelRunner` implementation."""
    runner_cls: type[XliteWrapper] = XliteWrapper
    """The class of the xlite model forward backend; consistent with :data:`runner_model`."""
    runner_model: XliteWrapper
    """The xlite model forward backend; consistent with :data:`runner_cls`."""
    _runner_enabled: bool = False
    """If :data:`runner_model` is the current model forward backend; otherwise, :data:`fallback_model`."""

    def get_model(self) -> nn.Module:
        """Returns the unwrapper fallback model. See :meth:`NPUModelRunner.get_model` for details."""
        with self._bypass_xlite_wrapper():
            return super().get_model()

    def load_model(self) -> None:
        super().load_model()
        self.fallback_model = self.model
        # NOTE: this will create a circular reference between XliteModelRunner and XliteWrapper instances,
        # but this should be fine since they are both long-lived objects
        self.model = self.runner_model = self.runner_cls(self, self.vllm_config, device=self.device)  # type: ignore[assignment]

    @contextmanager
    def _bypass_xlite_wrapper(self):
        """Temporarily route ``self.model`` to the native runnable."""
        if not self.runner_enabled:
            yield
            return

        self.model = self.fallback_model
        try:
            yield
        finally:
            self.model = self.runner_model  # type: ignore[assignment]

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_kv_cache(kv_cache_config)
        self.runner_model.register_kv_caches(self.kv_caches)  # type: ignore[arg-type]

    def _should_build_dummy_attn_metadata(
        self,
        force_attention: bool = False,
        is_profile: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
    ) -> bool:
        """
        Override to build attention metadata during dummy_run when xlite is enable.
        For xlite, we need to build metadata during DP dummy_run to ensure all ranks
        have consistent metadata, even when some ranks have no requests.
        """
        base_condition = super()._should_build_dummy_attn_metadata(force_attention, is_profile, cudagraph_runtime_mode)
        xlite_condition = self.ascend_config.xlite_graph_config.enabled and not is_profile
        return base_condition or xlite_condition

    @property
    def model(self) -> nn.Module:
        """The current model forward backend."""
        return self._model

    @model.setter
    def model(self, value: nn.Module) -> None:
        self._model = value
        self._runner_enabled = isinstance(value, self.runner_cls)

    @property
    def runner_enabled(self) -> bool:
        """If the current model forward backend is the xlite runner."""
        return self._runner_enabled
