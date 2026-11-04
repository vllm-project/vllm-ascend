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
#
"""xlite model runner for the V1 model runner.

The shared mixin and :class:`XliteModelRunnerV2` stay in
``vllm_ascend/xlite/xlite_model_runner.py``; only the V1 subclass lives here so that
importing the xlite package no longer pulls in ``vllm_ascend.worker.v1``.
"""

from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.worker.v1.model_runner import NPUModelRunner
from vllm_ascend.xlite.utils import _TensorRepr
from vllm_ascend.xlite.xlite_model_runner import XliteModelRunnerMixin


class XliteModelRunner(XliteModelRunnerMixin, NPUModelRunner):  # type: ignore[misc]
    """xlite model runner for model runner V1 (`vllm_ascend/worker/v1/model_runner.py`).

    Remove this module together with `NPUModelRunner` V1 when the V1 runner is dropped; the mixin and
    `XliteModelRunnerV2` stay in `vllm_ascend/xlite/xlite_model_runner.py`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8:
            logger.error("xlite does not currently support Sparse SFA/LI C8.")

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

    def _dummy_run(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Route dummy/profile runs to the native runnable, bypassing the xlite wrapper.

        See :meth:`_bypass_xlite_wrapper` for why dummy runs must not trigger the xlite forward. Delegates every
        argument to the base implementation unchanged. The guards read keyword arguments only: every existing
        caller passes them as keywords (the V1 base signatures even disagree on ``uniform_decode``'s positional
        slot, so positional form would be ambiguous), and a stray positional pass would at worst leave a guard
        unseen — degrading toward the pre-fix behavior, never crashing on re-passed keywords.
        """
        if not kwargs.get("is_profile", False) and not kwargs.get("is_graph_capturing", False):
            # DP `execute_dummy_batch` must be routed to xlite forward path to avoid out of sync issues
            if kwargs.get("uniform_decode") and self.runner_enabled:
                # xlite only needs one token for one request from a dummy rank; flag the run so the wrapper
                # clamps the dummy batch accordingly.
                with self.runner_model.dummy_batch():
                    return super()._dummy_run(*args, **kwargs)  # type: ignore[return-value]
            return super()._dummy_run(*args, **kwargs)  # type: ignore[return-value]

        with self._bypass_xlite_wrapper():
            return super()._dummy_run(*args, **kwargs)  # type: ignore[return-value]

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, kv_cache_allocation_context: AbstractContextManager | None = None
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context=kv_cache_allocation_context)
        kv_caches: list[torch.Tensor] = self._get_flattened_kv_caches(self.kv_caches)
        logger.info_once("xlite: registering kv caches to runner model: %s", _TensorRepr(kv_caches, max_it=None))
        self.runner_model.register_kv_caches(kv_caches)

    def _should_build_dummy_attn_metadata(
        self, force_attention: bool = False, is_profile: bool = False, *args: Any, **kwargs: Any
    ) -> bool:
        base_condition = super()._should_build_dummy_attn_metadata(force_attention, is_profile, *args, **kwargs)
        return base_condition or (self.runner_enabled and not is_profile)
