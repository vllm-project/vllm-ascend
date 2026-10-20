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
#
"""xlite model runner subclasses (MRV1/MRV2) for `vllm-ascend`"""

from collections.abc import Iterable
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2
from vllm_ascend.xlite.utils import _TensorRepr
from vllm_ascend.xlite.xlite import XliteWrapper


class XliteModelRunnerMixin:
    fallback_model: nn.Module
    """The fallback model from the native :class:`NPUModelRunner` implementation."""
    runner_cls: type[XliteWrapper] = XliteWrapper
    """The class of the xlite model forward backend; consistent with :data:`runner_model`."""
    runner_model: XliteWrapper
    """The xlite model forward backend; consistent with :data:`runner_cls`."""
    _runner_enabled: bool = False
    """If :data:`runner_model` is the current model forward backend; otherwise, :data:`fallback_model`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert get_ascend_config().xlite_graph_config.enabled, "xlite graph must be enabled to use XliteModelRunner"
        super().__init__(*args, **kwargs)

    @contextmanager
    def _bypass_xlite_wrapper(self):
        """Temporarily route ``self.model`` to the native runnable.

        Profile runs and dummy runs (including ``execute_dummy_batch`` and cudagraph capture warmups) should not enter
        the xlite forward path.
        """
        if not self.runner_enabled:
            yield
            return

        self.model = self.fallback_model
        try:
            yield
        finally:
            self.model = self.runner_model  # type: ignore[assignment]

    @property
    def model(self) -> nn.Module:
        """The current model forward backend, either :data:`runner_model` or :data:`fallback_model`."""
        return self._model

    @model.setter
    def model(self, value: nn.Module) -> None:
        self._model = value
        self._runner_enabled = isinstance(value, self.runner_cls)

    @model.deleter
    def model(self) -> None:
        self._runner_enabled = False
        del self._model
        del self.fallback_model
        del self.runner_model

    @property
    def runner_enabled(self) -> bool:
        """If the current model forward backend is the xlite runner."""
        return self._runner_enabled

    def _get_flattened_kv_caches(
        self, kv_caches: list[torch.Tensor] | list[Iterable[torch.Tensor]], /, *, check_is_contiguous: bool = True
    ) -> list[torch.Tensor]:
        """Flattens the KV cache list to a single list of tensors.

        Args:
            kv_caches (list[torch.Tensor] | list[Iterable[torch.Tensor]]): The KV cache list to flatten.
            check_is_contiguous (bool, optional): If True, checks that each retrieved tensor is contiguous in memory. If
                any tensor is not contiguous, a ValueError is raised. Default is True.

        Returns:
            list[torch.Tensor]: The flattened KV cache list.
        """
        flattened_kv_caches: list[torch.Tensor] = []
        for cache in kv_caches:
            if isinstance(cache, torch.Tensor):
                flattened_kv_caches.append(cache)
            elif isinstance(cache, Iterable) and all(isinstance(c, torch.Tensor) for c in cache):  # type: ignore[arg-type]
                flattened_kv_caches.extend(cache)
            else:
                raise TypeError(f"Unexpected kv_cache type: {type(cache)}")
        if check_is_contiguous and not all(tensor.is_contiguous() for tensor in flattened_kv_caches):
            raise ValueError("xlite: Found non-contiguous tensor in the KV caches.")
        return flattened_kv_caches


class XliteModelRunner(XliteModelRunnerMixin, NPUModelRunner):  # type: ignore[misc]
    """xlite model runner for model runner V1 (`vllm_ascend/worker/model_runner_v1.py`).

    Remove this class together with `NPUModelRunner` V1 when the V1 runner is dropped; the mixin and
    `XliteModelRunnerV2` below stay.
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

    def _dummy_run(
        self, *args: Any, is_profile: bool = False, is_graph_capturing: bool = False, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route dummy/profile runs to the native runnable, bypassing the xlite wrapper.

        See :meth:`_bypass_xlite_wrapper` for why dummy runs must not trigger the xlite forward. Delegates every
        argument to the base implementation unchanged.
        """
        if not is_profile and not is_graph_capturing:
            # DP `excute_dummy_batch` must be routed to xlite forward path to avoid out of sync issues
            return super()._dummy_run(*args, is_profile=is_profile, is_graph_capturing=is_graph_capturing, **kwargs)  # type: ignore[return-value]

        with self._bypass_xlite_wrapper():
            return super()._dummy_run(*args, is_profile=is_profile, is_graph_capturing=is_graph_capturing, **kwargs)  # type: ignore[return-value]

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, kv_cache_allocation_context: AbstractContextManager | None = None
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context=kv_cache_allocation_context)
        kv_caches: list[torch.Tensor] = self._get_flattened_kv_caches(self.kv_caches)
        logger.info_once("xlite: registering kv caches to runner model: %s", _TensorRepr(kv_caches, max_it=None))
        self.runner_model.register_kv_caches(kv_caches)

        # check attention metadata backend compatibility
        ascend_metadata_builder = self.attn_groups[0][-1].get_metadata_builder(0)
        ascend_metadata_cls = getattr(ascend_metadata_builder, "metadata_cls", ascend_metadata_builder)
        xlite_expects = self.runner_model.adapter_xlite_model._attn_metadata_type
        if ascend_metadata_cls != xlite_expects and (
            not isinstance(xlite_expects, tuple) or ascend_metadata_cls not in xlite_expects
        ):
            logger.error(
                "Attention metadata mismatch: xlite expects (one of) %s, but got %s. Be aware of runtime issues.",
                xlite_expects,
                ascend_metadata_cls,
            )

    def _should_build_dummy_attn_metadata(
        self, force_attention: bool = False, is_profile: bool = False, *args: Any, **kwargs: Any
    ) -> bool:
        base_condition = super()._should_build_dummy_attn_metadata(force_attention, is_profile, *args, **kwargs)
        return base_condition or (self.runner_enabled and not is_profile)


class XliteModelRunnerV2(XliteModelRunnerMixin, NPUModelRunnerV2):  # type: ignore[misc]
    """xlite model runner for model runner V2 (`vllm_ascend/worker/v2/model_runner.py`).

    Keeps all V2-only adaptations (KV cache regrouping, `decode_query_len` alias, sparse C8 guard,
    `load_model` signature) here so the V1 class above can be removed without touching it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        enable_sparse_sfa_c8 = self.vllm_config.cache_config.cache_dtype in ["fp8", "int8"]
        enable_sparse_li_c8 = self.vllm_config.attention_config.indexer_kv_dtype in ["fp8", "int8"]
        if enable_sparse_sfa_c8 or enable_sparse_li_c8:
            logger.error("xlite does not currently support Sparse SFA/LI C8.")

    @property
    def uniform_decode_query_len(self) -> int:
        """V1-compatibility alias for the V2 ``decode_query_len`` attribute.

        ``NPUWorker.execute_dummy_batch`` reads the V1 attribute name via ``getattr`` to size DP dummy
        batches; the attribute was renamed in V2, and without this alias the fallback would silently
        size them to a single token per rank.
        """
        return self.decode_query_len

    def get_model(self) -> nn.Module:
        """Returns the unwrapper fallback model. See :meth:`NPUModelRunner.get_model` for details."""
        with self._bypass_xlite_wrapper():
            return super().get_model()

    def load_model(self, load_dummy_weights: bool = False, *args: Any, **kwargs: Any) -> None:
        super().load_model(load_dummy_weights, *args, **kwargs)
        self.fallback_model = self.model
        # NOTE: this will create a circular reference between XliteModelRunner and XliteWrapper instances,
        # but this should be fine since they are both long-lived objects
        self.model = self.runner_model = self.runner_cls(self, self.vllm_config, device=self.device)  # type: ignore[assignment]

    def _dummy_run(self, *args: Any, is_profile: bool = False, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Route dummy/profile runs to the native runnable, bypassing the xlite wrapper.

        See :meth:`_bypass_xlite_wrapper` for why dummy runs must not trigger the xlite forward. Delegates every
        argument to the base implementation unchanged.
        """
        if not is_profile:
            # DP `excute_dummy_batch` must be routed to xlite forward path to avoid out of sync issues
            return super()._dummy_run(*args, is_profile=is_profile, **kwargs)  # type: ignore[return-value]

        with self._bypass_xlite_wrapper():
            return super()._dummy_run(*args, is_profile=is_profile, **kwargs)  # type: ignore[return-value]

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, kv_cache_allocation_context: AbstractContextManager | None = None
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context=kv_cache_allocation_context)
        kv_caches: list[torch.Tensor] = self._get_flattened_kv_caches(self.kv_caches, check_is_contiguous=False)
        logger.info_once("xlite: registering kv caches to runner model: %s", _TensorRepr(kv_caches, max_it=None))
        self.runner_model.register_kv_caches(kv_caches)

        # check attention metadata backend compatibility
        ascend_metadata_builder = self.attn_groups[0][-1].get_metadata_builder(0)
        ascend_metadata_cls = getattr(ascend_metadata_builder, "metadata_cls", ascend_metadata_builder)
        xlite_expects = self.runner_model.adapter_xlite_model._attn_metadata_type
        if ascend_metadata_cls != xlite_expects and (
            not isinstance(xlite_expects, tuple) or ascend_metadata_cls not in xlite_expects
        ):
            logger.error(
                "Attention metadata mismatch: xlite expects (one of) %s, but got %s. Be aware of runtime issues.",
                xlite_expects,
                ascend_metadata_cls,
            )
