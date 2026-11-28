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
"""xlite model runner mixin and the MRV2 subclass for `vllm-ascend`"""

from contextlib import AbstractContextManager, contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import is_hidden_state_cache_spec
from vllm_ascend.worker.v2.attn_utils import _get_layer_kv_cache_specs
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2
from vllm_ascend.xlite.utils import _TensorRepr
from vllm_ascend.xlite.xlite import XliteWrapper


def _hidden_state_cache_layers(kv_cache_config: KVCacheConfig) -> frozenset[str]:
    """Names of hidden-state-only cache layers (per-layer specs, as the MRV2 reshape reads them)."""
    return frozenset(
        name for name, spec in _get_layer_kv_cache_specs(kv_cache_config).items() if is_hidden_state_cache_spec(spec)
    )


class XliteModelRunnerMixin:
    fallback_model: nn.Module
    """The fallback model from the native :class:`NPUModelRunner` implementation."""
    runner_cls: type[XliteWrapper] = XliteWrapper
    """The class of the xlite model forward backend; consistent with :data:`runner_model`."""
    runner_model: XliteWrapper
    """The xlite model forward backend; consistent with :data:`runner_cls`."""
    _runner_enabled: bool = False
    """If :data:`runner_model` is the current model forward backend; otherwise, :data:`fallback_model`."""
    # Provided by the concrete runner base.
    kv_caches: list[torch.Tensor]
    device: torch.device

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

    def _attention_kv_caches(self, hidden_state_layers: frozenset[str]) -> list[torch.Tensor]:
        """Flat attention-cache tensors, hidden-state-only layers excluded by spec.

        The patched flatten (per-tensor device/None guards) over the name-keyed
        ``kv_caches_by_layer``; falls back to the unfiltered ``kv_caches`` without it.
        """
        caches_by_layer: dict[str, Any] | None = getattr(self, "kv_caches_by_layer", None)
        if caches_by_layer is None:
            logger.warning_once(
                "xlite: no name-keyed kv_caches_by_layer; registering the unfiltered "
                "kv_caches list, hidden-state-only caches included."
            )
            return self.kv_caches
        return [
            tensor
            for name, cache in caches_by_layer.items()
            if name not in hidden_state_layers
            for tensor in (cache if isinstance(cache, (tuple, list)) else (cache,))
            if tensor is not None and tensor.device == self.device
        ]


class XliteModelRunnerV2(XliteModelRunnerMixin, NPUModelRunnerV2):  # type: ignore[misc]
    """xlite model runner for model runner V2 (`vllm_ascend/worker/v2/model_runner.py`).

    Keeps all V2-only adaptations (KV cache regrouping, `decode_query_len` alias, sparse C8 guard,
    `load_model` signature) here so that the V1 subclass can be removed without touching it.
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

    def _dummy_run(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Route dummy/profile runs to the native runnable, bypassing the xlite wrapper.

        See :meth:`_bypass_xlite_wrapper` for why dummy runs must not trigger the xlite forward. Delegates every
        argument to the base implementation unchanged. The guards read keyword arguments only; ``uniform_decode``
        and ``is_profile`` are keyword-only in the V2 base signature, so they cannot arrive positionally here.
        """
        if not kwargs.get("is_profile", False):
            # DP `execute_dummy_batch` must be routed to xlite forward path to avoid out of sync issues
            if kwargs.get("uniform_decode") and self.runner_enabled:
                # xlite only needs one token for one request from a dummy rank
                with self.runner_model.dummy_batch():
                    return super()._dummy_run(*args, **kwargs)  # type: ignore[return-value]
            return super()._dummy_run(*args, **kwargs)  # type: ignore[return-value]

        with self._bypass_xlite_wrapper():
            return super()._dummy_run(*args, **kwargs)  # type: ignore[return-value]

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, kv_cache_allocation_context: AbstractContextManager | None = None
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context=kv_cache_allocation_context)
        # Drop hidden-state-only caches by spec (per-layer, as the MRV2 reshape does):
        # xlite registers attention K/V pairs only.
        hidden_state_layers = _hidden_state_cache_layers(kv_cache_config)
        kv_caches: list[torch.Tensor] = self._attention_kv_caches(hidden_state_layers)
        logger.info_once("xlite: registering kv caches to runner model: %s", _TensorRepr(kv_caches, max_it=None))
        self.runner_model.register_kv_caches(kv_caches)
