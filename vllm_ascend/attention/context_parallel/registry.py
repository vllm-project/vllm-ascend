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

"""Single registry and resolution point for context-parallel attention backends.

The registry itself imports no adapter module: each adapter registers its
provider at import time, and :func:`resolve_cp_classes` lazily imports the
built-in adapter modules so that simply importing the registry stays cheap and
free of vLLM/torch dependencies. Adding a backend therefore means adding its
own module (plus a ``register_cp_backend`` call) instead of editing the
existing backends or the platform table.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_ascend.attention.context_parallel.attn_hooks import AttnMetadataHooks
    from vllm_ascend.attention.context_parallel.topology import CPTopology


@dataclass(frozen=True)
class CPBackendProvider:
    """Declarative description of one context-parallel attention backend.

    ``base_*`` are the classes used when context parallelism is disabled,
    ``cp_*`` the specialization used when it is enabled. Backends with more than
    one CP layout (SFA) set ``cp_resolver`` instead of the static ``cp_*`` pair.
    """

    name: str
    base_builder_cls: type | None = None
    base_impl_cls: type | None = None
    cp_builder_cls: type | None = None
    cp_impl_cls: type | None = None
    supports_pcp: bool = False
    supports_dcp: bool = False
    validate: Callable[[Any, CPTopology], None] | None = None
    # Multi-layout hook used by backends such as SFA; when None, cp_* is used.
    cp_resolver: Callable[[Any, CPTopology], tuple[type | None, type | None]] | None = None
    # Optional per-backend hook for spec-decode drafting metadata; returns True
    # when it handled the update, False to fall back to the shared spec checks.
    spec_decode_drafting_updater: Callable[..., bool] | None = None
    # Optional assembly hooks for ``build_attn_metadata``; None keeps the plain path.
    attn_metadata_hooks: AttnMetadataHooks | None = None

    def uses_cp(self, topology: CPTopology) -> bool:
        """Whether a CP specialization should be selected for this topology."""
        return (topology.is_dcp_enabled and self.supports_dcp) or (topology.is_pcp_enabled and self.supports_pcp)

    def resolve_classes(self, vllm_config, topology: CPTopology) -> tuple[type, type]:
        """Resolve the (metadata builder, attention impl) pair for a topology.

        Backend-specific ``validate`` is intentionally not called here; routing
        validation through the provider is left to a follow-up change.
        """
        if not self.uses_cp(topology):
            builder_cls, impl_cls = self.base_builder_cls, self.base_impl_cls
            if builder_cls is None or impl_cls is None:
                raise NotImplementedError(f"Attention backend {self.name!r} has no base implementation.")
            return builder_cls, impl_cls

        if self.cp_resolver is not None:
            builder_cls, impl_cls = self.cp_resolver(vllm_config, topology)
        else:
            builder_cls, impl_cls = self.cp_builder_cls, self.cp_impl_cls
        if builder_cls is None or impl_cls is None:
            raise NotImplementedError(f"Attention backend {self.name!r} requires CP but has no CP implementation.")
        return builder_cls, impl_cls


_CP_BACKEND_REGISTRY: dict[str, CPBackendProvider] = {}

# provider name -> adapter module path. resolve_cp_classes imports these on
# demand so the registry itself never imports an adapter (no import cycles).
_CP_BACKEND_MODULES: dict[str, str] = {
    "gqa": "vllm_ascend.attention.context_parallel.attention_cp",
    "mla": "vllm_ascend.attention.context_parallel.mla_cp",
    "sfa": "vllm_ascend.attention.context_parallel.sfa_cp",
    "dsa": "vllm_ascend.attention.context_parallel.dsa_cp",
    "dsa_v41": "vllm_ascend.attention.context_parallel.dsa_v41_cp",
}


def register_cp_backend(provider: CPBackendProvider) -> CPBackendProvider:
    """Register one CP backend provider and return it for decorator-style use."""
    if provider.name in _CP_BACKEND_REGISTRY:
        raise ValueError(f"CP backend already registered: {provider.name}")
    _CP_BACKEND_REGISTRY[provider.name] = provider
    return provider


def get_cp_backend(name: str) -> CPBackendProvider | None:
    """Return the registered provider for ``name``, or None if absent."""
    return _CP_BACKEND_REGISTRY.get(name)


def list_cp_backends() -> tuple[str, ...]:
    """Return the registered CP backend names in a stable order."""
    return tuple(sorted(_CP_BACKEND_REGISTRY))


def resolve_cp_classes(name: str, vllm_config, topology: CPTopology) -> tuple[type, type]:
    """Resolve the (metadata builder, attention impl) pair of a CP backend.

    Unregistered built-in names are lazily loaded from their adapter module
    first; unknown names raise ``NotImplementedError``.
    """
    provider = _CP_BACKEND_REGISTRY.get(name)
    if provider is None:
        module_path = _CP_BACKEND_MODULES.get(name)
        if module_path is not None:
            importlib.import_module(module_path)
            provider = _CP_BACKEND_REGISTRY.get(name)
    if provider is None:
        raise NotImplementedError(f"Unknown context-parallel attention backend: {name!r}.")
    return provider.resolve_classes(vllm_config, topology)


def find_cp_backend_for_builder(attn_metadata_builder: Any | None) -> CPBackendProvider | None:
    """Find the provider owning a metadata builder class (None when unknown)."""
    if attn_metadata_builder is None:
        return None
    # Ensure the built-in adapters have registered their providers first; the
    # imports are idempotent and the registry itself stays free of adapters.
    for module_path in _CP_BACKEND_MODULES.values():
        importlib.import_module(module_path)
    for name in sorted(_CP_BACKEND_REGISTRY):
        provider = _CP_BACKEND_REGISTRY[name]
        builder_classes = [
            builder_cls
            for builder_cls in (provider.base_builder_cls, provider.cp_builder_cls)
            if builder_cls is not None
        ]
        if builder_classes and isinstance(attn_metadata_builder, tuple(builder_classes)):
            return provider
    return None
