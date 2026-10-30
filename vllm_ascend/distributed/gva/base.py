# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocols implemented by platform-specific MemFabric GVA adapters."""

from __future__ import annotations

from typing import Protocol

from vllm_ascend.distributed.gva.types import (
    GvaCapabilities,
    GvaObjectRef,
    GvaObjectSpec,
    GvaPoolMember,
    GvaPoolRef,
    GvaPoolSpec,
)


class GvaObject(Protocol):
    """A mapped or gathered logical object owned by a model adapter."""

    @property
    def ref(self) -> GvaObjectRef: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


class GvaPool(Protocol):
    """A fixed-membership pool for one model run."""

    @property
    def capabilities(self) -> GvaCapabilities: ...

    @property
    def ref(self) -> GvaPoolRef: ...

    def allocate(self, spec: GvaObjectSpec) -> GvaObject: ...

    def attach(self, ref: GvaObjectRef) -> GvaObject: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


class GvaBackend(Protocol):
    """Factory for one platform-specific GVA data plane."""

    @property
    def capabilities(self) -> GvaCapabilities: ...

    def create_pool(self, spec: GvaPoolSpec, member: GvaPoolMember) -> GvaPool: ...

    def attach_pool(self, ref: GvaPoolRef, spec: GvaPoolSpec, member: GvaPoolMember) -> GvaPool: ...
