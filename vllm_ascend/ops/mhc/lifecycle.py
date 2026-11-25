# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Model-owned mHC weight preparation at graph and loader boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

import torch

from .contracts import MhcHardwareCapability, MhcSeamSpec, PreparedMhcWeights, WeightFormat
from .registry import A3MegaMhcBackend, MhcBackend, resolve_mhc_device, select_backend


@dataclass(slots=True)
class _LayerWeights:
    load_canonical: Callable[[], torch.Tensor]
    version_tag: str
    consumers: frozenset[str]
    fp32_consumers: set[str]
    canonical: torch.Tensor | None
    binding: tuple[MhcBackend, PreparedMhcWeights] | None = None
    shape: tuple[int, int] | None = None


class MhcWeightLifecycleManager:
    """Own one hc_fn representation per weight group in the model.

    Register every reachable caller before staging. The loader transfers
    ownership of any resident canonical tensor, including release of model
    aliases. The reload callback supplies the exact checkpoint version for
    re-preparation after invalidation and must not retain its device storage.

    Lifecycle hooks run at serialized model boundaries. The graph owner
    retires and drains replays before a hook drops bindings, either through
    retire_graphs or before calling the hook. Captured graphs hold bindings
    for their generation and call validate_graph before replay.

    ``peak_staged_weight_bytes`` tracks the maximum single-layer packed weight
    size staged during sequential ``prepare_layer`` calls. When migrating a
    layer to a packed A3 layout, its canonical source is copied to CPU and the
    manager drops its canonical reference before allocating the packed device
    tensor, bounding transient per-layer device staging to one weight tensor.
    """

    def __init__(
        self,
        device: str,
        cap: MhcHardwareCapability,
        *,
        retire_graphs: Callable[[], None] | None = None,
        invalidate_device_constants: Callable[[torch.device], None] | None = None,
        native_adapter: Callable | None = None,
    ) -> None:
        self._device = resolve_mhc_device(device)
        self._cap = cap
        self._layers: dict[str, _LayerWeights] = {}
        self._retire_graphs = retire_graphs
        self._invalidate_device_constants = invalidate_device_constants
        self._native_adapter = native_adapter
        self._constants_ready = False
        self._asleep = False
        self.generation = 0
        self.peak_staged_weight_bytes = 0

    def register_layer(
        self,
        key: str,
        consumers: Iterable[str],
        *,
        load_canonical: Callable[[], torch.Tensor],
        version_tag: str,
        canonical: torch.Tensor | None = None,
    ) -> None:
        """Register a weight group and its initial FP32 callers."""
        names = frozenset(consumers)
        if key in self._layers or not names:
            raise ValueError(f"Weight group {key} must be new and have consumers")
        if not isinstance(version_tag, str) or not version_tag:
            raise ValueError("Weight groups require a nonempty version_tag")
        self._layers[key] = _LayerWeights(load_canonical, version_tag, names, set(names), canonical)

    def fp32_consumers(self, key: str) -> frozenset[str]:
        return frozenset(self._layers[key].fp32_consumers)

    def mark_consumer_migrated(self, key: str, consumer: str) -> None:
        """Record migration after the caller has a qualified packed path."""
        layer = self._layers[key]
        if consumer not in layer.consumers:
            raise ValueError(f"Unknown consumer {consumer} of {key}")
        layer.fp32_consumers.discard(consumer)

    def canonical_for(self, key: str, consumer: str) -> torch.Tensor:
        """Get the canonical weight for an entry or another unmigrated caller."""
        self._require_awake()
        layer = self._layers[key]
        if consumer not in layer.fp32_consumers:
            raise ValueError(f"Consumer {consumer} of {key} has no FP32 binding")
        source = layer.canonical if layer.canonical is not None else layer.load_canonical()
        if (
            not isinstance(source, torch.Tensor)
            or source.dtype != torch.float32
            or source.ndim != 2
            or not source.is_contiguous()
        ):
            raise ValueError(f"Canonical weight {key} must be a contiguous 2D float32 tensor")
        layer.canonical = source.to(self._device)
        return layer.canonical

    def prepare_layer(self, key: str, spec: MhcSeamSpec) -> tuple[MhcBackend, PreparedMhcWeights]:
        """Stage one weight group before capture and publish the selected binding."""
        self._require_awake()
        layer = self._layers[key]
        expected_shape = (spec.C * (spec.C + 2), spec.C * spec.D)
        if layer.shape is not None and layer.shape != expected_shape:
            raise ValueError(f"Weight group {key} must match seam shape {expected_shape}, got {layer.shape}")
        if layer.fp32_consumers:
            spec = replace(spec, weight_format=WeightFormat.FP32_CANONICAL)
        backend = select_backend(spec, self._cap)
        packed = isinstance(backend, A3MegaMhcBackend)
        layout = spec.weight_format if packed else WeightFormat.FP32_CANONICAL
        if layer.binding is not None:
            previous_backend, previous = layer.binding
            if previous_backend is backend and previous.weight_format == layout:
                return layer.binding
            if (
                packed
                and isinstance(previous_backend, A3MegaMhcBackend)
                and previous.weight_format == layout
                and previous.device == self._device
                and previous.version_tag == layer.version_tag
            ):
                layer.binding = (backend, previous)
                return layer.binding
            self._retire_generation()
            layer.binding = None
            del previous

        source = layer.canonical if layer.canonical is not None else layer.load_canonical()
        if (
            not isinstance(source, torch.Tensor)
            or source.dtype != torch.float32
            or source.ndim != 2
            or not source.is_contiguous()
        ):
            raise ValueError(f"Canonical weight {key} must be a contiguous 2D float32 tensor")
        actual_shape = tuple(source.shape)
        if actual_shape != expected_shape:
            raise ValueError(f"Weight group {key} must match seam shape {expected_shape}, got {actual_shape}")
        if packed:
            staged_source = source.cpu()
            # Release canonical device storage before backend.prepare allocates the packed
            # device tensor so transient per-layer device staging stays bounded to 1x weight size.
            del source
            layer.canonical = None
        else:
            staged_source = source.to(self._device)
        prepared = backend.prepare(
            {
                "hc_fn": staged_source,
                "weight_format": layout,
                "version_tag": layer.version_tag,
                "native_adapter": self._native_adapter,
            },
            str(self._device),
            self._cap,
        )
        if packed:
            del staged_source
            prepared = replace(prepared, source_released=True)
            self._constants_ready |= prepared.index_map_ready
            self.peak_staged_weight_bytes = max(self.peak_staged_weight_bytes, prepared.tensor.nbytes)
        else:
            layer.canonical = prepared.tensor
        layer.shape = expected_shape
        layer.binding = (backend, prepared)
        return layer.binding

    def validate_graph(self, generation: int) -> None:
        self._require_awake()
        if generation != self.generation:
            raise RuntimeError("mHC graph must be recaptured for the current weight generation")

    def on_sleep(self) -> None:
        if not self._asleep:
            self._invalidate_device()
            self._asleep = True

    def on_wake(self) -> None:
        """Resume preparation. Each layer is bound again before new capture."""
        self._asleep = False

    def on_device_move(self, device: str, cap: MhcHardwareCapability) -> None:
        """Retire device bindings and stage weights on the new device on demand."""
        target = resolve_mhc_device(device)
        if (target, cap) != (self._device, self._cap):
            self._invalidate_device()
            self._device, self._cap = target, cap

    def on_weight_update(
        self,
        key: str,
        *,
        canonical: torch.Tensor,
        version_tag: str,
        load_canonical: Callable[[], torch.Tensor],
    ) -> None:
        """Publish a new source and invalidate the captured weight binding."""
        layer = self._layers[key]
        if not isinstance(version_tag, str) or not version_tag or version_tag == layer.version_tag:
            raise ValueError("Weight update requires a new nonempty version_tag")
        if (
            not isinstance(canonical, torch.Tensor)
            or canonical.ndim != 2
            or canonical.dtype != torch.float32
            or not canonical.is_contiguous()
        ):
            raise ValueError(f"Canonical weight {key} must be a contiguous 2D float32 tensor")
        if layer.shape is not None and canonical.shape != layer.shape:
            raise ValueError(f"Updated weight group {key} must have shape {layer.shape}, got {tuple(canonical.shape)}")
        self._retire_generation()
        layer.binding = None
        layer.canonical, layer.version_tag, layer.load_canonical = canonical, version_tag, load_canonical

    def _invalidate_device(self) -> None:
        self._retire_generation()
        if self._constants_ready and self._invalidate_device_constants is not None:
            self._invalidate_device_constants(self._device)
        self._constants_ready = False
        for layer in self._layers.values():
            layer.binding = None
            layer.canonical = None

    def _retire_generation(self) -> None:
        if self._retire_graphs is not None:
            self._retire_graphs()
        self.generation += 1

    def _require_awake(self) -> None:
        if self._asleep:
            raise RuntimeError("mHC weights are asleep")
