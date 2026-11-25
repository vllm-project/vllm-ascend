# SPDX-License-Identifier: Apache-2.0

"""Default-off, restart-scoped Qwen3.5 decode weight prefetch configuration.

mode:
  "full"    issue real prefetch transfers on the side stream.
  "control" run the same stream/wait plumbing without data movement; a dry-run
            used to validate capture/replay ordering on real hardware.

next_input_join:
  "immediate" join the side stream right after the collective that carries the
              dependency, so the prefetch overlaps only with that collective.
  "deferred"  leave the join to the consumer layer; supported only for
              linear_attention targets, whose forward joins before the input
              projection. full_attention targets silently fall back to the
              immediate join.

min_decode_concurrency:
  skip all prefetch while the pure-decode batch is smaller than this many
  requests; small batches lose more to bandwidth contention than they gain.
  The gate is evaluated per captured graph size, so batches captured below the
  threshold simply never contain prefetch ops.
  Measured on Qwen3.5-35B-A3B TP2 (2026-10-11): concurrency 1 regresses TPOT
  by ~2%, concurrency 8 is neutral, concurrency 32 gains ~1.4%; hence the
  default of 8.
"""

from dataclasses import dataclass

NUM_HIDDEN_LAYERS = 40


@dataclass(frozen=True)
class Qwen35WeightPrefetchConfig:
    enabled: bool = False
    mode: str = "full"
    targets: tuple[str, ...] = ("next_input",)
    next_input_bytes: int | str = 2 * 1024 * 1024
    routed_w2_bytes: int = 4 * 1024 * 1024
    gdn_out_bytes: int = 2 * 1024 * 1024
    layers: tuple[int, ...] | None = None
    next_layer_types: tuple[str, ...] = ("linear_attention", "full_attention")
    routed_start: str = "after_dispatch"
    vector_cores: int = 2
    next_input_join: str = "immediate"
    min_decode_concurrency: int = 8

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise ValueError("qwen35_weight_prefetch must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown weight prefetch options: {sorted(unknown)}")
        options = dict(value)
        try:
            for name in ("targets", "layers", "next_layer_types"):
                if name in options and options[name] is not None:
                    options[name] = tuple(options[name])
            result = cls(**options)
        except TypeError as error:
            raise ValueError(f"invalid qwen35_weight_prefetch value: {error}") from error
        if type(result.enabled) is not bool or result.mode not in ("full", "control"):
            raise ValueError("enabled must be boolean; mode must be full or control")
        if not result.targets or set(result.targets) - {"next_input", "routed_w2", "gdn_out"}:
            raise ValueError("targets must contain next_input, routed_w2, and/or gdn_out")
        if len(set(result.targets)) != len(result.targets):
            raise ValueError("duplicate prefetch targets")
        for name in ("next_input_bytes", "routed_w2_bytes", "gdn_out_bytes"):
            size = getattr(result, name)
            if name == "next_input_bytes" and type(size) is str and size == "full":
                continue
            if type(size) is not int or size <= 0 or size % 32:
                raise ValueError(f"{name} must be positive and 32-byte aligned")
        if result.layers is not None and any(
            type(i) is not int or not 0 <= i < NUM_HIDDEN_LAYERS for i in result.layers
        ):
            raise ValueError(f"layers must contain integer layer indices in [0, {NUM_HIDDEN_LAYERS})")
        if set(result.next_layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError("unsupported next_layer_types")
        if result.routed_start not in ("after_dispatch", "after_gmm1"):
            raise ValueError("unsupported routed_start")
        if result.next_input_join not in ("immediate", "deferred"):
            raise ValueError("unsupported next_input_join")
        if type(result.min_decode_concurrency) is not int or result.min_decode_concurrency < 1:
            raise ValueError("min_decode_concurrency must be a positive integer")
        if type(result.vector_cores) is not int or result.vector_cores not in (1, 2, 4):
            raise ValueError("vector_cores must be 1, 2, or 4")
        return result
