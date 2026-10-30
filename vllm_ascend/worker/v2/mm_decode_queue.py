# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only proof that a request has crossed its complete media prompt.

An optimistic scheduler count cannot establish this fact. The producer tags
each corrected device-count snapshot with the request incarnation and media
layout that it describes; only the synchronized consumer may certify it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any


@dataclass(frozen=True, eq=False)
class ObjectIdentity:
    """Retain the object so its identity cannot be recycled under a proof."""

    value: Any = field(repr=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObjectIdentity) and self.value is other.value

    def __hash__(self) -> int:
        return id(self.value)


@dataclass(frozen=True)
class MMDecodeRequestKey:
    req_id: str
    slot: int
    prefill_len: int
    context: tuple
    features: tuple


def make_mm_decode_request_key(
    req_id: str,
    slot: int,
    prefill_len: int,
    context: tuple,
    mm_features: Sequence[Any],
) -> MMDecodeRequestKey | None:
    """Reject media layouts outside the audited non-realtime prompt path."""
    if prefill_len <= 0 or slot < 0:
        return None
    features = []
    previous_end = 0
    for feature in mm_features:
        position = feature.mm_position
        offset, length = position.offset, position.length
        if (
            not isinstance(offset, Integral)
            or not isinstance(length, Integral)
            or isinstance(offset, bool)
            or isinstance(length, bool)
        ):
            return None
        # Normalize NumPy integers before arithmetic to avoid fixed-width overflow.
        offset, length = int(offset), int(length)
        if (
            offset < previous_end
            or length <= 0
            or offset + length > prefill_len
            or feature.modality == "prompt_embeds"
            or position.is_embed is not None
        ):
            return None
        features.append(
            (
                ObjectIdentity(feature),
                ObjectIdentity(position),
                ObjectIdentity(feature.data),
                feature.modality,
                feature.identifier,
                offset,
                length,
            )
        )
        previous_end = offset + length
    return MMDecodeRequestKey(req_id, slot, prefill_len, context, (ObjectIdentity(mm_features), tuple(features)))


class MMDecodeCertificateState:
    """Track trusted prompt completion without reading in-flight D2H memory."""

    def __init__(self) -> None:
        self.snapshot_generation = 0
        self._snapshot_keys: dict[str, MMDecodeRequestKey] = {}
        self._certified: dict[str, MMDecodeRequestKey] = {}

    def invalidate(self) -> None:
        self._snapshot_keys.clear()
        self._certified.clear()

    def record_snapshot(self, keys: Sequence[MMDecodeRequestKey]) -> None:
        """Called after the exact D2H and its completion event are enqueued."""
        self.snapshot_generation += 1
        self._snapshot_keys = {key.req_id: key for key in keys}

    def certify(self, keys: Sequence[MMDecodeRequestKey], exact_counts: Sequence[int]) -> bool:
        """Called only after the producer's event has synchronized on the CPU."""
        if not keys or len(keys) != len(exact_counts):
            return False
        valid = (
            self.snapshot_generation > 0
            and len({key.req_id for key in keys}) == len(keys)
            and all(
                self._snapshot_keys.get(key.req_id) == key and count >= key.prefill_len
                for key, count in zip(keys, exact_counts)
            )
        )
        if valid:
            self._certified.update((key.req_id, key) for key in keys)
        else:
            for key in keys:
                self._certified.pop(key.req_id, None)
        return valid

    def is_certified(self, keys: Sequence[MMDecodeRequestKey]) -> bool:
        return bool(keys) and all(self._certified.get(key.req_id) == key for key in keys)
