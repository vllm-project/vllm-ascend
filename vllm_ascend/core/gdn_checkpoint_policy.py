# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass


@dataclass(frozen=True)
class GDNCheckpointPolicy:
    max_tokens: int
    preserve_physical_prefix: bool
    elide_singleton_warm_tail: bool

    @classmethod
    def from_config(cls, config) -> "GDNCheckpointPolicy | None":
        options = config.additional_config or {}
        if config.kv_transfer_config is not None or not options.get("gdn_native_prefill_checkpoints"):
            return None
        policy = cls(
            max_tokens=options.get("gdn_native_checkpoint_max_tokens", 0),
            preserve_physical_prefix=options.get("gdn_native_preserve_physical_prefix", False),
            elide_singleton_warm_tail=options.get("gdn_elide_singleton_warm_tail", False),
        )
        return policy if policy.max_tokens or policy.preserve_physical_prefix else None

    def allows_internal_checkpoint(self, position: int, end: int, block_size: int) -> bool:
        return (not self.max_tokens or end <= self.max_tokens) and not (
            self.preserve_physical_prefix and position % block_size == 0
        )

    def internal_position(
        self,
        position: int,
        start: int,
        end: int,
        block_size: int,
        alignment: int | None,
    ) -> int:
        if not self.allows_internal_checkpoint(position, end, block_size):
            return 0
        if (
            self.preserve_physical_prefix
            and alignment
            and block_size % alignment == 0
            and start == 0
            and block_size < position < end <= 2 * block_size
        ):
            # Preserve the first physical page on a cold, two-page prefill.
            return block_size
        return position
