# SPDX-License-Identifier: Apache-2.0
"""Host-only validation of offsets exported by the GDN chunk kernel."""

GDN_CHECKPOINT_ALIGNMENT = 64


def validate_checkpoint_plan(query_lens: list[int], offsets: list[int]) -> None:
    """Reject plans that cannot write every advertised internal checkpoint."""
    if len(query_lens) != len(offsets):
        raise ValueError("Checkpoint offsets and query lengths must have equal length")
    for length, offset in zip(query_lens, offsets):
        if length <= 0 or offset < 0 or offset >= length or offset % GDN_CHECKPOINT_ALIGNMENT:
            raise ValueError("GDN checkpoints require non-empty queries and aligned internal offsets")
