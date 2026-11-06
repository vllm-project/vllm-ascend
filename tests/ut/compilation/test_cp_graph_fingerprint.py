# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the cross-rank CP capture fingerprint."""

from vllm_ascend.compilation.cp_graph_fingerprint import (
    build_digest,
    collect_capture_entries,
)


def test_digest_is_order_independent():
    first = build_digest(
        [
            ("PIECEWISE", 16, 0, 0),
            ("PIECEWISE", 32, 0, 0),
        ]
    )
    second = build_digest(
        [
            ("PIECEWISE", 32, 0, 0),
            ("PIECEWISE", 16, 0, 0),
        ]
    )
    assert first.digest == second.digest
    assert first.piece_count == second.piece_count == 2
    assert first.token_extents == second.token_extents == (16, 32)


def test_digest_changes_with_piece_count_and_extent():
    base = build_digest([("PIECEWISE", 16, 0, 0)])
    more_pieces = build_digest([("PIECEWISE", 16, 0, 0), ("PIECEWISE", 16, 0, 0)])
    other_extent = build_digest([("PIECEWISE", 32, 0, 0)])
    other_mode = build_digest([("FULL", 16, 0, 0)])

    assert base.piece_count == 1
    assert more_pieces.piece_count == 2
    assert len({base.digest, more_pieces.digest, other_extent.digest, other_mode.digest}) == 4


def test_digest_is_positive_int64():
    digest = build_digest([("PIECEWISE", 64, 1, 2)]).digest
    assert digest >= 0
    assert digest < 2**63


def test_empty_capture_set_is_stable():
    empty = build_digest([])
    assert empty.piece_count == 0
    assert empty.token_extents == ()
    assert empty.describe().startswith("digest=0x")


def test_collect_capture_entries_reads_acl_wrappers():
    """A captured entry contributes, an uncaptured one does not."""

    class _Entry:
        def __init__(self, aclgraph):
            self.aclgraph = aclgraph

    class _Descriptor:
        def __init__(self, num_tokens):
            self.num_tokens = num_tokens
            self.has_lora = False
            self.num_active_loras = 0

    class _Wrapper:
        def __init__(self):
            self.runtime_mode = "PIECEWISE"
            self.concrete_aclgraph_entries = {
                _Descriptor(16): _Entry(aclgraph=object()),
                _Descriptor(32): _Entry(aclgraph=None),
            }

    import vllm_ascend.compilation.acl_graph as acl_graph

    wrapper = _Wrapper()
    original = acl_graph._acl_graph_wrappers
    acl_graph._acl_graph_wrappers = {wrapper}
    try:
        entries = collect_capture_entries()
    finally:
        acl_graph._acl_graph_wrappers = original

    assert entries == (("PIECEWISE", 16, 0, 0),)
