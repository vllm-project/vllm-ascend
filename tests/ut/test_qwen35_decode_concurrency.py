# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend import ascend_forward_context


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"kv0": SimpleNamespace(num_prefills=0, num_decodes=8)}, 8),
        (
            {
                "kv0": SimpleNamespace(num_prefills=0, num_decodes=8),
                "kv1": SimpleNamespace(num_prefills=0, num_decodes=5),
            },
            5,
        ),
        ({"kv0": SimpleNamespace(num_prefills=1, num_decodes=8)}, 0),
        ({"kv0": SimpleNamespace(num_prefills=0, num_decodes=0)}, 0),
        ({"kv0": object()}, 0),
        ({}, 0),
    ],
)
def test_decode_concurrency(metadata, expected, monkeypatch):
    context = SimpleNamespace(is_draft_model=False, attn_metadata=metadata)
    monkeypatch.setattr(ascend_forward_context, "get_forward_context", lambda: context)

    assert ascend_forward_context.decode_concurrency() == expected


def test_decode_concurrency_returns_zero_outside_forward_context(monkeypatch):
    def raise_without_context():
        raise AssertionError("forward context is not available")

    monkeypatch.setattr(ascend_forward_context, "get_forward_context", raise_without_context)

    assert ascend_forward_context.decode_concurrency() == 0


def test_decode_concurrency_skips_draft_model(monkeypatch):
    metadata = {"kv0": SimpleNamespace(num_prefills=0, num_decodes=8)}
    context = SimpleNamespace(is_draft_model=True, attn_metadata=metadata)
    monkeypatch.setattr(ascend_forward_context, "get_forward_context", lambda: context)

    assert ascend_forward_context.decode_concurrency() == 0
