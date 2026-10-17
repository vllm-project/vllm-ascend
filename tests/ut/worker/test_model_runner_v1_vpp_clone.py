# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_clone_vpp_attn_metadata`` (VPP attn-metadata deep clone).

Regression for the PR #18018 review finding: the previous recursive clone did
not clone ``torch.Tensor`` leaves that lived *inside* dicts / lists / nested
dataclasses (``_clone`` on a raw Tensor was a no-op). The VPP fold-back keeps
batches in flight across yields, so a later batch's ``_build_attention_metadata``
can overwrite the shared view buffers and fault the NPU attention kernel; every
reachable tensor must be cloned.
"""

from dataclasses import dataclass, field

import torch

from vllm_ascend.worker.model_runner_v1 import _clone_vpp_attn_metadata


@dataclass
class _Sub:
    pos: torch.Tensor


@dataclass
class _Meta:
    slot: torch.Tensor                         # direct dataclass field
    table: torch.Tensor | None = None          # None field must not break
    sub: _Sub | None = None                    # nested dataclass
    bucket: dict = field(default_factory=dict)  # dict of tensors
    seq: list = field(default_factory=list)     # list of tensors
    tag: object = None                          # non-tensor shared by reference


def _leaf(m: _Meta) -> dict[str, torch.Tensor]:
    return {
        "slot": m.slot,
        "sub": m.sub.pos,
        "bucket.k1": m.bucket["k1"],
        "bucket.k2": m.bucket["k2"],
        "seq.0": m.seq[0],
        "seq.1": m.seq[1],
    }


def _make():
    return _Meta(
        slot=torch.randn(2),
        table=None,
        sub=_Sub(pos=torch.randn(3)),
        bucket={"k1": torch.randn(1), "k2": torch.randn(2)},
        seq=[torch.randn(1), torch.randn(2)],
        tag=object(),
    )


def test_every_tensor_leaf_is_cloned():
    m = _make()
    before = _leaf(m)
    before_ids = {k: id(v) for k, v in before.items()}

    _clone_vpp_attn_metadata(m)

    after = _leaf(m)
    assert set(after) == set(before)
    for k, new in after.items():
        # A real deep copy (new object, not the same tensor).
        assert id(new) != before_ids[k], f"tensor {k} was not cloned"
        assert new.dtype == before[k].dtype
        assert new.shape == before[k].shape
        assert torch.allclose(new, before[k])


def test_nothing_is_aliased_under_the_hood():
    m = _make()
    before_ids = {k: id(v) for k, v in _leaf(m).items()}
    _clone_vpp_attn_metadata(m)
    after_ids = {k: id(v) for k, v in _leaf(m).items()}
    # Distinct clones, none aliasing the original tensors.
    assert all(after_ids[k] != before_ids[k] for k in before_ids)
    # And the clones are distinct from one another where expected.
    assert after_ids["bucket.k1"] != after_ids["bucket.k2"]


def test_non_tensor_leaf_shared_by_reference():
    m = _make()
    tag = m.tag
    _clone_vpp_attn_metadata(m)
    assert m.tag is tag  # same object reference


def test_none_fields_do_not_break_clone():
    m = _Meta(slot=torch.tensor([1.0]), table=None, sub=None, bucket={}, seq=[], tag=1)
    _clone_vpp_attn_metadata(m)  # no exception
    assert m.slot.shape == (1,)
    assert m.table is None


def test_cycle_is_safe():
    m = _make()
    m.bucket.clear()
    m.seq.clear()
    m.bucket["self"] = m
    m.seq.append(m.bucket)
    _clone_vpp_attn_metadata(m)  # terminates despite self-references
    assert m.bucket["self"] is m