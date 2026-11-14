# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the INT2 prefill execution mixin.

The prefill numeric path (HP writes, middle quantize, attention core) is
device-bound; its behavioral evidence is the device contract battery and
the serve parity legs. At module level we pin the structural contract.
"""

import inspect

from vllm_ascend.attention.int2.prefill import Int2PrefillMixin


def test_prefill_entry_points():
    for m in ("_forward_int2_prefill_threetier", "_attn_core_prefill"):
        assert callable(getattr(Int2PrefillMixin, m, None)), m


def test_explicit_import_blocks():
    import vllm_ascend.attention.int2.prefill as pmod

    assert "from vllm_ascend.attention.int2.ctx import (" in inspect.getsource(pmod)
