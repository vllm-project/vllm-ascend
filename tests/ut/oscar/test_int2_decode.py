# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the INT2 decode execution mixin.

The tier-merge numeric path (incl. the empty-segment arms) is device-bound;
its behavioral evidence is the device contract battery (including the
empty-segment arms) and the serve parity legs. At module level we pin the
structural contract (entry-point presence, explicit imports).
"""

import inspect

from vllm_ascend.attention.int2.decode import Int2DecodeMixin


def test_decode_entry_points():
    assert callable(getattr(Int2DecodeMixin, "_forward_int2_decode_threetier", None))


def test_explicit_import_blocks():
    import vllm_ascend.attention.int2.decode as dmod

    assert "from vllm_ascend.attention.int2.ctx import (" in inspect.getsource(dmod)
