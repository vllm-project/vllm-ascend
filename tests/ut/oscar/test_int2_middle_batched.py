# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the INT2 batched middle-operand mixin.

The batched S+R builder's numeric path is device-bound; behavioral evidence
is the device contract battery and the serve parity legs. At module level we
pin the entry points and the explicit import surface.
"""

import inspect

from vllm_ascend.attention.int2.middle_batched import Int2MiddleBatchedMixin


def test_exposes_batched_sr_entry_points():
    for m in ("_r196_batched_sr", "_r164_midprobe", "_r197_probe_record"):
        assert callable(getattr(Int2MiddleBatchedMixin, m, None)), m


def test_module_resolves_shared_state_through_explicit_imports():
    import vllm_ascend.attention.int2.middle_batched as mod

    src = inspect.getsource(mod)
    assert "from vllm_ascend.attention.int2.ctx import (" in src
