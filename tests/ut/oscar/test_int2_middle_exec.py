# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the INT2 middle-executor mixin.

The executor's numeric path is device-bound (op input layout, aclnn launcher,
OUT/LSE reception); behavioral evidence is the device contract battery and
the serve parity legs. At module level we pin: the entry point, the explicit
import surface, and the R218 once-counter riding the ctx registry (shared
with the candidate module under the finer split).
"""

import inspect

from vllm_ascend.attention.int2.middle_exec import Int2MiddleExecMixin


def test_exposes_middle_executor_entry_point():
    assert callable(getattr(Int2MiddleExecMixin, "_forward_int2_decode_threetier_middle", None))


def test_module_resolves_shared_state_through_explicit_imports():
    import vllm_ascend.attention.int2.middle_exec as mod

    src = inspect.getsource(mod)
    assert "from vllm_ascend.attention.int2.ctx import (" in src
    assert "from vllm_ascend.attention.int2.helpers import (" in src


def test_midc_once_counter_rides_the_registry():
    import vllm_ascend.attention.int2.middle_exec as mod

    src = inspect.getsource(mod)
    assert '_R2XX_G.setdefault("_R218_MIDC_N"' in src
    assert 'globals().setdefault("_R218_MIDC_N"' not in src
