# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the INT2 middle candidate mixin.

The candidate's numeric path is device-bound (NPU tensors, aclnn launcher);
behavioral evidence is the device contract battery and the serve parity legs.
At module level we pin: the entry point, the explicit import surface, and the
R218 once-counter riding the ctx registry (the finer split shares it with the
executor — bare module globals would silently isolate it).
"""

import inspect

from vllm_ascend.attention.int2.middle_candidate import Int2MiddleCandidateMixin


def test_exposes_candidate_entry_point():
    assert callable(getattr(Int2MiddleCandidateMixin, "_r109_middle_candidate", None))


def test_module_resolves_shared_state_through_explicit_imports():
    import vllm_ascend.attention.int2.middle_candidate as mod

    src = inspect.getsource(mod)
    assert "from vllm_ascend.attention.int2.ctx import (" in src
    assert "from vllm_ascend.attention.int2.helpers import (" in src


def test_midc_once_counter_rides_the_registry():
    # _R218_MIDC_N is touched by BOTH the candidate and the executor; under
    # the module split only the ctx registry keeps them consistent
    import vllm_ascend.attention.int2.middle_candidate as mod

    src = inspect.getsource(mod)
    assert '_R2XX_G.setdefault("_R218_MIDC_N"' in src
    assert 'globals().setdefault("_R218_MIDC_N"' not in src
