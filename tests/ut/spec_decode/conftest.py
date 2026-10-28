# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _stub_ascend_config(monkeypatch):
    # Bare-speculator harnesses reach lmhead TP gates (the DSpark draft-logits
    # wrap) that read the global ascend config; stub it as lmhead-off so the
    # gates no-op instead of raising on the uninitialized config read.
    from vllm_ascend import ascend_config as _ascend_config_module

    monkeypatch.setattr(
        _ascend_config_module,
        "_ASCEND_CONFIG",
        SimpleNamespace(
            finegrained_tp_config=SimpleNamespace(lmhead_tensor_parallel_size=0),
            ascend_compilation_config=object(),
            eplb_config=object(),
            draft_window_size=None,
        ),
    )
