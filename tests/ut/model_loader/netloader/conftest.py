# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Netloader UT setup.

The dynamic-EPLB guard in netloader.load_model reads the global Ascend config
(``vllm_ascend.ascend_config.get_ascend_config``). UTs never initialize it, so
the guard raised "Ascend config is not initialized" before reaching the code
under test. Provide an EPLB-disabled stand-in, mirroring how the rfork test
conftest handles the same guard.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _stub_ascend_config_for_eplb_guard(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
