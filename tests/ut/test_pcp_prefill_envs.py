# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_ascend import envs


@pytest.mark.parametrize(
    "name",
    [
        "VLLM_ASCEND_PCP_MAIN_KV_ASYNC",
        "VLLM_ASCEND_PCP_KPOOL_ASYNC",
        "VLLM_ASCEND_PCP_MOE_HIDDEN_EARLY",
        "VLLM_ASCEND_PCP_LOCAL_TOPK",
        "VLLM_ASCEND_KDA_PCP_REDUCE_SCATTER",
    ],
)
@pytest.mark.parametrize("value", [None, "0", "1", "true", "2"])
def test_pcp_prefill_flags_require_explicit_opt_in(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert getattr(envs, name) is (value == "1")
