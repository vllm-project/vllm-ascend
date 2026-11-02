# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.ops.triton.kda.recoverssm as recoverssm


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_verify_records_are_independent_of_physical_page_count(monkeypatch, capacity):
    records = recoverssm.KDARecoverSSMRecords.allocate(capacity, 2, 6, 128, torch.bfloat16, torch.device("cpu"))
    q = torch.zeros(1, 8, 2, 128, dtype=torch.bfloat16)
    beta = torch.zeros(1, 8, 2, dtype=q.dtype)
    state = torch.zeros(9, 2, 128, 128)
    starts, pages = torch.tensor([0, 6, 8], dtype=torch.int32), torch.tensor([8, 3], dtype=torch.int32)
    kernel = MagicMock()
    monkeypatch.setattr(recoverssm, "_kda_recoverssm_verify_kernel", kernel)
    monkeypatch.setattr(recoverssm, "is_950", lambda: False)
    monkeypatch.setattr(recoverssm.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(), raising=False)
    monkeypatch.setattr(recoverssm.triton, "cdiv", lambda n, d: (n + d - 1) // d, raising=False)
    args = (
        q,
        q,
        q,
        q,
        beta,
        torch.zeros(2),
        torch.zeros(256),
        -5.0,
        state,
        records.correction,
        records.key_gate,
        starts,
        pages,
        6,
    )
    if capacity < pages.numel():
        with pytest.raises(ValueError, match="every speculative request"):
            recoverssm.kda_recoverssm_verify(*args)
        kernel.__getitem__.assert_not_called()
    else:
        result = recoverssm.kda_recoverssm_verify(*args)
        assert result.shape == q.shape
        assert records.correction.dtype == torch.float32
        assert records.key_gate.dtype == q.dtype
        kernel.__getitem__.assert_called_once_with((2, 2, 2))
        launched = kernel.__getitem__.return_value.call_args.args
        assert launched[8] is records.correction and launched[9] is records.key_gate
        assert launched[12] is pages


@pytest.mark.parametrize("legacy_pages", [False, True])
def test_commit_rejects_missing_layer_records_or_legacy_record_pages(legacy_pages):
    layer = SimpleNamespace(kv_cache=(torch.empty(0),) * (4 if legacy_pages else 2), recoverssm_records=None)
    with pytest.raises(ValueError, match="only conv and SSM state" if legacy_pages else "independent verify records"):
        recoverssm.KDARecoverSSMCommitContext.create([layer], spec_query_len=6, max_num_reqs=2)
