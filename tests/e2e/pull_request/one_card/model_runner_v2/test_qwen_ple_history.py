# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState
from vllm_ascend.patch.worker import patch_uva


@pytest.mark.parametrize("mapped", [False, True])
def test_qwen_ple_history_on_npu(monkeypatch, mapped):
    if mapped:
        monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "pinned_mem_register:True")
    else:
        monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    history = patch_uva.UvaBufferWrapper((4, 16), torch.int32)
    if mapped and not history._use_real_uva:
        pytest.skip("Mapped host memory unavailable")
    for row in range(4):
        history.cpu[row] = torch.arange(16, dtype=torch.int32) + 100 * row
    tokens = history.uva()
    device = tokens.device
    state = object.__new__(AscendQwen4ExpModelState)
    state.ngram_eos_token_id = 999
    state.ngram_context_offsets = torch.arange(-2, 0, device=device)
    state.ngram_context = torch.full((4, 2), -1, dtype=torch.int32, device=device)
    requests = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=tokens),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([1, 0, 0, 5], device=device)),
    )
    batch = SimpleNamespace(num_reqs=3, num_reqs_after_padding=4, idx_mapping=torch.tensor([3, 0, 2], device=device))
    result = state._prepare_ngram_context(batch, requests)
    assert result.tolist() == [[303, 304], [999, 0], [999, 999], [999, 999]]
    assert result.data_ptr() == state.ngram_context.data_ptr()
    batch.num_reqs = 0
    assert state._prepare_ngram_context(batch, requests).tolist() == [[999, 999]] * 4
