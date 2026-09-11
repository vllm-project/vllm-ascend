# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.config as vllm_config_module

import vllm_ascend.platform as platform
from vllm_ascend.device.hardware_profile import AttentionBackendFamily, DeviceAdaptorFamily

# Routing and visible-length checks do not execute the external kernels.
with patch.dict(
    "sys.modules",
    {
        "vllm_ascend.ops.triton.glm5_next_kpool_state_compress": MagicMock(),
        "vllm_ascend.ops.triton.glm5_next_lightning_indexer": MagicMock(),
    },
):
    import vllm_ascend.attention.indexer_kpool_backend as backend_module
    from vllm_ascend.attention.indexer_kpool_backend import Glm5NextKPoolIndexerBackend


@pytest.mark.parametrize("kpool", [False, True])
@pytest.mark.parametrize("fp8_device", [False, True])
def test_sparse_models_select_shared_sfa(monkeypatch, kpool, fp8_device):
    config = SimpleNamespace(index_topk=2048)
    if kpool:
        config.index_kpool = 4
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config)),
    )
    monkeypatch.setattr(
        platform,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(
            attention_backend_family=AttentionBackendFamily.STANDARD,
            device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED if fp8_device else None,
        ),
    )
    monkeypatch.setattr(platform, "_validate_fa3_backend", lambda *args: False)
    selector = SimpleNamespace(use_mla=True, use_sparse=True, use_pcp=False)
    assert platform.NPUPlatform.get_attn_backend_cls(None, selector) == "vllm_ascend.attention.sfa_v1.AscendSFABackend"


@pytest.mark.parametrize("mode", ["310p", "pcp", "dcp"])
def test_kpool_unsupported_routes_fail_explicitly(monkeypatch, mode):
    monkeypatch.setattr(
        backend_module,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(
            attention_backend_family=AttentionBackendFamily.COMPATIBILITY
            if mode == "310p"
            else AttentionBackendFamily.STANDARD
        ),
    )
    source = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=2 if mode == "pcp" else 1,
                decode_context_parallel_size=2 if mode == "dcp" else 1,
            )
        )
    )
    with pytest.raises(NotImplementedError, match="requires Ascend|PCP or DCP"):
        Glm5NextKPoolIndexerBackend(source, qk_rope_head_dim=0)


def _indexer():
    backend = object.__new__(Glm5NextKPoolIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend.topk_tokens, backend.index_kpool = 2048, 4
    return backend


def test_indexer_owns_pool_boundary_lengths():
    indexer = _indexer()
    assert indexer.topk_output_width == 2051
    torch.testing.assert_close(
        indexer.get_topk_lengths(torch.tensor([0, 2, 3, 2047, 2048, 2050, 4095])),
        torch.tensor([1, 3, 4, 2048, 2049, 2051, 2048]),
    )
