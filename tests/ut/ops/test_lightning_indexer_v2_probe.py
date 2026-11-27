# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import lightning_indexer_v2 as li


@pytest.fixture
def native_extension(monkeypatch):
    # Use a separate namespace so a CPU worker exercises PyTorch's real schema
    # and dispatch-key lookup without touching the installed Ascend extension.
    namespace = "test_bf16_li_v2_probe"
    library = torch.library.Library(namespace, "FRAGMENT")
    ops = torch.ops.__getattr__(namespace)
    monkeypatch.setattr(torch.ops, "_C_ascend", ops)
    monkeypatch.setattr(li, "import_module", Mock(return_value=SimpleNamespace(__file__="/wheel/vllm_ascend_C.so")))
    li.get_lightning_indexer_v2_unavailable_reason.cache_clear()
    yield library, ops
    li.get_lightning_indexer_v2_unavailable_reason.cache_clear()


def _register_ops(library, *, padded=False, metadata_kernel=True):
    for name in ("npu_lightning_indexer_v2", "npu_lightning_indexer_v2_metadata"):
        extra = ", Tensor? seqused_q=None" if padded else ""
        library.define(f"{name}(Tensor q{extra}) -> Tensor")
        if metadata_kernel or not name.endswith("metadata"):
            library.impl(name, lambda q, seqused_q=None: q, "PrivateUse1")


def test_probe_reports_unloadable_library(native_extension, monkeypatch):
    monkeypatch.setattr(li, "find_spec", lambda name: SimpleNamespace(origin="/wheel/vllm_ascend_C.so"))
    li.import_module.side_effect = ImportError("missing libopapi.so")
    assert (
        li.get_lightning_indexer_v2_unavailable_reason()
        == "failed to load /wheel/vllm_ascend_C.so: missing libopapi.so"
    )


def test_probe_reports_missing_binding(native_extension):
    reason = li.get_lightning_indexer_v2_unavailable_reason()
    assert "/wheel/vllm_ascend_C.so" in reason and "_C_ascend::npu_lightning_indexer_v2" in reason


def test_probe_reports_missing_metadata_npu_kernel(native_extension):
    library, _ = native_extension
    _register_ops(library, metadata_kernel=False)
    reason = li.get_lightning_indexer_v2_unavailable_reason()
    assert "no NPU kernel" in reason and "_C_ascend::npu_lightning_indexer_v2_metadata" in reason


@pytest.mark.parametrize(
    "missing", ["", "aclnnLightningIndexerV2MetadataGetWorkspaceSize, aclnnLightningIndexerV2Metadata"]
)
def test_probe_uses_native_symbol_diagnostics(native_extension, monkeypatch, missing):
    library, ops = native_extension
    _register_ops(library, padded=True)
    probe = Mock(return_value=missing)
    monkeypatch.setattr(ops, "get_lightning_indexer_v2_missing_symbols", probe, raising=False)
    log = Mock()
    monkeypatch.setattr(li.logger, "info_once", log)
    for _ in range(2):
        reason = li.get_lightning_indexer_v2_unavailable_reason(require_seqused_q=True)
        if missing:
            assert missing in reason and "/wheel/vllm_ascend_C.so" in reason
        else:
            assert reason is None
    probe.assert_called_once_with()
    if missing:
        log.assert_not_called()
    else:
        log.assert_called_once()
        assert "stage=probe" in log.call_args.args[0] and log.call_args.args[1] == "/wheel/vllm_ascend_C.so"


def test_old_extension_can_still_run_unpadded_v2(native_extension, monkeypatch):
    library, ops = native_extension
    _register_ops(library)
    monkeypatch.setattr(ops, "is_lightning_indexer_v2_available", lambda: True, raising=False)
    assert li.has_lightning_indexer_v2()
    assert "seqused_q" in li.get_lightning_indexer_v2_unavailable_reason(require_seqused_q=True)
