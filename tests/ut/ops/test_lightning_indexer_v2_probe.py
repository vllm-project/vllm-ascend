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


def _register_ops(library, *, padded=False, metadata_kernel=True, metadata_size=True):
    for name in ("npu_lightning_indexer_v2", "npu_lightning_indexer_v2_metadata"):
        extra = ", Tensor? seqused_q=None" if padded else ""
        if metadata_size and name.endswith("metadata"):
            extra += ", int metadata_size=2048"
        library.define(f"{name}(Tensor q{extra}) -> Tensor")
        if metadata_kernel or not name.endswith("metadata"):
            library.impl(name, lambda q, *args: q, "PrivateUse1")


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


def test_binding_without_seqused_q_can_still_run_unpadded_v2(native_extension, monkeypatch):
    library, ops = native_extension
    _register_ops(library)
    monkeypatch.setattr(ops, "is_lightning_indexer_v2_available", lambda: True, raising=False)
    assert li.has_lightning_indexer_v2()
    assert "seqused_q" in li.get_lightning_indexer_v2_unavailable_reason(require_seqused_q=True)


def test_probe_reports_stale_metadata_binding(native_extension, monkeypatch):
    library, ops = native_extension
    _register_ops(library, padded=True, metadata_size=False)
    monkeypatch.setattr(ops, "is_lightning_indexer_v2_available", lambda: True, raising=False)
    for padded in (False, True):
        reason = li.get_lightning_indexer_v2_unavailable_reason(require_seqused_q=padded)
        assert "metadata_size" in reason and "/wheel/vllm_ascend_C.so" in reason


@pytest.fixture
def metadata_op(native_extension, monkeypatch):
    _, ops = native_extension
    probe = Mock()
    monkeypatch.setattr(ops, "npu_lightning_indexer_v2_metadata", probe, raising=False)
    li.get_lightning_indexer_v2_metadata_size.cache_clear()
    yield probe
    li.get_lightning_indexer_v2_metadata_size.cache_clear()


def _shape_error(required_size):
    return RuntimeError(
        "call aclnnLightningIndexerV2MetadataGetWorkspaceSize failed with status 161002, "
        "detail:[PID: 74476] AclNN_Parameter_Error(EZ1001): "
        f"The element num of metadata must be {required_size}, but got 1024"
    )


@pytest.mark.parametrize("required_size", [2048, 4096])
def test_metadata_size_is_probed_once_per_device(metadata_op, required_size):
    # CPU indices provide distinct device cache keys without an NPU dependency.
    device0, device1 = torch.device("cpu:0"), torch.device("cpu:1")
    metadata_op.side_effect = [
        torch.empty(1024, dtype=torch.int32),
        _shape_error(required_size),
        torch.empty(required_size, dtype=torch.int32),
    ]
    for device, expected in ((device0, 1024), (device1, required_size), (device0, 1024), (device1, required_size)):
        assert li.get_lightning_indexer_v2_metadata_size(device) == expected

    assert [call.kwargs["metadata_size"] for call in metadata_op.call_args_list] == [1024, 1024, required_size]
    for call in metadata_op.call_args_list:
        starts, lengths, heads, topk = call.args[:4]
        torch.testing.assert_close(starts, torch.tensor([0, 1], dtype=torch.int32))
        torch.testing.assert_close(lengths, torch.tensor([1], dtype=torch.int32))
        assert (heads, topk) == (32, 2048)


@pytest.mark.parametrize(
    "message",
    [
        "out of device memory",
        "call aclnnLightningIndexerV2Metadata failed\nThe element num of metadata must be 2048, but got 1024",
        "call aclnnLightningIndexerV2GetWorkspaceSize failed\nThe element num of metadata must be 2048, but got 1024",
        str(_shape_error(1280)),
        "call aclnnLightningIndexerV2MetadataGetWorkspaceSize failed with status 161002, detail:invalid layout\n"
        "TraceBack (most recent call last):\nThe element num of metadata must be 2048, but got 1024",
    ],
)
def test_metadata_size_propagates_other_failures(metadata_op, message):
    error = RuntimeError(message)
    metadata_op.side_effect = error
    with pytest.raises(RuntimeError) as raised:
        li.get_lightning_indexer_v2_metadata_size(torch.device("cpu"))
    assert raised.value is error
    metadata_op.assert_called_once()


def test_metadata_size_retry_failure_is_not_cached(metadata_op):
    retry_error = RuntimeError("failed to launch metadata kernel")
    metadata_op.side_effect = [_shape_error(2048), retry_error, torch.empty(1024, dtype=torch.int32)]
    device = torch.device("cpu")
    with pytest.raises(RuntimeError) as raised:
        li.get_lightning_indexer_v2_metadata_size(device)
    assert raised.value is retry_error
    assert li.get_lightning_indexer_v2_metadata_size(device) == 1024
    assert [call.kwargs["metadata_size"] for call in metadata_op.call_args_list] == [1024, 2048, 1024]
